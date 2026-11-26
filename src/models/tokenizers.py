"""MHD tubelet, PatchTST diagnostics, and geometry / Mach conditioners."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.diagnostics import PHYSICAL_QUANTITIES


class MHD3DPatchTokenizer(nn.Module):
    """
  3D spatiotemporal tubelet embedding: (B, T, C, Z, Y, X) -> (B, N, D).

  Uses Conv3d with tubelet size (T_t, Z_t, Y_t, X_t).
  """

    def __init__(
        self,
        in_channels: int = 7,
        d_model: int = 256,
        tubelet: tuple[int, int, int, int] = (1, 16, 16, 16),
    ) -> None:
        super().__init__()
        _t_t, z_t, y_t, x_t = tubelet
        self.tubelet = tubelet
        # Spatial tubelets only; time is folded outside Conv3d.
        self.proj = nn.Conv3d(in_channels, d_model, kernel_size=(z_t, y_t, x_t), stride=(z_t, y_t, x_t))
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, C, Z, Y, X) -> tokens (B, N, D), position_ids (B, N, 4)."""
        b, t, _c, _z, _y, _x = x.shape
        chunks: list[torch.Tensor] = []
        grid_shape: tuple[int, int, int] = (1, 1, 1)
        for ti in range(t):
            frame = x[:, ti]  # (B, C, Z, Y, X)
            emb = self.proj(frame)  # (B, D, Z', Y', X')
            grid_shape = emb.shape[-3:]
            chunks.append(emb.flatten(2).transpose(1, 2))
        tokens = torch.cat(chunks, dim=1)  # (B, N, D)
        n = tokens.shape[1]
        zp, yp, xp = grid_shape
        coords = torch.arange(n, device=x.device)
        pos = torch.stack(
            [
                coords // max(1, zp * yp * xp),
                (coords // max(1, yp * xp)) % max(1, zp),
                (coords // max(1, xp)) % max(1, yp),
                coords % max(1, xp),
            ],
            dim=-1,
        ).unsqueeze(0).expand(b, -1, -1)
        return tokens, pos


class ChannelIndependentTimeTokenizer(nn.Module):
    """
  PatchTST-style 1D tokenizer with validity and physical-quantity embeddings.

  values: (B, L, C) or padded (B, N_ch, patch_len)
  """

    def __init__(
        self,
        d_model: int = 256,
        patch_len: int = 8,
        n_quantities: int = len(PHYSICAL_QUANTITIES),
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.quantity_embed = nn.Embedding(n_quantities, d_model)
        self.valid_embed = nn.Embedding(2, d_model)

    def forward(
        self,
        patches: torch.Tensor,
        quantity_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
    patches: (B, N_ch, patch_len)
    quantity_ids: (B, N_ch)
    valid: (B, N_ch) in {0,1}
    Returns tokens (B, N_ch, D), key_padding_mask (B, N_ch) True=ignore
    """
        b, n_ch, pl = patches.shape
        tok = self.patch_proj(patches)  # (B, N_ch, D)
        tok = tok + self.quantity_embed(quantity_ids.clamp(0, self.quantity_embed.num_embeddings - 1))
        v_idx = valid.long().clamp(0, 1)
        tok = tok + self.valid_embed(v_idx)
        key_pad = valid < 0.5
        return tok, key_pad


class FourierFeatureMLP(nn.Module):
    """Fourier features sin/cos(2^k pi x) then MLP to tokens."""

    def __init__(self, in_dim: int, d_model: int, n_ctx: int, n_freq: int = 6) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.in_proj = nn.Linear(in_dim * 2 * n_freq, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.ctx_queries = nn.Parameter(torch.randn(n_ctx, d_model) * 0.02)
        self.n_ctx = n_ctx

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, in_dim) in [0,1]; mask (B, in_dim) zeros invalid coords."""
        b, d = x.shape
        if mask is not None:
            x = x * mask
        freqs = 2.0 ** torch.arange(self.n_freq, device=x.device, dtype=x.dtype)
        feats = []
        for f in freqs:
            feats.append(torch.sin(math.pi * f * x))
            feats.append(torch.cos(math.pi * f * x))
        h = torch.cat(feats, dim=-1)  # (B, d * 2 * n_freq)
        h = self.in_proj(h)  # (B, D)
        h = self.mlp(h)
        ctx = self.ctx_queries.unsqueeze(0).expand(b, -1, -1) + h.unsqueeze(1)
        return ctx  # (B, n_ctx, D)


class ReactorGeometryConditioner(FourierFeatureMLP):
    """Maps 6-D dimensionless reactor coordinates to [REACTOR_CTX] tokens."""

    def __init__(self, d_model: int = 256, n_ctx: int = 4) -> None:
        super().__init__(in_dim=6, d_model=d_model, n_ctx=n_ctx)

    def forward(self, geometry: torch.Tensor, geo_mask: torch.Tensor) -> torch.Tensor:
        """geometry (B, 6), geo_mask (B, 6)."""
        if geometry.dim() == 1:
            geometry = geometry.unsqueeze(0)
        if geo_mask.dim() == 1:
            geo_mask = geo_mask.unsqueeze(0)
        return self.encode(geometry, mask=geo_mask)


class MachRegimeConditioner(nn.Module):
    """Separate MLP on (M_s, M_A) for stage 1 — not weight-shared with reactor conditioner."""

    def __init__(self, d_model: int = 256, n_ctx: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.tokens = nn.Parameter(torch.randn(n_ctx, d_model) * 0.02)

    def forward(self, m_s: torch.Tensor, m_a: torch.Tensor) -> torch.Tensor:
        """m_s, m_a: (B, 1) -> (B, n_ctx, D)."""
        if m_s.dim() == 1:
            m_s = m_s.unsqueeze(-1)
        if m_a.dim() == 1:
            m_a = m_a.unsqueeze(-1)
        reg = torch.cat([m_s, m_a], dim=-1)  # (B, 2)
        h = self.net(reg)
        return self.tokens.unsqueeze(0).expand(reg.shape[0], -1, -1) + h.unsqueeze(1)