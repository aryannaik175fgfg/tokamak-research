"""Shared Pre-LN transformer with MHD, hazard, and profile heads."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tokenizers import (
    ChannelIndependentTimeTokenizer,
    MachRegimeConditioner,
    MHD3DPatchTokenizer,
    ReactorGeometryConditioner,
)


def apply_rope(x: torch.Tensor, pos: torch.Tensor, d_head: int) -> torch.Tensor:
    """
  Lightweight RoPE on last dim using position_ids (B, N, 4).

  x: (B, N, D)
  """
    _b, _n, d = x.shape
    half = d // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, 2, device=x.device, dtype=x.dtype) / max(half, 1)))
    angles = pos[..., :1].float() * freqs.view(1, 1, -1)  # (B, N, n_freq)
    if angles.shape[-1] < half:
        angles = F.pad(angles, (0, half - angles.shape[-1]))
    else:
        angles = angles[..., :half]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half : half * 2]
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if d > half * 2:
        rot = torch.cat([rot, x[..., half * 2 :]], dim=-1)
    return rot


class PreLNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        if rope_pos is not None and rope_pos.shape[1] == h.shape[1]:
            h = apply_rope(h, rope_pos, h.shape[-1] // max(1, self.attn.num_heads))
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MHDDecodeHead(nn.Module):
    """
  Decode token lattice to a 3D field via trilinear upsample + light Conv3d.

  Avoids stacked ConvTranspose3d (which can allocate tens of GB at 64^3).
  """

    def __init__(self, d_model: int, out_channels: int = 7, grid: int = 4) -> None:
        super().__init__()
        self.grid = grid
        self.proj = nn.Linear(d_model, d_model)
        self.post = nn.Sequential(
            nn.Conv3d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(d_model // 2, out_channels, kernel_size=1),
        )
        self.default_spatial = 64

    def forward(
        self,
        tokens: torch.Tensor,
        n_spatial: int,
        target_size: int | tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """tokens (B, N, D) -> (B, 1, C, Z, Y, X)."""
        b, n, d = tokens.shape
        g = 2
        while g**3 < n and g < 16:
            g += 1
        if g**3 > n:
            tokens = F.pad(tokens, (0, 0, 0, g**3 - n))
        h = self.proj(tokens).permute(0, 2, 1).reshape(b, d, g, g, g)
        if target_size is None:
            if isinstance(n_spatial, int) and n_spatial > 0:
                target_size = (n_spatial, n_spatial, n_spatial)
            else:
                target_size = (self.default_spatial,) * 3
        elif isinstance(target_size, int):
            target_size = (target_size, target_size, target_size)
        h = F.interpolate(h, size=target_size, mode="trilinear", align_corners=False)
        y = self.post(h)
        return y.unsqueeze(1)  # (B, 1, C, Z, Y, X)


class FusionTransformer(nn.Module):
    """
  Multi-task encoder with MHD next-step, disruption hazard, and profile heads.
  """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_len: int = 8,
        tubelet: tuple[int, int, int, int] = (1, 16, 16, 16),
        n_reactor_ctx: int = 4,
        n_mach_ctx: int = 2,
        n_rho: int = 32,
        n_horizons: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mhd_tok = MHD3DPatchTokenizer(d_model=d_model, tubelet=tubelet)
        self.time_tok = ChannelIndependentTimeTokenizer(d_model=d_model, patch_len=patch_len)
        self.reactor_cond = ReactorGeometryConditioner(d_model=d_model, n_ctx=n_reactor_ctx)
        self.mach_cond = MachRegimeConditioner(d_model=d_model, n_ctx=n_mach_ctx)
        self.blocks = nn.ModuleList([PreLNBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.mhd_head = MHDDecodeHead(d_model)
        self.hazard_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_horizons))
        self.ip_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.profile_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_rho * 2))
        self.n_rho = n_rho
        self.delta_t_enc = nn.Linear(1, d_model)

    def _encode(
        self,
        tokens: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_pos: Optional[torch.Tensor] = None,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ctx is not None:
            tokens = torch.cat([ctx, tokens], dim=1)
            if key_padding_mask is not None:
                false_ctx = torch.zeros(ctx.shape[0], ctx.shape[1], dtype=torch.bool, device=tokens.device)
                key_padding_mask = torch.cat([false_ctx, key_padding_mask], dim=1)
        if delta_t is not None:
            dt_emb = self.delta_t_enc(delta_t.unsqueeze(-1)).unsqueeze(1)
            tokens = tokens + dt_emb
        x = tokens
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask, rope_pos=rope_pos)
        return x

    def forward_mhd(
        self,
        x: torch.Tensor,
        m_s: torch.Tensor,
        m_a: torch.Tensor,
    ) -> torch.Tensor:
        """
    x: (B, T, 7, Z, Y, X)
    Returns (B, 1, 7, 64, 64, 64)
    """
        tok, pos = self.mhd_tok(x)
        ctx = self.mach_cond(m_s, m_a)
        enc = self._encode(tok, ctx=ctx, rope_pos=pos)
        spatial = enc[:, ctx.shape[1] :, :]
        z = x.shape[-3]
        return self.mhd_head(spatial, spatial.shape[1], target_size=z)

    def forward_tokamak(
        self,
        patches: torch.Tensor,
        quantity_ids: torch.Tensor,
        valid: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
        geo_mask: Optional[torch.Tensor] = None,
        use_geometry: bool = True,
        delta_t_ms: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
    Returns hazard_logits (B, 4), profiles (B, n_rho, 2)
    """
        tok, kpm = self.time_tok(patches, quantity_ids, valid)
        ctx = None
        if use_geometry and geometry is not None:
            gm = geo_mask if geo_mask is not None else torch.ones_like(geometry)
            ctx = self.reactor_cond(geometry, gm)
        dt = delta_t_ms
        enc = self._encode(tok, ctx=ctx, key_padding_mask=kpm, delta_t=dt)
        if ctx is not None:
            pooled = enc[:, ctx.shape[1] :, :].mean(dim=1)
            prof_pool = enc[:, : ctx.shape[1], :].mean(dim=1)
        else:
            pooled = enc.mean(dim=1)
            prof_pool = pooled
        hazard = self.hazard_head(pooled)
        prof_flat = self.profile_head(prof_pool)
        prof = prof_flat.view(-1, self.n_rho, 2)
        return hazard, prof

    def forward_ip(
        self,
        patches: torch.Tensor,
        quantity_ids: torch.Tensor,
        valid: torch.Tensor,
        geometry: Optional[torch.Tensor] = None,
        geo_mask: Optional[torch.Tensor] = None,
        use_geometry: bool = True,
    ) -> torch.Tensor:
        """Plasma current regression (B,) from multi-patch diagnostic tokens."""
        tok, kpm = self.time_tok(patches, quantity_ids, valid)
        ctx = None
        if use_geometry and geometry is not None:
            gm = geo_mask if geo_mask is not None else torch.ones_like(geometry)
            ctx = self.reactor_cond(geometry, gm)
        enc = self._encode(tok, ctx=ctx, key_padding_mask=kpm)
        if ctx is not None:
            pooled = enc[:, ctx.shape[1] :, :].mean(dim=1)
        else:
            pooled = enc.mean(dim=1)
        return self.ip_head(pooled).squeeze(-1)

    def load_encoder_from(self, other: "FusionTransformer") -> None:
        """Copy transformer block weights from a stage-1 checkpoint."""
        self.blocks.load_state_dict(other.blocks.state_dict())

    def encoder_parameters(self):
        """Parameters for reduced LR in stage 2."""
        return self.blocks.parameters()