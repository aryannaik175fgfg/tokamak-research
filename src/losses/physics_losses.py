"""Physics-informed losses for MHD pretraining and disruption forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def periodic_central_diff(f: torch.Tensor, dim: int, dx: float) -> torch.Tensor:
    """
  Second-order central difference along ``dim`` with periodic boundaries.

  f: arbitrary shape; diff shrinks that dimension by 0 (same shape via roll).
  """
    f_plus = torch.roll(f, shifts=-1, dims=dim)
    f_minus = torch.roll(f, shifts=1, dims=dim)
    return (f_plus - f_minus) / (2.0 * dx)


def divergence_b(
    b: torch.Tensor,
    dx: float = 1.0,
) -> torch.Tensor:
    """
  div B on grid ``b`` with shape (..., 3, Z, Y, X) or (..., 7, Z, Y, X) using channels 1:4.

  Returns divergence field (..., Z, Y, X).
  """
    if b.shape[-4] >= 7:
        bx, by, bz = b[:, 1], b[:, 2], b[:, 3]
    else:
        bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    # (B, Z, Y, X)
    if bx.dim() == 4:
        d_x = periodic_central_diff(bx, dim=3, dx=dx)
        d_y = periodic_central_diff(by, dim=2, dx=dx)
        d_z = periodic_central_diff(bz, dim=1, dx=dx)
    else:
        d_x = periodic_central_diff(bx, dim=-1, dx=dx)
        d_y = periodic_central_diff(by, dim=-2, dx=dx)
        d_z = periodic_central_diff(bz, dim=-3, dx=dx)
    return d_x + d_y + d_z


def divergence_free_magnetic_loss(
    b_pred: torch.Tensor,
    dx: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
  Dimensionless mean(div^2) normalized by <B^2>/dx^2.

  b_pred: (B, 3, Z, Y, X) magnetic field only.
  """
    if b_pred.shape[1] != 3:
        b_pred = b_pred[:, 1:4]
    padded = torch.zeros(b_pred.shape[0], 7, *b_pred.shape[2:], device=b_pred.device, dtype=b_pred.dtype)
    padded[:, 1:4] = b_pred
    div = divergence_b(padded, dx=dx)
    b2 = (b_pred**2).mean()
    return (div**2).mean() / (b2 / (dx**2) + eps)


def vrmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_var: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
  Well-style VRMSE: mean_c sqrt(MSE_c / (sigma_c^2 + eps)).

  pred, target: (B, C, ...)
  channel_var: (C,) detached training-set variances.
  """
    c = pred.shape[1]
    losses = []
    for i in range(c):
        diff2 = ((pred[:, i] - target[:, i]) ** 2).mean()
        losses.append(torch.sqrt(diff2 / (channel_var[i] + eps)))
    return torch.stack(losses).mean()


def continuity_residual_loss(
    rho_pred: torch.Tensor,
    rho_t: torch.Tensor,
    v_t: torch.Tensor,
    dt: float,
    dx: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
  ((rho_pred - rho_t)/dt + div(rho_t * v_t))^2 normalized by observed density change scale.

  rho_pred, rho_t: (B, Z, Y, X); v_t: (B, 3, Z, Y, X). Flux terms detached via caller.
  """
    drho_dt = (rho_pred - rho_t) / dt
    flux_x = rho_t * v_t[:, 0]
    flux_y = rho_t * v_t[:, 1]
    flux_z = rho_t * v_t[:, 2]
    div_flux = (
        periodic_central_diff(flux_x, dim=3, dx=dx)
        + periodic_central_diff(flux_y, dim=2, dx=dx)
        + periodic_central_diff(flux_z, dim=1, dx=dx)
    )
    resid = drho_dt + div_flux
    scale = ((rho_t - rho_t.mean(dim=(1, 2, 3), keepdim=True)) ** 2).mean() + eps
    return (resid**2).mean() / scale


def curl_v_cross_b(
    v: torch.Tensor,
    b: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """curl(v x B) for v, b shaped (B, 3, Z, Y, X)."""
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    cx = vy * bz - vz * by
    cy = vz * bx - vx * bz
    cz = vx * by - vy * bx
    curl_x = periodic_central_diff(cz, dim=2, dx=dx) - periodic_central_diff(cy, dim=1, dx=dx)
    curl_y = periodic_central_diff(cx, dim=1, dx=dx) - periodic_central_diff(cz, dim=3, dx=dx)
    curl_z = periodic_central_diff(cy, dim=3, dx=dx) - periodic_central_diff(cx, dim=2, dx=dx)
    return torch.stack([curl_x, curl_y, curl_z], dim=1)


def induction_residual_loss(
    b_pred: torch.Tensor,
    b_t: torch.Tensor,
    v_t: torch.Tensor,
    dt: float,
    dx: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """((B_pred - B_t)/dt - curl(v_t x B_t))^2 normalized by |dB|^2 scale."""
    db_dt = (b_pred - b_t) / dt
    curl_vxb = curl_v_cross_b(v_t.detach(), b_t.detach(), dx)
    resid = db_dt - curl_vxb
    scale = ((b_t - b_t.mean(dim=(2, 3, 4), keepdim=True)) ** 2).mean() + eps
    return (resid**2).mean() / scale


def mass_sum_mismatch_log(
    rho_pred: torch.Tensor,
    rho_t: torch.Tensor,
) -> torch.Tensor:
    """Logged global mass mismatch (not in main loss)."""
    return ((rho_pred.sum(dim=(1, 2, 3)) - rho_t.sum(dim=(1, 2, 3))) ** 2).mean()


def focal_hazard_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    lead_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
  Time-weighted focal loss over horizons.

  logits: (B, H); targets: (B, H) in {0,1}.
  lead_weights: (B,) detached multipliers on focal term only.
  """
    probs = torch.sigmoid(logits)
    b, h = logits.shape
    losses = []
    for hi in range(h):
        p = probs[:, hi]
        t = targets[:, hi]
        p_t = torch.where(t > 0.5, p, 1.0 - p)
        alpha_t = torch.where(t > 0.5, alpha, 1.0 - alpha)
        fl = -alpha_t * (1.0 - p_t) ** gamma * torch.log(p_t.clamp(min=1e-6))
        if lead_weights is not None:
            fl = fl * lead_weights
        losses.append(fl.mean())
    return torch.stack(losses).mean()


def monotonic_hazard_loss(probs: torch.Tensor) -> torch.Tensor:
    """
  sum_h max(0, p_h - p_{h+1})^2 for ordered horizons.

  probs: (B, H)
  """
    if probs.shape[1] < 2:
        return probs.new_zeros(())
    viol = F.relu(probs[:, :-1] - probs[:, 1:])
    return (viol**2).sum(dim=1).mean()


def masked_profile_loss(
    pred_te: torch.Tensor,
    pred_ne: torch.Tensor,
    te: torch.Tensor,
    ne: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
  MSE on Te and Ne with mask (B, N_rho, 2).

  pred_* : (B, N_rho)
  """
    m_te = mask[..., 0]
    m_ne = mask[..., 1]
    err_te = m_te * (pred_te - te) ** 2
    err_ne = m_ne * (pred_ne - ne) ** 2
    denom = m_te.sum() + m_ne.sum() + eps
    return (err_te.sum() + err_ne.sum()) / denom


def lead_time_weight(
    t_end: torch.Tensor,
    t_disrupt: torch.Tensor | None,
    disruptive: torch.Tensor,
    ref_ms: float = 20.0,
    tau_ms: float = 50.0,
    clip: tuple[float, float] = (0.05, 5.0),
) -> torch.Tensor:
    """
  w(t) = exp((ref_ms - (t_D - t)*1e3)/tau_ms) for disruptive windows, else 1.

  t_end, t_disrupt in seconds; disruptive mask (B,).
  """
    w = torch.ones_like(t_end)
    if t_disrupt is None:
        return w
    lead_ms = (t_disrupt - t_end) * 1000.0
    wd = torch.exp((ref_ms - lead_ms) / tau_ms)
    wd = wd.clamp(clip[0], clip[1])
    w = torch.where(disruptive > 0.5, wd, w)
    return w.detach()