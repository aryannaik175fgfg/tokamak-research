"""Thermal-quench onset and current-quench timing for MAST-style labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DisruptionTiming:
    """Per-shot disruption annotation."""

    t_disrupt_s: Optional[float]  # thermal quench onset; None if non-disruptive
    tau_80_to_20_s: Optional[float]  # current-quench duration when defined
    disruptive: bool


def _median_flattop_slope(times: np.ndarray, ip: np.ndarray) -> float:
    """Median |dIp/dt| on the flattop (central 50% of shot duration)."""
    if len(times) < 3:
        return float(np.nanmedian(np.abs(np.diff(ip) / np.diff(times))))
    i0, i1 = len(times) // 4, (3 * len(times)) // 4
    dt = np.diff(times[i0:i1])
    dip = np.diff(ip[i0:i1])
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.abs(dip / dt)
    slopes = slopes[np.isfinite(slopes)]
    return float(np.median(slopes)) if len(slopes) else 0.0


def detect_sxr_thermal_quench(
    times: np.ndarray,
    sxr: np.ndarray,
    drop_frac: float = 0.5,
    window_s: float = 0.002,
) -> Optional[float]:
    """
  Detect thermal quench from soft-X-ray collapse (>50% drop in 2 ms).

  Returns onset time in seconds or None.
  """
    if len(times) < 2:
        return None
    baseline = float(np.nanmax(sxr[: max(1, len(sxr) // 4)]))
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    threshold = (1.0 - drop_frac) * baseline
    for i in range(1, len(times)):
        if not np.isfinite(sxr[i]):
            continue
        if sxr[i] < threshold:
            dt = times[i] - times[i - 1]
            if dt <= window_s or (times[i] - times[max(0, i - 5)]) <= window_s:
                return float(times[i])
    return None


def detect_current_quench_onset(
    times: np.ndarray,
    ip: np.ndarray,
    quench_window_s: float = 0.015,
    slope_factor: float = 5.0,
) -> tuple[Optional[float], Optional[float]]:
    """
  ITER-style current quench: 80% flattop crossing with 20% within 15 ms.

  Slow ramp-down returns (None, None).
  """
    if len(times) < 4:
        return None, None
    ip_max = float(np.nanmax(ip))
    if not np.isfinite(ip_max) or ip_max <= 0:
        return None, None
    level_80 = 0.8 * ip_max
    level_20 = 0.2 * ip_max
    t_80: Optional[float] = None
    t_20: Optional[float] = None
    for i, val in enumerate(ip):
        if t_80 is None and np.isfinite(val) and val <= level_80:
            t_80 = float(times[i])
        if t_80 is not None and np.isfinite(val) and val <= level_20:
            t_20 = float(times[i])
            break
    if t_80 is None or t_20 is None:
        return None, None
    if t_20 - t_80 > quench_window_s:
        return None, None
    flattop_slope = _median_flattop_slope(times, ip)
    mask = (times >= t_80) & (times <= t_20)
    if mask.sum() < 2:
        return None, None
    dt = np.diff(times[mask])
    dip = np.diff(ip[mask])
    with np.errstate(divide="ignore", invalid="ignore"):
        quench_slope = float(np.nanmedian(np.abs(dip / dt)))
    if flattop_slope > 0 and quench_slope < slope_factor * flattop_slope:
        return None, None
    return t_80, t_20 - t_80


def detect_vde_runaway(
    times: np.ndarray,
    z: np.ndarray,
    z_flattop: float,
    threshold: float = 0.05,
) -> Optional[float]:
    """Earliest time |z - z_flattop| exceeds threshold (normalized displacement)."""
    for i, zi in enumerate(z):
        if np.isfinite(zi) and abs(zi - z_flattop) > threshold:
            return float(times[i])
    return None


def annotate_mast_shot(
    times: np.ndarray,
    ip: np.ndarray,
    sxr: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    z_flattop: float = 0.0,
) -> DisruptionTiming:
    """
  Combine SXR thermal quench, current quench, and VDE runaway into one t_D.
  """
    t_d: Optional[float] = None
    tau_q: Optional[float] = None
    if sxr is not None:
        t_d = detect_sxr_thermal_quench(times, sxr)
    if t_d is None:
        t_80, tau = detect_current_quench_onset(times, ip)
        if t_80 is not None:
            t_d = t_80
            tau_q = tau
    if z is not None:
        t_vde = detect_vde_runaway(times, z, z_flattop)
        if t_vde is not None and (t_d is None or t_vde < t_d):
            t_d = t_vde
    disruptive = t_d is not None
    return DisruptionTiming(t_disrupt_s=t_d, tau_80_to_20_s=tau_q, disruptive=disruptive)


def horizon_labels(
    t_end_s: float,
    t_disrupt_s: Optional[float],
    horizons_ms: tuple[int, ...] = (10, 25, 50, 100),
) -> np.ndarray:
    """
  Binary labels per horizon: 1 if 0 < t_D - t_end <= tau (seconds).

  Shape: (H,)
  """
    labels = np.zeros(len(horizons_ms), dtype=np.float32)
    if t_disrupt_s is None:
        return labels
    lead_s = t_disrupt_s - t_end_s
    if lead_s <= 0:
        return labels
    for h, tau_ms in enumerate(horizons_ms):
        if lead_s <= tau_ms * 1e-3:
            labels[h] = 1.0
    return labels