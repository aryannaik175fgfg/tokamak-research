"""ITPA DB5.2.3 globals and PR08 UFILE profile loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MISSING_MARKERS = {-9.999e-09, 9.999e-09, -9999999, 9.999e09}


def _is_missing(val: float) -> bool:
    if not np.isfinite(val):
        return True
    for m in MISSING_MARKERS:
        if abs(val - m) < abs(m) * 1e-3 + 1e-12:
            return True
    return False


def load_itpa_globals(path: str | Path) -> pd.DataFrame:
    """Load HDB5V2.3 xlsx/csv with standard column names uppercased."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def q_cylindrical(a_m: float, r_m: float, bt_t: float, ip_ma: float, kappa: float) -> float:
    """
  Cylindrical q estimate when Q95 is missing.

  q_cyl = 5 a^2 B_t (1+kappa^2) / (2 R I_p) with a,R in m, B_t in T, I_p in MA.
  """
    if _is_missing(a_m) or _is_missing(r_m) or _is_missing(bt_t) or _is_missing(ip_ma):
        return float("nan")
    if abs(ip_ma) < 1e-6:
        return float("nan")
    return 5.0 * a_m**2 * abs(bt_t) * (1.0 + kappa**2) / (2.0 * r_m * abs(ip_ma))


def row_to_dimensionless_geometry(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
  Six dimensionless conditioner coordinates and a validity mask (1=valid).

  Returns (vec shape (6,), mask shape (6,)).
  """
    r = float(row.get("RGEO", np.nan))
    a = float(row.get("AMIN", np.nan))
    bt = float(row.get("BT", np.nan))
    ip = float(row.get("IP", np.nan))
    kappa = float(row.get("KAREA", np.nan))
    delta = float(row.get("DELTA", np.nan))
    q95 = float(row.get("Q95", np.nan))

    mask = np.ones(6, dtype=np.float32)
    vec = np.zeros(6, dtype=np.float32)

    if _is_missing(a) or _is_missing(r) or a <= 0:
        mask[:] = 0.0
        return vec, mask

    vec[0] = r / a  # aspect ratio
    vec[1] = kappa if not _is_missing(kappa) else np.nan
    vec[2] = delta if not _is_missing(delta) else np.nan
    if not _is_missing(q95):
        vec[3] = q95
    else:
        qc = q_cylindrical(a, r, bt, ip, kappa if not _is_missing(kappa) else 1.0)
        vec[3] = qc
    if _is_missing(bt) or _is_missing(ip):
        vec[4] = np.nan
    else:
        vec[4] = abs(ip) / (a * abs(bt))  # I_N
    vec[5] = a  # meters; scaled later

    for i in range(6):
        if not np.isfinite(vec[i]):
            mask[i] = 0.0
    return vec, mask


def fit_geometry_scaler(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Min-max on training rows for coordinates with valid mask."""
    mats = []
    for _, row in rows.iterrows():
        v, m = row_to_dimensionless_geometry(row)
        if m.sum() > 0:
            mats.append(v)
    if not mats:
        lo = np.zeros(6)
        hi = np.ones(6)
    else:
        arr = np.stack(mats)
        lo = np.nanmin(arr, axis=0)
        hi = np.nanmax(arr, axis=0)
        hi = np.where(hi - lo < 1e-6, lo + 1.0, hi)
    return {"lo": lo.astype(np.float32), "hi": hi.astype(np.float32)}


def scale_geometry(vec: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    """Map to [0, 1] with train-set min/max."""
    lo, hi = scaler["lo"], scaler["hi"]
    return np.clip((vec - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def parse_pr08_ufile_te_ne(
    path: str | Path,
    n_rho: int = 32,
) -> list[dict[str, np.ndarray]]:
    """
  Parse concatenated ASCII UFILE blocks for TE and NE (simplified).

  Returns list of records with rho, te, ne, validity masks.
  """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    records: list[dict[str, np.ndarray]] = []
    blocks = re.split(r"\n\s*\n", text)
    rho_grid = np.linspace(0.0, 1.0, n_rho, dtype=np.float32)

    for block in blocks:
        upper = block.upper()
        if "TE" in upper and "RHO" in upper:
            kind = "te"
        elif "NE" in upper and "RHO" in upper:
            kind = "ne"
        else:
            continue
        nums = []
        for line in block.splitlines():
            for tok in line.split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    continue
        if len(nums) < 4:
            continue
        arr = np.asarray(nums, dtype=np.float64)
        # Heuristic: pairs after headers; take last 2*n_rho or resample
        vals = arr[-n_rho:] if len(arr) >= n_rho else arr
        profile = np.full(n_rho, np.nan, dtype=np.float32)
        mask = np.zeros(n_rho, dtype=np.float32)
        n = min(len(vals), n_rho)
        for i in range(n):
            if not _is_missing(vals[i]):
                profile[i] = vals[i]
                mask[i] = 1.0
        records.append({"kind": kind, "rho": rho_grid, "profile": profile, "mask": mask})
    return records


class ITPAProfileDataset(Dataset):
    """Synthetic-wrapped PR08 profiles for stage-2 auxiliary loss."""

    def __init__(
        self,
        profiles: Sequence[dict[str, np.ndarray]],
        geometry: np.ndarray,
        geo_mask: np.ndarray,
        core_norm: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self._profiles = list(profiles)
        self._geometry = geometry.astype(np.float32)
        self._geo_mask = geo_mask.astype(np.float32)
        self._te_norm, self._ne_norm = core_norm

    def __len__(self) -> int:
        return max(1, len(self._profiles))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not self._profiles:
            rho = torch.linspace(0, 1, 32)
            return {
                "geometry": torch.from_numpy(self._geometry),
                "geo_mask": torch.from_numpy(self._geo_mask),
                "te": torch.zeros(32),
                "ne": torch.zeros(32),
                "profile_mask": torch.zeros(32, 2),
                "rho": rho,
            }
        rec = self._profiles[idx % len(self._profiles)]
        prof = rec["profile"] / (self._te_norm if rec["kind"] == "te" else self._ne_norm)
        mask = rec["mask"]
        te = np.zeros(32, dtype=np.float32)
        ne = np.zeros(32, dtype=np.float32)
        m_te = np.zeros(32, dtype=np.float32)
        m_ne = np.zeros(32, dtype=np.float32)
        if rec["kind"] == "te":
            te = prof
            m_te = mask
        else:
            ne = prof
            m_ne = mask
        pmask = np.stack([m_te, m_ne], axis=-1)
        return {
            "geometry": torch.from_numpy(self._geometry),
            "geo_mask": torch.from_numpy(self._geo_mask),
            "te": torch.from_numpy(te),
            "ne": torch.from_numpy(ne),
            "profile_mask": torch.from_numpy(pmask),
            "rho": torch.from_numpy(rec["rho"]),
        }