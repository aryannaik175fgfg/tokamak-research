"""disruption-py export adapter for DIII-D and Alcator C-Mod evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.diagnostics import map_signal_to_quantity, quantity_index
from data.disruption_labels import horizon_labels


def load_disruption_export(path: str | Path) -> pd.DataFrame:
    """Load parquet/csv/h5 export with shot, time, and diagnostics."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    elif suffix in {".h5", ".hdf5"}:
        df = pd.read_hdf(path)
    else:
        raise ValueError(f"unsupported export format: {suffix}")
    if "shot" not in df.columns or "time" not in df.columns:
        raise ValueError("export must contain shot and time columns")
    if "intentional_disruption" in df.columns:
        df = df[df["intentional_disruption"].fillna(0).astype(int) == 0]
    return df


def build_disruption_windows(
    df: pd.DataFrame,
    window_s: float = 0.1,
    horizons_ms: tuple[int, ...] = (10, 25, 50, 100),
    tokamak: Optional[Literal["d3d", "cmod"]] = None,
) -> list[dict]:
    """
  Group by shot; label horizons from ``time_until_disrupt`` (Granetz time).

  Each window ends at label time t_end; t_D = time + time_until_disrupt when disrupted.
  """
    windows: list[dict] = []
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"shot", "time", "time_until_disrupt", "tokamak"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for shot, grp in df.groupby("shot"):
        grp = grp.sort_values("time")
        times = grp["time"].to_numpy(dtype=np.float64)
        if len(times) < 2:
            continue
        t_ud = grp["time_until_disrupt"].to_numpy(dtype=np.float64)
        disruptive = np.any(np.isfinite(t_ud) & (t_ud >= 0))
        t_d_shot = None
        if disruptive:
            idx = np.where(np.isfinite(t_ud) & (t_ud >= 0))[0]
            if len(idx):
                i = idx[0]
                t_d_shot = float(times[i] + t_ud[i])
        for i in range(len(times)):
            t_end = float(times[i])
            if disruptive and t_d_shot is not None and t_end >= t_d_shot:
                continue
            t0 = t_end - window_s
            mask = (times >= t0) & (times <= t_end)
            if mask.sum() < 2:
                continue
            labels = horizon_labels(t_end, t_d_shot, horizons_ms)
            values = []
            valid = []
            qids = []
            for col in numeric_cols:
                x = grp.loc[mask, col].to_numpy(dtype=np.float64)
                v = float(x[-1]) if len(x) else 0.0
                ok = np.isfinite(v)
                if not ok:
                    v = 0.0
                values.append(v)
                valid.append(float(ok))
                qids.append(quantity_index(map_signal_to_quantity(col)))
            windows.append(
                {
                    "shot": int(shot),
                    "t_end": t_end,
                    "labels": labels,
                    "values": np.asarray(values, dtype=np.float32),
                    "valid": np.asarray(valid, dtype=np.float32),
                    "quantity_ids": np.asarray(qids, dtype=np.int64),
                    "tokamak": tokamak or str(grp["tokamak"].iloc[0]) if "tokamak" in grp.columns else "unknown",
                    "tau_80_20": np.nan,
                }
            )
    return windows


class DisruptionPyWindowDataset(Dataset):
    """Windowed shots from a disruption-py dataframe export."""

    def __init__(self, windows: list[dict]) -> None:
        self._windows = windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        w = self._windows[idx]
        return {
            "values": torch.from_numpy(w["values"]),
            "valid": torch.from_numpy(w["valid"]),
            "quantity_ids": torch.from_numpy(w["quantity_ids"]),
            "labels": torch.from_numpy(w["labels"]),
            "t_end": torch.tensor(w["t_end"], dtype=torch.float32),
        }