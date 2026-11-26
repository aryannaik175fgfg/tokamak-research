"""Neerav-Gupta/tokamark-robustness-data NumPy disruption benchmark (~650 MB)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.diagnostics import PHYSICAL_QUANTITIES


def timeseries_to_multipatch(
    x: torch.Tensor,
    patch_len: int = 8,
    stride: int = 49,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert (T, C) diagnostics into PatchTST tokens along full history.

    Returns patches (N_tok, patch_len), quantity_ids (N_tok,), valid (N_tok,).
    """
    if x.ndim != 2:
        raise ValueError(f"expected (T, C), got {tuple(x.shape)}")
    t_steps, n_ch = x.shape
    n_q = len(PHYSICAL_QUANTITIES)
    patches_list: list[torch.Tensor] = []
    qids_list: list[int] = []
    for c in range(n_ch):
        series = x[:, c]
        for start in range(0, t_steps - patch_len + 1, stride):
            seg = series[start : start + patch_len]
            patches_list.append(seg.contiguous())
            qids_list.append(c % n_q)
    if not patches_list:
        pad = torch.zeros(patch_len, dtype=x.dtype)
        patches = pad.unsqueeze(0)
        qids = torch.zeros(1, dtype=torch.long)
        valid = torch.zeros(1, dtype=torch.float32)
        return patches, qids, valid
    patches = torch.stack(patches_list)
    qids = torch.tensor(qids_list, dtype=torch.long)
    valid = torch.isfinite(patches).all(dim=1).float()
    patches = torch.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)
    return patches, qids, valid


class TokamarkRobustnessDataset(Dataset):
    """
    Window-level time series from ``train_X_ts.npy`` / ``*_y.npy`` (or ``*_y_45.npy``).

    Expected layout under ``root/data/``: train/val/test splits with matching X and y files.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        label_file: Literal["y", "y_45"] = "y",
        use_feat: bool = False,
    ) -> None:
        self.root = Path(root)
        data = self.root / "data"
        prefix = {"train": "train", "val": "val", "test": "test"}[split]
        x_path = data / f"{prefix}_X_feat.npy" if use_feat else data / f"{prefix}_X_ts.npy"
        y_path = data / f"{prefix}_{label_file}.npy"
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"missing {x_path} or {y_path}")
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if len(self.x) != len(self.y):
            raise ValueError(f"X/y length mismatch: {len(self.x)} vs {len(self.y)}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.as_tensor(np.array(self.x[idx], copy=True), dtype=torch.float32)
        y = torch.as_tensor(np.array(self.y[idx], copy=True), dtype=torch.float32)
        if y.ndim == 0:
            y = y.unsqueeze(0)
        return {"x_ts": x, "labels": y}


class TokamarkRobustnessStage2Dataset(Dataset):
    """
    Map robustness windows (T, F) into ``FusionTransformer.forward_tokamak`` inputs.

    Uses the last ``patch_len`` timesteps per feature channel; no full TokaMark zarr install.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        patch_len: int = 8,
        horizons: int = 4,
        label_file: Literal["y", "y_45"] = "y",
        mast_geo: Optional[np.ndarray] = None,
        mast_geo_mask: Optional[np.ndarray] = None,
        itpa_geo: Optional[np.ndarray] = None,
        itpa_geo_mask: Optional[np.ndarray] = None,
        n_rho: int = 32,
    ) -> None:
        self._inner = TokamarkRobustnessDataset(
            root, split=split, label_file=label_file, use_feat=False
        )
        self.patch_len = patch_len
        self.horizons = horizons
        self.n_rho = n_rho
        n_q = len(PHYSICAL_QUANTITIES)
        self._mast_geo = torch.as_tensor(
            mast_geo if mast_geo is not None else np.zeros(6, dtype=np.float32),
            dtype=torch.float32,
        )
        self._mast_geo_mask = torch.as_tensor(
            mast_geo_mask if mast_geo_mask is not None else np.zeros(6, dtype=np.float32),
            dtype=torch.float32,
        )
        self._itpa_geo = torch.as_tensor(
            itpa_geo if itpa_geo is not None else self._mast_geo.numpy(),
            dtype=torch.float32,
        )
        self._itpa_geo_mask = torch.as_tensor(
            itpa_geo_mask if itpa_geo_mask is not None else self._mast_geo_mask.numpy(),
            dtype=torch.float32,
        )
        self._qids = torch.arange(int(self._inner.x.shape[-1])) % n_q

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._inner[idx]
        x: torch.Tensor = item["x_ts"]
        if x.shape[0] < self.patch_len:
            pad = torch.zeros(self.patch_len - x.shape[0], x.shape[1], dtype=x.dtype)
            x = torch.cat([pad, x], dim=0)
        patches = x[-self.patch_len :].T.contiguous()
        valid = torch.isfinite(patches).all(dim=1).float()
        patches = torch.nan_to_num(patches, nan=0.0, posinf=0.0, neginf=0.0)

        y = item["labels"].float().view(-1)
        if y.numel() == 1:
            labels = y.expand(self.horizons)
        else:
            labels = y[: self.horizons]
            if labels.numel() < self.horizons:
                labels = torch.nn.functional.pad(labels, (0, self.horizons - labels.numel()))

        return {
            "mast_patches": patches,
            "mast_qids": self._qids.clone(),
            "mast_valid": valid,
            "mast_labels": labels,
            "mast_geo": self._mast_geo.clone(),
            "mast_geo_mask": self._mast_geo_mask.clone(),
            "itpa_geo": self._itpa_geo.clone(),
            "itpa_geo_mask": self._itpa_geo_mask.clone(),
            "te": torch.zeros(self.n_rho),
            "ne": torch.zeros(self.n_rho),
            "profile_mask": torch.zeros(self.n_rho, 2),
        }


class TokamarkRobustnessIpDataset(Dataset):
    """Task 4-4: predict normalized Ip from full (600, 18) windows (robustness benchmark)."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        patch_len: int = 8,
        stride: int = 49,
        label_file: Literal["y", "y_45"] = "y",
        mast_geo: Optional[np.ndarray] = None,
        mast_geo_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._inner = TokamarkRobustnessDataset(root, split=split, label_file=label_file, use_feat=False)
        self.patch_len = patch_len
        self.stride = stride
        self._mast_geo = torch.as_tensor(
            mast_geo if mast_geo is not None else np.zeros(6, dtype=np.float32),
            dtype=torch.float32,
        )
        self._mast_geo_mask = torch.as_tensor(
            mast_geo_mask if mast_geo_mask is not None else np.zeros(6, dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._inner[idx]
        patches, qids, valid = timeseries_to_multipatch(item["x_ts"], self.patch_len, self.stride)
        y = item["labels"].float().view(-1)[0]
        return {
            "patches": patches,
            "qids": qids,
            "valid": valid,
            "y": y,
            "geo": self._mast_geo.clone(),
            "geo_mask": self._mast_geo_mask.clone(),
        }