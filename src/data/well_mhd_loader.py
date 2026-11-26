"""The Well MHD_64 loader with Windows-safe Hugging Face paths."""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from torch.utils.data import Dataset


def _mhd_hf_path(split: Literal["train", "valid", "test"]) -> str:
    return f"hf://datasets/polymathic-ai/MHD_64/data/{split}"


def open_mhd_dataset(
    split: Literal["train", "valid", "test"] = "train",
    well_base_path: Optional[str] = "hf://datasets/polymathic-ai/",
    well_dataset_name: str = "MHD_64",
    n_steps_input: int = 4,
    n_steps_output: int = 1,
    use_normalization: bool = False,
) -> Dataset:
    """
  Wrap ``the_well.data.WellDataset`` for MHD_64.

  On Windows, remote URIs must use ``path=`` with forward slashes only.
  """
    try:
        from the_well.data import WellDataset
    except ImportError as e:
        raise ImportError(
            "Install the_well: pip install 'the_well[benchmark]'"
        ) from e

    if well_base_path and "://" in str(well_base_path):
        path = _mhd_hf_path(split)
        ds = WellDataset(
            path=path,
            well_split_name=None,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
        )
    else:
        ds = WellDataset(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
            well_split_name=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
        )
    return ds


def well_item_to_channels_first(item: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
  Convert Well sample to ``(T, C, 64, 64, 64)`` with physics scalars.

  Input fields layout from Well: ``(T_in, Z, Y, X, F)`` with F=7.
  """
    x_in = item["input_fields"]  # (T, Z, Y, X, C)
    x_out = item["output_fields"]
    if not isinstance(x_in, torch.Tensor):
        x_in = torch.as_tensor(x_in)
        x_out = torch.as_tensor(x_out)
    # (T, Z, Y, X, C) -> (T, C, Z, Y, X)
    x_in = x_in.permute(0, 4, 1, 2, 3).contiguous()
    x_out = x_out.permute(0, 4, 1, 2, 3).contiguous()

    scalars = item.get("constant_scalars", {})

    def _one_scalar(val: object, default: float) -> float:
        if val is None:
            return default
        if isinstance(val, torch.Tensor):
            return float(val.reshape(-1)[0].item())
        return float(val)

    if isinstance(scalars, dict):
        m_s = _one_scalar(scalars.get("Ms", scalars.get("M_s")), 1.0)
        m_a = _one_scalar(scalars.get("Ma", scalars.get("M_A")), 0.7)
    elif isinstance(scalars, torch.Tensor) and scalars.numel() > 0:
        flat = scalars.reshape(-1)
        m_s = float(flat[0].item())
        m_a = float(flat[1].item()) if flat.numel() > 1 else 0.7
    else:
        m_s, m_a = 1.0, 0.7

    dt = 0.01
    if "output_time_grid" in item and item["output_time_grid"] is not None:
        tg = torch.as_tensor(item["output_time_grid"])
        if tg.numel() >= 2:
            dt = float((tg[-1] - tg[0]).item() / max(1, tg.numel() - 1))
    dx = 1.0

    return {
        "input_fields": x_in,
        "output_fields": x_out,
        "M_s": torch.tensor([m_s], dtype=torch.float32),
        "M_A": torch.tensor([m_a], dtype=torch.float32),
        "dt": torch.tensor(dt, dtype=torch.float32),
        "dx": torch.tensor(dx, dtype=torch.float32),
    }


class MHD64ChannelFirstDataset(Dataset):
    """Map-style dataset yielding channel-first MHD tensors."""

    def __init__(self, well_dataset: Dataset) -> None:
        self._ds = well_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return well_item_to_channels_first(self._ds[idx])