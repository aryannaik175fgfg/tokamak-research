"""CUDA / CPU device selection and dataloader settings."""

from __future__ import annotations

import argparse
from typing import Any

import torch


def resolve_device(choice: str | None = "auto") -> torch.device:
    """
  Resolve training device.

  ``auto`` uses CUDA when available, otherwise CPU.
  """
    if choice is None or choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch and check drivers with nvidia-smi."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def configure_cuda_runtime(device: torch.device) -> None:
    """Enable common GPU training optimizations."""
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def log_device(device: torch.device) -> None:
    """Print device name and memory for logging."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"Using GPU: {name} ({total_gb:.1f} GiB) [cuda:{idx}]")
    else:
        print("Using CPU (pass --device cuda when a GPU is available)")


def dataloader_kwargs(device: torch.device, num_workers: int) -> dict[str, Any]:
    """pin_memory and persistent_workers when training on CUDA."""
    kw: dict[str, Any] = {"num_workers": num_workers}
    if device.type == "cuda":
        kw["pin_memory"] = True
        if num_workers > 0:
            kw["persistent_workers"] = True
    return kw


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Compute device (default: auto = CUDA if available)",
    )


def empty_cache_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()