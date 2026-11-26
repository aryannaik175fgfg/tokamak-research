"""Run trained Ip model on one window (600×18) — for PCS-style preview / debugging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.itpa_loader import load_itpa_globals
from data.tokamark_robustness_loader import timeseries_to_multipatch
from models.fusion_transformer import FusionTransformer
from train_stage2_tokamak import _pick_itpa_geometry
from utils.config import load_config
from utils.device import configure_cuda_runtime, resolve_device


def load_window(split: str, index: int, data_root: Path) -> np.ndarray:
    x = np.load(data_root / "data" / f"{split}_X_ts.npy", mmap_mode="r")
    return np.array(x[index], copy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict normalized Ip 100ms ahead for one window.")
    parser.add_argument("--config", default="configs/train_tokamark_ip.yaml")
    parser.add_argument("--checkpoint", default="runs/stage2_ip/best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--npy", type=Path, default=None, help="Optional (600,18) .npy instead of split/index")
    parser.add_argument(
        "--alert-below",
        type=float,
        default=None,
        help="If set, print WARNING when prediction is below this normalized Ip",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    dcfg = cfg["dataset"]
    data_root = Path(dcfg["tokamark_robustness_path"])

    if args.npy:
        window = np.load(args.npy)
    else:
        window = load_window(args.split, args.index, data_root)
    if window.shape != (600, 18):
        raise SystemExit(f"expected window shape (600, 18), got {window.shape}")

    y_true = None
    if args.npy is None:
        y_all = np.load(data_root / "data" / f"{args.split}_y.npy", mmap_mode="r")
        y_true = float(y_all[args.index])

    patch_len = dcfg.get("patch_len", 8)
    stride = dcfg.get("time_stride", 49)
    patches, qids, valid = timeseries_to_multipatch(
        torch.as_tensor(window, dtype=torch.float32), patch_len, stride
    )

    mast_geo, mast_mask = np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)
    itpa_path = dcfg.get("itpa_globals_path")
    if itpa_path and Path(itpa_path).is_file():
        df = load_itpa_globals(itpa_path)
        mast_geo, mast_mask = _pick_itpa_geometry(df, "MAST")

    mcfg = cfg["model"]
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FusionTransformer(
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        mlp_ratio=mcfg.get("mlp_ratio", 4.0),
        patch_len=patch_len,
        n_reactor_ctx=mcfg.get("n_reactor_ctx", 4),
        n_rho=mcfg.get("n_rho", 32),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    use_itpa = bool(ckpt.get("flags", {}).get("use_itpa_conditioning", True))
    model.eval()

    with torch.no_grad():
        pred = model.forward_ip(
            patches.unsqueeze(0).to(device),
            qids.unsqueeze(0).to(device),
            valid.unsqueeze(0).to(device),
            geometry=torch.as_tensor(mast_geo).unsqueeze(0).to(device),
            geo_mask=torch.as_tensor(mast_mask).unsqueeze(0).to(device),
            use_geometry=use_itpa,
        )
    y_hat = float(pred.item())

    out = {
        "predicted_ip_normalized": y_hat,
        "horizon_ms": 100,
        "window_shape": list(window.shape),
        "split": args.split if not args.npy else None,
        "index": args.index if not args.npy else None,
    }
    if y_true is not None:
        out["true_ip_normalized"] = y_true
        out["error"] = y_hat - y_true

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"Predicted Ip (normalized, +100ms): {y_hat:.4f}")
        if y_true is not None:
            print(f"True Ip (normalized):              {y_true:.4f}  (error {out['error']:+.4f})")
        if args.alert_below is not None and y_hat < args.alert_below:
            print(f"WARNING: prediction below alert threshold {args.alert_below}")

    if args.alert_below is not None:
        out["alert"] = y_hat < args.alert_below
    return 0


if __name__ == "__main__":
    raise SystemExit(main())