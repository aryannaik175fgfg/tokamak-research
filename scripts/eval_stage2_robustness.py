"""Evaluate stage-2 checkpoints on TokaMark robustness val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.itpa_loader import load_itpa_globals, row_to_dimensionless_geometry
from data.tokamark_robustness_loader import TokamarkRobustnessIpDataset
from models.fusion_transformer import FusionTransformer
from train_stage2_ip import collate_ip
from train_stage2_tokamak import _pick_itpa_geometry
from utils.config import load_config
from utils.device import configure_cuda_runtime, dataloader_kwargs, resolve_device

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    mean_squared_error = None


def _geo_from_cfg(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dcfg = cfg["dataset"]
    mast_geo = np.zeros(6, dtype=np.float32)
    mast_mask = np.zeros(6, dtype=np.float32)
    itpa_geo, itpa_mask = mast_geo.copy(), mast_mask.copy()
    itpa_path = dcfg.get("itpa_globals_path")
    if itpa_path and Path(itpa_path).is_file():
        df = load_itpa_globals(itpa_path)
        mast_geo, mast_mask = _pick_itpa_geometry(df, "MAST")
        itpa_geo, itpa_mask = _pick_itpa_geometry(df, "JET")
        if float(itpa_mask.sum()) < 3:
            itpa_geo, itpa_mask = _pick_itpa_geometry(df, "ITER")
    return mast_geo, mast_mask, itpa_geo, itpa_mask


@torch.no_grad()
def run_eval(
    ckpt_path: Path,
    split: str,
    cfg: dict,
    device: torch.device,
    batch_size: int = 32,
    max_batches: int | None = None,
) -> dict[str, float]:
    mcfg = cfg["model"]
    ckpt_obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    flags = ckpt_obj.get("flags", {})
    use_geo = bool(flags.get("use_itpa_conditioning", True))
    task_ip = flags.get("task") == "ip_regression"

    mast_geo, mast_mask, _, _ = _geo_from_cfg(cfg)
    root = cfg["dataset"]["tokamark_robustness_path"]
    dcfg = cfg["dataset"]
    ds = TokamarkRobustnessIpDataset(
        root,
        split=split,  # type: ignore[arg-type]
        patch_len=dcfg.get("patch_len", mcfg.get("patch_len", 8)),
        stride=dcfg.get("time_stride", 49),
        label_file=dcfg.get("label_file", "y"),  # type: ignore[arg-type]
        mast_geo=mast_geo,
        mast_geo_mask=mast_mask,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ip,
        **dataloader_kwargs(device, 0),
    )

    model = FusionTransformer(
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        mlp_ratio=mcfg.get("mlp_ratio", 4.0),
        patch_len=mcfg.get("patch_len", 8),
        n_reactor_ctx=mcfg.get("n_reactor_ctx", 4),
        n_rho=mcfg.get("n_rho", 32),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    all_probs: list[float] = []
    all_labels: list[float] = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        patches = batch["patches"].to(device)
        qids = batch["qids"].to(device)
        valid = batch["valid"].to(device)
        geo = batch["geo"].to(device)
        gmask = batch["geo_mask"].to(device)
        labels = batch["y"].cpu().numpy()
        if task_ip or hasattr(model, "ip_head"):
            pred = model.forward_ip(patches, qids, valid, geometry=geo, geo_mask=gmask, use_geometry=use_geo).cpu().numpy()
        else:
            hazard, _ = model.forward_tokamak(
                patches, qids, valid, geometry=geo, geo_mask=gmask, use_geometry=use_geo
            )
            pred = hazard[:, 0].cpu().numpy()
        all_probs.extend(pred.tolist())
        all_labels.extend(labels.tolist())

    y = np.asarray(all_labels, dtype=np.float32)
    pred = np.asarray(all_probs, dtype=np.float32)
    baseline = np.full_like(y, float(np.mean(y)))
    out: dict[str, float] = {
        "n": float(len(y)),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }
    if mean_squared_error is not None:
        out["mse"] = float(mean_squared_error(y, pred))
        out["mae"] = float(mean_absolute_error(y, pred))
        out["r2"] = float(r2_score(y, pred))
        out["mse_baseline_mean"] = float(mean_squared_error(y, baseline))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_tokamark_ip.yaml")
    parser.add_argument(
        "--checkpoint",
        default="runs/stage2_ip/best.pt",
    )
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out",
        default="runs/stage2_tokamark/eval_metrics.json",
        help="Write JSON summary",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"missing {ckpt}")
    splits = ["val", "test"] if args.split == "both" else [args.split]
    report: dict[str, dict[str, float]] = {}
    for sp in splits:
        report[sp] = run_eval(ckpt, sp, cfg, device, batch_size=args.batch_size)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(
        json.dumps({"checkpoint": str(ckpt), "splits": report}, indent=2),
        encoding="utf-8",
    )
    print(f"checkpoint: {ckpt}")
    for sp, metrics in report.items():
        print(f"--- {sp} ---")
        for k, v in metrics.items():
            if k == "n":
                print(f"  {k}: {int(v)}")
            else:
                print(f"  {k}: {v:.4f}" if v == v else f"  {k}: nan")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()