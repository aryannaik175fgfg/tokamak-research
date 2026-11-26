"""TPR @ FAR < 5% (window-level) for Ip regression alarm scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_stage2_ip import build_ip_dataset, collate_ip
from models.fusion_transformer import FusionTransformer
from utils.config import load_config
from utils.device import add_device_arg, configure_cuda_runtime, dataloader_kwargs, resolve_device

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


def tpr_at_far(scores: np.ndarray, labels: np.ndarray, far_max: float = 0.05) -> tuple[float, float, float]:
    """
    Max TPR s.t. FPR <= far_max. Returns (tpr, fpr_at_threshold, threshold).

    labels: {0,1}; scores: higher => more likely positive.
    """
    order = np.sort(np.unique(scores))
    best_tpr, best_fpr, best_thr = 0.0, 1.0, float("nan")
    for thr in order:
        pred = scores >= thr
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tp = ((pred == 1) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)
        if fpr <= far_max + 1e-12:
            if tpr > best_tpr:
                best_tpr, best_fpr, best_thr = float(tpr), float(fpr), float(thr)
    return best_tpr, best_fpr, best_thr


@torch.no_grad()
def collect_scores(
    model: FusionTransformer,
    loader: DataLoader,
    device: torch.device,
    use_itpa: bool,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[float] = []
    ys: list[float] = []
    non_blocking = device.type == "cuda"
    for batch in loader:
        patches = batch["patches"].to(device, non_blocking=non_blocking)
        qids = batch["qids"].to(device, non_blocking=non_blocking)
        valid = batch["valid"].to(device, non_blocking=non_blocking)
        y = batch["y"]
        geo = batch["geo"].to(device, non_blocking=non_blocking)
        gmask = batch["geo_mask"].to(device, non_blocking=non_blocking)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model.forward_ip(patches, qids, valid, geometry=geo, geo_mask=gmask, use_geometry=use_itpa)
        preds.extend(pred.float().cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
    return np.asarray(preds, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="TPR @ FAR<5% with Ip quench-proxy labels")
    parser.add_argument("--config", default="configs/train_tokamark_ip.yaml")
    parser.add_argument("--checkpoint", default="runs/stage2_ip/best.pt")
    parser.add_argument("--far-max", type=float, default=0.05)
    parser.add_argument("--label-quantile", type=float, default=0.20, help="Train-y quantile for label=1 (low Ip)")
    parser.add_argument("--out", default="runs/stage2_ip/alarm_far.json")
    add_device_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"missing {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    flags = ckpt.get("flags", {})
    use_itpa = bool(flags.get("use_itpa_conditioning", True))
    mcfg = cfg["model"]
    model = FusionTransformer(
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        mlp_ratio=mcfg.get("mlp_ratio", 4.0),
        patch_len=mcfg.get("patch_len", 8),
        n_reactor_ctx=mcfg.get("n_reactor_ctx", 4),
        n_rho=mcfg.get("n_rho", 32),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    batch_size = cfg["dataset"].get("batch_size", 8)
    nw = cfg["dataset"].get("num_workers", 0)
    train_loader = DataLoader(
        build_ip_dataset(cfg, "train"),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ip,
        **dataloader_kwargs(device, nw),
    )
    _, y_train = collect_scores(model, train_loader, device, use_itpa, device.type == "cuda")
    thr_label = float(np.quantile(y_train, args.label_quantile))
    use_amp = device.type == "cuda"

    report: dict = {
        "checkpoint": str(ckpt_path),
        "far_max": args.far_max,
        "label_rule": f"y < train_q{args.label_quantile:.2f} ({thr_label:.4f})",
        "score_rule": "alarm_score = -predicted_Ip (lower prediction => higher alarm)",
        "note": "Window-level proxy; shot-level FAR needs test_raw_samples.pkl + t_cut.",
        "splits": {},
    }

    for split in ("val", "test"):
        loader = DataLoader(
            build_ip_dataset(cfg, split),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_ip,
            **dataloader_kwargs(device, nw),
        )
        pred, y = collect_scores(model, loader, device, use_itpa, use_amp)
        labels = (y < thr_label).astype(np.int32)
        scores = -pred
        tpr, fpr, thr = tpr_at_far(scores, labels, far_max=args.far_max)
        auc = float("nan")
        if roc_auc_score is not None and labels.sum() > 0 and labels.sum() < len(labels):
            auc = float(roc_auc_score(labels, scores))
        report["splits"][split] = {
            "n": int(len(y)),
            "positive_rate": float(labels.mean()),
            "auroc": auc,
            "tpr_at_far": tpr,
            "fpr_at_operating_point": fpr,
            "score_threshold": thr,
        }
        print(
            f"{split} PROXY low-Ip (not a disruption alarm): "
            f"pos_rate={labels.mean():.3f} AUROC={auc:.3f} "
            f"TPR@FAR<{args.far_max}={tpr:.3f} FPR={fpr:.4f} thr={thr:.4f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()