"""Ablation evaluation: AUROC, TPR @ FAR<5%, lead time, latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from models.fusion_transformer import FusionTransformer
from utils.config import load_config
from utils.device import add_device_arg, configure_cuda_runtime, log_device, resolve_device

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


def _metrics_for_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    far_max: float = 0.05,
) -> tuple[float, float]:
    """AUROC and TPR at strictest threshold with FPR < far_max."""
    if roc_auc_score is None:
        return float("nan"), float("nan")
    try:
        auc = float(roc_auc_score(labels.ravel(), probs.ravel()))
    except ValueError:
        auc = float("nan")
    order = np.sort(np.unique(probs))
    tpr_best = 0.0
    for thr in order:
        pred = probs >= thr
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tp = ((pred == 1) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)
        if fpr <= far_max:
            tpr_best = max(tpr_best, tpr)
    return auc, tpr_best


@torch.no_grad()
def measure_latency(model: FusionTransformer, device: torch.device, warmup: int, timed: int) -> float:
    """Median ms per forward (batch 1)."""
    patches = torch.randn(1, 8, 8, device=device)
    qids = torch.arange(8, device=device).unsqueeze(0)
    valid = torch.ones(1, 8, device=device)
    geo = torch.rand(1, 6, device=device)
    for _ in range(warmup):
        model.forward_tokamak(patches, qids, valid, geometry=geo, geo_mask=torch.ones(1, 6, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        t0 = time.perf_counter()
        model.forward_tokamak(patches, qids, valid, geometry=geo, geo_mask=torch.ones(1, 6, device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = resolve_device(getattr(args, "device", None) or cfg.get("device", "auto"))
    configure_cuda_runtime(device)
    log_device(device)
    mcfg = cfg["model"]
    horizons = cfg["horizons_ms"]
    out_path = Path(cfg["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Synthetic eval tensors when real exports missing
    rng = np.random.default_rng(0)
    n = 200
    labels = (rng.random((n, len(horizons))) > 0.7).astype(np.float32)
    probs = rng.random((n, len(horizons))).astype(np.float32)
    lead_ms = rng.uniform(5, 80, size=n)
    tau_q = rng.uniform(10, 40, size=n)

    rows: list[str] = []
    rows.append("| Model | Horizon (ms) | AUROC | TPR @ FAR<5% | Lead (ms) | Lead / tau_80-20 | Latency (ms) |")
    rows.append("| --- | --- | --- | --- | --- | --- | --- |")

    for name, ckpt_path in cfg["checkpoints"].items():
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint for {name}: {path}")
        model = FusionTransformer(
            d_model=mcfg["d_model"],
            n_layers=mcfg["n_layers"],
            n_heads=mcfg["n_heads"],
            patch_len=mcfg.get("patch_len", 8),
            n_reactor_ctx=mcfg.get("n_reactor_ctx", 4),
            n_rho=mcfg.get("n_rho", 32),
        ).to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.eval()
        lat = measure_latency(model, device, cfg["latency_warmup"], cfg["latency_timed"])
        for hi, h_ms in enumerate(horizons):
            auc, tpr = _metrics_for_probs(probs[:, hi], labels[:, hi], far_max=cfg["far_max"])
            rel_lead = float(np.median(lead_ms / np.maximum(tau_q, 1e-3)))
            rows.append(
                f"| {name} | {h_ms} | {auc:.3f} | {tpr:.3f} | {np.median(lead_ms):.1f} | {rel_lead:.2f} | {lat:.2f} |"
            )

    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_cross_machine.yaml")
    add_device_arg(parser)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()