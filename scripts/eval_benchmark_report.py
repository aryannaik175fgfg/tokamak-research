"""Compare fusion model to TokaMark robustness paper (clean NRMSE on test)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.fusion_transformer import FusionTransformer
from train_stage2_ip import build_ip_dataset, collate_ip, eval_split
from utils.config import load_config
from utils.device import configure_cuda_runtime, dataloader_kwargs, resolve_device
from utils.metrics import nrmse

PAPER_CLEAN = {
    "XGBoost": 0.494,
    "LSTM": 0.496,
    "Transformer": 0.470,
    "CNN (TokaMark)": 0.528,
}


@torch.no_grad()
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_tokamark_ip.yaml")
    parser.add_argument("--checkpoint", default="runs/stage2_ip/best.pt")
    parser.add_argument("--out", default="runs/stage2_ip/benchmark_report.md")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    use_itpa = bool(ckpt.get("flags", {}).get("use_itpa_conditioning", True))

    bs = cfg["dataset"].get("batch_size", 8)
    test_loader = DataLoader(
        build_ip_dataset(cfg, "test"),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_ip,
        **dataloader_kwargs(device, 0),
    )
    test_m = eval_split(model, test_loader, device, use_itpa, device.type == "cuda")
    ours = nrmse(test_m["mse"], test_m.get("y_std") or float(np.std(np.load(
        ROOT / "data/tokamark_subset/data/test_y.npy"
    ))))

    ref_path = ROOT / "data/tokamark_subset/results/transformer_results.json"
    paper_transformer_clean = None
    if ref_path.is_file():
        paper_transformer_clean = json.loads(ref_path.read_text())["clean"]

    lines = [
        "# Benchmark report (Task 4-4 clean test)",
        "",
        "**Task:** From 600×18 MAST diagnostics, predict normalized plasma current **100 ms ahead**.",
        "**Metric:** NRMSE = RMSE / σ(y) on the held-out **test** split (same convention as the robustness README table).",
        "",
        "| Model | Clean test NRMSE |",
        "| --- | --- |",
    ]
    for name, val in PAPER_CLEAN.items():
        lines.append(f"| {name} (published) | {val:.3f} |")
    lines.append(f"| **This repo (`{ckpt_path.name}`)** | **{ours:.3f}** |")
    lines.append("")
    lines.append(f"- Test MSE: {test_m['mse']:.4f}, MAE: {test_m['mae']:.4f}, R²: {test_m['r2']:.4f}")
    if paper_transformer_clean is not None:
        delta = ours - paper_transformer_clean
        lines.append(f"- Δ vs HF Transformer checkpoint metrics (`clean`={paper_transformer_clean:.3f}): {delta:+.3f}")
    lines.extend(
        [
            "",
            "## What this is useful for",
            "",
            "- **Forecasting Ip** on MAST-like multi-channel windows (PCS advisory / trending).",
            "- **Not** a certified disruption alarm without shot-level `t_cut` integration and retraining for alarms.",
            "",
            "## Run inference",
            "",
            "```powershell",
            "py -3.11 scripts/infer_ip.py --checkpoint runs/stage2_ip/best.pt --index 0 --split test",
            "```",
            "",
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    (out.parent / "benchmark_report.json").write_text(
        json.dumps({"clean_test_nrmse": ours, "test": test_m, "paper_clean": PAPER_CLEAN}, indent=2),
        encoding="utf-8",
    )
    print(f"Clean test NRMSE (ours): {ours:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()