"""Plot training curves and held-out scores for stage2_ip."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "stage2_ip"
OUT = RUN / "plots"


def load_epochs() -> list[dict[str, float]]:
    path = RUN / "metrics.csv"
    rows: list[dict[str, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(row[k]) for k in row if k != "epoch" and row[k]} | {"epoch": int(row["epoch"])})
    return rows


def plot_training(rows: list[dict]) -> None:
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Stage 2 IP training (batch 256, resumed)", fontsize=13)

    ax = axes[0, 0]
    ax.plot(ep, [r["train_mse"] for r in rows], label="train MSE", color="#2563eb")
    ax.plot(ep, [r["val_mse"] for r in rows], label="val MSE", color="#dc2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, [r["val_r2"] for r in rows], color="#059669", label="val R²")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ep, [r["val_mae"] for r in rows], color="#7c3aed", label="val MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "vram_reserved_mb" in rows[0]:
        ax.plot(ep, [r.get("vram_reserved_mb", 0) for r in rows], color="#64748b")
        ax.set_ylabel("VRAM reserved (MB)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_eval_bars() -> None:
    eval_path = RUN / "eval_metrics.json"
    if not eval_path.is_file():
        return
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    splits = data["splits"]
    metrics = ("mse", "mae", "r2")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle("Held-out regression (best.pt)", fontsize=13)
    colors = {"val": "#2563eb", "test": "#dc2626"}
    for ax, metric in zip(axes, metrics, strict=True):
        names = list(splits)
        vals = [splits[s][metric] for s in names]
        bars = ax.bar(names, vals, color=[colors.get(n, "#334155") for n in names])
        ax.set_title(metric.upper())
        ax.set_ylim(0, max(vals) * 1.15 if metric != "r2" else 1.05)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "eval_regression.png", dpi=150)
    plt.close(fig)


def plot_alarm_bars() -> None:
    path = RUN / "alarm_far.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    splits = data["splits"]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Alarm proxy @ FAR < 5%", fontsize=13)
    x = list(splits.keys())
    w = 0.35
    auroc = [splits[s]["auroc"] for s in x]
    tpr = [splits[s]["tpr_at_far"] for s in x]
    import numpy as np

    xi = np.arange(len(x))
    ax.bar(xi - w / 2, auroc, w, label="AUROC", color="#2563eb")
    ax.bar(xi + w / 2, tpr, w, label="TPR @ FAR<5%", color="#059669")
    ax.set_xticks(xi, x)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "eval_alarm.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_epochs()
    plot_training(rows)
    plot_eval_bars()
    plot_alarm_bars()
    print(f"Wrote plots under {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()