"""Shot-level disruption alarms using t_cut, matching the robustness benchmark.

FAR is not reported: every test shot in that pickle disrupts, so there is no
negative class. The published numbers are TPR and mean warning time.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.tokamark_robustness_loader import timeseries_to_multipatch
from models.fusion_transformer import FusionTransformer
from train_stage2_ip import _geo_from_cfg
from utils.config import load_config
from utils.device import configure_cuda_runtime, resolve_device

PKL = ROOT / "data" / "tokamark_subset" / "data" / "test_raw_samples.pkl"


def _window_end_s(sample: dict) -> float:
    first = next(iter(sample["input"]))
    times = np.asarray(sample["input"][first]["time"], dtype=np.float64)
    return float(times[-1])


@torch.no_grad()
def predict_ip(
    model: FusionTransformer,
    windows: np.ndarray,
    device: torch.device,
    patch_len: int,
    stride: int,
    use_geometry: bool,
    batch_size: int,
    geo: torch.Tensor | None = None,
    geo_mask: torch.Tensor | None = None,
) -> np.ndarray:
    preds: list[float] = []
    model.eval()
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        patches, qids, valid = [], [], []
        for row in chunk:
            p, q, v = timeseries_to_multipatch(torch.as_tensor(row, dtype=torch.float32), patch_len, stride)
            patches.append(p)
            qids.append(q)
            valid.append(v)
        b = len(chunk)
        geometry = geo.expand(b, -1).to(device) if geo is not None else None
        mask = geo_mask.expand(b, -1).to(device) if geo_mask is not None else None
        pred = model.forward_ip(
            torch.stack(patches).to(device),
            torch.stack(qids).to(device),
            torch.stack(valid).to(device),
            geometry=geometry,
            geo_mask=mask,
            use_geometry=use_geometry,
        )
        preds.extend(pred.float().cpu().tolist())
    return np.asarray(preds, dtype=np.float64)


def sweep(samples: list[dict], preds: np.ndarray) -> dict[str, float]:
    by_shot: dict[int, dict] = {}
    for i, sample in enumerate(samples):
        sid = int(sample["shot_id"])
        bucket = by_shot.setdefault(sid, {"t_cut": float(sample["t_cut"]), "times": [], "preds": []})
        bucket["times"].append(_window_end_s(sample))
        bucket["preds"].append(float(preds[i]))

    best = {"tpr": 0.0, "mean_warning_time_ms": float("nan"), "threshold": 0.0, "n_shots": float(len(by_shot))}
    for thresh in np.linspace(0.05, 0.60, 30):
        hits = []
        warnings = []
        for bucket in by_shot.values():
            times = np.asarray(bucket["times"])
            values = np.asarray(bucket["preds"])
            order = np.argsort(times)
            times, values = times[order], values[order]
            scale = float(np.abs(values).mean())
            if scale < 1e-6:
                continue
            alarm = values < (1.0 - float(thresh)) * scale
            before = alarm & (times < bucket["t_cut"])
            if before.any():
                warn_ms = (bucket["t_cut"] - float(times[before][0])) * 1000.0
                if warn_ms > 0:
                    hits.append(1)
                    warnings.append(warn_ms)
                    continue
            hits.append(0)
        tpr = float(np.mean(hits)) if hits else 0.0
        mwt = float(np.mean(warnings)) if warnings else float("nan")
        better_tpr = tpr > best["tpr"]
        same_tpr_longer = tpr == best["tpr"] and warnings and (
            not np.isfinite(best["mean_warning_time_ms"]) or mwt > best["mean_warning_time_ms"]
        )
        if better_tpr or same_tpr_longer:
            best = {
                "tpr": tpr,
                "mean_warning_time_ms": mwt,
                "threshold": float(thresh),
                "n_shots": float(len(by_shot)),
            }
    best["far"] = None
    best["far_note"] = "Not computable: every test shot in test_raw_samples.pkl disrupts."
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_tokamark_ip.yaml")
    parser.add_argument("--checkpoint", default="runs/stage2_ip/best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="runs/stage2_ip/shot_alarms.json")
    args = parser.parse_args()
    if not PKL.is_file():
        print(
            "Shot-level TPR needs data/tokamark_subset/data/test_raw_samples.pkl "
            "(~21 GB). Download is separate from the 650 MB subset."
        )
        return 2

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    with PKL.open("rb") as handle:
        samples = pickle.load(handle)
    x_test = np.load(ROOT / "data" / "tokamark_subset" / "data" / "test_X_ts.npy", mmap_mode="r")
    if len(samples) != len(x_test):
        raise SystemExit(f"pickle has {len(samples)} rows, test_X_ts has {len(x_test)}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    use_geo = bool(ckpt.get("flags", {}).get("use_itpa_conditioning", True))
    dcfg = cfg["dataset"]
    geo_np, mask_np = _geo_from_cfg(cfg)
    print(f"scoring {len(samples)} windows on {device}...")
    preds = predict_ip(
        model,
        np.asarray(x_test),
        device,
        patch_len=dcfg.get("patch_len", 8),
        stride=dcfg.get("time_stride", 49),
        use_geometry=use_geo,
        batch_size=args.batch_size,
        geo=torch.as_tensor(geo_np, dtype=torch.float32).unsqueeze(0),
        geo_mask=torch.as_tensor(mask_np, dtype=torch.float32).unsqueeze(0),
    )
    metrics = sweep(samples, preds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "definition": "Alarm when predicted Ip falls below (1-threshold)*mean(|pred|) before t_cut. Threshold swept.",
        **{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()},
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mwt = payload["mean_warning_time_ms"]
    mwt_s = "nan" if mwt is None else f"{mwt:.1f}"
    print(f"shots={int(metrics['n_shots'])} TPR={metrics['tpr']:.3f} mean_warning_ms={mwt_s} threshold={metrics['threshold']:.2f}")
    print("FAR is not computable: all test shots disrupt.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())