"""Stage 2: TokaMark Task 4-4 plasma-current regression with per-epoch logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.itpa_loader import load_itpa_globals, row_to_dimensionless_geometry
from data.tokamark_robustness_loader import TokamarkRobustnessIpDataset
from models.fusion_transformer import FusionTransformer
from train_stage2_tokamak import _pick_itpa_geometry
from utils.config import load_config
from utils.device import add_device_arg, configure_cuda_runtime, dataloader_kwargs, log_device, resolve_device
from utils.epoch_log import EpochLogger


def collate_ip(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


def _geo_from_cfg(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    dcfg = cfg["dataset"]
    mast_geo = np.zeros(6, dtype=np.float32)
    mast_mask = np.zeros(6, dtype=np.float32)
    itpa_path = dcfg.get("itpa_globals_path")
    if itpa_path and Path(itpa_path).is_file():
        df = load_itpa_globals(itpa_path)
        mast_geo, mast_mask = _pick_itpa_geometry(df, "MAST")
    return mast_geo, mast_mask


def build_ip_dataset(cfg: dict, split: str) -> Dataset:
    dcfg = cfg["dataset"]
    root = dcfg["tokamark_robustness_path"]
    mast_geo, mast_mask = _geo_from_cfg(cfg)
    return TokamarkRobustnessIpDataset(
        root,
        split=split,  # type: ignore[arg-type]
        patch_len=dcfg.get("patch_len", cfg["model"].get("patch_len", 8)),
        stride=dcfg.get("time_stride", 49),
        label_file=dcfg.get("label_file", "y"),  # type: ignore[arg-type]
        mast_geo=mast_geo,
        mast_geo_mask=mast_mask,
    )


@torch.no_grad()
def eval_split(
    model: FusionTransformer,
    loader: DataLoader,
    device: torch.device,
    use_itpa: bool,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    sse = 0.0
    sae = 0.0
    n = 0
    sum_y = 0.0
    sum_y2 = 0.0
    non_blocking = device.type == "cuda"
    for batch in loader:
        patches = batch["patches"].to(device, non_blocking=non_blocking)
        qids = batch["qids"].to(device, non_blocking=non_blocking)
        valid = batch["valid"].to(device, non_blocking=non_blocking)
        y = batch["y"].to(device, non_blocking=non_blocking)
        geo = batch["geo"].to(device, non_blocking=non_blocking)
        gmask = batch["geo_mask"].to(device, non_blocking=non_blocking)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model.forward_ip(patches, qids, valid, geometry=geo, geo_mask=gmask, use_geometry=use_itpa)
        diff = pred.float() - y.float()
        b = y.numel()
        sse += float((diff**2).sum().item())
        sae += float(diff.abs().sum().item())
        n += b
        sum_y += float(y.sum().item())
        sum_y2 += float((y**2).sum().item())
    model.train()
    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0.0}
    mean_y = sum_y / n
    ss_tot = sum_y2 - n * mean_y**2
    mse = sse / n
    mae = sae / n
    if ss_tot > 1e-12:
        ss_res = sse
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")
    return {"mse": mse, "mae": mae, "r2": r2, "n": float(n)}


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = resolve_device(getattr(args, "device", None) or cfg.get("device", "auto"))
    configure_cuda_runtime(device)
    log_device(device)

    use_mhd = args.use_mhd_pretrain if args.use_mhd_pretrain is not None else cfg["flags"].get("use_mhd_pretrain", False)
    use_itpa = args.use_itpa_conditioning if args.use_itpa_conditioning is not None else cfg["flags"].get("use_itpa_conditioning", True)

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

    resume = getattr(args, "resume", None)
    if resume and Path(resume).is_file():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        print(f"Resumed weights from {resume} (epoch {state.get('epoch', '?')})")

    stage1_path = cfg["optim"].get("stage1_checkpoint")
    if use_mhd and stage1_path and Path(stage1_path).exists():
        ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
        enc_state = {k.replace("blocks.", "", 1): v for k, v in ckpt["model"].items() if k.startswith("blocks.")}
        model.blocks.load_state_dict(enc_state, strict=True)
        print(f"Loaded stage-1 encoder blocks from {stage1_path} ({len(enc_state)} tensors)")

    dcfg = cfg["dataset"]
    batch_size = args.batch_size or dcfg.get("batch_size", dcfg.get("batch_size_mast", 8))
    nw = dcfg.get("num_workers", 0)
    train_ds = build_ip_dataset(cfg, "train")
    val_ds = build_ip_dataset(cfg, "val")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ip, **dataloader_kwargs(device, nw)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ip, **dataloader_kwargs(device, nw)
    )

    enc_lr = cfg["optim"]["lr"] * (cfg["optim"].get("encoder_lr_scale", 0.1) if use_mhd else 1.0)
    head_params = list(model.time_tok.parameters()) + list(model.ip_head.parameters())
    if use_itpa:
        head_params += list(model.reactor_cond.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder_parameters(), "lr": enc_lr},
            {"params": head_params, "lr": cfg["optim"]["lr"]},
        ],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    ckpt_dir = Path(cfg["log"]["checkpoint_dir"])
    logger = EpochLogger(ckpt_dir, cfg["log"].get("metrics_csv", "metrics.csv"), cfg["log"].get("train_log", "train.log"))
    logger.save_config(cfg)
    steps_per_epoch = max(1, (len(train_ds) + batch_size - 1) // batch_size)
    n_tok = 18 * max(1, (600 - mcfg.get("patch_len", 8)) // dcfg.get("time_stride", 49) + 1)
    epochs = args.epochs or cfg["optim"].get("epochs", 25)
    use_amp = bool(getattr(args, "amp", False) or cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    non_blocking = device.type == "cuda"
    best_val = float("inf")
    patience = int(cfg["optim"].get("early_stop_patience", 8))
    patience_left = patience

    logger.log_message(
        f"task=ip_regression train={len(train_ds)} val={len(val_ds)} batch={batch_size} "
        f"steps_per_epoch={steps_per_epoch} tokens_per_sample~{n_tok}"
    )
    if device.type == "cuda":
        warm = next(iter(train_loader))
        patches = warm["patches"][:1].to(device)
        qids = warm["qids"][:1].to(device)
        valid = warm["valid"][:1].to(device)
        geo = warm["geo"][:1].to(device)
        gmask = warm["geo_mask"][:1].to(device)
        torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast("cuda", enabled=use_amp):
            model.forward_ip(patches, qids, valid, geometry=geo, geo_mask=gmask, use_geometry=use_itpa)
        logger.log_message(
            f"cuda_warmup batch=1 reserved_MB={torch.cuda.max_memory_reserved() / 1024**2:.0f}"
        )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch in train_loader:
            patches = batch["patches"].to(device, non_blocking=non_blocking)
            qids = batch["qids"].to(device, non_blocking=non_blocking)
            valid = batch["valid"].to(device, non_blocking=non_blocking)
            y = batch["y"].to(device, non_blocking=non_blocking)
            geo = batch["geo"].to(device, non_blocking=non_blocking)
            gmask = batch["geo_mask"].to(device, non_blocking=non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model.forward_ip(patches, qids, valid, geometry=geo, geo_mask=gmask, use_geometry=use_itpa)
                loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss_sum += float(loss.item())
            train_batches += 1

        train_mse = train_loss_sum / max(train_batches, 1)
        val_m = eval_split(model, val_loader, device, use_itpa, use_amp)
        lr = opt.param_groups[-1]["lr"]
        vram_mb = 0.0
        if device.type == "cuda":
            vram_mb = torch.cuda.max_memory_reserved() / 1024**2
        metrics: dict[str, Any] = {
            "train_mse": train_mse,
            "val_mse": val_m["mse"],
            "val_mae": val_m["mae"],
            "val_r2": val_m["r2"],
            "lr": lr,
            "vram_reserved_mb": vram_mb,
        }
        logger.log_epoch(epoch, metrics)

        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_mse": val_m["mse"],
            "flags": {"use_mhd_pretrain": use_mhd, "use_itpa_conditioning": use_itpa, "task": "ip_regression"},
        }
        torch.save(payload, ckpt_dir / "last.pt")
        if val_m["mse"] < best_val:
            best_val = val_m["mse"]
            torch.save(payload, ckpt_dir / "best.pt")
            patience_left = patience
            logger.log_message(f"new best val_mse={best_val:.6f} -> best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.log_message(f"early_stop at epoch={epoch} (no val improvement for {patience} epochs)")
                break

    logger.log_message(f"finished training; best val_mse={best_val:.6f}")
    print(f"logs: {ckpt_dir / 'train.log'}  |  {ckpt_dir / 'metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_tokamark_ip.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use_mhd_pretrain", action="store_true", default=None)
    parser.add_argument("--no_mhd_pretrain", action="store_false", dest="use_mhd_pretrain")
    parser.add_argument("--use_itpa_conditioning", action="store_true", default=None)
    parser.add_argument("--no_itpa_conditioning", action="store_false", dest="use_itpa_conditioning")
    parser.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")
    parser.add_argument("--resume", default=None, help="Load model weights from checkpoint (e.g. runs/stage2_ip/best.pt)")
    add_device_arg(parser)
    train(parser.parse_args())


if __name__ == "__main__":
    main()