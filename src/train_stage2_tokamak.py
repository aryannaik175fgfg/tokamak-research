"""Stage 2 TokaMark + ITPA fine-tuning with ablation flags."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np

from data.itpa_loader import load_itpa_globals, row_to_dimensionless_geometry
from data.tokamark_robustness_loader import TokamarkRobustnessStage2Dataset
from losses.physics_losses import (
    focal_hazard_loss,
    masked_profile_loss,
    monotonic_hazard_loss,
)
from models.fusion_transformer import FusionTransformer
from utils.config import load_config
from utils.device import add_device_arg, configure_cuda_runtime, dataloader_kwargs, log_device, resolve_device


class SyntheticTokamakBatch(Dataset):
    """Synthetic paired MAST / ITPA batches for smoke tests."""

    def __init__(self, n: int = 32, n_ch: int = 8, patch_len: int = 8) -> None:
        self.n = n
        self.n_ch = n_ch
        self.patch_len = patch_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "mast_patches": torch.randn(self.n_ch, self.patch_len),
            "mast_qids": torch.arange(self.n_ch) % 10,
            "mast_valid": torch.ones(self.n_ch),
            "mast_labels": torch.tensor([0.0, 0.0, 1.0, 1.0]),
            "mast_geo": torch.rand(6),
            "mast_geo_mask": torch.ones(6),
            "itpa_geo": torch.rand(6),
            "itpa_geo_mask": torch.ones(6),
            "te": torch.rand(32),
            "ne": torch.rand(32),
            "profile_mask": torch.ones(32, 2),
        }


def collate_synthetic(samples: list[dict]) -> dict[str, torch.Tensor]:
    out = {}
    for k in samples[0]:
        out[k] = torch.stack([s[k] for s in samples])
    return out


def _pick_itpa_geometry(df, device_hint: str = "MAST") -> tuple:
    sub = df
    for col in ("DEVICE", "TOK", "DEVICE_NAME"):
        if col in df.columns:
            m = df[col].astype(str).str.upper().str.contains(device_hint, na=False)
            if m.any():
                sub = df.loc[m]
                break
    for _, row in sub.iterrows():
        geo, mask = row_to_dimensionless_geometry(row)
        if float(mask.sum()) >= 3:
            return geo, mask
    row = df.iloc[0]
    return row_to_dimensionless_geometry(row)


def build_stage2_dataset(cfg: dict) -> Dataset:
    dcfg = cfg["dataset"]
    robust = dcfg.get("tokamark_robustness_path")
    if robust and Path(robust).is_dir():
        itpa_path = dcfg.get("itpa_globals_path")
        mast_geo, mast_mask = np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)
        itpa_geo, itpa_mask = mast_geo.copy(), mast_mask.copy()
        if itpa_path and Path(itpa_path).is_file():
            df = load_itpa_globals(itpa_path)
            mast_geo, mast_mask = _pick_itpa_geometry(df, "MAST")
            itpa_geo, itpa_mask = _pick_itpa_geometry(df, "JET")
            if float(itpa_mask.sum()) < 3:
                itpa_geo, itpa_mask = _pick_itpa_geometry(df, "ITER")
        horizons = len(dcfg.get("horizons_ms", [10, 25, 50, 100]))
        print(f"Stage-2 data: TokamarkRobustnessStage2Dataset ({robust})")
        return TokamarkRobustnessStage2Dataset(
            robust,
            split="train",
            patch_len=cfg["model"].get("patch_len", 8),
            horizons=horizons,
            mast_geo=mast_geo,
            mast_geo_mask=mast_mask,
            itpa_geo=itpa_geo,
            itpa_geo_mask=itpa_mask,
            n_rho=cfg["model"].get("n_rho", 32),
        )
    print("Stage-2 data: SyntheticTokamakBatch (set dataset.tokamark_robustness_path for real data)")
    return SyntheticTokamakBatch()


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = resolve_device(getattr(args, "device", None) or cfg.get("device", "auto"))
    configure_cuda_runtime(device)
    log_device(device)

    use_mhd = args.use_mhd_pretrain if args.use_mhd_pretrain is not None else cfg["flags"]["use_mhd_pretrain"]
    use_itpa = args.use_itpa_conditioning if args.use_itpa_conditioning is not None else cfg["flags"]["use_itpa_conditioning"]
    use_phys = args.use_physics_loss if args.use_physics_loss is not None else cfg["flags"]["use_physics_loss"]

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

    stage1_path = cfg["optim"].get("stage1_checkpoint")
    if use_mhd and stage1_path and Path(stage1_path).exists():
        ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
        enc_state = {
            k.replace("blocks.", "", 1): v
            for k, v in ckpt["model"].items()
            if k.startswith("blocks.")
        }
        model.blocks.load_state_dict(enc_state, strict=True)
        print(f"Loaded stage-1 encoder blocks from {stage1_path} ({len(enc_state)} tensors)")

    ds = build_stage2_dataset(cfg)
    nw = cfg["dataset"].get("num_workers", 0)
    loader = DataLoader(
        ds,
        batch_size=cfg["dataset"]["batch_size_mast"],
        collate_fn=collate_synthetic,
        **dataloader_kwargs(device, nw),
    )

    enc_lr = cfg["optim"]["lr"] * (cfg["optim"].get("encoder_lr_scale", 0.1) if use_mhd else 1.0)
    head_params = list(model.time_tok.parameters()) + list(model.hazard_head.parameters()) + list(model.profile_head.parameters())
    if use_itpa:
        head_params += list(model.reactor_cond.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder_parameters(), "lr": enc_lr},
            {"params": head_params, "lr": cfg["optim"]["lr"]},
        ],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    lcfg = cfg["loss"]
    lam_mono = lcfg["lambda_mono"] if use_phys else 0.0
    lam_prof = lcfg["lambda_prof"] if use_itpa else 0.0

    ckpt_dir = Path(cfg["log"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    max_steps = args.max_steps or cfg["optim"].get("max_steps") or 20
    use_amp = bool(getattr(args, "amp", False) or cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    model.train()
    non_blocking = device.type == "cuda"

    while step < max_steps:
        for batch in loader:
            patches = batch["mast_patches"].to(device, non_blocking=non_blocking)
            qids = batch["mast_qids"].to(device, non_blocking=non_blocking)
            valid = batch["mast_valid"].to(device, non_blocking=non_blocking)
            labels = batch["mast_labels"].to(device, non_blocking=non_blocking)
            geo = batch["mast_geo"].to(device, non_blocking=non_blocking)
            gmask = batch["mast_geo_mask"].to(device, non_blocking=non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                hazard, _prof = model.forward_tokamak(
                    patches,
                    qids,
                    valid,
                    geometry=geo,
                    geo_mask=gmask,
                    use_geometry=use_itpa,
                )
                probs = torch.sigmoid(hazard)
                loss_f = focal_hazard_loss(
                    hazard,
                    labels,
                    alpha=lcfg["focal_alpha"],
                    gamma=lcfg["focal_gamma"],
                )
                loss_m = monotonic_hazard_loss(probs) if lam_mono > 0 else hazard.new_zeros(())
                te = batch["te"].to(device, non_blocking=non_blocking)
                ne = batch["ne"].to(device, non_blocking=non_blocking)
                pm = batch["profile_mask"].to(device, non_blocking=non_blocking)
                itpa_geo = batch["itpa_geo"].to(device, non_blocking=non_blocking)
                itpa_mask = batch["itpa_geo_mask"].to(device, non_blocking=non_blocking)
                _, prof_itpa = model.forward_tokamak(
                    patches[:, :1].expand(-1, patches.shape[1], -1),
                    qids[:, :1].expand(-1, qids.shape[1]),
                    valid,
                    geometry=itpa_geo,
                    geo_mask=itpa_mask,
                    use_geometry=use_itpa,
                )
                loss_p = (
                    masked_profile_loss(prof_itpa[..., 0], prof_itpa[..., 1], te, ne, pm)
                    if lam_prof > 0
                    else hazard.new_zeros(())
                )
                loss = loss_f + lam_mono * loss_m + lam_prof * loss_p

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % 5 == 0:
                print(f"step={step} loss={loss.item():.4f} focal={loss_f.item():.4f}")

            step += 1
            if step >= max_steps:
                break

    tag = "mhd_itpa" if use_mhd and use_itpa else "itpa" if use_itpa else "vanilla"
    torch.save({"model": model.state_dict(), "flags": {"use_mhd_pretrain": use_mhd, "use_itpa_conditioning": use_itpa}}, ckpt_dir / f"{tag}.pt")
    print(f"saved {ckpt_dir / f'{tag}.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_tokamark.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--use_mhd_pretrain", action="store_true", default=None)
    parser.add_argument("--no_mhd_pretrain", action="store_false", dest="use_mhd_pretrain")
    parser.add_argument("--use_itpa_conditioning", action="store_true", default=None)
    parser.add_argument("--no_itpa_conditioning", action="store_false", dest="use_itpa_conditioning")
    parser.add_argument("--use_physics_loss", action="store_true", default=None)
    parser.add_argument("--no_physics_loss", action="store_false", dest="use_physics_loss")
    parser.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")
    add_device_arg(parser)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()