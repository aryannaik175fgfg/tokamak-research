"""Stage 1 ideal-MHD pretraining on The Well MHD_64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from losses.physics_losses import (
    continuity_residual_loss,
    divergence_free_magnetic_loss,
    induction_residual_loss,
    mass_sum_mismatch_log,
    vrmse_loss,
)
from models.fusion_transformer import FusionTransformer
from utils.config import load_config
from utils.device import add_device_arg, configure_cuda_runtime, dataloader_kwargs, log_device, resolve_device


class SyntheticMHD64Dataset(Dataset):
    """Small synthetic cubes for smoke tests when MHD_64 is unavailable."""

    def __init__(self, n: int = 16, t_in: int = 4) -> None:
        self.n = n
        self.t_in = t_in

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.randn(self.t_in, 7, 8, 8, 8) * 0.1 + 1.0
        y = torch.randn(1, 7, 8, 8, 8) * 0.1 + 1.0
        return {
            "input_fields": x,
            "output_fields": y,
            "M_s": torch.tensor([1.0]),
            "M_A": torch.tensor([0.7]),
            "dt": torch.tensor(0.01),
            "dx": torch.tensor(1.0),
        }


def fit_channel_variance(loader: DataLoader, n_batches: int, device: torch.device) -> torch.Tensor:
    """Training-set channel variances for VRMSE, shape (7,)."""
    sums = torch.zeros(7)
    sums2 = torch.zeros(7)
    count = 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        y = batch["output_fields"][:, 0].to(device)  # (B, C, Z, Y, X)
        for c in range(7):
            flat = y[:, c].reshape(-1)
            sums[c] += flat.sum().cpu()
            sums2[c] += (flat**2).sum().cpu()
            count += flat.numel()
    mean = sums / max(count // 7, 1)
    var = (sums2 / max(count // 7, 1)) - mean**2
    return var.clamp(min=1e-6).to(device)


@torch.no_grad()
def rollout_log(
    model: FusionTransformer,
    batch: dict[str, torch.Tensor],
    channel_var: torch.Tensor,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    """Autoregressive rollout diagnostics (no grad)."""
    x = batch["input_fields"].to(device)
    dt = float(batch["dt"][0].item())
    dx = float(batch["dx"][0].item())
    state = x
    logs: dict[str, float] = {}
    for step in range(1, steps + 1):
        pred = model.forward_mhd(state, batch["M_s"].to(device), batch["M_A"].to(device))
        target = batch["output_fields"][:, 0].to(device) if step == 1 else pred[:, 0].detach()
        vr = vrmse_loss(pred[:, 0], target.unsqueeze(1)[:, 0] if target.dim() == 4 else target, channel_var)
        b = pred[:, 0, 1:4]
        div = divergence_free_magnetic_loss(b, dx=dx)
        logs[f"rollout_vrmse_{step}"] = float(vr.item())
        logs[f"rollout_div_{step}"] = float(div.item())
        # slide window (teacher-forced target only step 1)
        next_frame = pred[:, 0]  # (B, C, Z, Y, X)
        if state.shape[1] > 1:
            state = torch.cat([state[:, 1:], next_frame.unsqueeze(1)], dim=1)
        else:
            state = next_frame.unsqueeze(1)
    return logs


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = resolve_device(getattr(args, "device", None) or cfg.get("device", "auto"))
    configure_cuda_runtime(device)
    log_device(device)
    mcfg = cfg["model"]
    lcfg = cfg["loss"]
    tubelet = tuple(mcfg.get("tubelet", [1, 16, 16, 16]))

    require_real = bool(getattr(args, "require_real", False))
    dcfg = cfg["dataset"]
    train_ds = None
    use_synth = True
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            from data.well_mhd_loader import MHD64ChannelFirstDataset, open_mhd_dataset

            train_ds = MHD64ChannelFirstDataset(
                open_mhd_dataset(
                    "train",
                    well_base_path=dcfg.get("well_base_path"),
                    well_dataset_name=dcfg.get("well_dataset_name", "MHD_64"),
                    n_steps_input=dcfg["n_steps_input"],
                    n_steps_output=dcfg.get("n_steps_output", 1),
                    use_normalization=dcfg.get("use_normalization", False),
                )
            )
            use_synth = False
            break
        except Exception as exc:
            last_exc = exc
            print(f"MHD_64 open attempt {attempt + 1}/3 failed: {exc}")
    if use_synth:
        if require_real:
            raise SystemExit(f"Real MHD_64 required but unavailable: {last_exc}")
        print(f"MHD_64 unavailable ({last_exc}); using synthetic dataset.")
        train_ds = SyntheticMHD64Dataset()

    nw = cfg["dataset"].get("num_workers", 0)
    loader = DataLoader(
        train_ds,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=bool(dcfg.get("shuffle", not use_synth)),
        **dataloader_kwargs(device, nw),
    )

    if use_synth:
        tubelet = (1, 4, 4, 4)
    model = FusionTransformer(
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        mlp_ratio=mcfg.get("mlp_ratio", 4.0),
        tubelet=tubelet,
        n_mach_ctx=mcfg.get("n_mach_ctx", 2),
    ).to(device)

    n_var_batches = cfg["log"].get("variance_fit_batches", 50)

    opt = torch.optim.AdamW(
        list(model.mhd_head.parameters())
        + list(model.mhd_tok.parameters())
        + list(model.mach_cond.parameters())
        + list(model.blocks.parameters()),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    use_amp = bool(getattr(args, "amp", False) or cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = Path(cfg["log"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    var_path = ckpt_dir / "channel_var.json"
    if var_path.is_file() and not getattr(args, "refit_variance", False):
        channel_var = torch.tensor(json.loads(var_path.read_text()), dtype=torch.float32, device=device)
        print(f"Loaded channel variances from {var_path}")
    else:
        channel_var = fit_channel_variance(loader, n_var_batches, device)
        var_path.write_text(json.dumps(channel_var.cpu().tolist()), encoding="utf-8")

    max_steps = args.max_steps or cfg["optim"].get("max_steps") or 10
    step = 0
    model.train()
    while step < max_steps:
        for batch in loader:
            non_blocking = device.type == "cuda"
            x = batch["input_fields"].to(device, non_blocking=non_blocking)
            y = batch["output_fields"][:, 0].to(device, non_blocking=non_blocking)  # (B, C, Z, Y, X)
            rho_t = x[:, -1, 0].detach()
            v_t = x[:, -1, 4:7].detach()
            b_t = x[:, -1, 1:4].detach()
            dt = float(batch["dt"][0].item())
            dx = float(batch["dx"][0].item())

            m_s = batch["M_s"].to(device, non_blocking=non_blocking)
            m_a = batch["M_A"].to(device, non_blocking=non_blocking)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model.forward_mhd(x, m_s, m_a)
                pred_c = pred[:, 0]
                loss_v = vrmse_loss(pred_c, y, channel_var, eps=lcfg["epsilon"])
                loss_div = divergence_free_magnetic_loss(pred_c[:, 1:4], dx=dx, eps=lcfg["epsilon"])
                loss_c = continuity_residual_loss(
                    pred_c[:, 0], rho_t, v_t, dt=dt, dx=dx, eps=lcfg["epsilon"]
                )
                loss_i = induction_residual_loss(
                    pred_c[:, 1:4], b_t, v_t, dt=dt, dx=dx, eps=lcfg["epsilon"]
                )
                loss = (
                    loss_v
                    + lcfg["lambda_div"] * loss_div
                    + lcfg["lambda_cont"] * loss_c
                    + lcfg["lambda_ind"] * loss_i
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            mass_log = mass_sum_mismatch_log(pred_c[:, 0], rho_t)
            if step % 5 == 0:
                print(
                    f"step={step} loss={loss.item():.4f} vrmse={loss_v.item():.4f} "
                    f"div={loss_div.item():.4f} cont={loss_c.item():.4f} ind={loss_i.item():.4f} "
                    f"mass_log={mass_log.item():.4e}"
                )
                if step % 10 == 0 and not getattr(args, "skip_rollout", False):
                    rlogs = rollout_log(model, batch, channel_var, cfg["log"].get("rollout_steps", 5), device)
                    print(f"  rollout: {rlogs}")

            step += 1
            if step >= max_steps:
                break

    torch.save(
        {
            "model": model.state_dict(),
            "channel_var": channel_var.cpu(),
            "use_synth": use_synth,
            "tubelet": list(tubelet),
        },
        ckpt_dir / "stage1_last.pt",
    )
    print(f"checkpoint -> {ckpt_dir / 'stage1_last.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_mhd.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Mixed precision on CUDA (recommended on GPU)")
    parser.add_argument("--skip_rollout", action="store_true", help="Skip autoregressive rollout logging (saves VRAM)")
    parser.add_argument("--refit_variance", action="store_true", help="Recompute channel_var.json from streaming data")
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Exit if MHD_64 cannot be opened. Do not train on the synthetic cube.",
    )
    add_device_arg(parser)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()