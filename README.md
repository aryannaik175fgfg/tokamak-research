# Physics-Pretrained, Geometry-Conditioned Transformer for Cross-Tokamak Disruption Forecasting

Research repository for Stage 1 ideal-MHD pretraining on [The Well `MHD_64`](https://polymathic-ai.org/the_well/datasets/MHD_64/) and Stage 2 disruption forecasting on MAST (TokaMark) with ITPA geometry conditioning and held-out DIII-D / Alcator C-Mod evaluation via disruption-py exports.

## Physical scope

**MHD_64** is driven isothermal MHD turbulence in a periodic cube (CATS / Cho–Lazarian). It is **not** a tokamak. Stage 1 teaches divergence-free \(\mathbf{B}\), continuity, and ideal induction only. The momentum equation is **not** used as a loss because the drive \(\mathbf{f}\) is unobserved.

**Stage 2** transfers through dimensionless reactor geometry and cross-machine diagnostic quantities, not raw amplitudes. This does not claim the turbulent cube “knows” \(q_{95}\).

## Setup

```powershell
cd d:\coding\tokamak
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

```bash
cd tokamak
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

### Data sources

| Source | Notes |
|--------|--------|
| `MHD_64` | Hugging Face `polymathic-ai/MHD_64` via `the_well` |
| TokaMark | `tokamark` + MAST zarr/S3; configure `mast_local_path` |
| ITPA DB5.2.3 | `HDB5V2.3.xlsx` from Princeton DataSpace |
| PR08 UFILEs | ITPA multi-machine profile bundle |
| disruption-py | Parquet/CSV exports with `shot`, `time`, `time_until_disrupt` |

**Windows HF URI:** use `path="hf://datasets/polymathic-ai/MHD_64/data/<split>"` (see `well_mhd_loader.py`).

**Splits:** TokaMark uses the published shot-level 80/10/10 lists. ITPA and cross-machine exports use `shot_level_split` in `src/data/splits.py`. Never shuffle windows across shots.

## GPU training

This matches the setup you already use in [`d:\coding\PRISM`](../PRISM): `get_device()` resolves **`cuda` when `torch.cuda.is_available()`**, then models are moved with `.to(device)`. PRISM trains via `scripts/train.py` and `TrainConfig.device: auto` the same way as tokamak’s `--device auto`.

**Fast path (same Python 3.11 + cu121 as your working GPU stack):**

```powershell
cd d:\coding\tokamak
.\scripts\setup_gpu_env.ps1
```

Training scripts use **`--device auto`** (CUDA when PyTorch sees a GPU). On start you should see e.g. `Using GPU: NVIDIA GeForce RTX 3050 ...`.

**Important:** the default `pip install torch` wheel is often **CPU-only**. If `python -c "import torch; print(torch.cuda.is_available())"` prints `False` while `nvidia-smi` works, reinstall CUDA PyTorch inside your venv:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**6 GB laptop GPU (e.g. RTX 3050):** use `batch_size: 1` for MHD_64 at 64³, add `--amp`, and `--skip_rollout` on Stage 1 to limit VRAM. Stage 2 is light and can use larger `batch_size_mast`.

```powershell
python src/train_stage1_mhd.py --config configs/pretrain_mhd.yaml --device cuda --amp --skip_rollout
python src/train_stage2_tokamak.py --config configs/train_tokamark.yaml --device cuda --amp --use_mhd_pretrain --use_itpa_conditioning --use_physics_loss
python src/evaluate_ablation.py --config configs/eval_cross_machine.yaml --device cuda
```

## Training

```powershell
python src/train_stage1_mhd.py --config configs/pretrain_mhd.yaml --max_steps 100
python src/train_stage2_tokamak.py --config configs/train_tokamark.yaml --use_mhd_pretrain --use_itpa_conditioning --use_physics_loss
python src/evaluate_ablation.py --config configs/eval_cross_machine.yaml
```

## Tests

```powershell
pytest tests/test_shapes_and_leakage.py -q
```

Tests use synthetic tensors and fixtures; they do not download MHD_64 (~72 GB).

## Layout

- `configs/` — YAML for pretrain, fine-tune, evaluation
- `src/data/` — loaders and shot-level split utilities
- `src/models/` — tokenizers, `FusionTransformer`
- `src/losses/` — VRMSE, \(\nabla\cdot B\), continuity, induction, focal hazard
- `src/train_stage1_mhd.py`, `src/train_stage2_tokamak.py`, `src/evaluate_ablation.py`