# MAST plasma-current forecast (TokaMark Task 4-4)

Short-horizon forecaster: **600 timesteps × 18 diagnostic channels → normalized plasma current 100 ms ahead**.

The number this repo can stand behind is **clean test NRMSE ≈ 0.478** on the [TokaMark robustness subset](https://huggingface.co/datasets/Neerav-Gupta/tokamark-robustness-data), next to the published Transformer (**0.470**) and LSTM (**0.496**) on the same task. Reproduce it with the commands below after downloading the subset. Weights are not in git. Download `best.pt` from the [stage2-ip release](https://github.com/aryannaik175fgfg/tokamak-research/releases/tag/stage2-ip) into `runs/stage2_ip/`.

This is **not** a cross-machine disruption alarm. Shot-level true-positive rate needs disruption timestamps (`t_cut`) that are not in the default download. MHD pretraining on The Well `MHD_64` is implemented and **not** part of the reported score (`configs/train_tokamark_ip.yaml` sets `use_mhd_pretrain: false`). A finished stage-1 run on real MHD_64 is still outstanding: the last local attempt fell back to a synthetic cube after a Hugging Face timeout.

## What is in the box

| Piece | Status |
| --- | --- |
| Ip regression (`src/train_stage2_ip.py`) | Trained. Early stopping is on (`early_stop_patience: 8`). |
| Benchmark table (`scripts/eval_benchmark_report.py`) | Compares clean-test NRMSE to the published models. |
| Inference (`scripts/infer_ip.py`) | One window in, one normalized Ip out. |
| MHD stage 1 (`src/train_stage1_mhd.py`) | Code only. Not used by the Ip checkpoint. |
| Disruption / DIII-D / C-Mod (`src/evaluate_ablation.py`) | Disabled. It used to score random labels. |
| Shot-level alarms | `scripts/eval_shot_alarms.py` exits until `test_raw_samples.pkl` is present. |

## Setup

```powershell
cd d:\coding\tokamak
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

GPU wheels are separate from the default PyTorch install. On a 6 GB laptop, Ip training uses mixed precision and `batch_size: 256` in `configs/train_tokamark_ip.yaml` (about 4 GB reserved). MHD at 64³ needs `batch_size: 1`, `--amp`, and `--skip_rollout`.

## Data

The NumPy subset is about 650 MB. Full TokaMark zarr (~565 GB) and `test_raw_samples.pkl` (~21 GB) are intentionally not fetched.

```powershell
py -3.11 scripts\fetch_tokamark_subset.py
```

ITPA globals (`data/itpa/HDB5V2.3.xlsx`) are optional. Training copies one MAST geometry row onto every window; that is a conditioner, not per-shot equilibrium data. See `data/README.md`.

## Train, score, predict

```powershell
$env:PYTHONPATH = "src"
py -3.11 src\train_stage2_ip.py --config configs\train_tokamark_ip.yaml --device cuda --amp
py -3.11 scripts\eval_benchmark_report.py --checkpoint runs\stage2_ip\best.pt
py -3.11 scripts\infer_ip.py --checkpoint runs\stage2_ip\best.pt --split test --index 0
```

Logs land in `runs/stage2_ip/train.log` and `runs/stage2_ip/metrics.csv` (train MSE, val MSE, val MAE, val R², VRAM). The saved checkpoint is the best validation epoch, not the last epoch.

`scripts/eval_tpr_far.py` builds a **proxy** label (true Ip below the training 20th percentile). It is not the paper's shot-level TPR.

## Tests

```powershell
$env:PYTHONPATH = "src"
pytest tests -q
```

CI on GitHub runs the same suite on CPU. Tests do not download MHD_64 or the robustness arrays.

## Layout

- `configs/train_tokamark_ip.yaml` — the run that matches the NRMSE above
- `src/models/fusion_transformer.py` — shared transformer; Ip head is `forward_ip`
- `src/data/tokamark_robustness_loader.py` — full 600-step windows as channel patches
- `docs/MODEL_USE.md` — inference notes
- `src/train_stage1_mhd.py`, `src/train_stage2_tokamak.py` — MHD scaffold and the old hazard path; neither produced the reported number

## License

MIT. See `LICENSE`.
