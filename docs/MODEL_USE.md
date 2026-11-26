# Using the trained Ip forecaster

## What you have

Checkpoint `runs/stage2_ip/best.pt` is a **MAST-style plasma-current short-horizon forecaster**:

- **Input:** last **600** time steps × **18** channels (magnetics, kinetics, actuators — see `data/tokamark_subset/README.md`).
- **Output:** **normalized Ip ~100 ms ahead** (TokaMark Task 4-4, same as the Neerav-Gupta robustness benchmark).

On **clean test**, **NRMSE ≈ 0.48**, in line with the published Transformer (**0.47**) and LSTM (**0.50**) on the same task. That is a **usable** regression model for trending / soft alarms — **not** a replacement for a facility-certified disruption system until shot-level `t_cut` metrics are added.

## Quick inference

```powershell
cd d:\coding\tokamak
$env:PYTHONPATH="src"
py -3.11 scripts\infer_ip.py --checkpoint runs\stage2_ip\best.pt --split test --index 0
py -3.11 scripts\infer_ip.py --npy path\to\window_600x18.npy --alert-below 0.0 --json
```

`--alert-below` is optional: flag when predicted normalized Ip is below a threshold (tune on your ops needs).

## Benchmark vs paper

```powershell
py -3.11 scripts\eval_benchmark_report.py
```

Writes `runs/stage2_ip/benchmark_report.md` with a table vs XGBoost / LSTM / Transformer / CNN.

## Retrain (stable)

```powershell
py -3.11 src\train_stage2_ip.py --config configs\train_tokamark_ip.yaml --device cuda --amp
```

Uses early stopping (`early_stop_patience: 8`), saves only the best val checkpoint.

## What is not a result

`results/ablation.md` used to list AUROC from random labels. That file now says so. `scripts/eval_shot_alarms.py` refuses to invent a shot-level false-alarm rate without `test_raw_samples.pkl`.

## What still needs work for disruption forecasting

| Goal | Status |
|------|--------|
| Ip forecast on benchmark windows | **Done** (`train_stage2_ip.py`) |
| Compare to published NRMSE | **Done** (`eval_benchmark_report.py`) |
| Shot-level disruption alarm (TPR/FAR with `t_cut`) | Needs `test_raw_samples.pkl` or TokaMark zarr pipeline |
| Cross-machine (DIII-D / C-Mod) | Needs disruption-py exports + `train_stage2_tokamak.py` hazard path |
| MHD pretrain transfer | Optional; IP model currently trains without MHD blocks |

Next high-value step for **disruption** use: keep Ip model for physics context, add a **hazard head** trained on real binary labels from your local `data/tokamark/restored_zarrs` (200 shots) or a small `t_cut` export — not the regression `y` alone.