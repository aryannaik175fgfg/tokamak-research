# Benchmark report (Task 4-4 clean test)

**Task:** From 600×18 MAST diagnostics, predict normalized plasma current **100 ms ahead**.
**Metric:** NRMSE = RMSE / σ(y) on the held-out **test** split (same convention as the robustness README table).

| Model | Clean test NRMSE |
| --- | --- |
| XGBoost (published) | 0.494 |
| LSTM (published) | 0.496 |
| Transformer (published) | 0.470 |
| CNN (TokaMark) (published) | 0.528 |
| **This repo (`best.pt`)** | **0.478** |

- Test MSE: 0.0969, MAE: 0.1576, R²: 0.7715
- Δ vs HF Transformer checkpoint metrics (`clean`=0.470): +0.008

## What this is useful for

- **Forecasting Ip** on MAST-like multi-channel windows (PCS advisory / trending).
- **Not** a certified disruption alarm without shot-level `t_cut` integration and retraining for alarms.

## Run inference

```powershell
py -3.11 scripts/infer_ip.py --checkpoint runs/stage2_ip/best.pt --index 0 --split test
```
