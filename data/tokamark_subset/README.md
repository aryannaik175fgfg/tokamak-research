---
license: mit
task_categories:
- time-series-forecasting
language:
- en
tags:
- plasma-physics
- fusion
- tokamak
- robustness
- sensor-failure
- MAST
- TokaMark
size_categories:
- 10G<n<100G
---

# TokaMark Sensor Robustness Benchmark Data

**Associated paper:** Benchmarking Sensor Robustness in Plasma Diagnostic Models: A Systematic Evaluation on TokaMark  
**Author:** Neerav Gupta  
**Code:** [github.com/Neerav-Gupta/tokamark-robustness](https://github.com/Neerav-Gupta/tokamark-robustness)

---

## Dataset Description

This dataset contains pre-processed numpy arrays, trained model checkpoints, and experiment results from the first systematic robustness benchmark of plasma diagnostic ML models under realistic sensor failure, using the [TokaMark benchmark](https://arxiv.org/abs/2602.10132) on MAST tokamak data.

We evaluate four architectures (XGBoost, LSTM, Transformer, and the TokaMark CNN baseline) across six physically-motivated failure scenarios and three imputation strategies, and compute shot-level alarm metrics using ground-truth disruption timestamps from FAIR-MAST.

The raw data originates from the [FAIR-MAST dataset](https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark) provided by UKAEA, IBM Research, and STFC. This dataset contains derived arrays prepared for robustness benchmarking and is not a redistribution of the raw FAIR-MAST data.

---

## Key Results

| Model | Clean NRMSE | Robustness Score (RS) | Clean TPR | TPR (proximate 25%, zero-fill) |
|---|---|---|---|---|
| XGBoost | 0.494 | 0.841 | 0.40 | — |
| LSTM | 0.496 | 0.808 | 0.52 | 0.00 |
| Transformer | 0.470 | 0.765 | 0.48 | 0.08 |
| CNN (TokaMark baseline) | 0.528 | 0.764 | 0.60 | 0.46 |

**Key finding:** Under disruption-proximate sensor failure, LSTM alarm detection collapses to TPR = 0.00 with zero-fill imputation, but recovers to TPR = 1.00 with mean-fill — the opposite of its effect on NRMSE.

---

## Dataset Structure

```
tokamark-robustness-data/
├── data/
│   ├── train_X_feat.npy          # XGBoost feature vectors, train
│   ├── train_X_ts.npy            # Time series tensors, train
│   ├── train_y.npy               # Target labels, train
│   ├── val_X_feat.npy            # XGBoost feature vectors, val
│   ├── val_X_ts.npy              # Time series tensors, val
│   ├── val_y.npy                 # Target labels, val
│   ├── test_X_feat.npy           # XGBoost feature vectors, test
│   ├── test_X_ts.npy             # Time series tensors, test
│   ├── test_y.npy                # Target labels, test
│   ├── test_raw_samples.pkl      # Raw test samples with t_cut timestamps
│   └── feature_names.json        # Feature names for X_feat columns
├── checkpoints/
│   ├── xgboost_clean.pkl         # Trained XGBoost model
│   ├── lstm_clean.pt             # Trained LSTM model
│   ├── transformer_clean.pt      # Trained Transformer model
│   └── cnn_clean.pt              # Trained CNN baseline model
└── results/
    ├── xgboost_results.json
    ├── lstm_results.json
    ├── transformer_results.json
    ├── cnn_results.json
    ├── shot_level_metrics.json
    ├── alarm_under_corruption.json
    └── alarm_mitigation_proximate.json
```

---

## File Descriptions

### Data Arrays

| File | Description | Shape |
|---|---|---|
| `train_X_feat.npy` | Training feature vectors for XGBoost | (9950, 142) |
| `train_X_ts.npy` | Training time series for LSTM/Transformer/CNN | (9950, 600, 18) |
| `train_y.npy` | Training targets (normalized plasma current) | (9950,) |
| `val_X_feat.npy` | Validation feature vectors | (2500, 142) |
| `val_X_ts.npy` | Validation time series | (2500, 600, 18) |
| `val_y.npy` | Validation targets | (2500,) |
| `test_X_feat.npy` | Test feature vectors | (2420, 142) |
| `test_X_ts.npy` | Test time series | (2420, 600, 18) |
| `test_y.npy` | Test targets | (2420,) |
| `test_raw_samples.pkl` | Raw test samples including `t_cut` disruption timestamps and signal time arrays | 2420 samples |
| `feature_names.json` | Feature names for the 142 X_feat columns | 142 names |

### Checkpoints

| File | Description |
|---|---|
| `checkpoints/xgboost_clean.pkl` | Trained XGBoost model (clean data) |
| `checkpoints/lstm_clean.pt` | Trained LSTM model (clean data) |
| `checkpoints/transformer_clean.pt` | Trained Transformer model (clean data) |
| `checkpoints/cnn_clean.pt` | Trained TokaMark CNN baseline (clean data) |

### Results

| File | Description |
|---|---|
| `results/xgboost_results.json` | Full robustness results for XGBoost |
| `results/lstm_results.json` | Full robustness results for LSTM |
| `results/transformer_results.json` | Full robustness results for Transformer |
| `results/cnn_results.json` | Full robustness results for CNN baseline |
| `results/shot_level_metrics.json` | Clean-data shot-level TPR and MWT |
| `results/alarm_under_corruption.json` | Shot-level alarm metrics under sensor failure |
| `results/alarm_mitigation_proximate.json` | Alarm metrics under proximate failure with each imputation strategy |

---

## Data Details

| Split | Shots | Windows |
|---|---|---|
| Train | 200 | 9,950 |
| Val | 50 | 2,500 |
| Test | 50 | 2,420 |

**Task:** Task 4-4 from TokaMark — plasma current quench prediction. Given 150ms of diagnostic history across 14 input signals and 4 actuator signals (18 total channels), predict plasma current 100ms into the future.

**All 50 test shots disrupted** (finite `t_cut` in `test_raw_samples.pkl`), enabling real shot-level alarm metric computation.

---

## Signal Order in X_ts (18 channels)

| Index | Signal | Category |
|---|---|---|
| 0 | interferometer-n_e_line | Kinetics |
| 1 | magnetics-b_field_pol_probe_ccbv_field | Magnetics |
| 2 | magnetics-b_field_pol_probe_obr_field | Magnetics |
| 3 | magnetics-b_field_pol_probe_obv_field | Magnetics |
| 4 | magnetics-b_field_pol_probe_omv_voltage | Magnetics |
| 5 | magnetics-b_field_tor_probe_cc_field | Magnetics |
| 6 | magnetics-b_field_tor_probe_saddle_voltage | Magnetics |
| 7 | magnetics-flux_loop_flux | Magnetics |
| 8 | pf_active-coil_current | Active coils |
| 9 | pf_active-solenoid_current | Active coils |
| 10 | soft_x_rays-horizontal_cam_lower | Radiatives |
| 11 | soft_x_rays-horizontal_cam_upper | Radiatives |
| 12 | spectrometer_visible-filter_spectrometer_dalpha_voltage | Kinetics |
| 13 | summary-ip | Plasma current |
| 14 | gas_injection-total_injected | Actuator |
| 15 | pulse_schedule-i_plasma | Actuator |
| 16 | pulse_schedule-n_e_line | Actuator |
| 17 | summary-power_nbi | Actuator |

---

## Loading the Data

```python
import numpy as np
import pickle
from huggingface_hub import snapshot_download

# Download everything
snapshot_download(
    repo_id="Neerav-Gupta/tokamark-robustness-data",
    repo_type="dataset",
    local_dir="./tokamark_robustness_data"
)

# Load arrays
X_train_ts = np.load("./tokamark_robustness_data/data/train_X_ts.npy")
y_train    = np.load("./tokamark_robustness_data/data/train_y.npy")
X_test_ts  = np.load("./tokamark_robustness_data/data/test_X_ts.npy")
y_test     = np.load("./tokamark_robustness_data/data/test_y.npy")

print(f"Train: {X_train_ts.shape}")  # (9950, 600, 18)
print(f"Test:  {X_test_ts.shape}")   # (2420, 600, 18)

# Load raw test samples (includes t_cut disruption timestamps)
with open("./tokamark_robustness_data/data/test_raw_samples.pkl", "rb") as f:
    test_samples = pickle.load(f)

# Each sample has: shot_id, window_index, input, actuator, output, t_cut
print(f"Sample keys: {list(test_samples[0].keys())}")
print(f"t_cut (disruption time): {test_samples[0]['t_cut']:.4f}s")

# Load trained LSTM checkpoint
import torch
ckpt = torch.load(
    "./tokamark_robustness_data/checkpoints/lstm_clean.pt",
    map_location="cpu"
)
print(f"LSTM n_features: {ckpt['n_features']}")

# Load trained CNN checkpoint
ckpt_cnn = torch.load(
    "./tokamark_robustness_data/checkpoints/cnn_clean.pt",
    map_location="cpu"
)
print(f"CNN n_channels: {ckpt_cnn['n_channels']}, "
      f"input_len: {ckpt_cnn['input_len']}, "
      f"backbone_hidden: {ckpt_cnn['backbone_hidden']}")
```

---

## Reproducing Results

```bash
# Clone the code repository
git clone https://github.com/Neerav-Gupta/tokamark-robustness.git
cd tokamark-robustness

# Download this dataset
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Neerav-Gupta/tokamark-robustness-data',
    repo_type='dataset',
    local_dir='fusion_research/data'
)
"

# Train all four models and run robustness evaluation
python scripts/train_xgboost.py
python scripts/train_lstm.py
python scripts/train_transformer.py
python scripts/train_cnn_baseline.py

# Compute shot-level alarm metrics
python scripts/compute_alarm_metrics.py

# Generate all 9 figures
python scripts/analyze_results.py
```

---

## Citation

If you use this dataset please cite:

```bibtex
@misc{gupta2026tokamark_robustness,
  title         = {Benchmarking Sensor Robustness in Plasma Diagnostic
                   Models: A Systematic Evaluation on TokaMark},
  author        = {Gupta, Neerav},
  year          = {2026},
  eprint        = {2607.11915},
  archivePrefix = {arXiv},
  primaryClass  = {physics.plasm-ph}
}
```

Please also cite the original TokaMark benchmark:

```bibtex
@article{rousseau2026tokamark,
  title   = {TokaMark: A Comprehensive Benchmark for MAST Tokamak
             Plasma Models},
  author  = {Rousseau, C{\'e}cile and Jackson, Samuel and
             Ordonez-Hurtado, Rodrigo H. and Amorisco, Nicola C. and
             Boschi, Tobia and Holt, George K. and Loreti, Andrea and
             Sz{\'e}kely, Eszter and Whittle, Alexander and
             Agnello, Adriano and Pamela, Stanislas and
             Pascale, Alessandra and Akers, Robert and
             Bernabe Moreno, Juan and Thorne, Sue and
             Zayats, Mykhaylo},
  journal = {arXiv preprint arXiv:2602.10132},
  year    = {2026}
}
```

---

## License

[MIT License](https://opensource.org/licenses/MIT)

---

## Acknowledgements

Raw plasma data sourced from the FAIR-MAST dataset provided by UKAEA, IBM Research, and STFC. This benchmark dataset was prepared independently using the TokaMark data loading infrastructure.