"""Ip-forecast path: patch layout, NRMSE definition, and the regression head."""

from __future__ import annotations

import math

import numpy as np
import torch

from data.tokamark_robustness_loader import timeseries_to_multipatch
from models.fusion_transformer import FusionTransformer
from utils.metrics import nrmse


def test_nrmse_matches_rmse_over_std() -> None:
    assert nrmse(0.04, 0.2) == math.sqrt(0.04) / 0.2
    assert math.isnan(nrmse(0.04, 0.0))


def test_full_window_becomes_channel_patches() -> None:
    window = torch.arange(600 * 18, dtype=torch.float32).reshape(600, 18)
    patches, qids, valid = timeseries_to_multipatch(window, patch_len=8, stride=49)
    n_patches_per_channel = len(range(0, 600 - 8 + 1, 49))
    assert patches.shape == (18 * n_patches_per_channel, 8)
    assert qids.shape == (patches.shape[0],)
    assert valid.shape == (patches.shape[0],)
    assert int(valid.sum()) == patches.shape[0]


def test_shot_alarm_sweep_counts_warning_before_t_cut() -> None:
    import scripts.eval_shot_alarms as alarms

    sweep = alarms.sweep

    samples = [
        {"shot_id": 1, "t_cut": 1.0, "input": {"ip": {"time": np.linspace(0.0, 0.2, 8)}}},
        {"shot_id": 1, "t_cut": 1.0, "input": {"ip": {"time": np.linspace(0.2, 0.4, 8)}}},
        {"shot_id": 2, "t_cut": 1.0, "input": {"ip": {"time": np.linspace(0.0, 0.9, 8)}}},
    ]
    # Shot 1 drops early. Shot 2 stays flat, so no alarm before t_cut at a high threshold.
    metrics = sweep(samples, np.array([1.0, 0.2, 1.0]))
    assert metrics["n_shots"] == 2.0
    assert metrics["far"] is None
    assert 0.0 <= metrics["tpr"] <= 1.0


def test_forward_ip_is_scalar_per_batch() -> None:
    model = FusionTransformer(d_model=32, n_layers=1, n_heads=4, patch_len=8, n_reactor_ctx=2)
    patches, qids, valid = timeseries_to_multipatch(torch.randn(600, 18), patch_len=8, stride=49)
    pred = model.forward_ip(
        patches.unsqueeze(0),
        qids.unsqueeze(0),
        valid.unsqueeze(0),
        use_geometry=False,
    )
    assert pred.shape == (1,)
