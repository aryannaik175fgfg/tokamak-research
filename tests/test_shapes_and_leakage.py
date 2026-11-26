"""Shape, physics residual, label, and leakage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from data.disruption_labels import (
    annotate_mast_shot,
    detect_current_quench_onset,
    horizon_labels,
)
from data.disruption_py_adapter import load_disruption_export
from data.itpa_loader import q_cylindrical, row_to_dimensionless_geometry
from data.splits import assert_disjoint_shots, shot_level_split
from losses.physics_losses import (
    continuity_residual_loss,
    divergence_b,
    divergence_free_magnetic_loss,
    induction_residual_loss,
    monotonic_hazard_loss,
)
from models.fusion_transformer import FusionTransformer


@pytest.fixture
def small_model() -> FusionTransformer:
    return FusionTransformer(
        d_model=64,
        n_layers=2,
        n_heads=4,
        tubelet=(1, 4, 4, 4),
        patch_len=4,
        n_reactor_ctx=2,
        n_mach_ctx=2,
        n_rho=16,
        n_horizons=4,
    )


def test_forward_all_heads_variable_channels(small_model: FusionTransformer) -> None:
    b = 2
    x = torch.randn(b, 4, 7, 8, 8, 8)
    pred_mhd = small_model.forward_mhd(x, torch.ones(b, 1), torch.full((b, 1), 0.7))
    assert pred_mhd.shape[0] == b and pred_mhd.shape[2] == 7

    for n_ch in (3, 11):
        patches = torch.randn(b, n_ch, 4)
        qids = torch.arange(n_ch).unsqueeze(0).expand(b, -1) % 10
        valid = torch.ones(b, n_ch)
        valid[0, -1] = 0.0  # masked channel
        hazard, prof = small_model.forward_tokamak(patches, qids, valid, use_geometry=False)
        assert hazard.shape == (b, 4)
        assert prof.shape == (b, 16, 2)

    patches0 = torch.randn(b, 3, 4)
    valid0 = torch.zeros(b, 3)
    hazard0, _ = small_model.forward_tokamak(patches0, torch.zeros(b, 3, dtype=torch.long), valid0, use_geometry=False)
    assert hazard0.shape == (b, 4)


def test_div_b_zero_solenoidal() -> None:
    z, y, x = 8, 8, 8
    phi = torch.randn(1, 1, z, y, x)
    dx = 1.0
    from losses.physics_losses import periodic_central_diff

    b_const = torch.ones(1, 3, z, y, x)
    padded = torch.zeros(1, 7, z, y, x)
    padded[:, 1:4] = b_const
    div_c = divergence_b(padded, dx=dx)
    assert torch.allclose(div_c, torch.zeros_like(div_c), atol=1e-5)
    loss = divergence_free_magnetic_loss(b_const, dx=dx)
    assert float(loss.item()) < 1e-4


def test_continuity_and_induction_manufactured() -> None:
    z, y, x = 8, 8, 8
    dt, dx = 0.01, 1.0
    rho_t = torch.ones(1, z, y, x)
    v_t = torch.zeros(1, 3, z, y, x)
    rho_pred = rho_t.clone()
    assert float(continuity_residual_loss(rho_pred, rho_t, v_t, dt, dx).item()) < 1e-6

    b_t = torch.randn(1, 3, z, y, x)
    v_t = torch.randn(1, 3, z, y, x) * 0.01
    from losses.physics_losses import curl_v_cross_b

    db = dt * curl_v_cross_b(v_t, b_t, dx)
    b_pred = b_t + db
    ind = induction_residual_loss(b_pred, b_t, v_t, dt=dt, dx=dx)
    assert float(ind.item()) < 1e-2


def test_monotonic_hazard() -> None:
    p = torch.tensor([[0.1, 0.3, 0.5, 0.8]])
    assert float(monotonic_hazard_loss(p).item()) == 0.0
    p_bad = torch.tensor([[0.9, 0.3, 0.5, 0.8]])
    assert float(monotonic_hazard_loss(p_bad).item()) > 0.0


def test_mast_labels() -> None:
    times = np.linspace(0, 1, 1000)
    ip_slow = np.linspace(1.0, 0.15, 1000)
    t80, tau = detect_current_quench_onset(times, ip_slow, quench_window_s=0.015, slope_factor=5.0)
    assert t80 is None

    ip_fast = np.ones(1000)
    ip_fast[985:] = np.linspace(0.75, 0.05, 15)
    t80f, _tauf = detect_current_quench_onset(times, ip_fast)
    assert t80f is not None

    ann = annotate_mast_shot(times, ip_fast)
    assert ann.disruptive


def test_disruption_py_filter(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "shot": [1, 1, 2],
            "time": [0.0, 0.01, 0.0],
            "time_until_disrupt": [0.05, 0.04, np.nan],
            "ip": [1e6, 0.9e6, 0.5e6],
            "intentional_disruption": [0, 0, 1],
        }
    )
    path = tmp_path / "exp.csv"
    df.to_csv(path, index=False)
    loaded = load_disruption_export(path)
    assert (loaded["shot"] != 2).all()


def test_q_cyl_and_geometry_sign() -> None:
    q = q_cylindrical(a_m=0.5, r_m=1.5, bt_t=-2.0, ip_ma=-1.0, kappa=1.8)
    assert np.isfinite(q) and q > 0
    row = pd.Series({"RGEO": 1.5, "AMIN": 0.5, "BT": -2.0, "IP": -1.0, "KAREA": 1.8, "DELTA": 0.3})
    v1, _ = row_to_dimensionless_geometry(row)
    row2 = row.copy()
    row2["BT"] = 2.0
    v2, _ = row_to_dimensionless_geometry(row2)
    assert np.allclose(v1[3:], v2[3:])  # q95 path uses abs for I,B in I_N; BT sign flips I_N same with abs


def test_shot_splits_disjoint() -> None:
    shots = list(range(100))
    labels = [i % 3 == 0 for i in shots]
    tr, va, te = shot_level_split(shots, labels, seed=0)
    assert_disjoint_shots(tr, va, te)
    with pytest.raises(ValueError):
        assert_disjoint_shots(tr, tr, te)


def test_horizon_labels() -> None:
    y = horizon_labels(t_end_s=0.90, t_disrupt_s=0.95, horizons_ms=(10, 25, 50, 100))
    assert y[0] == 0.0 and y[1] == 0.0 and y[2] == 1.0 and y[3] == 1.0