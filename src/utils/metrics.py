"""Metrics shared by evaluation scripts and tests."""

from __future__ import annotations

import math


def nrmse(mse: float, y_std: float) -> float:
    """RMSE divided by the standard deviation of the target (robustness-paper convention)."""
    if y_std < 1e-12 or not math.isfinite(mse) or not math.isfinite(y_std):
        return float("nan")
    return math.sqrt(mse) / y_std
