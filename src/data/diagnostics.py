"""Canonical physical-quantity vocabulary for cross-machine tokenization."""

from __future__ import annotations

from typing import Final

# Shared across MAST, DIII-D, C-Mod exports (soft X-ray is MAST-only).
PHYSICAL_QUANTITIES: Final[tuple[str, ...]] = (
    "ip",
    "kappa",
    "q95",
    "li",
    "beta_n",
    "greenwald_fraction",
    "z_over_a",
    "n1_amplitude",
    "v_loop",
    "radiated_fraction",
    "soft_xray",
    "unknown",
)

QUANTITY_TO_IDX: dict[str, int] = {q: i for i, q in enumerate(PHYSICAL_QUANTITIES)}

DIMENSIONLESS_QUANTITIES: Final[frozenset[str]] = frozenset(
    {
        "kappa",
        "q95",
        "li",
        "beta_n",
        "greenwald_fraction",
        "z_over_a",
        "n1_amplitude",
        "radiated_fraction",
    }
)

# TokaMark / disruption-py name fragments -> canonical quantity
SIGNAL_ALIASES: dict[str, str] = {
    "equilibrium-ip": "ip",
    "equilibrium-plasma_current": "ip",
    "ip": "ip",
    "equilibrium-elongation": "kappa",
    "kappa": "kappa",
    "equilibrium-q95": "q95",
    "q95": "q95",
    "equilibrium-li": "li",
    "li": "li",
    "equilibrium-beta_normal": "beta_n",
    "beta_n": "beta_n",
    "Greenwald_fraction": "greenwald_fraction",
    "greenwald_fraction": "greenwald_fraction",
    "equilibrium-z": "z_over_a",
    "z_error": "z_over_a",
    "n_equal_1_normalized": "n1_amplitude",
    "Mirnov_norm_btor": "n1_amplitude",
    "v_loop": "v_loop",
    "v_loop_efit": "v_loop",
    "radiated_fraction": "radiated_fraction",
    "soft_x_rays": "soft_xray",
    "soft_xray": "soft_xray",
}


def map_signal_to_quantity(name: str) -> str:
    """Map a dataset-specific channel name to a canonical physical quantity."""
    key = name.strip().lower()
    for alias, qty in SIGNAL_ALIASES.items():
        if alias.lower() in key or key == alias.lower():
            return qty
    for alias, qty in SIGNAL_ALIASES.items():
        if key.endswith(alias.lower().split("-")[-1]):
            return qty
    return "unknown"


def quantity_index(quantity: str) -> int:
    """Index into the shared quantity embedding table."""
    return QUANTITY_TO_IDX.get(quantity, QUANTITY_TO_IDX["unknown"])