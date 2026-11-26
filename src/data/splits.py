"""Shot-level train/val/test partitioning without time-series leakage."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def shot_level_split(
    shot_ids: Sequence[int],
    labels: Sequence[int],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
  Partition shots into train/val/test with stratification on binary labels.

  Parameters
  ----------
  shot_ids : sequence of shot identifiers
  labels : per-shot binary label (e.g. ever disruptive)
  ratios : (train, val, test) fractions summing to 1
  seed : RNG seed for reproducibility

  Returns
  -------
  train_ids, val_ids, test_ids : disjoint shot lists
  """
    if len(shot_ids) != len(labels):
        raise ValueError("shot_ids and labels must have the same length")
    r_train, r_val, r_test = ratios
    if abs(r_train + r_val + r_test - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    rng = np.random.default_rng(seed)
    shot_ids = list(shot_ids)
    labels = np.asarray(labels, dtype=int)

    pos = [s for s, y in zip(shot_ids, labels) if y == 1]
    neg = [s for s, y in zip(shot_ids, labels) if y == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def _split_bucket(bucket: list[int]) -> tuple[list[int], list[int], list[int]]:
        n = len(bucket)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = max(0, n - n_train)
        return (
            bucket[:n_train],
            bucket[n_train : n_train + n_val],
            bucket[n_train + n_val :],
        )

    t_p, v_p, te_p = _split_bucket(pos)
    t_n, v_n, te_n = _split_bucket(neg)
    train = t_p + t_n
    val = v_p + v_n
    test = te_p + te_n
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    assert_disjoint_shots(train, val, test)
    return train, val, test


def assert_disjoint_shots(
    train: Iterable[int],
    val: Iterable[int],
    test: Iterable[int],
) -> None:
    """Raise if any shot_id appears in more than one split."""
    st, sv, se = set(train), set(val), set(test)
    overlap = (st & sv) | (st & se) | (sv & se)
    if overlap:
        raise ValueError(f"shot-level leakage: overlapping ids {sorted(overlap)}")