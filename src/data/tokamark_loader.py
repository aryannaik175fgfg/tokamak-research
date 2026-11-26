"""TokaMark MAST wrapper with thermal-quench labels and validity masks."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np
import torch

from data.diagnostics import map_signal_to_quantity, quantity_index
from data.disruption_labels import annotate_mast_shot, horizon_labels


def _import_tokamark() -> Any:
    try:
        import tokamark  # noqa: F401
        from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset
        from tokamark.data_split import get_train_test_val_shots
        from tokamark.tasks import get_signals_metadata, get_task_config, get_task_metadata
    except ImportError as e:
        raise ImportError(
            "Install tokamark: pip install git+https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark.git"
        ) from e
    return {
        "initialize_MAST_dataset": initialize_MAST_dataset,
        "initialize_TokaMark_dataset": initialize_TokaMark_dataset,
        "get_train_test_val_shots": get_train_test_val_shots,
        "get_task_config": get_task_config,
        "get_task_metadata": get_task_metadata,
        "get_signals_metadata": get_signals_metadata,
    }


def get_official_shot_splits() -> tuple[list[int], list[int], list[int]]:
    """Published TokaMark 80/10/10 shot lists (train, test, val)."""
    tm = _import_tokamark()
    train, test, val = tm["get_train_test_val_shots"]()
    return train, val, test  # reorder to train, val, test


def _signal_names_from_tasks(task_mag: str, task_dis: str) -> list[str]:
    tm = _import_tokamark()
    cfg_mag = tm["get_task_config"](task_mag)
    cfg_dis = tm["get_task_config"](task_dis)
    names: set[str] = set()
    for cfg in (cfg_mag, cfg_dis):
        for key in ("input", "input_signals", "signals"):
            if key in cfg:
                val = cfg[key]
                if isinstance(val, list):
                    names.update(str(s) for s in val)
                elif isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, list):
                            names.update(str(s) for s in v)
    return sorted(names)


def build_tokamark_iterable(
    shots: list[int],
    task_disruption: str = "task_4-1",
    task_magnetics: str = "task_1-1",
    local_flag: bool = True,
    mast_local_path: Optional[str] = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """
  Lazily yield windowed samples with binary validity masks.

  Requires local or S3 MAST data; fails with install/path message if unavailable.
  """
    tm = _import_tokamark()
    config = tm["get_task_config"](task_disruption)
    task_md = tm["get_task_metadata"](config, verbose=False)
    sig_md = tm["get_signals_metadata"]()
    mast_ds = tm["initialize_MAST_dataset"](
        config_task=config,
        shots_list=shots,
        local_flag=local_flag,
    )
    tok_ds = tm["initialize_TokaMark_dataset"](
        dataset=mast_ds,
        task_metadata=task_md,
        config_metadata=sig_md,
    )
    horizons = (10, 25, 50, 100)
    for sample in tok_ds:
        yield _normalize_tokamark_sample(sample, horizons)


def _normalize_tokamark_sample(
    sample: dict[str, Any],
    horizons_ms: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Map raw TokaMark window to tensors with validity masks (best-effort)."""
    out: dict[str, torch.Tensor] = {}
    labels = sample.get("labels", sample.get("y", None))
    if labels is None:
        labels = torch.zeros(len(horizons_ms))
    elif not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)

    inputs = sample.get("input", sample.get("x", None))
    if inputs is None:
        values = torch.zeros(1)
        valid = torch.zeros(1)
        qids = torch.zeros(1, dtype=torch.long)
    else:
        if isinstance(inputs, dict):
            vals, valids, qids_list = [], [], []
            for name, arr in inputs.items():
                x = torch.as_tensor(arr, dtype=torch.float32).flatten()
                for v in x[-8:]:
                    ok = torch.isfinite(v)
                    vals.append(v if ok else torch.tensor(0.0))
                    valids.append(ok.float())
                    qids_list.append(quantity_index(map_signal_to_quantity(name)))
            values = torch.stack(vals) if vals else torch.zeros(1)
            valid = torch.stack(valids) if valids else torch.zeros(1)
            qids = torch.tensor(qids_list, dtype=torch.long) if qids_list else torch.zeros(1, dtype=torch.long)
        else:
            x = torch.as_tensor(inputs, dtype=torch.float32)
            values = torch.nan_to_num(x.flatten(), nan=0.0)
            valid = (~torch.isnan(x.flatten())).float()
            qids = torch.zeros(values.shape[0], dtype=torch.long)

    geo = sample.get("geometry", torch.zeros(6))
    if not isinstance(geo, torch.Tensor):
        geo = torch.as_tensor(geo, dtype=torch.float32)
    geo_mask = sample.get("geo_mask", torch.ones_like(geo))

    out["values"] = values
    out["valid"] = valid
    out["quantity_ids"] = qids
    out["labels"] = labels[: len(horizons_ms)] if labels.numel() >= len(horizons_ms) else horizon_labels(0.0, None, horizons_ms)
    out["geometry"] = geo[:6] if geo.numel() >= 6 else torch.nn.functional.pad(geo, (0, max(0, 6 - geo.numel())))
    out["geo_mask"] = geo_mask[:6] if geo_mask.numel() >= 6 else torch.ones(6)
    out["lead_weight"] = torch.tensor(1.0)
    return out


def geometry_from_equilibrium(
    rgeo: float,
    amin: float,
    bt: float,
    ip: float,
    kappa: float,
    delta: float,
    q95: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dimensionless geometry vector from equilibrium scalars (MAST window)."""
    import pandas as pd

    from data.itpa_loader import row_to_dimensionless_geometry

    row = pd.Series(
        {
            "RGEO": rgeo,
            "AMIN": amin,
            "BT": bt,
            "IP": ip,
            "KAREA": kappa,
            "DELTA": delta,
            "Q95": q95 if q95 is not None else np.nan,
        }
    )
    return row_to_dimensionless_geometry(row)