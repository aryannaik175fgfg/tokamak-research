"""Per-epoch training metrics to CSV and human-readable log."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EpochLogger:
    def __init__(self, log_dir: str | Path, csv_name: str = "metrics.csv", log_name: str = "train.log") -> None:
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / csv_name
        self.log_path = self.dir / log_name
        self._fields: list[str] | None = None
        if not self.csv_path.is_file():
            self._write_log(f"# tokamak training started {self._ts()}\n")

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _write_log(self, line: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line if line.endswith("\n") else line + "\n")

    def log_epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        row = {"epoch": epoch, **{k: metrics[k] for k in sorted(metrics)}}
        if self._fields is None:
            self._fields = list(row.keys())
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow(row)

        parts = [f"epoch={epoch}"]
        for k in sorted(metrics):
            v = metrics[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:.6f}")
            else:
                parts.append(f"{k}={v}")
        self._write_log(f"[{self._ts()}] " + " ".join(parts))

    def log_message(self, msg: str) -> None:
        self._write_log(f"[{self._ts()}] {msg}")

    def save_config(self, cfg: dict[str, Any]) -> None:
        (self.dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")