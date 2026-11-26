"""Print which datasets are present under data/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _mb(p: Path) -> str:
    if not p.exists():
        return "missing"
    if p.is_file():
        return f"{p.stat().st_size / 1e6:.1f} MB"
    total = 0
    n = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
                n += 1
        except OSError:
            continue
    return f"{total / 1e9:.2f} GB ({n} files)"


def main() -> int:
    checks = [
        ("ITPA DB5.2.3", DATA / "itpa" / "HDB5V2.3.xlsx"),
        ("ITPA PR08 dir", DATA / "itpa" / "pr08"),
        ("MHD_64 cache", DATA / "mhd_64"),
        ("TokaMark", DATA / "tokamark"),
        ("TokaMark robustness (~650MB)", DATA / "tokamark_subset" / "data"),
        ("disruption exports", DATA / "disruption_exports"),
    ]
    ok = 0
    for name, path in checks:
        status = _mb(path)
        present = path.exists() and (path.is_file() or any(path.iterdir()) if path.is_dir() else True)
        if present and status != "missing":
            ok += 1
        print(f"  [{('OK' if present and status != 'missing' else '--')}] {name}: {status}")
    print(f"\n{ok}/{len(checks)} roots have data (see data/README.md for manual steps).")
    return 0


if __name__ == "__main__":
    sys.exit(main())