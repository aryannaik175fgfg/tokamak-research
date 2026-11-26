"""Shot-level disruption alarms need disruption timestamps we do not ship."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "data" / "tokamark_subset" / "data" / "test_raw_samples.pkl"


def main() -> int:
    if not PKL.is_file():
        print(
            "Shot-level TPR / false-alarm rate is not computed in this repo.\n"
            f"Missing {PKL.relative_to(ROOT)} (~21 GB; scripts/fetch_tokamark_subset.py skips it).\n"
            "That file has per-window t_cut. Without it, an alarm rate would be invented.\n"
            "\n"
            "What you can report today:\n"
            "  py -3.11 scripts/eval_benchmark_report.py\n"
            "    clean-test NRMSE for normalized Ip, 100 ms ahead.\n"
            "\n"
            "scripts/eval_tpr_far.py is a proxy (low true Ip), not a disruption alarm.\n"
            "Do not put those numbers in a paper table next to the robustness TPR column."
        )
        return 2
    print(
        f"Found {PKL}. Wire this file to the robustness repo's compute_alarm_metrics.py "
        "before quoting shot-level TPR."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())