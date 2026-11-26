"""Download ~650 MB TokaMark robustness NumPy arrays (skip 21 GB pickle)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "tokamark_subset",
    )
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="Neerav-Gupta/tokamark-robustness-data",
        repo_type="dataset",
        local_dir=str(args.local_dir),
        ignore_patterns=["*test_raw_samples.pkl", "checkpoints/*", "results/*"],
    )
    n = len(list((args.local_dir / "data").glob("*.npy")))
    print(f"Done: {n} .npy files under {args.local_dir / 'data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())