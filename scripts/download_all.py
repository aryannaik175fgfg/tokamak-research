"""Queue all automatable dataset downloads (respects free disk on D:)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ITPA_OSF = {
    "HDB5V2.3.xlsx": "https://osf.io/download/km7fy/",
    "STD5.csv": "https://osf.io/download/38htb/",
    "STD5-IL.csv": "https://osf.io/download/ck9t7/",
}

TOKAMARK_REPO = "UKAEA-IBM-STFC/tokamark-v1"
MHD_REPO = "polymathic-ai/MHD_64"


def _free_gb(drive: str = "D:") -> float:
    usage = shutil.disk_usage(Path(drive + "\\"))
    return usage.free / (1024**3)


def _run(cmd: list[str], *, cwd: Path | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd or ROOT)


def download_itpa() -> None:
    import urllib.request

    dest_dir = DATA / "itpa"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # csv xlsx
    for name, url in ITPA_OSF.items():
        out = dest_dir / name
        if out.exists() and out.stat().st_size > 1000:
            print(f"[skip] {name}")
            continue
        print(f"[itpa] {name}")
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f"  warn: {name}: {e}")
    # DB5.2.3.csv (hrqcf)
    csv_out = dest_dir / "DB5.2.3.csv"
    if not csv_out.exists() or csv_out.stat().st_size < 1_000_000:
        try:
            meta = urllib.request.urlopen(
                "https://api.osf.io/v2/nodes/hrqcf/files/osfstorage/?page%5Bsize%5D=20"
            ).read()
            import json

            files = json.loads(meta)["data"]
            for f in files:
                if f["attributes"]["name"] == "DB5.2.3.csv":
                    urllib.request.urlretrieve(f["links"]["download"], csv_out)
                    break
        except Exception as e:
            print(f"  warn: DB5.2.3.csv: {e}")
    pr08 = DATA / "itpa" / "pr08"
    pr08.mkdir(parents=True, exist_ok=True)
    print("[itpa] PR08: manual UFILE drop at data/itpa/pr08/ (profile DB)")


def download_mhd_hf() -> None:
    import os

    from huggingface_hub import snapshot_download

    # Xet backend can fail mid-transfer on unstable links; classic HF CDN resumes better.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    target = DATA / "mhd_64" / "datasets" / "MHD_64"
    target.mkdir(parents=True, exist_ok=True)
    print("[mhd] HuggingFace snapshot ->", target)
    for attempt in range(1, 6):
        try:
            snapshot_download(
                repo_id=MHD_REPO,
                repo_type="dataset",
                local_dir=str(target),
                allow_patterns=["data/**", "MHD_64.yaml"],
                max_workers=2,
            )
            break
        except Exception as e:
            print(f"[mhd] attempt {attempt}/5 failed: {e}")
            if attempt == 5:
                raise
    # stats at repo data/stats.yaml -> datasets/MHD_64/stats.yaml for the_well layout
    stats_src = target / "data" / "stats.yaml"
    stats_dst = target / "stats.yaml"
    if stats_src.exists() and not stats_dst.exists():
        shutil.copy2(stats_src, stats_dst)


def download_mhd_well_splits(splits: list[str]) -> None:
    from the_well.utils.download import well_download

    base = DATA / "mhd_64"
    for sp in splits:
        print(f"[mhd] the_well curl: split={sp}")
        well_download(base_path=str(base), dataset="MHD_64", split=sp)


def download_tokamark_batches(
    start: int,
    end: int,
    *,
    unpack: bool,
    delete_tar: bool,
) -> None:
    from huggingface_hub import hf_hub_download

    tok_dir = DATA / "tokamark"
    zarr_dir = tok_dir / "restored_zarrs"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    unpack_py = tok_dir / "unpack_all.py"
    if unpack and not unpack_py.exists():
        hf_hub_download(
            repo_id=TOKAMARK_REPO,
            filename="unpack_all.py",
            repo_type="dataset",
            local_dir=str(tok_dir),
        )

    for i in range(start, end + 1):
        name = f"batch_{i:05d}.tar"
        tar = tok_dir / name
        if tar.exists() and tar.stat().st_size > 1e6:
            print(f"[tokamark] have {name}")
        else:
            free = _free_gb()
            if free < 8:
                print(f"[tokamark] stop {name}: only {free:.1f} GB free")
                break
            print(f"[tokamark] downloading {name} ({free:.1f} GB free)")
            hf_hub_download(
                repo_id=TOKAMARK_REPO,
                filename=name,
                repo_type="dataset",
                local_dir=str(tok_dir),
            )
        if unpack and unpack_py.exists():
            print(f"[tokamark] unpack {name}")
            code = _run(
                [sys.executable, str(unpack_py), str(tok_dir), str(zarr_dir), "--workers", "4"]
            )
            if code != 0:
                print(f"[tokamark] unpack failed for {name}")
            elif delete_tar and tar.exists():
                tar.unlink()
                print(f"[tokamark] removed {name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--itpa", action="store_true", help="ITPA OSF files + pr08 dir")
    p.add_argument(
        "--mhd",
        choices=("hf", "well", "both"),
        default="hf",
        help="MHD_64 source (hf recommended on slow/blocked Flatiron)",
    )
    p.add_argument(
        "--mhd-splits",
        default="valid,test,train",
        help="well_download splits (comma-separated)",
    )
    p.add_argument("--tokamark-start", type=int, default=1)
    p.add_argument("--tokamark-end", type=int, default=57)
    p.add_argument("--no-unpack", action="store_true")
    p.add_argument("--keep-tar", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=12.0)
    args = p.parse_args()

    print(f"D: free ~{_free_gb():.1f} GB")
    if _free_gb() < args.min_free_gb:
        print("WARNING: low disk; TokaMark loop will stop early.")

    if args.itpa:
        download_itpa()
    if args.mhd in ("hf", "both"):
        if _free_gb() < 75:
            print("[mhd] skip HF full snapshot: need ~72 GB (use --mhd well or free space)")
        else:
            download_mhd_hf()
    if args.mhd in ("well", "both"):
        download_mhd_well_splits([s.strip() for s in args.mhd_splits.split(",") if s.strip()])
    if args.tokamark_end >= args.tokamark_start:
        download_tokamark_batches(
            args.tokamark_start,
            args.tokamark_end,
            unpack=not args.no_unpack,
            delete_tar=not args.keep_tar,
        )

    _run([sys.executable, str(ROOT / "scripts" / "verify_data.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())