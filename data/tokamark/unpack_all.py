#!/usr/bin/env python3
"""
Reverse of zip_stores.py + batch_archives.py:
  1. Extract each batch_XXXXX.tar to recover the individual .zarr.zip files
  2. Unzip each .zarr.zip back into a real .zarr directory

Usage:
    python unpack_all.py /path/to/batches /path/to/restored_zarrs --workers 8
"""
import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_batch(args):
    tar_path, zips_dir = args
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(zips_dir)
    return tar_path


def unzip_one(args):
    zip_path, dest_dir = args
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return zip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batches_dir", help="folder containing batch_XXXXX.tar files")
    ap.add_argument("output_dir", help="folder where restored .zarr stores will appear")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--keep-intermediate-zips", action="store_true",
                     help="keep the extracted per-store .zip files instead of deleting them after unzip")
    args = ap.parse_args()

    batches_dir = Path(args.batches_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = out_dir / "_zips_tmp"
    zips_dir.mkdir(parents=True, exist_ok=True)

    tars = sorted(batches_dir.glob("*.tar"))
    print(f"Found {len(tars)} batch archives")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_batch, (str(t), str(zips_dir))): t for t in tars}
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            print(f"[{i}/{len(tars)}] extracted {futs[fut].name}")

    zips = sorted(zips_dir.glob("*.zip"))
    print(f"Extracted {len(zips)} zip files from batches. Now unzipping stores...")

    ok, failed = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(unzip_one, (str(z), str(out_dir))): z for z in zips}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                ok += 1
            except Exception as e:
                failed += 1
                print(f"FAILED: {futs[fut]}: {e}")
            if i % 200 == 0 or i == len(zips):
                print(f"[{i}/{len(zips)}] unzipped (ok={ok} failed={failed})")

    if not args.keep_intermediate_zips:
        shutil.rmtree(zips_dir, ignore_errors=True)
        print("Removed intermediate zip files")

    print("Done. Restored zarr stores in:", out_dir)


if __name__ == "__main__":
    main()
