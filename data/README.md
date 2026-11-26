# Dataset layout

| Path | Dataset | Approx. size | Status |
|------|---------|--------------|--------|
| `itpa/HDB5V2.3.xlsx` | ITPA DB5.2.3 globals | ~13 MB | `scripts/fetch_data.ps1 -Stage itpa` or [OSF DB5.2.3](https://osf.io/hrqcf/) |
| `itpa/pr08/` | PR08 UFILE bundle | varies | manual — [profile DB](https://tokamak-profiledb.ccfe.ac.uk/) |
| `mhd_64/` | The Well MHD_64 (optional local cache) | ~72 GB | stream via HF or `-Stage mhd` |
| `tokamark/` | MAST zarr stores | ~565 GB full | `-Stage tokamark` (subset) or S3 remote |
| `tokamark_subset/` | [robustness NumPy](https://huggingface.co/datasets/Neerav-Gupta/tokamark-robustness-data) | **~650 MB** | `py -3.11 scripts/fetch_tokamark_subset.py` (skips 21 GB `test_raw_samples.pkl`) |
| `disruption_exports/` | disruption-py CSV/parquet | MB–GB | MIT/GA access or team exports |

**Recommended (low disk):**

- **MHD_64:** do **not** cache locally — `configs/pretrain_mhd.yaml` uses `well_base_path: hf://datasets/polymathic-ai/` (streams per batch; ~0 GB on disk).
- **MAST / disruption:** `scripts/fetch_tokamark_subset.py` (~652 MB `.npy`), or keep `data/tokamark/restored_zarrs` for full zarr training.

**Optional heavy cache:** `py -3.11 scripts/download_all.py --mhd hf` (~72 GB MHD) + TokaMark batches (needs **600+ GB** for everything).

**Automated queue:** `py -3.11 scripts/download_all.py --itpa --mhd hf --tokamark-start 1 --tokamark-end 57`

After fetch, paths are written to `data/paths.yaml` and referenced from training configs.