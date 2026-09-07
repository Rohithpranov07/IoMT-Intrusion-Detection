"""Download IoTID20 from Kaggle via kagglehub and report where it landed.

Usage:
    .venv/bin/python scripts/download_data.py

kagglehub caches under ~/.cache/kagglehub, so re-runs are cheap. The path it returns is
printed and also written to data/raw/IOTID20_PATH.txt so the notebooks can pick it up
without hardcoding a machine-specific path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kagglehub  # noqa: E402

from src.config import IOTID20_KAGGLE_SLUG, RAW_DIR  # noqa: E402


def download_iotid20() -> Path:
    """Download the IoTID20 dataset and return the local directory kagglehub cached it in."""
    path = Path(kagglehub.dataset_download(IOTID20_KAGGLE_SLUG))
    print("Path to dataset files:", path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "IOTID20_PATH.txt").write_text(str(path) + "\n", encoding="utf-8")

    csvs = sorted(path.rglob("*.csv"))
    print(f"\n{len(csvs)} CSV file(s) found:")
    for csv in csvs:
        size_mb = csv.stat().st_size / (1024 * 1024)
        print(f"  {csv.relative_to(path)}  ({size_mb:.1f} MB)")
    return path


if __name__ == "__main__":
    download_iotid20()
