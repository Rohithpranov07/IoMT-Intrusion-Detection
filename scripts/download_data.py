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

#: Edge-IIoTset (T3.6). The full dataset is 1.63 GB because it bundles raw pcaps; a whole-archive
#: download failed at 12% with a broken pipe. kagglehub's `path=` argument fetches the single
#: ML-ready CSV instead, which is all this project needs.
EDGE_IIOTSET_SLUG = "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
EDGE_IIOTSET_FILE = (
    "Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"
)


def download_edge_iiotset() -> Path:
    """Download the Edge-IIoTset ML-ready CSV and record its path.

    Returns:
        Path to the downloaded CSV.
    """
    path = Path(kagglehub.dataset_download(EDGE_IIOTSET_SLUG, path=EDGE_IIOTSET_FILE))
    print("Path to Edge-IIoTset CSV:", path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "EDGE_IIOTSET_PATH.txt").write_text(str(path) + "\n", encoding="utf-8")
    print(f"  {path.stat().st_size / (1024 * 1024):.1f} MB")
    return path


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
    if "--edge-iiotset" in sys.argv:
        download_edge_iiotset()
    else:
        download_iotid20()
