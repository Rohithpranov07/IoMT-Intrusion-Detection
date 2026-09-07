"""Project-wide constants.

Single source of truth for values that must not drift between modules (Build-Instructions §A.3):

- RANDOM_STATE = 42                : the one seed used for every split, resample, and weight init.
- SPLIT_RATIOS  = 0.80/0.10/0.10   : train/test/validation, matching the base paper's reported
                                     ratios (TRD.md §6.2) so our numbers stay comparable to theirs.
- POSITIVE_CLASS_NAME = "Attack"   : THIS PROJECT'S positive-class convention (TRD.md §2.3).
                                     The base paper uses "Normal" as positive; we deliberately do
                                     not, and every reported metric in this repo says so.
- POSITIVE_LABEL = 1               : integer encoding of the positive class in binary labels.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE: int = 42

# --- Split ratios (TRD.md §6.2) -------------------------------------------
TRAIN_RATIO: float = 0.80
TEST_RATIO: float = 0.10
VAL_RATIO: float = 0.10

# --- Positive-class convention (TRD.md §2.3) ------------------------------
# Attack is the positive class for EVERY metric this project reports.
POSITIVE_CLASS_NAME: str = "Attack"
NEGATIVE_CLASS_NAME: str = "Normal"
POSITIVE_LABEL: int = 1
NEGATIVE_LABEL: int = 0

# --- Paths -----------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
DOCS_DIR: Path = REPO_ROOT / "docs"
REPORTS_DIR: Path = REPO_ROOT / "reports"
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"

# --- Dataset identifiers ---------------------------------------------------
IOTID20_KAGGLE_SLUG: str = "rohulaminlabid/iotid20-dataset"

# Base paper's feature counts, for reference only (TRD.md §6.1).
# We do NOT claim to reproduce its PSO; see docs/feature_selection_decision.md.
IOTID20_RAW_FEATURE_COUNT: int = 83
IOTID20_SELECTED_FEATURE_COUNT: int = 62
EDGE_IIOTSET_RAW_FEATURE_COUNT: int = 61
EDGE_IIOTSET_SELECTED_FEATURE_COUNT: int = 46
