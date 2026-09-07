"""IoTID20 loading and cleaning (Build-Instructions T1.1; base paper §II.G.1).

Exact, non-negotiable numbers used by this module
-------------------------------------------------
Raw IoTID20 CSV                 : 625,783 rows x 86 columns
                                  = 83 feature columns + 3 label columns
                                    (`Label`, `Cat`, `Sub_Cat`)
                                  83 matches the base paper's stated raw feature count (TRD.md §6.1).

Feature funnel (each step is explicit; no silent drops)
    83 raw feature columns
  -  4 identifier columns moved to metadata : Flow_ID, Src_IP, Dst_IP, Timestamp
  = 79
  - 10 zero-variance columns dropped        : Fwd_PSH_Flags, Fwd_URG_Flags, Fwd_Byts/b_Avg,
                                              Fwd_Pkts/b_Avg, Fwd_Blk_Rate_Avg, Bwd_Byts/b_Avg,
                                              Bwd_Pkts/b_Avg, Bwd_Blk_Rate_Avg,
                                              Init_Fwd_Win_Byts, Fwd_Seg_Size_Min
  = 69 usable numeric features handed to feature selection
       (feature_select.py then takes the top 62 -> the base paper's post-selection count)

Row cleaning
    - `Flow_Byts/s` and `Flow_Pkts/s` contain +/-inf. inf -> NaN, then rows with NaN are dropped
      (the base paper's §II.G.1 "drop NaN/irrelevant indices" step).
    - The raw 86-column file contains 163,946 exact duplicate rows; after the identifier and
      zero-variance columns are removed, 363,884 of the surviving 625,415 rows (58.2%) are exact
      duplicates in the 69-feature space. The base paper never mentions removing
      them, so `drop_duplicates` DEFAULTS TO FALSE to stay faithful to the pipeline under critique.
      Duplicates are an *independent* train/test leakage vector from the SMOTE-ordering bug this
      project targets; `clean_iotid20()` reports the count so the notebooks can discuss it, and the
      flag can be flipped for a sensitivity check.

Label convention (TRD.md §2.3) -- THIS PROJECT USES **ATTACK AS THE POSITIVE CLASS**
    Raw `Label` column values : "Anomaly" (585,710 rows) / "Normal" (40,073 rows)
    Binary encoding           : Anomaly -> 1 (POSITIVE, "Attack")
                                Normal  -> 0 (negative)
    This is deliberately the OPPOSITE of the base paper, which treats Normal as positive
    (PRD.md §2.1.3). Every metric in this repo is computed under the convention above.

Determinism: this module performs no randomised operation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NEGATIVE_LABEL, POSITIVE_LABEL, RAW_DIR

logger = logging.getLogger(__name__)

# --- Column groups (named constants; see the feature funnel in the module docstring) --------
LABEL_COLUMNS: tuple[str, ...] = ("Label", "Cat", "Sub_Cat")

#: Identifier / non-predictive columns. Excluded from the FEATURE matrix but retained as
#: metadata: `Src_IP`/`Dst_IP` are needed for the GNN go/no-go graph (T2.7) and `Timestamp` +
#: `Flow_ID` are needed to order records into sequences (T2.1).
IDENTIFIER_COLUMNS: tuple[str, ...] = ("Flow_ID", "Src_IP", "Dst_IP", "Timestamp")

#: Columns holding a single value across all 625,783 rows -- zero information, dropped.
ZERO_VARIANCE_COLUMNS: tuple[str, ...] = (
    "Fwd_PSH_Flags",
    "Fwd_URG_Flags",
    "Fwd_Byts/b_Avg",
    "Fwd_Pkts/b_Avg",
    "Fwd_Blk_Rate_Avg",
    "Bwd_Byts/b_Avg",
    "Bwd_Pkts/b_Avg",
    "Bwd_Blk_Rate_Avg",
    "Init_Fwd_Win_Byts",
    "Fwd_Seg_Size_Min",
)

#: Raw `Label` value that maps to the POSITIVE class (Attack) under TRD.md §2.3.
ATTACK_LABEL_VALUE: str = "Anomaly"
NORMAL_LABEL_VALUE: str = "Normal"

#: `Timestamp` format in the raw CSV, e.g. "25/07/2019 03:25:53 AM" (day-first, 12-hour).
TIMESTAMP_FORMAT: str = "%d/%m/%Y %I:%M:%S %p"

EXPECTED_RAW_COLUMNS: int = 86
EXPECTED_RAW_FEATURE_COLUMNS: int = 83


def resolve_iotid20_csv(dataset_dir: str | Path | None = None) -> Path:
    """Locate the main IoTID20 CSV.

    Args:
        dataset_dir: kagglehub download directory. If None, read the path recorded by
            `scripts/download_data.py` in `data/raw/IOTID20_PATH.txt`.

    Returns:
        Path to `IoT Network Intrusion Dataset.csv`.

    Raises:
        FileNotFoundError: if the path file or the CSV itself is missing.
    """
    if dataset_dir is None:
        pointer = RAW_DIR / "IOTID20_PATH.txt"
        if not pointer.exists():
            raise FileNotFoundError(
                f"{pointer} not found. Run: python scripts/download_data.py"
            )
        dataset_dir = pointer.read_text(encoding="utf-8").strip()

    csv_path = Path(dataset_dir) / "IoT Network Intrusion Dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"IoTID20 CSV not found at {csv_path}")
    return csv_path


def load_iotid20(
    dataset_dir: str | Path | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Load the raw IoTID20 CSV with no transformation applied.

    Args:
        dataset_dir: kagglehub download directory, or None to auto-resolve.
        nrows: optional row cap, for fast smoke tests only. Never use for reported results.

    Returns:
        The raw DataFrame (86 columns).
    """
    csv_path = resolve_iotid20_csv(dataset_dir)
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    logger.info("Loaded IoTID20: %d rows x %d columns from %s", len(df), df.shape[1], csv_path)

    if nrows is None and df.shape[1] != EXPECTED_RAW_COLUMNS:
        logger.warning(
            "Expected %d columns (83 features + 3 labels), found %d",
            EXPECTED_RAW_COLUMNS,
            df.shape[1],
        )
    return df


def encode_binary_labels(labels: pd.Series) -> pd.Series:
    """Map the raw `Label` column to this project's binary convention.

    POSITIVE CLASS = **Attack** (raw value "Anomaly") -> 1
    negative class = Normal                            -> 0
    This is TRD.md §2.3's convention and is the OPPOSITE of the base paper's (PRD.md §2.1.3).

    Args:
        labels: the raw `Label` column.

    Returns:
        Integer Series of 0/1 with the mapping above.

    Raises:
        ValueError: if the column contains a value other than "Anomaly"/"Normal".
    """
    unexpected = set(labels.unique()) - {ATTACK_LABEL_VALUE, NORMAL_LABEL_VALUE}
    if unexpected:
        raise ValueError(f"Unexpected values in Label column: {sorted(unexpected)}")

    return labels.map(
        {ATTACK_LABEL_VALUE: POSITIVE_LABEL, NORMAL_LABEL_VALUE: NEGATIVE_LABEL}
    ).astype(np.int8)


def clean_iotid20(
    df: pd.DataFrame, drop_duplicates: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Clean the raw IoTID20 frame and split it into features / labels / metadata.

    Steps, in order (base paper §II.G.1, expanded to be explicit about every drop):
      1. Separate the 3 label columns and the 4 identifier columns from the 83 raw features.
      2. Replace +/-inf with NaN (affects `Flow_Byts/s`, `Flow_Pkts/s`).
      3. Drop rows containing NaN in any remaining feature.
      4. Drop the 10 zero-variance columns.
      5. Optionally drop exact duplicate feature rows (OFF by default -- see module docstring).
      6. Encode labels with **Attack (Anomaly) = 1 = positive class**.

    Args:
        df: raw DataFrame from `load_iotid20`.
        drop_duplicates: drop exact duplicate feature rows. Defaults to False to match the
            base paper's described pipeline, which never mentions this step.

    Returns:
        Tuple of:
          - X: numeric feature matrix (69 columns for the full dataset).
          - y: binary labels, 1 = Attack (positive class per TRD.md §2.3).
          - meta: retained identifier columns (`Flow_ID`, `Src_IP`, `Dst_IP`, `Timestamp` as a
            parsed datetime) plus `Cat`/`Sub_Cat`, index-aligned with X and y. Needed by the
            sequence builder (T2.1) and the GNN sparsity check (T2.7).
    """
    n_start = len(df)

    labels_present = [c for c in LABEL_COLUMNS if c in df.columns]
    if "Label" not in labels_present:
        raise ValueError("Raw frame is missing the required `Label` column")

    ids_present = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in labels_present and c not in ids_present]
    logger.info(
        "Feature funnel: %d raw -> %d after removing %d identifier columns",
        len(feature_cols) + len(ids_present),
        len(feature_cols),
        len(ids_present),
    )

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Step 2: +/-inf is not a value any downstream scaler or model can use.
    n_inf = int(np.isinf(X.to_numpy(dtype=np.float64, na_value=np.nan)).sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    logger.info("Replaced %d infinite values with NaN", n_inf)

    # Step 3: drop rows with any NaN.
    keep_mask = X.notna().all(axis=1)
    X = X[keep_mask]
    logger.info("Dropped %d rows containing NaN (%d remain)", n_start - len(X), len(X))

    # Step 4: drop zero-variance columns (declared constants, verified against this frame).
    to_drop = [c for c in ZERO_VARIANCE_COLUMNS if c in X.columns]
    unexpected_constants = [c for c in X.columns if c not in to_drop and X[c].nunique() <= 1]
    if unexpected_constants:
        logger.warning(
            "Columns constant in this frame but not in ZERO_VARIANCE_COLUMNS: %s "
            "(kept, so the declared constant list stays the source of truth)",
            unexpected_constants,
        )
    X = X.drop(columns=to_drop)
    logger.info("Dropped %d zero-variance columns -> %d features", len(to_drop), X.shape[1])

    # Step 5: optional duplicate removal (see module docstring for why it is off by default).
    n_duplicates = int(X.duplicated().sum())
    logger.info(
        "Exact duplicate feature rows present: %d (drop_duplicates=%s)",
        n_duplicates,
        drop_duplicates,
    )
    if drop_duplicates:
        X = X[~X.duplicated()]
        logger.info("Dropped duplicates -> %d rows remain", len(X))

    # Step 6: labels + metadata, index-aligned with the surviving rows.
    y = encode_binary_labels(df.loc[X.index, "Label"])

    meta = df.loc[X.index, ids_present + [c for c in ("Cat", "Sub_Cat") if c in df.columns]].copy()
    if "Timestamp" in meta.columns:
        meta["Timestamp"] = pd.to_datetime(
            meta["Timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
        )
        n_unparsed = int(meta["Timestamp"].isna().sum())
        if n_unparsed:
            logger.warning("%d Timestamp values did not match %s", n_unparsed, TIMESTAMP_FORMAT)

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    logger.info(
        "Clean complete: X=%s, class balance -> Attack(1)=%d, Normal(0)=%d",
        X.shape,
        int((y == POSITIVE_LABEL).sum()),
        int((y == NEGATIVE_LABEL).sum()),
    )
    return X, y, meta
