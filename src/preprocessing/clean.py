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


# ---------------------------------------------------------------------------
# Edge-IIoTset (Build-Instructions T3.6; TRD.md §6.1)
# ---------------------------------------------------------------------------
# NOTE ON REPO LAYOUT: T3.6's Files line names only `notebooks/04_train_edge_iiotset.ipynb`. The
# cleaning logic lives here rather than in the notebook for the same reason `clean_iotid20` does:
# a second dataset's schema handling is code, and code in a notebook cannot be unit-tested.
#
# Edge-IIoTset is a PACKET-level capture (tshark fields), not a flow-statistics dataset like
# IoTID20. Its 63 columns are 61 features + 2 labels (`Attack_label`, `Attack_type`), which matches
# the 61 raw features `TRD.md §6.1` attributes to it.
#
# THREE DEFECTS IN THIS FILE, ALL MEASURED (see `reports/t3_6_edge_iiotset.md`)
# ----------------------------------------------------------------------------
# 1. PLACEHOLDER SPELLING LEAKS THE LABEL. Normal rows encode an absent field as the string "0";
#    attack rows encode it as "0.0". The two sets never overlap, so the SPELLING of "missing"
#    identifies the class. Five columns are affected -- `mqtt.topic`, `mqtt.protoname`,
#    `mqtt.conack.flags`, `mqtt.msg`, `dns.qry.name.len` -- and three of them reach **100% solo
#    accuracy** with a depth-6 decision tree against an 84.6% majority baseline. This is an
#    artefact of preprocessing the normal and attack captures separately, not a property of
#    attacks. `PLACEHOLDER_TOKENS` normalises every spelling to one token, which removes the
#    artefact at the root; afterwards no feature solo-predicts above 0.95.
#
# 2. `frame.time` IS CORRUPTED. Values are fragments like "6.0", "0.0", or
#    " 2021 22:14:30.939803000 " -- the date has been split away by the original CSV's commas.
#    90% retain a clock time but none retain a usable date, so records cannot be ordered.
#
# 3. THE FILE IS 15 CONTIGUOUS PER-ATTACK BLOCKS, not capture order. Row order therefore cannot
#    substitute for a timestamp.
#
# Consequences 2 and 3 together mean **no valid sequence construction is possible on this file**,
# so T3.6 runs the record-level leakage-free pipeline only. Building windows from row order would
# produce sessions that are label-pure by construction -- an artefact of how the file was
# concatenated, which would inflate results rather than measure anything.
#
# Label convention is unchanged: `Attack_label` 1 = Attack = POSITIVE (`TRD.md §2.3`).

#: Label columns.
EDGE_LABEL_COLUMNS: tuple[str, ...] = ("Attack_label", "Attack_type")

#: Every spelling of "this field was absent" seen in the file. Normalised to one token so the
#: SPELLING cannot encode the class (defect 1 above).
PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {"0", "0.0", "0.00", "", "nan", "NaN", "None", "none"}
)
PLACEHOLDER_NORMALISED: str = "__MISSING__"

#: Addresses and timestamps. Excluded from features, retained as metadata: the host columns are
#: needed for T3.6's GNN sparsity re-check, and `frame.time` is kept only to evidence defect 2.
EDGE_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
)

#: Per-connection nonces and ephemeral identifiers. These identify a specific TCP conversation and
#: carry no meaning across conversations, so a model using them memorises connections rather than
#: learning attack behaviour -- the same reason `Flow_ID` is dropped from IoTID20. Measured solo
#: accuracy before removal: tcp.seq 0.919, tcp.ack 0.916.
#:
#: `tcp.srcport` and `tcp.dstport` are included here, which needs justifying because ports ARE
#: normally informative -- IoTID20 keeps both. The distinction is cardinality. On the 96,429
#: cleaned rows, `tcp.srcport` holds 32,182 distinct values (33.4% of rows) and `tcp.dstport`
#: 23,188 (24.0%). At that ratio they are ephemeral per-connection numbers, not service
#: identifiers; IoTID20's `Dst_Port` has 655 distinct values over 625,415 rows (0.1%), which is
#: what a service port looks like. Keeping them takes a Random Forest to **0.9999 accuracy with
#: zero errors** -- a perfect score is a defect report, not a result. Dropping them gives 0.9839.
#: `udp.port` is KEPT: 32 distinct values, genuinely a service identifier.
EDGE_NONCE_COLUMNS: tuple[str, ...] = (
    "tcp.seq", "tcp.ack", "tcp.ack_raw", "tcp.checksum",
    "icmp.checksum", "icmp.seq_le", "icmp.transmit_timestamp",
    "udp.stream", "dns.qry.name",
    "tcp.srcport", "tcp.dstport",
)

#: Raw payload and request content. `http.request.full_uri` and `tcp.payload` map to exactly ONE
#: attack type for 100% of their non-placeholder values -- they contain the attack strings
#: themselves (SQL injection payloads, XSS vectors). A model using them memorises attack text; it
#: does not detect attacks. A Random Forest on four of these columns alone scores 0.9845 against an
#: 0.8460 baseline.
EDGE_CONTENT_COLUMNS: tuple[str, ...] = (
    "tcp.payload", "tcp.options", "http.file_data", "http.request.uri.query",
    "http.request.full_uri", "http.referer", "mqtt.msg",
)

#: Columns holding a single value across all 157,800 rows.
EDGE_ZERO_VARIANCE_COLUMNS: tuple[str, ...] = (
    "icmp.unused", "http.tls_port", "dns.qry.type", "dns.retransmit_request_in",
    "mqtt.msg_decoded_as", "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
)

EDGE_CSV_RELATIVE_PATH: str = (
    "Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"
)


def resolve_edge_iiotset_csv(dataset_path: str | Path | None = None) -> Path:
    """Locate the Edge-IIoTset ML-ready CSV.

    Args:
        dataset_path: direct path to the CSV, or None to read the pointer written by
            `scripts/download_data.py`.

    Returns:
        Path to `ML-EdgeIIoT-dataset.csv`.

    Raises:
        FileNotFoundError: if the pointer or the CSV is missing.
    """
    if dataset_path is not None:
        path = Path(dataset_path)
        if path.is_dir():
            path = path / EDGE_CSV_RELATIVE_PATH
        if not path.exists():
            raise FileNotFoundError(f"Edge-IIoTset CSV not found at {path}")
        return path

    pointer = RAW_DIR / "EDGE_IIOTSET_PATH.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            f"{pointer} not found. Run: python scripts/download_data.py --edge-iiotset"
        )
    return Path(pointer.read_text(encoding="utf-8").strip())


def load_edge_iiotset(
    dataset_path: str | Path | None = None, nrows: int | None = None
) -> pd.DataFrame:
    """Load the raw Edge-IIoTset CSV with no transformation applied.

    Args:
        dataset_path: path to the CSV or its directory, or None to auto-resolve.
        nrows: optional row cap for smoke tests. Never use for reported results.

    Returns:
        The raw DataFrame (63 columns).
    """
    csv_path = resolve_edge_iiotset_csv(dataset_path)
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    logger.info("Loaded Edge-IIoTset: %d rows x %d columns", len(df), df.shape[1])
    return df


def normalise_placeholders(series: pd.Series) -> pd.Series:
    """Collapse every spelling of "field absent" into one token.

    This removes defect 1 (see the section header): the file spells an absent field "0" in normal
    rows and "0.0" in attack rows, so the spelling alone identifies the class. Normalising is
    preferred to dropping the affected columns, because it removes the artefact while keeping
    whatever legitimate signal the column carries when the field IS present.

    Args:
        series: the column to normalise.

    Returns:
        The column with all placeholder spellings replaced by `PLACEHOLDER_NORMALISED`.
    """
    text = series.astype(str).str.strip()
    return text.where(~text.isin(PLACEHOLDER_TOKENS), PLACEHOLDER_NORMALISED)


def clean_edge_iiotset(
    df: pd.DataFrame, drop_duplicates: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Clean Edge-IIoTset into features / labels / metadata.

    Steps, in order:
      1. Normalise placeholder spellings across every column (defect 1).
      2. Separate labels and identifier columns.
      3. Drop per-connection nonces, raw content columns, and zero-variance columns.
      4. Ordinal-encode the surviving low-cardinality categoricals; coerce the rest to numeric.
      5. Encode labels with **Attack = 1 = positive** (`TRD.md §2.3`).

    Args:
        df: raw DataFrame from `load_edge_iiotset`.
        drop_duplicates: drop exact duplicate feature rows. Defaults to False, matching the
            `clean_iotid20` convention. Edge-IIoTset has only 814 duplicates (0.5%), against
            IoTID20's 58.2%, so the choice matters far less here.

    Returns:
        Tuple of `(X, y, meta)`. `y` is 1 = Attack. `meta` carries the identifier columns plus
        `Attack_type`.

    Raises:
        ValueError: if `Attack_label` is missing.
    """
    if "Attack_label" not in df.columns:
        raise ValueError("Edge-IIoTset frame is missing the required `Attack_label` column")

    # Step 1: kill the placeholder artefact before anything reads these columns.
    normalised = df.copy()
    for column in normalised.columns:
        if column not in EDGE_LABEL_COLUMNS and normalised[column].dtype == object:
            normalised[column] = normalise_placeholders(normalised[column])

    dropped = (
        set(EDGE_LABEL_COLUMNS) | set(EDGE_IDENTIFIER_COLUMNS)
        | set(EDGE_NONCE_COLUMNS) | set(EDGE_CONTENT_COLUMNS)
        | set(EDGE_ZERO_VARIANCE_COLUMNS)
    )
    feature_columns = [c for c in normalised.columns if c not in dropped]
    logger.info(
        "Edge-IIoTset feature funnel: %d raw -> %d "
        "(-%d identifiers, -%d nonces, -%d content, -%d constant)",
        len(normalised.columns) - len(EDGE_LABEL_COLUMNS), len(feature_columns),
        len(EDGE_IDENTIFIER_COLUMNS), len(EDGE_NONCE_COLUMNS),
        len(EDGE_CONTENT_COLUMNS), len(EDGE_ZERO_VARIANCE_COLUMNS),
    )

    # Step 4: numeric where possible; ordinal-encode the remaining categoricals.
    X = pd.DataFrame(index=normalised.index)
    for column in feature_columns:
        numeric = pd.to_numeric(normalised[column], errors="coerce")
        if numeric.notna().mean() > 0.99:
            X[column] = numeric.fillna(0.0)
        else:
            X[column] = normalised[column].astype("category").cat.codes.astype(np.float64)

    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    X = X[keep]

    n_duplicates = int(X.duplicated().sum())
    logger.info("Exact duplicate feature rows: %d (drop_duplicates=%s)", n_duplicates, drop_duplicates)
    if drop_duplicates:
        X = X[~X.duplicated()]

    y = df.loc[X.index, "Attack_label"].astype(np.int8)
    meta_columns = [c for c in EDGE_IDENTIFIER_COLUMNS if c in df.columns]
    if "Attack_type" in df.columns:
        meta_columns.append("Attack_type")
    meta = df.loc[X.index, meta_columns].copy()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    logger.info(
        "Edge-IIoTset clean complete: X=%s, Attack(1)=%d, Normal(0)=%d",
        X.shape, int((y == POSITIVE_LABEL).sum()), int((y == NEGATIVE_LABEL).sum()),
    )
    return X, y, meta
