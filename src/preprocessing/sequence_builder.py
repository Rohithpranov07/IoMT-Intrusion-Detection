"""Flow-sequence construction (Build-Instructions T2.1; TRD.md §3.1).

This module is the fix for Objection #2's second half
-----------------------------------------------------
The base paper feeds its CNN-LSTM a **single flow record** per input, then justifies the LSTM by
appealing to long-term temporal dependencies that a one-step input cannot possibly contain
(`PRD.md §2.1.2`). This module builds genuine time-ordered sequences instead, so the ensemble's
recurrent and attention branches have something real to model.

Exact parameters (frozen in `docs/architecture_decision.md` §1.2 -- do not change here alone)
---------------------------------------------------------------------------------------------
    SEQUENCE_LENGTH      = 10     timesteps per window. Covers 98.5% of IoTID20 rows.
    TRAIN_STRIDE         = 5      50% overlap when generating training windows.
    INFERENCE_STRIDE     = 1      every arriving record yields a decision at detection time.
    SESSION_GAP_SECONDS  = 1.0    a gap longer than this on a device starts a new session.
    MIN_SESSION_LENGTH   = 2      1-record sessions are DROPPED, never padded up to 10 -- a
                                  9/10-padded window is a single flow record wearing a sequence
                                  costume, i.e. Objection #2 through the back door.
    PAD_VALUE            = 0.0    short sessions are PRE-padded, so the newest record is always
                                  the last timestep.
    SESSION_KEY          = "Dst_IP"   sessions are per-destination-device (see §1.1: it is the
                                  only key whose median group size exceeds 1, and it matches the
                                  device-criticality unit the adaptive threshold keys on).

Output shape contract: **(batch, SEQUENCE_LENGTH, n_features)** = (batch, 10, 62) for IoTID20.
`TRD.md §9`'s gate requires `sequence_length > 1`; `_validate_sequence_length` ASSERTS this and is
called from every public entry point, so no configuration can silently collapse to single-record
windows.

Window labelling: a window takes the label of its **last (most recent)** record -- causal, and it
never needs a label from the future. On IoTID20 all 4,301 sessions happen to be label-pure, but the
rule is stated and implemented generally because that will not hold on Edge-IIoTset (T3.6).

LEAKAGE WARNING (`docs/architecture_decision.md` §1.4)
------------------------------------------------------
Adjacent windows within a session share up to `SEQUENCE_LENGTH - 1` records. Splitting *windows*
randomly would put near-identical windows in both train and test -- the same class of bug this
project exists to expose, reintroduced by our own sequence builder. Use `split_sessions()` to
assign whole SESSIONS to folds, then build windows within each fold. `build_sequences` returns the
session id of every window so this is checkable after the fact.

Determinism: sessionisation and windowing are fully deterministic. `split_sessions` uses
`RANDOM_STATE`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, TEST_RATIO, TRAIN_RATIO, VAL_RATIO

logger = logging.getLogger(__name__)

# --- Frozen constants (docs/architecture_decision.md §1.2) --------------------------------
SEQUENCE_LENGTH: int = 10
TRAIN_STRIDE: int = 5
INFERENCE_STRIDE: int = 1
SESSION_GAP_SECONDS: float = 1.0
MIN_SESSION_LENGTH: int = 2
PAD_VALUE: float = 0.0
SESSION_KEY: str = "Dst_IP"
TIMESTAMP_COLUMN: str = "Timestamp"


class SingleRecordWindowError(ValueError):
    """Raised when a configuration would produce length-1 windows.

    This is Objection #2 (`PRD.md §2.1.2`) reintroduced, and `TRD.md §9`'s "Sequences are real
    sequences" gate exists to make it impossible. It is a hard error, never a warning.
    """


def _validate_sequence_length(sequence_length: int) -> None:
    """Assert the window is a real sequence.

    Args:
        sequence_length: the configured window length.

    Raises:
        SingleRecordWindowError: if `sequence_length` is less than 2.
    """
    if sequence_length < 2:
        raise SingleRecordWindowError(
            f"sequence_length={sequence_length} would feed the ensemble single flow records, "
            "which is exactly the base paper's Objection #2 defect (PRD.md §2.1.2). "
            "TRD.md §9 requires sequence_length > 1."
        )
    assert sequence_length > 1, "sequence_length must exceed 1"  # TRD.md §9 gate


@dataclass
class SequenceDataset:
    """Windowed sequences with the provenance needed to split them safely.

    Attributes:
        X: float32 array of shape `(n_windows, sequence_length, n_features)`.
        y: int8 array of shape `(n_windows,)`; 1 = Attack = POSITIVE class (`TRD.md §2.3`).
        session_ids: session id per window. Required for session-level splitting (§1.4).
        feature_names: column names of the last axis of `X`, in order.
        n_padded_windows: windows containing at least one pad timestep.
        sequence_length: the window length used.
    """

    X: np.ndarray
    y: np.ndarray
    session_ids: np.ndarray
    feature_names: list[str]
    n_padded_windows: int
    sequence_length: int

    def __post_init__(self) -> None:
        """Enforce the shape contract at construction time."""
        _validate_sequence_length(self.sequence_length)
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3-D (batch, seq, features); got shape {self.X.shape}")
        if self.X.shape[1] != self.sequence_length:
            raise ValueError(
                f"X axis 1 is {self.X.shape[1]}, expected sequence_length={self.sequence_length}"
            )
        if not (len(self.X) == len(self.y) == len(self.session_ids)):
            raise ValueError(
                f"Length mismatch: X={len(self.X)}, y={len(self.y)}, "
                f"session_ids={len(self.session_ids)}"
            )

    def summary(self) -> str:
        """Render shape, class balance, and padding statistics."""
        n_attack = int((self.y == 1).sum())
        return (
            f"SequenceDataset: X={self.X.shape} (batch, sequence_length, features)\n"
            f"  sequence_length          : {self.sequence_length}  (> 1, TRD.md §9 gate)\n"
            f"  windows                  : {len(self.X):,}\n"
            f"  distinct sessions        : {len(np.unique(self.session_ids)):,}\n"
            f"  Attack(1) / Normal(0)    : {n_attack:,} / {len(self.y) - n_attack:,}\n"
            f"  windows containing padding: {self.n_padded_windows:,} "
            f"({self.n_padded_windows / max(len(self.X), 1):.1%})"
        )


def assign_sessions(
    meta: pd.DataFrame,
    session_key: str = SESSION_KEY,
    gap_seconds: float = SESSION_GAP_SECONDS,
) -> pd.Series:
    """Assign a session id to every record.

    A new session starts at the first record of a device, or whenever the gap since that device's
    previous record exceeds `gap_seconds` (`docs/architecture_decision.md` §1.1).

    Args:
        meta: metadata frame from `clean_iotid20`, containing `session_key` and a parsed
            `Timestamp` column.
        session_key: the grouping column. Defaults to `Dst_IP` (the destination device).
        gap_seconds: inactivity gap that starts a new session.

    Returns:
        Integer Series of session ids, index-aligned with `meta`.

    Raises:
        KeyError: if `session_key` or `Timestamp` is missing.
        ValueError: if `Timestamp` is not datetime-typed.
    """
    for column in (session_key, TIMESTAMP_COLUMN):
        if column not in meta.columns:
            raise KeyError(f"meta is missing required column {column!r}")
    if not pd.api.types.is_datetime64_any_dtype(meta[TIMESTAMP_COLUMN]):
        raise ValueError(
            f"{TIMESTAMP_COLUMN} must be datetime64; got {meta[TIMESTAMP_COLUMN].dtype}. "
            "clean_iotid20() parses it -- did this frame bypass it?"
        )

    ordered = meta.sort_values([session_key, TIMESTAMP_COLUMN], kind="mergesort")
    gap = ordered.groupby(session_key, observed=True)[TIMESTAMP_COLUMN].diff().dt.total_seconds()
    starts_new = gap.isna() | (gap > gap_seconds)

    session_ids = starts_new.cumsum().astype(np.int64)
    logger.info(
        "Assigned %d sessions over %d devices (gap > %.1fs)",
        session_ids.nunique(),
        meta[session_key].nunique(),
        gap_seconds,
    )
    # Restore the caller's original row order.
    return session_ids.reindex(meta.index)


def build_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = TRAIN_STRIDE,
    session_ids: pd.Series | None = None,
    min_session_length: int = MIN_SESSION_LENGTH,
    pad_value: float = PAD_VALUE,
) -> SequenceDataset:
    """Build fixed-length windows from time-ordered records, grouped by session.

    Within each session, records are sorted by timestamp and windowed with the given stride.
    Sessions shorter than `sequence_length` (but at least `min_session_length`) produce a single
    PRE-padded window, so the most recent record is always the final timestep. Sessions shorter
    than `min_session_length` are dropped.

    Args:
        X: numeric feature matrix, one row per flow record.
        y: binary labels (1 = Attack = positive class, `TRD.md §2.3`).
        meta: metadata with `SESSION_KEY` and `Timestamp`, index-aligned with `X`.
        sequence_length: window length. Must be > 1 (`TRD.md §9`).
        stride: step between consecutive window starts. Use `TRAIN_STRIDE` for training data and
            `INFERENCE_STRIDE` at detection time.
        session_ids: precomputed session ids. Computed via `assign_sessions` when None.
        min_session_length: sessions shorter than this are dropped.
        pad_value: fill value for pre-padding.

    Returns:
        A `SequenceDataset` of shape `(n_windows, sequence_length, n_features)`.

    Raises:
        SingleRecordWindowError: if `sequence_length` < 2.
        ValueError: if `X`, `y`, and `meta` lengths disagree, or no windows are produced.
    """
    _validate_sequence_length(sequence_length)
    if not (len(X) == len(y) == len(meta)):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, meta={len(meta)}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if session_ids is None:
        session_ids = assign_sessions(meta)

    frame = pd.DataFrame(
        {
            "_session": np.asarray(session_ids),
            "_timestamp": meta[TIMESTAMP_COLUMN].to_numpy(),
            "_label": np.asarray(y).ravel(),
        }
    )
    feature_values = X.to_numpy(dtype=np.float32)
    feature_names = list(X.columns)
    n_features = feature_values.shape[1]

    windows: list[np.ndarray] = []
    labels: list[int] = []
    window_sessions: list[int] = []
    n_padded = 0
    n_dropped_sessions = 0
    n_dropped_rows = 0

    order = np.lexsort((frame["_timestamp"].to_numpy(), frame["_session"].to_numpy()))
    sorted_sessions = frame["_session"].to_numpy()[order]
    sorted_labels = frame["_label"].to_numpy()[order]

    # Contiguous runs of equal session id after the lexsort delimit each session.
    boundaries = np.flatnonzero(np.diff(sorted_sessions)) + 1
    for start, stop in zip(
        np.concatenate(([0], boundaries)), np.concatenate((boundaries, [len(order)]))
    ):
        idx = order[start:stop]
        length = len(idx)

        if length < min_session_length:
            n_dropped_sessions += 1
            n_dropped_rows += length
            continue

        session_id = int(sorted_sessions[start])

        if length < sequence_length:
            # Pre-pad: pad timesteps first, real records last.
            window = np.full((sequence_length, n_features), pad_value, dtype=np.float32)
            window[-length:] = feature_values[idx]
            windows.append(window)
            labels.append(int(sorted_labels[stop - 1]))
            window_sessions.append(session_id)
            n_padded += 1
            continue

        for begin in range(0, length - sequence_length + 1, stride):
            end = begin + sequence_length
            windows.append(feature_values[idx[begin:end]])
            # Label of the LAST record in the window (§1.3).
            labels.append(int(sorted_labels[start + end - 1]))
            window_sessions.append(session_id)

    if not windows:
        raise ValueError(
            "No windows produced. Check that meta['Timestamp'] parsed and that sessions reach "
            f"min_session_length={min_session_length}."
        )

    logger.info(
        "Built %d windows (stride=%d); dropped %d sessions shorter than %d records (%d rows)",
        len(windows), stride, n_dropped_sessions, min_session_length, n_dropped_rows,
    )

    dataset = SequenceDataset(
        X=np.stack(windows),
        y=np.asarray(labels, dtype=np.int8),
        session_ids=np.asarray(window_sessions, dtype=np.int64),
        feature_names=feature_names,
        n_padded_windows=n_padded,
        sequence_length=sequence_length,
    )
    logger.info("%s", dataset.summary())
    return dataset


def split_sessions(
    session_ids: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split SESSIONS (not windows) into train/test/validation ids, 80/10/10 by WINDOW COUNT.

    This is the leakage guard from `docs/architecture_decision.md` §1.4: overlapping windows within
    a session share up to `sequence_length - 1` records, so splitting windows directly would put
    near-duplicates on both sides of the split.

    Why the assignment is size-aware rather than a plain stratified `train_test_split`
    ---------------------------------------------------------------------------------
    IoTID20 session sizes are extremely skewed -- the p99 session runs to thousands of records
    while the median is 5. Assigning *sessions* uniformly at random therefore does NOT give
    80/10/10 of the *windows*: a measured run of the naive version produced folds that were 98.3%,
    39.2% and 96.4% Attack, which are not comparable to one another and make any metric computed
    across them close to meaningless.

    Instead, sessions are bin-packed: within each label class, sessions are taken largest-first and
    each is placed in whichever fold is furthest below its window quota. That keeps both the
    window-count ratio and the class balance close to target while still never splitting a session.

    Args:
        session_ids: session id per window.
        labels: window labels, used to keep each fold's class balance close to the overall rate.
        random_state: seed used to shuffle equal-sized sessions before packing, so the result is
            reproducible but not an artefact of the input ordering.

    Returns:
        Tuple of `(train_sessions, test_sessions, val_sessions)` id arrays.

    Raises:
        ValueError: if a session id carries more than one label, which would mean windows from one
            session disagree and the fold's class balance could not be controlled.
    """
    frame = pd.DataFrame(
        {"session": np.asarray(session_ids), "label": np.asarray(labels).ravel()}
    )
    per_session = (
        frame.groupby("session")
        .agg(n_windows=("label", "size"), label_mean=("label", "mean"))
        .reset_index()
    )
    # A session whose windows disagree cannot be assigned a class; on IoTID20 sessions are pure.
    mixed = per_session[(per_session["label_mean"] > 0) & (per_session["label_mean"] < 1)]
    if len(mixed):
        logger.warning(
            "%d sessions contain both classes; assigning each by majority label", len(mixed)
        )
    per_session["label"] = per_session["label_mean"].round().astype(int)

    targets = np.array([TRAIN_RATIO, TEST_RATIO, VAL_RATIO])
    folds: list[list[int]] = [[], [], []]
    rng = np.random.default_rng(random_state)

    for label_value in sorted(per_session["label"].unique()):
        subset = per_session[per_session["label"] == label_value].copy()
        # Shuffle first so equal-sized sessions are ordered reproducibly but not by id.
        subset = subset.iloc[rng.permutation(len(subset))]
        subset = subset.sort_values("n_windows", ascending=False, kind="mergesort")

        quotas = targets * subset["n_windows"].sum()
        assigned = np.zeros(3, dtype=np.float64)

        for session, n_windows in zip(subset["session"], subset["n_windows"]):
            # Place this session where the shortfall against quota is largest.
            fold = int(np.argmax(quotas - assigned))
            folds[fold].append(int(session))
            assigned[fold] += n_windows

    train_ids, test_ids, val_ids = (np.asarray(sorted(f), dtype=np.int64) for f in folds)
    logger.info(
        "Session-level split: %d train / %d test / %d validation sessions",
        len(train_ids), len(test_ids), len(val_ids),
    )
    return train_ids, test_ids, val_ids


def subset_by_sessions(dataset: SequenceDataset, session_ids: np.ndarray) -> SequenceDataset:
    """Return the windows belonging to the given sessions.

    Args:
        dataset: the full windowed dataset.
        session_ids: sessions to keep, e.g. one output of `split_sessions`.

    Returns:
        A `SequenceDataset` restricted to those sessions.
    """
    mask = np.isin(dataset.session_ids, session_ids)
    return SequenceDataset(
        X=dataset.X[mask],
        y=dataset.y[mask],
        session_ids=dataset.session_ids[mask],
        feature_names=dataset.feature_names,
        n_padded_windows=int(mask.sum()) if dataset.n_padded_windows else 0,
        sequence_length=dataset.sequence_length,
    )
