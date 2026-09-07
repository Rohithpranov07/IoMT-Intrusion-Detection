"""Unit tests for the sequence builder (Build-Instructions T2.1 VERIFY block).

The VERIFY condition is: *"Unit test passes; assertion present and cannot be silently bypassed by a
window-length-1 configuration."* `test_sequence_length_one_is_rejected` and
`test_dataclass_rejects_length_one` cover the second half from both directions -- the builder
entry point and the dataclass contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sequence_builder import (
    MIN_SESSION_LENGTH,
    SEQUENCE_LENGTH,
    SequenceDataset,
    SingleRecordWindowError,
    assign_sessions,
    build_sequences,
    split_sessions,
    subset_by_sessions,
)

N_FEATURES = 4
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]


def make_synthetic(
    n_devices: int = 3, records_per_device: int = 25, gap_after: int | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build a small synthetic dataset with known structure.

    Args:
        n_devices: number of distinct destination devices.
        records_per_device: records emitted per device, 0.1 s apart.
        gap_after: if given, insert a 10 s gap after this many records on each device, which
            should split that device's traffic into exactly two sessions.

    Returns:
        Tuple of (X, y, meta) in the same shape `clean_iotid20` returns.
    """
    rows, labels, metas = [], [], []
    base = pd.Timestamp("2019-07-25 03:00:00")

    for device in range(n_devices):
        clock = base
        for record in range(records_per_device):
            if gap_after is not None and record == gap_after:
                clock += pd.Timedelta(seconds=10)
            rows.append(np.full(N_FEATURES, float(device * 1000 + record)))
            labels.append(device % 2)  # alternating label per device; sessions stay pure
            metas.append({"Dst_IP": f"10.0.0.{device}", "Timestamp": clock})
            clock += pd.Timedelta(seconds=0.1)

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels, dtype=np.int8, name="Label")
    meta = pd.DataFrame(metas)
    return X, y, meta


# --- The core shape contract (TRD.md §9 "Sequences are real sequences") --------------------


def test_output_shape_is_batch_seq_features() -> None:
    """Output must be (batch, sequence_length, n_features), not (batch, n_features)."""
    X, y, meta = make_synthetic()
    dataset = build_sequences(X, y, meta, sequence_length=10, stride=5)

    assert dataset.X.ndim == 3, "Input must be 3-D or the BiLSTM/attention branches are pointless"
    assert dataset.X.shape[1] == 10
    assert dataset.X.shape[2] == N_FEATURES
    assert dataset.sequence_length == 10


def test_default_sequence_length_matches_architecture_decision() -> None:
    """The default must equal the value frozen in docs/architecture_decision.md §1.2."""
    assert SEQUENCE_LENGTH == 10


def test_window_contents_are_consecutive_records() -> None:
    """A window must hold consecutive records from ONE session, in time order."""
    X, y, meta = make_synthetic(n_devices=1, records_per_device=12)
    dataset = build_sequences(X, y, meta, sequence_length=10, stride=1)

    first = dataset.X[0]
    expected = np.stack([np.full(N_FEATURES, float(i)) for i in range(10)])
    np.testing.assert_allclose(first, expected)


# --- The length-1 guard, from both entry points -------------------------------------------


@pytest.mark.parametrize("bad_length", [0, 1, -1])
def test_sequence_length_one_is_rejected(bad_length: int) -> None:
    """A length-1 window is Objection #2 reintroduced; it must raise, not warn."""
    X, y, meta = make_synthetic()
    with pytest.raises(SingleRecordWindowError):
        build_sequences(X, y, meta, sequence_length=bad_length)


def test_dataclass_rejects_length_one() -> None:
    """The guard must also hold when a SequenceDataset is constructed directly."""
    with pytest.raises(SingleRecordWindowError):
        SequenceDataset(
            X=np.zeros((5, 1, N_FEATURES), dtype=np.float32),
            y=np.zeros(5, dtype=np.int8),
            session_ids=np.zeros(5, dtype=np.int64),
            feature_names=FEATURE_NAMES,
            n_padded_windows=0,
            sequence_length=1,
        )


def test_dataclass_rejects_two_dimensional_input() -> None:
    """(batch, features) is exactly the base paper's single-flow-record shape."""
    with pytest.raises(ValueError, match="3-D"):
        SequenceDataset(
            X=np.zeros((5, N_FEATURES), dtype=np.float32),
            y=np.zeros(5, dtype=np.int8),
            session_ids=np.zeros(5, dtype=np.int64),
            feature_names=FEATURE_NAMES,
            n_padded_windows=0,
            sequence_length=10,
        )


# --- Sessionisation ------------------------------------------------------------------------


def test_time_gap_starts_a_new_session() -> None:
    """A gap beyond SESSION_GAP_SECONDS must split one device's traffic into two sessions."""
    X, y, meta = make_synthetic(n_devices=1, records_per_device=20, gap_after=10)
    sessions = assign_sessions(meta, gap_seconds=1.0)
    assert sessions.nunique() == 2

    without_gap_meta = make_synthetic(n_devices=1, records_per_device=20)[2]
    assert assign_sessions(without_gap_meta, gap_seconds=1.0).nunique() == 1


def test_sessions_do_not_span_devices() -> None:
    """Each device's traffic must form its own session, never a shared one."""
    X, y, meta = make_synthetic(n_devices=3, records_per_device=15)
    assert assign_sessions(meta).nunique() == 3


def test_windows_never_span_two_sessions() -> None:
    """Every window must come from exactly one session."""
    X, y, meta = make_synthetic(n_devices=3, records_per_device=15)
    dataset = build_sequences(X, y, meta, sequence_length=10, stride=1)

    sessions = assign_sessions(meta).to_numpy()
    for window_index, window in enumerate(dataset.X):
        # Feature values encode device*1000 + record, so a window spanning devices would show a
        # value range far exceeding the per-session record count.
        assert window.max() - window.min() < 1000, f"window {window_index} spans devices"
    assert set(np.unique(dataset.session_ids)).issubset(set(np.unique(sessions)))


# --- Padding and dropping ------------------------------------------------------------------


def test_short_session_is_pre_padded() -> None:
    """A session shorter than the window is PRE-padded, newest record last."""
    X, y, meta = make_synthetic(n_devices=1, records_per_device=4)
    dataset = build_sequences(X, y, meta, sequence_length=10, stride=5, pad_value=0.0)

    assert len(dataset.X) == 1
    assert dataset.n_padded_windows == 1
    window = dataset.X[0]
    np.testing.assert_allclose(window[:6], 0.0)          # padding at the FRONT
    np.testing.assert_allclose(window[-1], np.full(N_FEATURES, 3.0))  # newest record LAST


def test_single_record_sessions_are_dropped_not_padded() -> None:
    """A 1-record session padded to 10 would be a single flow record in disguise."""
    X, y, meta = make_synthetic(n_devices=1, records_per_device=1)
    with pytest.raises(ValueError, match="No windows produced"):
        build_sequences(X, y, meta, sequence_length=10, min_session_length=MIN_SESSION_LENGTH)


# --- Labelling and splitting ---------------------------------------------------------------


def test_window_takes_label_of_last_record() -> None:
    """The window label is the most recent record's label (causal, §1.3)."""
    X, y, meta = make_synthetic(n_devices=1, records_per_device=12)
    y = pd.Series([0] * 11 + [1], dtype=np.int8)  # only the final record is an Attack

    dataset = build_sequences(X, y, meta, sequence_length=10, stride=1)
    assert dataset.y[0] == 0     # window ending at record 9
    assert dataset.y[-1] == 1    # window ending at record 11, the Attack


def test_session_split_shares_no_session_between_folds() -> None:
    """Session-level splitting is the §1.4 leakage guard; folds must be disjoint."""
    X, y, meta = make_synthetic(n_devices=40, records_per_device=15)
    dataset = build_sequences(X, y, meta, sequence_length=10, stride=5)

    train, test, val = split_sessions(dataset.session_ids, dataset.y)
    assert not (set(train) & set(test))
    assert not (set(train) & set(val))
    assert not (set(test) & set(val))

    train_windows = subset_by_sessions(dataset, train)
    test_windows = subset_by_sessions(dataset, test)
    assert not (set(train_windows.session_ids) & set(test_windows.session_ids))
    assert len(train_windows.X) + len(test_windows.X) <= len(dataset.X)


def test_stride_controls_window_count() -> None:
    """Stride 1 must yield more windows than stride 5 over the same records."""
    X, y, meta = make_synthetic(n_devices=1, records_per_device=30)
    dense = build_sequences(X, y, meta, sequence_length=10, stride=1)
    sparse = build_sequences(X, y, meta, sequence_length=10, stride=5)
    assert len(dense.X) == 21   # 30 - 10 + 1
    assert len(sparse.X) == 5   # begins at 0, 5, 10, 15, 20
    assert len(dense.X) > len(sparse.X)
