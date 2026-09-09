"""Tests for Edge-IIoTset cleaning (Build-Instructions T3.6).

The substantive T3.6 results are dataset findings, measured by
`scripts/evaluate_edge_iiotset.py` and reported in `reports/t3_6_edge_iiotset.md`. These tests pin
the cleaning behaviour those findings depend on — above all the placeholder normalisation, since
without it three columns classify the entire dataset perfectly and every downstream number is
meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sequence_builder import SESSION_GAP_SECONDS, assign_sessions
from src.preprocessing.clean import (
    EDGE_BLOCK_OFFSET_SECONDS,
    EDGE_CONTENT_COLUMNS,
    EDGE_IDENTIFIER_COLUMNS,
    EDGE_NONCE_COLUMNS,
    PLACEHOLDER_NORMALISED,
    PLACEHOLDER_TOKENS,
    build_edge_iiotset_timeline,
    clean_edge_iiotset,
    normalise_placeholders,
    parse_edge_iiotset_clock,
)


def synthetic_edge_frame(n: int = 200) -> pd.DataFrame:
    """Build a frame reproducing Edge-IIoTset's structure and its placeholder defect.

    Normal rows spell an absent field "0"; attack rows spell it "0.0" — exactly the artefact
    measured in the real file.

    Args:
        n: rows to generate.

    Returns:
        A DataFrame with the columns the cleaner cares about.
    """
    rng = np.random.default_rng(0)
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    return pd.DataFrame({
        "frame.time": ["6.0"] * n,
        "ip.src_host": [f"192.168.0.{i % 20}" for i in range(n)],
        "ip.dst_host": [f"192.168.0.{(i * 3) % 15}" for i in range(n)],
        "arp.src.proto_ipv4": ["0.0"] * n,
        "arp.dst.proto_ipv4": ["0.0"] * n,
        # THE DEFECT: placeholder spelling differs by class.
        "mqtt.topic": ["0" if y == 0 else "0.0" for y in labels],
        "mqtt.protoname": ["0" if y == 0 else "0.0" for y in labels],
        "tcp.payload": ["0" if y == 0 else "0.0" for y in labels],
        "tcp.seq": rng.integers(0, 10_000, n).astype(float),
        "tcp.srcport": rng.integers(1024, 65535, n).astype(float),
        "tcp.dstport": rng.integers(1024, 65535, n).astype(float),
        "tcp.len": rng.integers(0, 1500, n).astype(float),
        "tcp.flags": rng.integers(0, 8, n).astype(float),
        "udp.port": rng.choice([53, 123, 1883], n).astype(float),
        "icmp.unused": np.zeros(n),  # constant
        "Attack_label": labels,
        "Attack_type": ["Normal" if y == 0 else "DDoS_UDP" for y in labels],
    })


# --- The placeholder defect ---------------------------------------------------------------------


def test_placeholder_spellings_are_collapsed_to_one_token() -> None:
    """"0" and "0.0" must become indistinguishable, or the spelling encodes the class."""
    series = pd.Series(["0", "0.0", "nan", "", "Temperature_and_Humidity"])
    normalised = normalise_placeholders(series)

    assert list(normalised[:4]) == [PLACEHOLDER_NORMALISED] * 4
    assert normalised.iloc[4] == "Temperature_and_Humidity"  # real values survive


def test_cleaning_removes_the_placeholder_leak() -> None:
    """THE test. Before cleaning, `mqtt.topic` alone separates the classes perfectly.

    Three columns in the real file reach 100% solo accuracy against an 84.6% majority baseline,
    purely because normal and attack captures were preprocessed separately. If this regresses,
    every Edge-IIoTset number in `reports/` becomes meaningless.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.tree import DecisionTreeClassifier

    frame = synthetic_edge_frame()
    y = frame["Attack_label"].to_numpy()

    def solo(series: pd.Series) -> float:
        encoded = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(series.astype(str).to_frame())
        return float(cross_val_score(
            DecisionTreeClassifier(max_depth=4, random_state=42), encoded, y, cv=3
        ).mean())

    assert solo(frame["mqtt.topic"]) > 0.99, "fixture does not reproduce the defect"
    assert solo(normalise_placeholders(frame["mqtt.topic"])) < 0.99, (
        "normalisation failed to remove the placeholder leak"
    )


def test_all_known_placeholder_spellings_are_covered() -> None:
    """The real file uses "0" and "0.0"; the set also guards the obvious neighbours."""
    assert {"0", "0.0"} <= PLACEHOLDER_TOKENS
    assert "nan" in PLACEHOLDER_TOKENS


# --- The drop lists --------------------------------------------------------------------------------


def test_content_and_nonce_columns_are_excluded_from_features() -> None:
    """Payload columns hold the attack strings; nonces identify one connection."""
    X, y, meta = clean_edge_iiotset(synthetic_edge_frame())

    for column in EDGE_CONTENT_COLUMNS + EDGE_NONCE_COLUMNS + EDGE_IDENTIFIER_COLUMNS:
        assert column not in X.columns, f"{column} must not be a feature"


def test_ephemeral_ports_are_treated_as_identifiers() -> None:
    """On this dataset `tcp.srcport` takes a distinct value on 20% of rows.

    That is an ephemeral per-connection number, not a service identifier. Keeping both port
    columns took the Random Forest to 0.9999 with zero errors — a perfect score on a held-out
    fold is a defect report, not a result. `udp.port` (32 distinct values) is kept.
    """
    assert "tcp.srcport" in EDGE_NONCE_COLUMNS
    assert "tcp.dstport" in EDGE_NONCE_COLUMNS
    assert "udp.port" not in EDGE_NONCE_COLUMNS

    X, _, _ = clean_edge_iiotset(synthetic_edge_frame())
    assert "udp.port" in X.columns


def test_identifiers_are_retained_as_metadata_for_the_graph_check() -> None:
    """`ip.src_host`/`ip.dst_host` leave the feature matrix but must remain available."""
    X, y, meta = clean_edge_iiotset(synthetic_edge_frame())
    assert "ip.src_host" in meta.columns and "ip.dst_host" in meta.columns
    assert "Attack_type" in meta.columns


def test_constant_columns_are_dropped() -> None:
    """Zero-variance columns carry no information."""
    X, _, _ = clean_edge_iiotset(synthetic_edge_frame())
    assert "icmp.unused" not in X.columns


# --- The label convention -----------------------------------------------------------------------


def test_attack_is_the_positive_class() -> None:
    """`TRD.md §2.3`, consistently with IoTID20."""
    frame = synthetic_edge_frame()
    X, y, meta = clean_edge_iiotset(frame)

    assert set(np.unique(y)) <= {0, 1}
    assert int((y == 1).sum()) == int((frame["Attack_label"] == 1).sum())


def test_features_labels_and_metadata_stay_aligned() -> None:
    """A misalignment would silently pair rows with the wrong labels."""
    X, y, meta = clean_edge_iiotset(synthetic_edge_frame())
    assert len(X) == len(y) == len(meta)


def test_missing_label_column_is_rejected() -> None:
    """Failing loudly beats producing an unlabelled feature matrix."""
    frame = synthetic_edge_frame().drop(columns=["Attack_label"])
    with pytest.raises(ValueError, match="Attack_label"):
        clean_edge_iiotset(frame)


def test_deduplication_is_available_and_off_by_default() -> None:
    """Edge-IIoTset is 98.95% duplicates once identifiers are removed, so this flag matters."""
    frame = synthetic_edge_frame()
    kept, _, _ = clean_edge_iiotset(frame, drop_duplicates=False)
    deduped, _, _ = clean_edge_iiotset(frame, drop_duplicates=True)
    assert len(deduped) <= len(kept)


def test_all_features_are_numeric() -> None:
    """Downstream scaling and models require a fully numeric matrix."""
    X, _, _ = clean_edge_iiotset(synthetic_edge_frame())
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)
    assert not X.isna().any().any()


# --- Timeline recovery (T3.6 sequence half) ------------------------------------------------
#
# These pin the measurement that unblocked T3.6's ensemble half: the clock inside a damaged
# `frame.time` is recoverable, and the two corrections applied on top of it behave as documented.


def timed_edge_frame() -> pd.DataFrame:
    """Build a frame reproducing the real file's damaged `frame.time` shapes.

    Three blocks: one normal block that crosses midnight, one attack block with an intact clock,
    and one attack block whose `frame.time` holds an IP address instead of a time — the exact
    three shapes measured in the published CSV.

    Returns:
        A DataFrame with the columns `build_edge_iiotset_timeline` requires.
    """
    times = (
        [" 2021 23:59:58.100000000 ", " 2021 23:59:59.100000000 ", " 2021 00:00:01.100000000 "]
        + [f" 2021 10:00:0{i}.000000000 " for i in range(4)]
        + ["185.101.177.190", "185.101.177.191"]
    )
    attack_types = ["Normal"] * 3 + ["DDoS_HTTP"] * 4 + ["DDoS_UDP"] * 2
    return pd.DataFrame({
        "frame.time": times,
        "ip.src_host": ["192.168.0.1"] * len(times),
        "ip.dst_host": ["192.168.0.128"] * len(times),
        "Attack_type": attack_types,
    })


def test_clock_is_recovered_from_the_damaged_timestamp() -> None:
    """The surviving HH:MM:SS tail parses; a value with no clock yields NaN."""
    parsed = parse_edge_iiotset_clock(
        pd.Series([" 2021 22:14:30.939803000 ", "6.0", "185.101.177.190"])
    )
    assert parsed.iloc[0] == pytest.approx(22 * 3600 + 14 * 60 + 30.939803)
    assert np.isnan(parsed.iloc[1])
    assert np.isnan(parsed.iloc[2])


def test_rows_without_a_clock_are_dropped_never_imputed() -> None:
    """An invented timestamp would fabricate the ordering the ensemble half depends on."""
    frame = timed_edge_frame()
    timeline, keep_mask = build_edge_iiotset_timeline(frame)

    assert keep_mask.sum() == 7, "the two IP-valued frame.time rows must be dropped"
    assert len(timeline) == 7
    assert "DDoS_UDP" not in set(timeline["Attack_type"]), "untimed class must not survive"
    assert timeline["Timestamp"].notna().all(), "no NaT may reach the sequence builder"


def test_midnight_wrap_does_not_move_time_backwards() -> None:
    """23:59:59 -> 00:00:01 is a day boundary, not a 24-hour backward jump."""
    timeline, _ = build_edge_iiotset_timeline(timed_edge_frame())
    normal = timeline[timeline["Attack_type"] == "Normal"]["Timestamp"]

    assert normal.is_monotonic_increasing
    step = (normal.iloc[2] - normal.iloc[1]).total_seconds()
    assert step == pytest.approx(2.0), f"wrap should leave a 2 s step, got {step}"


def test_separate_capture_blocks_cannot_merge_into_one_session() -> None:
    """Two blocks sharing an hour of the day are different captures, not concurrent traffic."""
    timeline, _ = build_edge_iiotset_timeline(timed_edge_frame())
    first_block_end = timeline[timeline["Attack_type"] == "Normal"]["Timestamp"].max()
    second_block_start = timeline[timeline["Attack_type"] == "DDoS_HTTP"]["Timestamp"].min()

    gap = (second_block_start - first_block_end).total_seconds()
    assert gap >= EDGE_BLOCK_OFFSET_SECONDS - 86400.0, "blocks must not overlap in time"
    assert gap > SESSION_GAP_SECONDS, "the gap must exceed the sessionisation threshold"


def test_timeline_is_usable_by_the_sequence_builder() -> None:
    """The whole point: the output feeds `assign_sessions` unchanged, as IoTID20's meta does."""
    timeline, _ = build_edge_iiotset_timeline(timed_edge_frame())

    assert "Dst_IP" in timeline.columns, "sequence_builder keys sessions on Dst_IP"
    assert pd.api.types.is_datetime64_any_dtype(timeline["Timestamp"])

    sessions = assign_sessions(timeline)
    assert len(sessions) == len(timeline)
    assert sessions.nunique() >= 2, "the block offset must start a new session per block"


def test_missing_column_is_rejected() -> None:
    """A frame that bypassed `clean_edge_iiotset` must fail loudly, not silently mis-order."""
    with pytest.raises(KeyError):
        build_edge_iiotset_timeline(timed_edge_frame().drop(columns=["Attack_type"]))
