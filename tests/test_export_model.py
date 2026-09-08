"""Unit tests for the TFLite export (Build-Instructions T4.1).

T4.1's VERIFY block — *"exported model file loads successfully in a standalone inference script
without requiring the full training environment"* — is exercised end to end by
`scripts/export_for_pi.py` against the trained ensemble, and reported in `reports/t4_1_export.md`.

These tests cover the contract on small untrained models, and one thing in particular:
`test_export_refuses_to_write_an_unfaithful_graph`. During T4.1 a conversion route produced a 16 KB
file that loaded fine and emitted NaN, agreeing with the Keras model on 0% of windows. Conversion
succeeding is not verification, and the exporter must refuse rather than write such a graph out.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.config import POSITIVE_CLASS_NAME
from src.deployment.export_model import (
    DEFAULT_PRECISION,
    MIN_ARGMAX_AGREEMENT,
    PROBABILITY_TOLERANCE,
    export_bilstm_for_tflite,
    export_ensemble,
    verify_tflite,
)
from src.models import bilstm_branch
from src.models.bilstm_branch import build_bilstm_branch
from src.models.cnn_branch import build_cnn_branch

SEQUENCE_LENGTH = 10
N_FEATURES = 6
FEATURE_NAMES = [f"Feature_{i}" for i in range(N_FEATURES)]


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    """Deterministic verification windows."""
    warnings.filterwarnings("ignore")
    return np.random.default_rng(0).random(
        (16, SEQUENCE_LENGTH, N_FEATURES)
    ).astype(np.float32)


# --- The BiLSTM unrolling fix ------------------------------------------------------------------


def test_unrolled_bilstm_is_numerically_identical(sample) -> None:
    """Unrolling is a graph transformation, not an architecture change.

    If this ever drifts, the deployed BiLSTM would differ from the evaluated one — and the
    evaluated one is what every report in this repository describes.
    """
    trained = build_bilstm_branch(SEQUENCE_LENGTH, N_FEATURES)
    unrolled = export_bilstm_for_tflite(trained)

    difference = np.abs(
        unrolled.predict(sample, verbose=0) - trained.predict(sample, verbose=0)
    ).max()
    assert difference == 0.0, "unrolling must be exact, not merely close"


def test_unrolled_bilstm_keeps_the_documented_architecture(sample) -> None:
    """2 layers, 64 units per direction — Objection #2's fix must survive export."""
    unrolled = export_bilstm_for_tflite(build_bilstm_branch(SEQUENCE_LENGTH, N_FEATURES))

    bidirectional = [l for l in unrolled.layers if l.__class__.__name__ == "Bidirectional"]
    assert len(bidirectional) == bilstm_branch.LSTM_LAYER_COUNT == 2
    for layer in bidirectional:
        assert layer.forward_layer.units == bilstm_branch.LSTM_UNITS == 64
        assert layer.forward_layer.unroll is True


def test_unrolled_bilstm_preserves_weights(sample) -> None:
    """Weight transfer must be complete — the discarded export route lost them silently."""
    trained = build_bilstm_branch(SEQUENCE_LENGTH, N_FEATURES)
    unrolled = export_bilstm_for_tflite(trained)

    assert len(unrolled.get_weights()) == len(trained.get_weights())
    for exported, original in zip(unrolled.get_weights(), trained.get_weights()):
        np.testing.assert_array_equal(exported, original)


# --- Export and verification --------------------------------------------------------------------


def test_export_writes_verified_graphs_and_a_manifest(tmp_path: Path, sample) -> None:
    """The happy path: every branch converts, verifies, and is described in the manifest."""
    models = {
        "cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES),
        "bilstm": build_bilstm_branch(SEQUENCE_LENGTH, N_FEATURES),
    }
    export = export_ensemble(models, tmp_path, sample, FEATURE_NAMES)

    assert export.all_verified()
    for branch in export.branches:
        assert branch.path.exists() and branch.size_bytes > 0
        assert branch.argmax_agreement >= MIN_ARGMAX_AGREEMENT
        assert branch.max_output_difference <= PROBABILITY_TOLERANCE[DEFAULT_PRECISION]

    assert export.manifest_path is not None and export.manifest_path.exists()


def test_bilstm_is_marked_as_unrolled(tmp_path: Path, sample) -> None:
    """The manifest must record that the BiLSTM was transformed, not silently swapped."""
    models = {"bilstm": build_bilstm_branch(SEQUENCE_LENGTH, N_FEATURES)}
    export = export_ensemble(models, tmp_path, sample, FEATURE_NAMES)

    assert export.branches[0].unrolled is True
    manifest = json.loads(export.manifest_path.read_text(encoding="utf-8"))
    assert manifest["branches"][0]["unrolled"] is True


def test_export_refuses_to_write_an_unfaithful_graph(tmp_path: Path, sample, monkeypatch) -> None:
    """A graph that does not reproduce the evaluated model must not be written out as usable.

    This is the guard against the real failure seen during T4.1: a conversion route produced a
    16 KB file that loaded, emitted NaN, and agreed with the Keras model on 0% of windows.
    """
    import src.deployment.export_model as export_module

    monkeypatch.setattr(export_module, "verify_tflite", lambda *a, **k: (1.0, 0.0))

    with pytest.raises(ValueError, match="must not be deployed"):
        export_module.export_ensemble(
            {"cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)},
            tmp_path, sample, FEATURE_NAMES,
        )


def test_unknown_precision_is_rejected(tmp_path: Path, sample) -> None:
    """An unrecognised precision must fail loudly rather than silently pick a default."""
    with pytest.raises((ValueError, KeyError)):
        export_ensemble(
            {"cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)},
            tmp_path, sample, FEATURE_NAMES, precision="int4",
        )


def test_verify_detects_a_mismatched_reference(tmp_path: Path, sample) -> None:
    """`verify_tflite` must actually compare, not just report success."""
    from src.deployment.export_model import _convert

    model = build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)
    other = build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES, seed=999)

    difference, _ = verify_tflite(_convert(model, "float32"), model, sample)
    assert difference <= PROBABILITY_TOLERANCE["float32"]

    # Compared against a DIFFERENT model, the same graph must not look faithful.
    wrong_difference, _ = verify_tflite(_convert(model, "float32"), other, sample)
    assert wrong_difference > difference


# --- The manifest is the Pi's only contract ------------------------------------------------------


def test_manifest_carries_everything_the_pi_needs(tmp_path: Path, sample) -> None:
    """T4.2 must reproduce the deployed decision from this file alone."""
    models = {"cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)}
    export = export_ensemble(models, tmp_path, sample, FEATURE_NAMES)
    manifest = json.loads(export.manifest_path.read_text(encoding="utf-8"))

    assert manifest["sequence_length"] == SEQUENCE_LENGTH
    assert manifest["n_features"] == N_FEATURES
    # Feature ORDER matters: a silent reordering would corrupt every prediction undetectably.
    assert manifest["feature_names"] == FEATURE_NAMES

    fusion = manifest["fusion"]
    assert fusion["rule"] == "confidence_weighted"
    assert "gamma" in fusion and "branch_priors" in fusion and "epsilon" in fusion

    # The positive-class convention must travel with the model (TRD.md §2.3).
    assert manifest["classes"]["1"] == POSITIVE_CLASS_NAME
    assert manifest["classes"]["positive_label"] == 1


def test_manifest_records_the_verification_evidence(tmp_path: Path, sample) -> None:
    """A deployment should carry proof it was checked, not just a claim that it was."""
    export = export_ensemble(
        {"cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)}, tmp_path, sample, FEATURE_NAMES
    )
    verification = json.loads(export.manifest_path.read_text(encoding="utf-8"))["verification"]

    assert verification["min_argmax_agreement"] == MIN_ARGMAX_AGREEMENT
    assert verification["results"]["cnn"]["argmax_agreement"] >= MIN_ARGMAX_AGREEMENT


def test_tolerance_is_precision_aware() -> None:
    """One tolerance for both formats rejected float16 for being float16."""
    assert PROBABILITY_TOLERANCE["float16"] > PROBABILITY_TOLERANCE["float32"]
    assert DEFAULT_PRECISION == "float32"
