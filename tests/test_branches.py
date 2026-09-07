"""Unit tests for the three ensemble branches (Build-Instructions T2.2, T2.3, T2.4).

The VERIFY blocks for these tasks are documentation checks: *"Docstring numbers match
`docs/architecture_decision.md` exactly"* (T2.2/T2.4) and *"Docstring states unit and layer counts
in plain numbers; no vague language like 'a few layers'"* (T2.3). Those are asserted here against
the decision document itself, so the two cannot drift apart silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.models import bilstm_branch, cnn_branch, transformer_branch

SEQUENCE_LENGTH = 10
N_FEATURES = 62
N_CLASSES = 2
BATCH = 4

DECISION_DOC = (
    Path(__file__).resolve().parents[1] / "docs" / "architecture_decision.md"
).read_text(encoding="utf-8")


def _sample_batch() -> np.ndarray:
    """Return a deterministic dummy input batch of the contracted shape."""
    rng = np.random.default_rng(0)
    return rng.random((BATCH, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)


# --- Shape contract, shared by all three branches (TRD.md §9) ------------------------------


@pytest.mark.parametrize(
    "builder",
    [cnn_branch.build_cnn_branch, bilstm_branch.build_bilstm_branch,
     transformer_branch.build_transformer_branch],
)
def test_branch_accepts_sequence_input_and_emits_softmax(builder) -> None:
    """Every branch maps (batch, 10, 62) to a (batch, 2) probability distribution."""
    model = builder(SEQUENCE_LENGTH, N_FEATURES)
    assert model.input_shape == (None, SEQUENCE_LENGTH, N_FEATURES)
    assert model.output_shape == (None, N_CLASSES)

    output = model.predict(_sample_batch(), verbose=0)
    assert output.shape == (BATCH, N_CLASSES)
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-5)
    assert (output >= 0).all()


@pytest.mark.parametrize(
    "builder",
    [cnn_branch.build_cnn_branch, bilstm_branch.build_bilstm_branch,
     transformer_branch.build_transformer_branch],
)
@pytest.mark.parametrize("bad_length", [0, 1])
def test_branch_rejects_single_record_input(builder, bad_length: int) -> None:
    """A length-1 window is Objection #2; every branch must refuse it (TRD.md §9)."""
    with pytest.raises(ValueError):
        builder(bad_length, N_FEATURES)


# --- T2.2: CNN branch numbers ---------------------------------------------------------------


def test_cnn_constants_match_decision_document() -> None:
    """3 conv layers, 64/128/64 filters, kernel 3, ReLU, dropout 0.3 (§2.1)."""
    assert cnn_branch.CONV_LAYER_COUNT == 3
    assert cnn_branch.CONV_FILTERS == (64, 128, 64)
    assert cnn_branch.CONV_KERNEL_SIZE == 3
    assert cnn_branch.CONV_ACTIVATION == "relu"
    assert cnn_branch.DROPOUT_RATE == 0.3
    assert cnn_branch.EMBEDDING_DIM == 64

    assert "64, 128, 64" in DECISION_DOC or "64/128/64" in DECISION_DOC
    assert "GlobalMaxPooling1D" in DECISION_DOC


def test_cnn_has_exactly_three_conv_layers() -> None:
    """The built graph must contain the documented layer count, not merely declare it."""
    model = cnn_branch.build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)
    conv_layers = [l for l in model.layers if l.__class__.__name__ == "Conv1D"]
    assert len(conv_layers) == cnn_branch.CONV_LAYER_COUNT == 3
    assert [l.filters for l in conv_layers] == list(cnn_branch.CONV_FILTERS)
    assert all(l.kernel_size == (cnn_branch.CONV_KERNEL_SIZE,) for l in conv_layers)


def test_cnn_embedding_model_exposes_documented_width() -> None:
    """The pooled embedding must be EMBEDDING_DIM wide, shared with the other branches."""
    model = cnn_branch.build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)
    embedder = cnn_branch.build_cnn_embedding_model(model)
    assert embedder.output_shape == (None, cnn_branch.EMBEDDING_DIM)


# --- T2.3: BiLSTM branch numbers — the direct fix for Objection #2 ---------------------------


def test_bilstm_layer_and_unit_counts_are_documented_numbers() -> None:
    """T2.3 VERIFY: 2 layers and 64 units per direction, stated as plain numbers."""
    assert bilstm_branch.LSTM_LAYER_COUNT == 2
    assert bilstm_branch.LSTM_UNITS == 64
    assert bilstm_branch.RECURRENT_DROPOUT == 0.0

    docstring = bilstm_branch.__doc__ or ""
    assert "LAYER COUNT = 2" in docstring
    assert "UNITS = 64 PER DIRECTION" in docstring


def test_bilstm_docstring_contains_no_vague_language() -> None:
    """T2.3 VERIFY explicitly forbids phrasing like 'a few layers'.

    Checked against `describe_architecture()` rather than the raw docstring: the docstring
    legitimately *names* the forbidden phrases in order to prohibit them, so a naive substring
    search over it would flag the prohibition itself. `describe_architecture()` is the text that
    actually gets quoted into reports and slides, so it is the string that must be specific.
    """
    description = bilstm_branch.describe_architecture().lower()
    for phrase in ("a few layers", "several layers", "a couple of layers",
                   "some units", "several units", "tbd"):
        assert phrase not in description, f"vague phrase {phrase!r} in describe_architecture()"

    # And the numbers must actually be present, as digits.
    assert re.search(r"\b2 bidirectional lstm layers\b", description)
    assert re.search(r"\b64 units per direction\b", description)


def test_bilstm_graph_has_two_bidirectional_layers_of_64_units() -> None:
    """The built graph must match the documented counts."""
    model = bilstm_branch.build_bilstm_branch(SEQUENCE_LENGTH, N_FEATURES)
    bidirectional = [l for l in model.layers if l.__class__.__name__ == "Bidirectional"]
    assert len(bidirectional) == 2
    for layer in bidirectional:
        assert layer.forward_layer.units == 64
        assert layer.backward_layer.units == 64
    # Layer 1 returns sequences (feeds layer 2), layer 2 does not.
    assert bidirectional[0].forward_layer.return_sequences is True
    assert bidirectional[1].forward_layer.return_sequences is False


def test_bilstm_description_matches_constants() -> None:
    """describe_architecture() is the single source quoted by reports and slides."""
    description = bilstm_branch.describe_architecture()
    assert "2 Bidirectional LSTM layers" in description
    assert "64 units per direction" in description
    assert "128-d concatenated state" in description


# --- T2.4: Transformer branch numbers -------------------------------------------------------


def test_transformer_constants_match_decision_document() -> None:
    """4 heads, 2 encoder layers, embed dim 64, key_dim 16, FFN 128 (§2.3)."""
    assert transformer_branch.NUM_ATTENTION_HEADS == 4
    assert transformer_branch.NUM_ENCODER_LAYERS == 2
    assert transformer_branch.EMBED_DIM == 64
    assert transformer_branch.KEY_DIM == 16
    assert transformer_branch.FFN_DIM == 128
    # The head-splitting relation the decision document states.
    assert (
        transformer_branch.NUM_ATTENTION_HEADS * transformer_branch.KEY_DIM
        == transformer_branch.EMBED_DIM
    )


def test_transformer_graph_has_documented_attention_configuration() -> None:
    """The built graph must contain 2 MultiHeadAttention layers of 4 heads each."""
    model = transformer_branch.build_transformer_branch(SEQUENCE_LENGTH, N_FEATURES)
    attention = [l for l in model.layers if l.__class__.__name__ == "MultiHeadAttention"]
    assert len(attention) == transformer_branch.NUM_ENCODER_LAYERS == 2
    for layer in attention:
        assert layer._num_heads == 4
        assert layer._key_dim == 16


def test_positional_encoding_is_deterministic_and_bounded() -> None:
    """Sinusoidal encoding: parameter-free, reproducible, and within [-1, 1]."""
    first = transformer_branch.sinusoidal_positional_encoding(SEQUENCE_LENGTH, 64)
    second = transformer_branch.sinusoidal_positional_encoding(SEQUENCE_LENGTH, 64)

    assert first.shape == (SEQUENCE_LENGTH, 64)
    np.testing.assert_array_equal(first, second)
    assert first.min() >= -1.0 and first.max() <= 1.0
    # Distinct positions must receive distinct encodings, or position carries no information.
    assert not np.allclose(first[0], first[1])


def test_positional_encoding_rejects_odd_width() -> None:
    """Sine/cosine pairing requires an even width."""
    with pytest.raises(ValueError, match="even"):
        transformer_branch.sinusoidal_positional_encoding(SEQUENCE_LENGTH, 63)


def test_transformer_rejects_indivisible_head_configuration() -> None:
    """embed_dim must split evenly across heads."""
    with pytest.raises(ValueError, match="divisible"):
        transformer_branch.build_transformer_branch(SEQUENCE_LENGTH, N_FEATURES, num_heads=7)


# --- Cross-branch consistency ----------------------------------------------------------------


def test_all_branches_share_the_same_embedding_width() -> None:
    """A shared 64-d embedding is what keeps the branches comparable in fusion."""
    assert (
        cnn_branch.EMBEDDING_DIM
        == bilstm_branch.EMBEDDING_DIM
        == transformer_branch.EMBED_DIM
        == 64
    )


def test_decision_document_has_no_tbd_placeholders() -> None:
    """T1.5 VERIFY: 'No "TBD" placeholders — every number must be a specific choice.'

    Occurrences inside quotation marks are ignored: the document states that it contains no
    "TBD"s, and that sentence must not fail its own check.
    """
    unquoted = re.sub(r"[\"\u201c\u201d`]TBD[\"\u201c\u201d`]", "", DECISION_DOC, flags=re.IGNORECASE)
    matches = re.findall(r"\bTBD\b", unquoted, flags=re.IGNORECASE)
    assert not matches, f"unresolved TBD placeholder(s) in architecture_decision.md: {matches}"


def test_decision_document_fixes_every_number_phase_two_needs() -> None:
    """T1.5 VERIFY: no Phase 2 task should need to invent a number not already decided."""
    for required in (
        "SEQUENCE_LENGTH", "TRAIN_STRIDE", "INFERENCE_STRIDE", "SESSION_GAP_SECONDS",
        "MIN_SESSION_LENGTH", "CONFIDENCE_SHARPNESS", "BRANCH_PRIORS",
    ):
        assert required in DECISION_DOC, f"{required} is not fixed in architecture_decision.md"
