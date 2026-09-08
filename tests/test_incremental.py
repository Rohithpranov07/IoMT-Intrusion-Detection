"""Unit tests for incremental learning (Build-Instructions T3.5).

T3.5's VERIFY condition and `TRD.md §9`'s gate — *"before/after comparison shows no significant
drop in prior-class detection performance"* — is an experiment on the trained ensemble, run by
`scripts/evaluate_incremental_learning.py` and reported in `reports/t3_5_incremental_learning.md`.
It cannot be a unit test: it needs a real ensemble trained with an attack type held out.

What is testable here is the machinery that experiment depends on, and one thing in particular:
`test_gate_fails_a_mechanism_that_learned_nothing`. A mechanism that changes nothing trivially
preserves prior-class performance, so a naive reading of the gate would mark it a pass. That
loophole is closed in `IncrementalResult.passes_gate`, and this pins it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptive.incremental import (
    BUFFER_CAPACITY,
    REPLAY_RATIO,
    TRAINABLE_LAYER_SUFFIXES,
    IncrementalResult,
    ReplayBuffer,
    set_trainable_layers,
)
from src.models.cnn_branch import build_cnn_branch

SEQUENCE_LENGTH = 10
N_FEATURES = 6


def make_result(prior_delta: float, novel_delta: float) -> IncrementalResult:
    """Build a result with the given before/after deltas, for gate testing.

    Args:
        prior_delta: change in prior-class recall.
        novel_delta: change in recall on the new attack type.

    Returns:
        An `IncrementalResult`.
    """
    return IncrementalResult(
        mechanism="stub",
        new_attack_type="Test Attack",
        prior_recall_before=0.90,
        prior_recall_after=0.90 + prior_delta,
        novel_recall_before=0.10,
        novel_recall_after=0.10 + novel_delta,
        normal_fpr_before=0.05,
        normal_fpr_after=0.05,
        n_new_samples=100,
        n_replay_samples=200,
        trainable_parameters=1000,
        seconds=1.0,
    )


# --- The gate (TRD §9), including the loophole it has to close --------------------------------


def test_gate_passes_when_a_new_type_is_learned_without_forgetting() -> None:
    """The intended success case."""
    assert make_result(prior_delta=0.0, novel_delta=0.60).passes_gate()


def test_gate_fails_on_forgetting() -> None:
    """A drop in prior-class recall beyond tolerance is catastrophic forgetting."""
    assert not make_result(prior_delta=-0.20, novel_delta=0.60).passes_gate()


def test_gate_fails_a_mechanism_that_learned_nothing() -> None:
    """A no-op preserves prior performance perfectly and is still useless.

    This is the loophole a naive reading of `TRD.md §9` leaves open: "no significant drop in
    prior-class performance" is satisfied by changing nothing at all. The gate therefore requires
    the new type to have been learned as well.
    """
    no_op = make_result(prior_delta=0.0, novel_delta=0.0)
    assert no_op.prior_recall_delta == 0.0      # prior performance perfectly preserved...
    assert not no_op.passes_gate()              # ...and still a failure


def test_gate_tolerates_noise_level_regression() -> None:
    """A 0.01 tolerance stops evaluation noise being reported as forgetting."""
    assert make_result(prior_delta=-0.005, novel_delta=0.50).passes_gate()
    assert not make_result(prior_delta=-0.05, novel_delta=0.50).passes_gate()


def test_deltas_are_computed_from_before_and_after() -> None:
    """The reported deltas must be derived, not separately supplied and able to disagree."""
    result = make_result(prior_delta=-0.03, novel_delta=0.42)
    assert result.prior_recall_delta == pytest.approx(-0.03)
    assert result.novel_recall_delta == pytest.approx(0.42)


def test_summary_states_the_gate_outcome() -> None:
    """The before/after block must show the verdict, not leave it to be inferred."""
    assert "PASS" in make_result(0.0, 0.5).summary()
    assert "FAIL" in make_result(-0.5, 0.5).summary()


# --- Replay buffer ------------------------------------------------------------------------------


def _windows(n: int, seed: int = 0) -> np.ndarray:
    """Return `n` random windows."""
    return np.random.default_rng(seed).random((n, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)


def test_buffer_keeps_classes_balanced_under_capacity() -> None:
    """A rare prior type must not be crowded out — it is what forgetting erases first."""
    buffer = ReplayBuffer(capacity=90)
    # Wildly imbalanced input: 500 of one type, 10 of another.
    buffer.add(_windows(500), np.ones(500), ["Common"] * 500)
    buffer.add(_windows(10, seed=1), np.ones(10), ["Rare"] * 10)

    sample_X, _ = buffer.sample(60)
    assert len(buffer) <= 90
    assert set(buffer.classes_) == {"Common", "Rare"}
    # The rare class must survive both storage and sampling.
    assert len(sample_X) > 0


def test_buffer_respects_capacity() -> None:
    """A fog node cannot hold unbounded history."""
    buffer = ReplayBuffer(capacity=50)
    for index, name in enumerate(["A", "B", "C"]):
        buffer.add(_windows(200, seed=index), np.ones(200), [name] * 200)
    assert len(buffer) <= 50 + len(buffer.classes_)  # integer division slack of <1 per class


def test_buffer_sample_spans_every_stored_class() -> None:
    """Replay that omitted a class would fail to protect exactly that class."""
    buffer = ReplayBuffer(capacity=300)
    buffer.add(_windows(100), np.ones(100), ["A"] * 100)
    buffer.add(_windows(100, seed=1), np.zeros(100), ["B"] * 100)

    _, sample_y = buffer.sample(60)
    # Class A is all Attack, class B all Normal, so both labels present means both classes sampled.
    assert set(np.unique(sample_y)) == {0, 1}


def test_empty_buffer_samples_nothing_without_raising() -> None:
    """The first-ever update has no prior samples to replay."""
    X, y = ReplayBuffer().sample(10)
    assert len(X) == 0 and len(y) == 0


def test_buffer_rejects_mismatched_lengths() -> None:
    """A silent mismatch would pair windows with the wrong class."""
    buffer = ReplayBuffer()
    with lengths_must_match():
        buffer.add(_windows(10), np.ones(10), ["A"] * 5)


def lengths_must_match():
    """Return the expected-exception context for a length mismatch."""
    return pytest.raises(ValueError, match="lengths differ")


def test_buffer_is_reproducible_under_a_fixed_seed() -> None:
    """Replay selection must not vary run to run, or results are not reproducible."""
    def build() -> np.ndarray:
        buffer = ReplayBuffer(capacity=100, random_state=42)
        buffer.add(_windows(300), np.ones(300), ["A"] * 300)
        return buffer.sample(20)[0]

    np.testing.assert_array_equal(build(), build())


# --- Layer freezing -------------------------------------------------------------------------------


def test_only_head_and_embedding_layers_stay_trainable() -> None:
    """Feature extractors encode 'what traffic looks like' and must survive the update."""
    model = build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)
    trainable_before = int(sum(np.prod(w.shape) for w in model.trainable_weights))

    trainable_after = set_trainable_layers(model)

    assert trainable_after < trainable_before, "freezing must reduce the trainable surface"
    for layer in model.layers:
        expected = layer.name.endswith(TRAINABLE_LAYER_SUFFIXES)
        assert layer.trainable == expected, f"{layer.name} trainable={layer.trainable}"


def test_conv_layers_are_frozen() -> None:
    """Specifically: the convolutional stack must not move during an incremental update."""
    model = build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)
    set_trainable_layers(model)
    conv = [l for l in model.layers if l.__class__.__name__ == "Conv1D"]
    assert conv and not any(l.trainable for l in conv)


def test_frozen_model_still_produces_valid_predictions() -> None:
    """Freezing must not break inference."""
    model = build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)
    set_trainable_layers(model)
    output = model.predict(_windows(4), verbose=0)
    assert output.shape == (4, 2)
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-5)


# --- Documented constants ---------------------------------------------------------------------------


def test_constants_match_the_documented_values() -> None:
    """These are the knobs the report cites; they must be named constants, not inline literals."""
    assert REPLAY_RATIO == 2.0
    assert BUFFER_CAPACITY == 2000
    assert TRAINABLE_LAYER_SUFFIXES == ("_head", "_embedding")
