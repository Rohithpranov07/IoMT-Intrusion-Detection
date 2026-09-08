"""Unit tests for temperature scaling (`docs/calibration_decision.md`).

Calibration was **measured and rejected** — it is not in the pipeline. These tests exist anyway,
because the module is the instrument that produced the rejection: if it were wrong, the rejection
would be worthless, and T3.6 has to re-run it on Edge-IIoTset before assuming the conclusion
transfers.

The most important test here is `test_temperature_scaling_never_changes_predictions`. That property
is what made calibration safe to *evaluate* against a frozen architecture, and it is what lets
`fit_temperature` treat any accuracy change as a bug rather than a trade-off.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibration import (
    ECE_N_BINS,
    PROBABILITY_EPSILON,
    apply_temperature,
    expected_calibration_error,
    fit_all_branches,
    fit_temperature,
    negative_log_likelihood,
)


def make_overconfident(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build probabilities that are far more confident than they are accurate.

    Args:
        n: number of samples.
        seed: RNG seed.

    Returns:
        Tuple of `(n, 2)` probabilities and `(n,)` labels. Roughly 20% of the confident
        predictions are wrong, so a well-fitted temperature should exceed 1.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    correct = rng.random(n) > 0.20  # 80% accurate...
    predicted = np.where(correct, labels, 1 - labels)
    confidence = rng.uniform(0.97, 0.999, size=n)  # ...but claims ~98% certainty
    attack = np.where(predicted == 1, confidence, 1.0 - confidence)
    return np.stack([1.0 - attack, attack], axis=1), labels


# --- The property that makes calibration safe to evaluate ------------------------------------


@pytest.mark.parametrize("temperature", [0.25, 0.5, 1.0, 2.0, 10.0, 50.0])
def test_temperature_scaling_never_changes_predictions(temperature: float) -> None:
    """Dividing every logit by one positive scalar is monotonic, so argmax cannot move.

    This is what made it legitimate to evaluate calibration against a FROZEN architecture: it
    cannot degrade a branch's own accuracy, only rescale the confidences downstream components
    read.
    """
    probabilities, _ = make_overconfident()
    calibrated = apply_temperature(probabilities, temperature)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), calibrated.argmax(axis=1))


def test_temperature_one_is_the_identity() -> None:
    """T = 1 must be a no-op, or the fit cannot report 'already calibrated'."""
    probabilities, _ = make_overconfident()
    np.testing.assert_allclose(apply_temperature(probabilities, 1.0), probabilities, atol=1e-6)


def test_output_remains_a_valid_distribution() -> None:
    """Calibrated outputs must still be probabilities."""
    probabilities, _ = make_overconfident()
    calibrated = apply_temperature(probabilities, 3.0)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_high_temperature_softens_and_low_sharpens() -> None:
    """T > 1 must reduce confidence; T < 1 must increase it."""
    probabilities, _ = make_overconfident()
    base = probabilities.max(axis=1).mean()
    assert apply_temperature(probabilities, 5.0).max(axis=1).mean() < base
    assert apply_temperature(probabilities, 0.5).max(axis=1).mean() > base


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.5])
def test_non_positive_temperature_is_rejected(bad: float) -> None:
    """A non-positive temperature would invert or destroy the ordering — not calibration."""
    probabilities, _ = make_overconfident()
    with pytest.raises(ValueError, match="must be positive"):
        apply_temperature(probabilities, bad)


def test_saturated_probabilities_do_not_produce_infinities() -> None:
    """A saturated branch emits exact 0.0 and 1.0, whose logit difference is infinite."""
    probabilities = np.array([[0.0, 1.0], [1.0, 0.0]])
    calibrated = apply_temperature(probabilities, 2.0)
    assert np.isfinite(calibrated).all()
    assert PROBABILITY_EPSILON > 0


# --- Fitting -----------------------------------------------------------------------------------


def test_fit_recovers_a_temperature_above_one_for_an_overconfident_branch() -> None:
    """The fit must detect genuine overconfidence."""
    probabilities, labels = make_overconfident()
    result = fit_temperature(probabilities, labels, "overconfident")
    assert result.temperature > 1.0
    assert result.mean_confidence_after < result.mean_confidence_before


def test_fit_can_report_under_confidence() -> None:
    """The search is two-sided, so a T < 1 branch surfaces rather than being assumed away.

    This is not hypothetical: the trained BiLSTM fitted T = 0.79
    (`docs/calibration_decision.md` §2.1), which is what refuted the assumed defect.
    """
    probabilities, labels = make_overconfident()
    # Over-soften first, so the correct fix is to sharpen.
    under_confident = apply_temperature(probabilities, 12.0)
    result = fit_temperature(under_confident, labels, "under-confident")
    assert result.temperature < 1.0


def test_fitting_improves_the_objective_it_minimises() -> None:
    """NLL is the fitted objective, so it must not get worse."""
    probabilities, labels = make_overconfident()
    result = fit_temperature(probabilities, labels, "branch")
    assert result.nll_after <= result.nll_before


def test_fitting_improves_calibration_error() -> None:
    """ECE is not the fitted objective, so improving it is evidence the fit means something."""
    probabilities, labels = make_overconfident()
    result = fit_temperature(probabilities, labels, "branch")
    assert result.ece_after < result.ece_before


def test_fit_preserves_accuracy_exactly() -> None:
    """`fit_temperature` raises on any accuracy change, so this pins the guarantee."""
    probabilities, labels = make_overconfident()
    result = fit_temperature(probabilities, labels, "branch")
    assert result.accuracy_before == result.accuracy_after


def test_mismatched_lengths_are_rejected() -> None:
    """Silently truncating would fit a temperature against the wrong labels."""
    probabilities, labels = make_overconfident(n=50)
    with pytest.raises(ValueError, match="probabilities"):
        fit_temperature(probabilities, labels[:10], "branch")


def test_each_branch_is_fitted_independently() -> None:
    """One shared scalar would assume the branches are miscalibrated identically. They are not."""
    a, labels = make_overconfident(seed=1)
    b = apply_temperature(a, 6.0)  # deliberately different miscalibration

    results = fit_all_branches({"a": a, "b": b}, labels)
    assert set(results) == {"a", "b"}
    assert results["a"].temperature != results["b"].temperature


# --- The metrics themselves ---------------------------------------------------------------------


def test_ece_is_zero_for_a_perfectly_calibrated_predictor() -> None:
    """A predictor whose confidence equals its accuracy must score 0."""
    # 100 samples at exactly 0.8 confidence, 80 of them correct.
    attack = np.full(100, 0.8)
    probabilities = np.stack([1.0 - attack, attack], axis=1)
    labels = np.array([1] * 80 + [0] * 20)
    assert expected_calibration_error(probabilities, labels, n_bins=ECE_N_BINS) < 0.01


def test_ece_is_large_for_a_confidently_wrong_predictor() -> None:
    """Maximum confidence with zero accuracy is the worst case."""
    attack = np.full(100, 0.99)
    probabilities = np.stack([1.0 - attack, attack], axis=1)
    labels = np.zeros(100, dtype=int)  # every prediction wrong
    assert expected_calibration_error(probabilities, labels) > 0.9


def test_nll_rewards_correct_confident_predictions() -> None:
    """NLL is a proper scoring rule — being confidently right must beat being confidently wrong."""
    labels = np.ones(10, dtype=int)
    right = np.tile([0.01, 0.99], (10, 1))
    wrong = np.tile([0.99, 0.01], (10, 1))
    assert negative_log_likelihood(right, labels) < negative_log_likelihood(wrong, labels)


def test_calibration_is_not_wired_into_the_pipeline() -> None:
    """`docs/calibration_decision.md` REJECTS calibration. Nothing may import it silently.

    The module is kept as the instrument that produced the rejection, but if it ever appears in the
    fusion, threshold, or deployment path, that is a decision requiring a new record — not
    something to happen by accident.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src"
    for path in root.rglob("*.py"):
        if path.name == "calibration.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "from src.models.calibration" not in text, (
            f"{path} imports the rejected calibration module; see docs/calibration_decision.md"
        )
        assert "import calibration" not in text, f"{path} imports the rejected calibration module"
