"""Unit tests for the adaptive threshold (Build-Instructions T3.4).

T3.4's VERIFY condition — and `TRD.md §9`'s "Adaptive threshold responds to criticality" gate — is:

    "A test with two synthetic criticality levels (e.g. 'critical device' vs. 'non-critical device')
     produces two different effective threshold values, not one shared constant."

That is `test_critical_and_non_critical_devices_get_different_thresholds`, which checks both
thresholds this module produces AND asserts against the base paper's flat 500 ms, since a scheme
that varied but happened to land on one value everywhere would satisfy a naive reading of the gate
while fixing nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptive.threshold import (
    BASE_PAPER_FIXED_THRESHOLD_MS,
    CONTEXT_SENSITIVITY,
    DETECTION_BASE,
    DETECTION_CEILING,
    DETECTION_FLOOR,
    INTER_MESSAGE_BASE_MS,
    LOAD_NEUTRAL,
    Criticality,
    adaptive_detection_threshold,
    adaptive_inter_message_threshold,
    apply_threshold,
    threshold_table,
)


# --- T3.4 / TRD §9 VERIFY -------------------------------------------------------------------


def test_critical_and_non_critical_devices_get_different_thresholds() -> None:
    """THE gate: two criticality levels must give two different thresholds, not one constant."""
    critical = adaptive_detection_threshold(Criticality.LIFE_CRITICAL)
    non_critical = adaptive_detection_threshold(Criticality.NON_CLINICAL)

    assert critical.effective != non_critical.effective
    # And in the clinically correct direction: the life-sustaining device is MORE sensitive.
    assert critical.effective < non_critical.effective

    critical_ms = adaptive_inter_message_threshold(Criticality.LIFE_CRITICAL)
    non_critical_ms = adaptive_inter_message_threshold(Criticality.NON_CLINICAL)
    assert critical_ms.effective != non_critical_ms.effective
    assert critical_ms.effective < non_critical_ms.effective


def test_no_two_criticality_classes_share_a_threshold() -> None:
    """A scheme collapsing to one value would satisfy the gate's letter but fix nothing."""
    values = [
        adaptive_detection_threshold(criticality).effective for criticality in Criticality
    ]
    assert len(set(values)) == len(list(Criticality))


def test_thresholds_differ_from_the_base_papers_flat_rule() -> None:
    """The point of Objection #4 is that 500 ms applied to every device. It must not any more."""
    timings = {
        criticality: adaptive_inter_message_threshold(criticality).effective
        for criticality in Criticality
    }
    differing = [
        criticality for criticality, value in timings.items()
        if value != BASE_PAPER_FIXED_THRESHOLD_MS
    ]
    assert len(differing) == len(list(Criticality)) - 1, (
        "only the STANDARD reference class should coincide with the base paper's 500 ms"
    )
    assert timings[Criticality.STANDARD] == BASE_PAPER_FIXED_THRESHOLD_MS


# --- Ordering and monotonicity ----------------------------------------------------------------


def test_sensitivity_decreases_monotonically_with_criticality() -> None:
    """More critical devices must never be less sensitive than less critical ones."""
    order = [
        Criticality.LIFE_CRITICAL,
        Criticality.HIGH,
        Criticality.STANDARD,
        Criticality.NON_CLINICAL,
    ]
    values = [adaptive_detection_threshold(c).effective for c in order]
    assert values == sorted(values), "threshold must rise as criticality falls"


def test_standard_class_at_neutral_load_reproduces_the_base() -> None:
    """STANDARD is the reference class by definition, so it must be a no-op."""
    decision = adaptive_detection_threshold(Criticality.STANDARD, load=LOAD_NEUTRAL)
    assert decision.effective == pytest.approx(DETECTION_BASE)
    assert decision.criticality_weight == 1.0
    assert decision.context_factor == pytest.approx(1.0)


def test_standard_class_reproduces_the_base_papers_500ms() -> None:
    """Keeping the reference class at 500 ms makes the comparison like-for-like."""
    decision = adaptive_inter_message_threshold(Criticality.STANDARD, load=LOAD_NEUTRAL)
    assert decision.effective == pytest.approx(INTER_MESSAGE_BASE_MS)


# --- Network context ----------------------------------------------------------------------------


def test_load_raises_the_threshold_and_idle_lowers_it() -> None:
    """Congestion makes benign traffic look anomalous, so the bar rises under load."""
    idle = adaptive_detection_threshold(Criticality.STANDARD, load=0.0).effective
    neutral = adaptive_detection_threshold(Criticality.STANDARD, load=LOAD_NEUTRAL).effective
    busy = adaptive_detection_threshold(Criticality.STANDARD, load=1.0).effective
    assert idle < neutral < busy


def test_context_cannot_override_clinical_criticality() -> None:
    """A transient condition must never outrank a stable clinical property.

    A life-sustaining device on a saturated network must still be more sensitive than a
    non-clinical device on an idle one. This is why CONTEXT_SENSITIVITY is far smaller than the
    criticality range, and the test pins that relationship rather than trusting the constants.
    """
    critical_under_load = adaptive_detection_threshold(
        Criticality.LIFE_CRITICAL, load=1.0
    ).effective
    non_clinical_idle = adaptive_detection_threshold(
        Criticality.NON_CLINICAL, load=0.0
    ).effective
    assert critical_under_load < non_clinical_idle


def test_context_factor_stays_within_its_documented_band() -> None:
    """The documented band is 1 +/- CONTEXT_SENSITIVITY/2."""
    for load in (0.0, 0.25, 0.5, 0.75, 1.0):
        factor = adaptive_detection_threshold(Criticality.STANDARD, load).context_factor
        assert 1 - CONTEXT_SENSITIVITY / 2 <= factor <= 1 + CONTEXT_SENSITIVITY / 2


@pytest.mark.parametrize("bad_load", [-0.01, 1.01, 2.0, -1.0])
def test_out_of_range_load_is_rejected(bad_load: float) -> None:
    """Silently clamping a bad load would hide a broken telemetry feed."""
    with pytest.raises(ValueError, match="load must be in"):
        adaptive_detection_threshold(Criticality.STANDARD, load=bad_load)


def test_non_positive_base_is_rejected() -> None:
    """A zero or negative base would make every threshold meaningless."""
    with pytest.raises(ValueError, match="must be positive"):
        adaptive_detection_threshold(Criticality.STANDARD, base=0.0)


# --- Clamping -------------------------------------------------------------------------------------


def test_detection_threshold_stays_a_valid_probability() -> None:
    """Multiplying two factors can leave [0, 1]; a threshold no probability can cross is a bug."""
    for criticality in Criticality:
        for load in (0.0, 0.5, 1.0):
            value = adaptive_detection_threshold(criticality, load).effective
            assert DETECTION_FLOOR <= value <= DETECTION_CEILING


def test_clamping_is_reported_not_hidden() -> None:
    """When the floor or ceiling binds, the caller must be able to see it."""
    # An extreme base drives the raw value past the ceiling.
    decision = adaptive_detection_threshold(Criticality.NON_CLINICAL, load=1.0, base=0.99)
    assert decision.effective == pytest.approx(DETECTION_CEILING)
    assert decision.clamped is True

    normal = adaptive_detection_threshold(Criticality.STANDARD)
    assert normal.clamped is False


# --- Applying thresholds to real predictions ------------------------------------------------------


def test_same_probability_flags_differently_by_criticality() -> None:
    """The whole point: identical evidence, different verdicts, because the devices differ."""
    borderline = np.array([0.35, 0.35])
    predictions = apply_threshold(
        borderline, [Criticality.LIFE_CRITICAL, Criticality.NON_CLINICAL]
    )
    assert predictions[0] == 1, "0.35 must clear the life-critical bar of 0.30"
    assert predictions[1] == 0, "0.35 must not clear the non-clinical bar of 0.65"


def test_apply_threshold_accepts_per_sample_loads() -> None:
    """Load can vary per sample; a scalar must broadcast."""
    probabilities = np.array([0.5, 0.5, 0.5])
    criticalities = [Criticality.STANDARD] * 3

    scalar = apply_threshold(probabilities, criticalities, loads=0.5)
    per_sample = apply_threshold(probabilities, criticalities, loads=np.array([0.5, 0.5, 0.5]))
    np.testing.assert_array_equal(scalar, per_sample)


def test_apply_threshold_rejects_mismatched_lengths() -> None:
    """A silent length mismatch would apply the wrong device's threshold to a detection."""
    with pytest.raises(ValueError, match="criticalities"):
        apply_threshold(np.array([0.5, 0.5]), [Criticality.STANDARD])
    with pytest.raises(ValueError, match="loads"):
        apply_threshold(
            np.array([0.5, 0.5]), [Criticality.STANDARD] * 2, loads=np.array([0.5])
        )


def test_predictions_use_the_projects_positive_class_convention() -> None:
    """1 = Attack = positive (TRD.md §2.3)."""
    predictions = apply_threshold(
        np.array([0.99, 0.01]), [Criticality.STANDARD, Criticality.STANDARD]
    )
    assert predictions[0] == 1 and predictions[1] == 0


# --- Explainability of the threshold itself --------------------------------------------------------


def test_decision_explains_how_it_was_reached() -> None:
    """A threshold should be as explainable as a detection (PRD.md §4)."""
    explanation = adaptive_detection_threshold(Criticality.LIFE_CRITICAL, load=0.9).explain()
    assert "LIFE_CRITICAL" in explanation
    assert "more sensitive" in explanation
    assert "infusion pump" in explanation  # plain language, not just a class name


def test_threshold_table_contrasts_with_the_base_paper() -> None:
    """Any table of these values must show what it is replacing."""
    table = threshold_table()
    for criticality in Criticality:
        assert criticality.name in table
    assert "500" in table and "flat" in table.lower()


def test_every_criticality_has_a_plain_language_description() -> None:
    """Operator-facing output must never show only an enum name."""
    for criticality in Criticality:
        assert len(criticality.description) > 10
        assert criticality.description.islower() or "(" in criticality.description
