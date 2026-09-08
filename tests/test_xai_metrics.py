"""Unit tests for the explainability metrics (Build-Instructions T3.3).

T3.3's VERIFY block is a human check — *"have a team member unfamiliar with the model internals read
one sample explanation output and correctly state, in their own words, why that flow was flagged"* —
which no test can perform. `reports/t3_1_shap_examples.md` is the artefact for that reading.

What IS testable, and what these tests cover, is that the metrics themselves are sound. That matters
more than usual here, because a defect in this module gave this project a WRONG answer about its own
explainability layer once already: measuring fidelity at a single k rated SHAP "moderate" when the
full curve shows it is strongly faithful. `test_single_k_fidelity_can_be_misleading_on_saturated_models`
pins that lesson.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.xai.metrics import (
    AOPC_KS,
    CERTIFY_KS,
    TRUST_MIN_STABILITY,
    TRUST_MIN_TOP1_GAIN,
    Certification,
    FidelityCurve,
    ablate_features,
    certify_explanation,
    measure_fidelity_curve,
    measure_stability,
)

SEQUENCE_LENGTH = 6
N_FEATURES = 8
DRIVING_FEATURE = 3  # the one feature our synthetic model actually uses


# --- Synthetic model with a KNOWN ground truth ------------------------------------------------
# Real explainer tests cannot check correctness, because the true attribution of a trained network
# is unknown. Here the model is constructed so that exactly one feature drives it, which means a
# faithful explanation is knowable and the metrics can be checked against it.


def driving_predict_fn(windows: np.ndarray) -> np.ndarray:
    """A model whose decision depends ONLY on `DRIVING_FEATURE`.

    Args:
        windows: `(n, sequence_length, n_features)`.

    Returns:
        `(n, 2)` probabilities; Attack rises with the driving feature's mean.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim == 2:
        windows = windows[np.newaxis, ...]
    # Threshold 0.7 sits ABOVE the background mean (~0.5), so ablating the driving feature to
    # that mean flips the decision decisively. A threshold below the background mean would leave
    # the ablated window still classified as Attack, and the fixture would be too weak to
    # distinguish a faithful explanation from an unfaithful one.
    signal = windows[:, :, DRIVING_FEATURE].mean(axis=1)
    attack = 1.0 / (1.0 + np.exp(-12.0 * (signal - 0.7)))
    return np.stack([1.0 - attack, attack], axis=1)


class StubExplanation:
    """Minimal stand-in for `Explanation`, carrying only what the metrics read."""

    def __init__(self, feature_names: list[str], ranked_indices: list[int], method: str = "stub"):
        self.feature_names = feature_names
        self.method = method
        self.predicted_label = 1
        self.top_features = [
            type("F", (), {
                "feature_name": feature_names[i],
                "attribution": float(len(ranked_indices) - rank),
                "share_of_total": 1.0 / len(ranked_indices),
            })()
            for rank, i in enumerate(ranked_indices)
        ]


class StubExplainer:
    """An explainer returning a fixed ranking, so metric behaviour is fully determined."""

    def __init__(self, feature_names: list[str], ranking: list[int]):
        self.feature_names = feature_names
        self.ranking = ranking

    def explain(self, window: np.ndarray, top_n: int = 5) -> StubExplanation:
        """Return the fixed ranking, truncated to `top_n`."""
        return StubExplanation(self.feature_names, self.ranking[:top_n])


@pytest.fixture
def feature_names() -> list[str]:
    """Readable column names."""
    return [f"Feature_{i}" for i in range(N_FEATURES)]


@pytest.fixture
def background() -> np.ndarray:
    """Background windows with all features near 0.5."""
    return np.random.default_rng(0).uniform(
        0.4, 0.6, (30, SEQUENCE_LENGTH, N_FEATURES)
    ).astype(np.float32)


@pytest.fixture
def attack_windows() -> np.ndarray:
    """Windows the synthetic model classifies as Attack, driven by `DRIVING_FEATURE`."""
    rng = np.random.default_rng(1)
    windows = rng.uniform(0.4, 0.6, (6, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)
    windows[:, :, DRIVING_FEATURE] = 0.95
    return windows


# --- Ablation ---------------------------------------------------------------------------------


def test_ablation_replaces_with_background_mean_not_zero(background) -> None:
    """Zeroing would measure sensitivity to an extreme, not to removal.

    Features are min-max scaled, so 0.0 is the observed MINIMUM — a meaningful extreme value.
    """
    window = np.full((SEQUENCE_LENGTH, N_FEATURES), 0.9, dtype=np.float32)
    background_mean = background.mean(axis=(0, 1))

    ablated = ablate_features(window, [2], background_mean)
    np.testing.assert_allclose(ablated[:, 2], background_mean[2], atol=1e-6)
    assert not np.allclose(ablated[:, 2], 0.0), "ablation must not zero the feature"
    np.testing.assert_allclose(ablated[:, 3], 0.9)  # untouched


def test_ablation_applies_across_every_timestep(background) -> None:
    """A feature is removed from the whole window, not one timestep."""
    window = np.full((SEQUENCE_LENGTH, N_FEATURES), 0.9, dtype=np.float32)
    ablated = ablate_features(window, [1], background.mean(axis=(0, 1)))
    assert len(set(np.round(ablated[:, 1], 6))) == 1


def test_keep_only_inverts_the_selection(background) -> None:
    """`sufficiency` needs the complement: keep the cited features, ablate the rest."""
    window = np.full((SEQUENCE_LENGTH, N_FEATURES), 0.9, dtype=np.float32)
    background_mean = background.mean(axis=(0, 1))

    kept = ablate_features(window, [0, 1], background_mean, keep_only=True)
    np.testing.assert_allclose(kept[:, 0], 0.9)
    np.testing.assert_allclose(kept[:, 1], 0.9)
    np.testing.assert_allclose(kept[:, 2], background_mean[2], atol=1e-6)


# --- Fidelity: a correct explainer must beat a wrong one --------------------------------------


def test_faithful_explainer_scores_above_random_control(
    feature_names, background, attack_windows
) -> None:
    """An explainer naming the truly-driving feature must show a large positive gain."""
    good = StubExplainer(feature_names, [DRIVING_FEATURE] + [
        i for i in range(N_FEATURES) if i != DRIVING_FEATURE
    ])
    curve = measure_fidelity_curve(
        good, driving_predict_fn, attack_windows, background, "good", ks=(1, 3)
    )
    assert curve.gains[1] > TRUST_MIN_TOP1_GAIN
    assert curve.is_significant(1)


def test_wrong_explainer_scores_near_zero(feature_names, background, attack_windows) -> None:
    """An explainer citing irrelevant features must NOT pass. This is the metric's whole job."""
    wrong_ranking = [i for i in range(N_FEATURES) if i != DRIVING_FEATURE]
    bad = StubExplainer(feature_names, wrong_ranking)

    curve = measure_fidelity_curve(
        bad, driving_predict_fn, attack_windows, background, "bad", ks=(1, 3)
    )
    assert curve.gains[1] < TRUST_MIN_TOP1_GAIN, (
        "an explanation citing features the model ignores must not score as faithful"
    )


def test_single_k_fidelity_can_be_misleading_on_saturated_models(
    feature_names, background, attack_windows
) -> None:
    """Pins the measurement bug that gave this project a wrong answer about its own layer.

    Fidelity at k=1 and at a larger k can disagree sharply, because the random control rises with
    k while a saturated model's probability has limited headroom left to fall. A single-k number
    is therefore not a safe summary, which is why `measure_fidelity_curve` and AOPC exist.
    """
    good = StubExplainer(feature_names, [DRIVING_FEATURE] + [
        i for i in range(N_FEATURES) if i != DRIVING_FEATURE
    ])
    curve = measure_fidelity_curve(
        good, driving_predict_fn, attack_windows, background, "good", ks=(1, 5)
    )
    # The k=1 gain must be the stronger signal; the curve exists so this is visible.
    assert curve.gains[1] >= curve.gains[5]
    assert set(curve.ks) == {1, 5}


def test_aopc_summarises_the_curve(feature_names, background, attack_windows) -> None:
    """AOPC must be the mean gain over AOPC_KS, so no favourable k can be cherry-picked."""
    good = StubExplainer(feature_names, list(range(N_FEATURES)))
    curve = measure_fidelity_curve(
        good, driving_predict_fn, attack_windows, background, "good", ks=AOPC_KS
    )
    assert abs(curve.aopc() - float(np.mean([curve.gains[k] for k in AOPC_KS]))) < 1e-9


def test_significance_requires_two_standard_errors() -> None:
    """A gain smaller than twice its standard error must not be called significant."""
    curve = FidelityCurve(
        method="stub", ks=(1, 3),
        top_drops={1: 0.5, 3: 0.2}, random_drops={1: 0.0, 3: 0.1},
        gains={1: 0.5, 3: 0.1}, gain_standard_errors={1: 0.05, 3: 0.09},
        n_windows=10,
    )
    assert curve.is_significant(1)        # 0.5 > 2 * 0.05
    assert not curve.is_significant(3)    # 0.1 < 2 * 0.09


# --- Stability ---------------------------------------------------------------------------------


def test_a_deterministic_explainer_is_perfectly_stable(
    feature_names, background, attack_windows
) -> None:
    """A fixed ranking cannot vary, so Jaccard must be 1.0 — a sanity floor for the metric."""
    stable = StubExplainer(feature_names, list(range(N_FEATURES)))
    result = measure_stability(
        stable, driving_predict_fn, attack_windows[:3], "stub", k=3, repeats=2
    )
    assert result.jaccard == pytest.approx(1.0)
    assert result.jaccard >= TRUST_MIN_STABILITY


def test_stability_tracks_the_models_own_flip_rate(
    feature_names, background, attack_windows
) -> None:
    """If the MODEL changes its mind under noise, the explainer must not be blamed for changing."""
    stable = StubExplainer(feature_names, list(range(N_FEATURES)))
    result = measure_stability(
        stable, driving_predict_fn, attack_windows[:3], "stub", k=3, repeats=2
    )
    assert 0.0 <= result.prediction_flip_rate <= 1.0


# --- Certification: the mechanism that makes the layer trustworthy ----------------------------


def test_certification_passes_a_faithful_explanation(feature_names, background, attack_windows):
    """An explanation naming the truly-driving feature must certify, at k=1."""
    explanation = StubExplanation(feature_names, [DRIVING_FEATURE, 0, 1])
    certification = certify_explanation(
        explanation, driving_predict_fn, attack_windows[0], background
    )
    assert certification.trusted
    assert certification.certified_k == 1
    assert "[VERIFIED]" in certification.banner()


def test_certification_rejects_an_unsupported_explanation(
    feature_names, background, attack_windows
) -> None:
    """An explanation citing features the model ignores must be labelled UNVERIFIED.

    This is the guard that stops an operator being handed a rationale the model does not support.
    """
    irrelevant = [i for i in range(N_FEATURES) if i != DRIVING_FEATURE][:5]
    explanation = StubExplanation(feature_names, irrelevant)

    certification = certify_explanation(
        explanation, driving_predict_fn, attack_windows[0], background
    )
    assert not certification.trusted
    assert certification.certified_k is None
    assert "[UNVERIFIED]" in certification.banner()
    assert certification.reasons


def test_certification_is_adaptive_over_k(feature_names, background, attack_windows) -> None:
    """Certifying only at k=1 was too strict; the smallest passing k is used instead."""
    # The driving feature is ranked SECOND, so k=1 fails but k=2 should succeed.
    explanation = StubExplanation(feature_names, [0, DRIVING_FEATURE, 1, 2, 4])
    certification = certify_explanation(
        explanation, driving_predict_fn, attack_windows[0], background
    )
    assert certification.trusted
    assert certification.certified_k == 2, "certification must widen k rather than fail outright"
    assert certification.top1_gain < TRUST_MIN_TOP1_GAIN


def test_certification_records_every_k_it_tried(feature_names, background, attack_windows) -> None:
    """A failure must be diagnosable, not just reported."""
    explanation = StubExplanation(feature_names, list(range(N_FEATURES)))
    certification = certify_explanation(
        explanation, driving_predict_fn, attack_windows[0], background
    )
    assert set(certification.gains_by_k).issubset(set(CERTIFY_KS))
    assert certification.gains_by_k


def test_certification_handles_an_empty_explanation(feature_names, background, attack_windows):
    """Citing nothing must fail closed, not raise."""
    explanation = StubExplanation(feature_names, [])
    explanation.top_features = []
    certification = certify_explanation(
        explanation, driving_predict_fn, attack_windows[0], background
    )
    assert not certification.trusted
    assert "cites no features" in certification.banner()


def test_uncertified_explanations_are_labelled_unchecked() -> None:
    """An unchecked explanation must never read as though it had been verified."""
    from src.xai.shap_explainer import Explanation, FeatureAttribution

    explanation = Explanation(
        predicted_label=1,
        confidence=0.99,
        top_features=[FeatureAttribution(1, "Flow_Duration", 0.5, 1.0, 0)],
        branch_weights=np.array([0.4, 0.3, 0.3]),
        branch_names=["cnn", "bilstm", "transformer"],
        timestep_attributions=np.zeros((SEQUENCE_LENGTH, 1)),
        feature_names=["Flow_Duration"],
    )
    assert explanation.certification is None
    assert "[UNCHECKED]" in explanation.to_summary()


def test_certified_explanation_shows_its_verdict() -> None:
    """The banner must lead the operator-facing summary."""
    from src.xai.shap_explainer import Explanation, FeatureAttribution

    explanation = Explanation(
        predicted_label=1,
        confidence=0.99,
        top_features=[FeatureAttribution(1, "Flow_Duration", 0.5, 1.0, 0)],
        branch_weights=np.array([0.4, 0.3, 0.3]),
        branch_names=["cnn", "bilstm", "transformer"],
        timestep_attributions=np.zeros((SEQUENCE_LENGTH, 1)),
        feature_names=["Flow_Duration"],
        certification=Certification(
            trusted=True, certified_k=1, top1_gain=0.4, certified_gain=0.4,
            certified_drop=0.45, random_drop=0.02, gains_by_k={1: 0.4},
            method="shap", reasons=[], elapsed_seconds=0.05,
        ),
    )
    summary = explanation.to_summary()
    assert summary.startswith("[VERIFIED]")
    assert "[UNCHECKED]" not in summary
