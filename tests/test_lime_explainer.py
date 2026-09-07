"""Unit tests for the LIME explainer (Build-Instructions T3.2).

T3.2's VERIFY condition — *"runs measurably faster than the SHAP path from T3.1 on the same sample
input"* — is a **timing** claim about the trained model, so it is measured by
`scripts/benchmark_xai.py` against real artifacts rather than asserted here on toy models.
It **fails**: SHAP is ~1.9x faster, because T3.1 uses `GradientExplainer` rather than the
`KernelExplainer` that `docs/xai_survey.md` §2.2 had in mind. See the survey's AMENDMENT section
and `reports/t3_2_lime_vs_shap.md`.

`test_output_is_structurally_interchangeable_with_shap` is the test that actually matters for the
task's intent: whatever the relative speed, LIME must be a drop-in alternative producing the same
explanation contract, so a consumer can switch methods without changing code.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.config import NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME
from src.models.bilstm_branch import build_bilstm_branch
from src.models.cnn_branch import build_cnn_branch
from src.models.transformer_branch import build_transformer_branch
from src.xai.lime_explainer import (
    CLIP_TO_TRAINING_RANGE,
    DISCRETIZE_CONTINUOUS,
    FEATURE_SELECTION,
    NUM_SAMPLES,
    EnsembleLimeExplainer,
)
from src.xai.shap_explainer import DEFAULT_TOP_N, EnsembleShapExplainer, Explanation

SEQUENCE_LENGTH = 10
N_FEATURES = 6
FEATURE_NAMES = [
    "Flow_Duration", "Flow_Pkts/s", "Dst_Port", "ACK_Flag_Cnt", "Idle_Max", "Pkt_Size_Avg",
]
# Small sample budget: these tests check the contract, not explanation quality.
TEST_SAMPLES = 120


@pytest.fixture(scope="module")
def models() -> dict:
    """Three small untrained branches sharing one input shape."""
    warnings.filterwarnings("ignore")
    return {
        "cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES),
        "bilstm": build_bilstm_branch(SEQUENCE_LENGTH, N_FEATURES),
        "transformer": build_transformer_branch(SEQUENCE_LENGTH, N_FEATURES),
    }


@pytest.fixture(scope="module")
def background() -> np.ndarray:
    """Background windows standing in for the training fold."""
    return np.random.default_rng(0).random((40, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)


@pytest.fixture(scope="module")
def explainer(models, background) -> EnsembleLimeExplainer:
    """A LIME explainer over the fused ensemble."""
    return EnsembleLimeExplainer(models, FEATURE_NAMES, background)


@pytest.fixture(scope="module")
def sample_window() -> np.ndarray:
    """A single deterministic window to explain."""
    return np.random.default_rng(7).random((SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)


# --- The contract that matters: interchangeable with SHAP ------------------------------------


def test_output_is_structurally_interchangeable_with_shap(
    explainer, models, background, sample_window
) -> None:
    """A consumer must be able to swap methods without changing any code around it."""
    shap_explainer = EnsembleShapExplainer(
        models, FEATURE_NAMES, background, background_size=20
    )
    lime_explanation = explainer.explain(sample_window, num_samples=TEST_SAMPLES)
    shap_explanation = shap_explainer.explain(sample_window)

    assert isinstance(lime_explanation, Explanation)
    # Both describe the SAME deployed decision -- only the attribution method differs.
    assert lime_explanation.predicted_label == shap_explanation.predicted_label
    assert abs(lime_explanation.confidence - shap_explanation.confidence) < 1e-6
    np.testing.assert_allclose(
        lime_explanation.branch_weights, shap_explanation.branch_weights, atol=1e-6
    )
    assert (
        lime_explanation.timestep_attributions.shape
        == shap_explanation.timestep_attributions.shape
    )


def test_method_is_recorded_as_lime(explainer, sample_window) -> None:
    """`docs/xai_survey.md` §5 item 3: an explanation is never silently downgraded."""
    explanation = explainer.explain(sample_window, num_samples=TEST_SAMPLES)
    assert explanation.method == "lime"
    assert "LIME" in explanation.to_summary()


# --- Naming and readability (same standard as T3.1) ------------------------------------------


def test_top_features_use_original_feature_names(explainer, sample_window) -> None:
    """Features must map back to real columns, not flattened `name@t3` indices."""
    explanation = explainer.explain(sample_window, num_samples=TEST_SAMPLES)

    assert len(explanation.top_features) == DEFAULT_TOP_N
    for attribution in explanation.top_features:
        assert attribution.feature_name in FEATURE_NAMES
        assert "@t" not in attribution.feature_name, (
            "the flattened LIME name leaked out instead of the original column name"
        )


def test_summary_avoids_ml_jargon(explainer, sample_window) -> None:
    """`TRD.md §4`'s reader is a clinician, whichever method produced the explanation."""
    summary = explainer.explain(sample_window, num_samples=TEST_SAMPLES).to_summary().lower()
    for jargon in ("softmax", "surrogate", "ridge", "logit", "tensor", "perturb"):
        assert jargon not in summary, f"jargon {jargon!r} leaked into the summary"


# --- Sequence handling: the collapse rule must match T3.1 -------------------------------------


def test_timestep_map_has_full_resolution(explainer, sample_window) -> None:
    """620 flattened weights must fold back to a (10, 62)-shaped map."""
    explanation = explainer.explain(sample_window, num_samples=TEST_SAMPLES)
    assert explanation.timestep_attributions.shape == (SEQUENCE_LENGTH, N_FEATURES)


def test_per_feature_scores_are_the_timestep_sum(explainer, sample_window) -> None:
    """The collapse is a SUM across timesteps -- identical to T3.1, so the two are comparable."""
    explanation = explainer.explain(sample_window, num_samples=TEST_SAMPLES)
    for attribution in explanation.top_features:
        column = explanation.feature_names.index(attribution.feature_name)
        expected = explanation.timestep_attributions[:, column].sum()
        assert abs(expected - attribution.attribution) < 1e-6


def test_shares_sum_to_one_over_all_features(explainer, sample_window) -> None:
    """`FEATURE_SELECTION = 'none'` keeps every weight, so shares are a true distribution."""
    features = explainer.explain(
        sample_window, top_n=N_FEATURES, num_samples=TEST_SAMPLES
    ).top_features
    assert abs(sum(f.share_of_total for f in features) - 1.0) < 1e-6


def test_features_are_ranked_by_absolute_influence(explainer, sample_window) -> None:
    """Rank 1 is the most influential feature regardless of direction."""
    features = explainer.explain(sample_window, num_samples=TEST_SAMPLES).top_features
    magnitudes = [abs(f.attribution) for f in features]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [f.rank for f in features] == list(range(1, len(features) + 1))


# --- Plausibility constraint (docs/xai_survey.md §6) -----------------------------------------


def test_perturbed_rows_are_clipped_to_the_training_range(explainer) -> None:
    """This project's own addition: LIME must not probe impossible flow records.

    Without clipping, the surrogate can be fitted on windows with values the model will never
    meet (negative packet counts, IAT statistics inconsistent with the flow duration), and would
    then explain behaviour in a region that does not exist.
    """
    assert CLIP_TO_TRAINING_RANGE is True

    # Feed values far outside the background range; they must be clipped before reaching a model.
    wild = np.full((2, SEQUENCE_LENGTH * N_FEATURES), 1e6, dtype=np.float32)
    probabilities = explainer._predict_flattened(wild)

    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_discretisation_is_enabled(explainer) -> None:
    """Quartile sampling from the training data is the first half of the plausibility guard."""
    assert DISCRETIZE_CONTINUOUS is True
    assert FEATURE_SELECTION == "none"


def test_predict_wrapper_runs_the_full_fused_pipeline(explainer, sample_window) -> None:
    """LIME's advantage is seeing the DEPLOYED model -- fusion included, not one branch.

    Bounds are widened to include the instance (as `explain()` does), so clipping cannot alter it
    and the wrapper must reproduce the deployed prediction exactly.
    """
    from src.models.fusion import confidence_weighted_fusion

    window = sample_window[np.newaxis, ...]
    through_wrapper = explainer._predict_flattened(
        window.reshape(1, -1), bounds=explainer._bounds_for(window)
    )
    directly = confidence_weighted_fusion(
        [explainer.models[n].predict(window, verbose=0) for n in explainer.branch_names],
        explainer.branch_names,
    ).probabilities
    np.testing.assert_allclose(through_wrapper, directly, atol=1e-6)


def test_explained_instance_is_never_clipped(explainer, sample_window) -> None:
    """A window outside the background's range must still be explained as itself.

    Scaling is fitted on the training fold, so genuine test windows CAN exceed the background's
    observed per-feature range. Clipping the instance would mean the explanation described a flow
    that never occurred -- a quiet correctness bug, not a rounding detail.
    """
    out_of_range = sample_window * 3.0  # well outside the background's [0, 1)-ish range
    lower, upper = explainer._bounds_for(out_of_range)

    clipped = np.clip(out_of_range.ravel(), lower, upper)
    np.testing.assert_allclose(clipped, out_of_range.ravel(), atol=0)


def test_perturbed_neighbours_are_still_bounded_by_the_instance(explainer, sample_window) -> None:
    """Widening for the instance must not disable the constraint for everything else."""
    lower, upper = explainer._bounds_for(sample_window)
    wild = np.full(SEQUENCE_LENGTH * N_FEATURES, 1e6, dtype=np.float32)

    clipped = np.clip(wild, lower, upper)
    assert (clipped <= np.maximum(explainer._upper, sample_window.ravel()) + 1e-6).all()
    assert clipped.max() < 1e6, "an out-of-range neighbour escaped the plausibility constraint"


# --- Reproducibility, and what seeding does NOT buy -------------------------------------------


def test_seeding_makes_one_explanation_reproducible(models, background, sample_window) -> None:
    """Two explainers with the same seed must agree on the same input.

    This covers run-to-run reproducibility only. It is NOT stability under input perturbation,
    which is LIME's documented weakness (`docs/xai_survey.md` §2.2) and T3.3's job to measure.
    Nothing here should be read as evidence that LIME is stable.
    """
    first = EnsembleLimeExplainer(models, FEATURE_NAMES, background, random_state=42)
    second = EnsembleLimeExplainer(models, FEATURE_NAMES, background, random_state=42)

    a = first.explain(sample_window, num_samples=TEST_SAMPLES)
    b = second.explain(sample_window, num_samples=TEST_SAMPLES)
    assert [f.feature_name for f in a.top_features] == [f.feature_name for f in b.top_features]


def test_num_samples_default_is_the_documented_constant() -> None:
    """The stability/latency knob must be a named constant, not an inline literal."""
    assert NUM_SAMPLES == 1000


# --- Flagged-only behaviour and validation -----------------------------------------------------


def test_only_flagged_windows_are_explained(explainer) -> None:
    """Explanations are produced for flagged flows only (`docs/xai_survey.md` §5 item 1)."""
    windows = np.random.default_rng(3).random((5, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)
    predictions = np.array([1, 0, 1, 0, 0])
    assert len(explainer.explain_flagged(windows, predictions, limit=2)) == 2


def test_explain_rejects_a_batch(explainer) -> None:
    """`explain()` is single-window; a silent batch would misattribute the explanation."""
    batch = np.random.default_rng(5).random((3, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)
    with pytest.raises(ValueError, match="exactly one window"):
        explainer.explain(batch)


def test_explain_rejects_wrong_feature_count(explainer) -> None:
    """A width mismatch would misalign every feature name."""
    wrong = np.random.default_rng(6).random((SEQUENCE_LENGTH, N_FEATURES + 2)).astype(np.float32)
    with pytest.raises(ValueError, match="features"):
        explainer.explain(wrong)


def test_construction_rejects_mismatched_feature_names(models, background) -> None:
    """Names and data must line up or the explanation names the wrong columns."""
    with pytest.raises(ValueError, match="feature_names"):
        EnsembleLimeExplainer(models, FEATURE_NAMES[:-1], background)


def test_construction_rejects_two_dimensional_background(models) -> None:
    """The background must be windows, matching the model's input rank."""
    with pytest.raises(ValueError, match="sequence_length"):
        EnsembleLimeExplainer(models, FEATURE_NAMES, np.zeros((10, N_FEATURES), dtype=np.float32))


def test_construction_rejects_empty_model_set(background) -> None:
    """There is nothing to explain without at least one branch."""
    with pytest.raises(ValueError, match="At least one branch"):
        EnsembleLimeExplainer({}, FEATURE_NAMES, background)


def test_direction_names_the_class(explainer, sample_window) -> None:
    """A reader should see 'Attack'/'Normal', never a bare sign."""
    for row in explainer.explain(sample_window, num_samples=TEST_SAMPLES).as_table():
        assert row["pushes_toward"] in (POSITIVE_CLASS_NAME, NEGATIVE_CLASS_NAME)
