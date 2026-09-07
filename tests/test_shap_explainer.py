"""Unit tests for the SHAP explainer (Build-Instructions T3.1 VERIFY block).

The VERIFY condition is: *"Given a sample flagged detection, output includes a ranked feature list a
reader can map back to the original feature names (not just indices)."*
`test_top_features_use_original_feature_names` and `test_summary_is_readable_without_ml_background`
cover that from both the structured and the prose side.

These tests use small, untrained branches on synthetic data. They check the explainer's *contract* —
shapes, naming, ranking, fusion consistency — not attribution quality, which depends on a trained
model and is measured by the fidelity/stability metrics in T3.3.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.config import NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME, POSITIVE_LABEL
from src.models.bilstm_branch import build_bilstm_branch
from src.models.cnn_branch import build_cnn_branch
from src.models.transformer_branch import build_transformer_branch
from src.xai.shap_explainer import (
    DEFAULT_TOP_N,
    EnsembleShapExplainer,
    Explanation,
    FeatureAttribution,
)

SEQUENCE_LENGTH = 10
N_FEATURES = 8
FEATURE_NAMES = [
    "Flow_Duration", "Flow_Pkts/s", "Dst_Port", "Src_Port",
    "ACK_Flag_Cnt", "Idle_Max", "Bwd_Pkts/s", "Init_Bwd_Win_Byts",
]


@pytest.fixture(scope="module")
def explainer() -> EnsembleShapExplainer:
    """Build a three-branch explainer over small untrained models."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    background = rng.random((40, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)
    models = {
        "cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES),
        "bilstm": build_bilstm_branch(SEQUENCE_LENGTH, N_FEATURES),
        "transformer": build_transformer_branch(SEQUENCE_LENGTH, N_FEATURES),
    }
    return EnsembleShapExplainer(models, FEATURE_NAMES, background, background_size=20)


@pytest.fixture(scope="module")
def sample_window() -> np.ndarray:
    """A single deterministic window to explain."""
    return np.random.default_rng(7).random((SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)


# --- T3.1 VERIFY: names, not indices ---------------------------------------------------------


def test_top_features_use_original_feature_names(explainer, sample_window) -> None:
    """T3.1 VERIFY: every listed feature must be mappable back to the original data."""
    explanation = explainer.explain(sample_window)

    assert len(explanation.top_features) == DEFAULT_TOP_N
    for attribution in explanation.top_features:
        assert attribution.feature_name in FEATURE_NAMES, (
            "a feature was reported that does not correspond to a real column"
        )
        # An index dressed up as a name would defeat the whole task.
        assert not attribution.feature_name.isdigit()
        assert not attribution.feature_name.startswith("feature_")


def test_summary_is_readable_without_ml_background(explainer, sample_window) -> None:
    """`TRD.md §4`'s output contract targets a clinician, not an ML engineer."""
    summary = explainer.explain(sample_window).to_summary()

    # Names the reader can act on must be present...
    assert any(name in summary for name in FEATURE_NAMES)
    assert POSITIVE_CLASS_NAME in summary
    # ...and ML jargon must not be.
    for jargon in ("softmax", "shap value", "logit", "tensor", "argmax", "gradientexplainer"):
        assert jargon not in summary.lower(), f"jargon {jargon!r} leaked into the summary"


def test_summary_states_which_method_produced_it(explainer, sample_window) -> None:
    """`docs/xai_survey.md` §5 item 3: an explanation is never silently downgraded."""
    explanation = explainer.explain(sample_window)
    assert explanation.method == "shap"
    assert "SHAP" in explanation.to_summary()


# --- Ranking and attribution contract --------------------------------------------------------


def test_features_are_ranked_by_absolute_influence(explainer, sample_window) -> None:
    """Rank 1 must be the most influential feature, regardless of direction."""
    features = explainer.explain(sample_window).top_features

    assert [f.rank for f in features] == list(range(1, len(features) + 1))
    magnitudes = [abs(f.attribution) for f in features]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shares_are_fractions_of_the_total(explainer, sample_window) -> None:
    """`share_of_total` must be a genuine proportion, so 'X% of the evidence' is truthful."""
    features = explainer.explain(sample_window, top_n=N_FEATURES).top_features

    shares = [f.share_of_total for f in features]
    assert all(0.0 <= s <= 1.0 for s in shares)
    # Explaining every feature must account for all the evidence.
    assert abs(sum(shares) - 1.0) < 1e-6


def test_peak_timestep_is_within_the_window(explainer, sample_window) -> None:
    """'When in the window' must point at a real record."""
    for attribution in explainer.explain(sample_window).top_features:
        assert 0 <= attribution.peak_timestep < SEQUENCE_LENGTH


def test_timestep_map_is_retained_at_full_resolution(explainer, sample_window) -> None:
    """The per-feature collapse must not destroy the per-timestep detail (§ collapse rule)."""
    explanation = explainer.explain(sample_window)
    assert explanation.timestep_attributions.shape == (SEQUENCE_LENGTH, N_FEATURES)

    # The collapse is a SUM across timesteps -- verify that literally.
    for attribution in explanation.top_features:
        column = explanation.feature_names.index(attribution.feature_name)
        expected = explanation.timestep_attributions[:, column].sum()
        assert abs(expected - attribution.attribution) < 1e-5


# --- Consistency with the deployed fusion ----------------------------------------------------


def test_prediction_matches_the_fusion_layer(explainer, sample_window) -> None:
    """The explained prediction must be the ensemble's actual decision, not a branch's."""
    from src.models.fusion import confidence_weighted_fusion

    explanation = explainer.explain(sample_window)
    window = sample_window[np.newaxis, ...]
    fusion = confidence_weighted_fusion(
        [explainer.models[n].predict(window, verbose=0) for n in explainer.branch_names],
        explainer.branch_names,
    )

    assert explanation.predicted_label == int(fusion.predictions[0])
    assert abs(explanation.confidence - float(fusion.confidence[0])) < 1e-6
    np.testing.assert_allclose(explanation.branch_weights, fusion.branch_weights[0], atol=1e-6)


def test_branch_weights_sum_to_one(explainer, sample_window) -> None:
    """The reported detector agreement must be a distribution."""
    explanation = explainer.explain(sample_window)
    assert len(explanation.branch_weights) == len(explanation.branch_names) == 3
    assert abs(explanation.branch_weights.sum() - 1.0) < 1e-6


# --- Flagged-only behaviour (docs/xai_survey.md §5 item 1) -----------------------------------


def test_only_flagged_windows_are_explained(explainer) -> None:
    """Explanations are produced for flagged flows only, never for all traffic."""
    windows = np.random.default_rng(3).random((6, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)
    predictions = np.array([1, 0, 1, 0, 0, 1])

    explanations = explainer.explain_flagged(windows, predictions)
    assert len(explanations) == 3


def test_flagged_explanations_respect_the_limit(explainer) -> None:
    """A cap keeps a large flagged batch from blocking the detection path."""
    windows = np.random.default_rng(4).random((6, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)
    predictions = np.ones(6, dtype=int)
    assert len(explainer.explain_flagged(windows, predictions, limit=2)) == 2


# --- Input validation --------------------------------------------------------------------------


def test_explain_accepts_two_or_three_dimensional_single_window(explainer, sample_window) -> None:
    """Both (seq, features) and (1, seq, features) are valid ways to pass one window."""
    flat = explainer.explain(sample_window)
    batched = explainer.explain(sample_window[np.newaxis, ...])
    assert flat.predicted_label == batched.predicted_label


def test_explain_rejects_a_batch(explainer) -> None:
    """`explain()` is single-window; a silent batch would produce a misattributed explanation."""
    batch = np.random.default_rng(5).random((3, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)
    with pytest.raises(ValueError, match="exactly one window"):
        explainer.explain(batch)


def test_explain_rejects_wrong_feature_count(explainer) -> None:
    """A width mismatch would silently misalign every feature name."""
    wrong = np.random.default_rng(6).random((SEQUENCE_LENGTH, N_FEATURES + 1)).astype(np.float32)
    with pytest.raises(ValueError, match="features"):
        explainer.explain(wrong)


def test_construction_rejects_mismatched_feature_names() -> None:
    """Names and data must line up, or the explanation names the wrong columns."""
    background = np.random.default_rng(0).random((10, SEQUENCE_LENGTH, N_FEATURES)).astype(np.float32)
    with pytest.raises(ValueError, match="feature_names"):
        EnsembleShapExplainer(
            {"cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)},
            FEATURE_NAMES[:-1],
            background,
        )


def test_construction_rejects_two_dimensional_background() -> None:
    """The background must be windows, matching the model's input rank."""
    with pytest.raises(ValueError, match="sequence_length"):
        EnsembleShapExplainer(
            {"cnn": build_cnn_branch(SEQUENCE_LENGTH, N_FEATURES)},
            FEATURE_NAMES,
            np.zeros((10, N_FEATURES), dtype=np.float32),
        )


def test_construction_rejects_empty_model_set() -> None:
    """There is nothing to explain without at least one branch."""
    background = np.zeros((10, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
    with pytest.raises(ValueError, match="At least one branch"):
        EnsembleShapExplainer({}, FEATURE_NAMES, background)


# --- Presentation helpers ----------------------------------------------------------------------


def test_direction_names_the_class_not_a_sign() -> None:
    """A reader should see 'Attack'/'Normal', never '+' or '-'."""
    positive = FeatureAttribution(1, "Flow_Duration", 0.5, 0.5, 3)
    negative = FeatureAttribution(2, "Idle_Max", -0.5, 0.5, 4)
    assert positive.direction() == POSITIVE_CLASS_NAME
    assert negative.direction() == NEGATIVE_CLASS_NAME


def test_as_table_exposes_names_and_directions(explainer, sample_window) -> None:
    """The structured form must carry the same names the prose does."""
    rows = explainer.explain(sample_window).as_table()
    assert len(rows) == DEFAULT_TOP_N
    for row in rows:
        assert row["feature"] in FEATURE_NAMES
        assert row["pushes_toward"] in (POSITIVE_CLASS_NAME, NEGATIVE_CLASS_NAME)


def test_predicted_class_name_uses_the_project_convention(explainer, sample_window) -> None:
    """`TRD.md §2.3`: label 1 is Attack, the positive class."""
    explanation = explainer.explain(sample_window)
    expected = (
        POSITIVE_CLASS_NAME if explanation.predicted_label == POSITIVE_LABEL
        else NEGATIVE_CLASS_NAME
    )
    assert explanation.predicted_class_name() == expected
