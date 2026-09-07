"""SHAP explanations for the ensemble (Build-Instructions T3.1; TRD.md §4).

This module is the first half of Contribution 3, the fix for the explainability half of
Objection #4 (`PRD.md §2.1.4`): HIDS-IoMT flags traffic without telling anyone *why*, in a setting
where the person reading the alert may be a hospital IT lead rather than an ML engineer.

Method, and why (decided in `docs/xai_survey.md`, T1.4)
-------------------------------------------------------
SHAP is the **primary** method; LIME (T3.2) is the faster fallback. SHAP was chosen for its
additive consistency: attributions sum to `prediction - baseline`, so "these five features account
for most of this alert" is a literally true statement rather than a suggestive one. That property is
what makes a per-detection number defensible when a clinician asks how much a feature mattered.

HOW SHAP IS APPLIED TO A CONFIDENCE-WEIGHTED ENSEMBLE -- this project's own decision
------------------------------------------------------------------------------------
The deployed prediction is `P = sum_b w_b * p_b` where `w_b` depends on each branch's own softmax
output (`src/models/fusion.py`). That makes the fusion non-differentiable end to end, so SHAP
cannot be run over the ensemble as a single graph. `docs/xai_survey.md` §2.1 records the route
taken instead:

    1. Run `shap.GradientExplainer` on EACH branch separately, against that branch's own output.
    2. Recombine the three attribution maps using THE SAME confidence weights `w_b` that the fusion
       layer used for that sample.

This is coherent rather than a fudge: the fused probability is a weighted sum of branch outputs, so
by linearity of expectation a weighted sum of branch attributions is the consistent attribution for
that fused output. It is nonetheless **this project's own construction**, not off-the-shelf SHAP
behaviour, and any report using these attributions must say so.

Sequence collapse rule (`docs/xai_survey.md` §5 item 4)
-------------------------------------------------------
Raw attributions have shape `(sequence_length, n_features)` = `(10, 62)`. A clinician cannot read a
620-cell grid, so per-feature scores are obtained by **summing across timesteps** -- summing, not
averaging, because SHAP attributions are additive and a sum preserves the additivity property that
motivated choosing SHAP. The full per-timestep map is retained in `Explanation.timestep_attributions`
so "*when* in the window" stays answerable.

Fixed parameters
----------------
    DEFAULT_TOP_N         = 5      features listed in an explanation. Small enough to read at a
                                   glance; `docs/xai_survey.md` §5 makes explanation length the
                                   comprehensibility proxy standing in for the out-of-scope user
                                   study (`PRD.md §5.3`).
    DEFAULT_BACKGROUND_SIZE = 100  background windows summarising "normal" for the explainer.
                                   GradientExplainer cost grows with this; 100 is the standard
                                   default and is fixed here rather than left implicit.
    ATTRIBUTION_EPSILON   = 1e-12  guards division when normalising attribution shares.

NO LATENCY BUDGET IS SET HERE. `docs/xai_survey.md` §5 specifies that the SHAP-vs-LIME cutover
threshold is tuned in Phase 3 against the **measured** Raspberry Pi numbers from T4.3. Those
measurements do not exist yet, and `Build-Instructions.md` §A.1 rule 3 forbids inventing one.
`explain()` reports its own elapsed time so the budget can be set from data later.

Positive class: index 1 = **Attack** = the positive class (`TRD.md §2.3`). Explanations are
generated for flows flagged as Attack, and attributions are reported **toward that class**.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import shap
from tensorflow import keras

from src.config import NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME, POSITIVE_LABEL, RANDOM_STATE
from src.models.fusion import CONFIDENCE_SHARPNESS, confidence_weighted_fusion

logger = logging.getLogger(__name__)

DEFAULT_TOP_N: int = 5
DEFAULT_BACKGROUND_SIZE: int = 100
ATTRIBUTION_EPSILON: float = 1e-12


@dataclass
class FeatureAttribution:
    """One feature's contribution to one detection.

    Attributes:
        rank: 1-based position in the explanation, most influential first.
        feature_name: the ORIGINAL column name (e.g. `Flow_Duration`), never an index. T3.1's
            VERIFY block requires a reader be able to map every listed feature back to the data.
        attribution: signed SHAP value summed across the window's timesteps. Positive pushes the
            model toward **Attack**; negative pushes toward Normal.
        share_of_total: this feature's |attribution| as a share of all features' |attribution|.
        peak_timestep: the timestep (0 = oldest, 9 = most recent) contributing most of it.
    """

    rank: int
    feature_name: str
    attribution: float
    share_of_total: float
    peak_timestep: int

    def direction(self) -> str:
        """Return which class this feature pushed the prediction toward."""
        return POSITIVE_CLASS_NAME if self.attribution > 0 else NEGATIVE_CLASS_NAME


@dataclass
class Explanation:
    """A complete per-detection explanation (`TRD.md §4`'s output contract).

    Attributes:
        predicted_label: 1 = Attack (positive class), 0 = Normal.
        confidence: the ensemble's fused confidence in that prediction.
        top_features: ranked `FeatureAttribution`s, most influential first.
        branch_weights: the fusion weight each branch received for this sample.
        branch_names: branch names, matching `branch_weights` order.
        timestep_attributions: full `(sequence_length, n_features)` map, retained so "when in the
            window" remains answerable after the per-feature collapse.
        feature_names: all feature names, in `timestep_attributions` column order.
        method: always "shap" here; `Explanation` is shared with the LIME fallback (T3.2), which
            sets "lime". An explanation must never be silently downgraded without saying so.
        elapsed_seconds: wall-clock time to produce this explanation. Feeds the SHAP-vs-LIME
            cutover threshold once Pi measurements exist (T4.3).
    """

    predicted_label: int
    confidence: float
    top_features: list[FeatureAttribution]
    branch_weights: np.ndarray
    branch_names: list[str]
    timestep_attributions: np.ndarray = field(repr=False)
    feature_names: list[str] = field(repr=False)
    method: str = "shap"
    elapsed_seconds: float = 0.0

    def predicted_class_name(self) -> str:
        """Return the human-readable predicted class."""
        return POSITIVE_CLASS_NAME if self.predicted_label == POSITIVE_LABEL else NEGATIVE_CLASS_NAME

    def to_summary(self) -> str:
        """Render the plain-language summary `TRD.md §4` requires.

        Target reader is a hospital IT lead or clinician, not an ML engineer (`PRD.md §4`), so the
        text avoids "SHAP value", "softmax", "attribution", and feature indices. Every feature is
        named, and the sentence structure states what the model saw and how strongly.

        Returns:
            A multi-line, self-contained explanation string.
        """
        verdict = self.predicted_class_name().upper()
        lines = [
            f"ALERT: this traffic window was classified as {verdict} "
            f"with {self.confidence:.0%} confidence.",
            "",
            f"The {len(self.top_features)} measurements that most influenced this decision:",
        ]

        for attribution in self.top_features:
            pushed = (
                "raised the suspicion score"
                if attribution.attribution > 0
                else "lowered the suspicion score"
            )
            lines.append(
                f"  {attribution.rank}. {attribution.feature_name} — {pushed} "
                f"({attribution.share_of_total:.0%} of the total evidence), "
                f"most strongly at record {attribution.peak_timestep + 1} of "
                f"{self.timestep_attributions.shape[0]} in this window."
            )

        leading = int(np.argmax(self.branch_weights))
        lines += [
            "",
            f"Detector agreement: the {self.branch_names[leading]} detector carried the most weight "
            f"({self.branch_weights[leading]:.0%}); weights across all detectors were "
            + ", ".join(
                f"{name} {weight:.0%}"
                for name, weight in zip(self.branch_names, self.branch_weights)
            )
            + ".",
            f"(Explanation produced by {self.method.upper()} in {self.elapsed_seconds:.2f}s. "
            f"'{POSITIVE_CLASS_NAME}' is the positive class.)",
        ]
        return "\n".join(lines)

    def as_table(self) -> list[dict[str, object]]:
        """Return the ranked features as plain dicts, for a DataFrame or a report table."""
        return [
            {
                "rank": a.rank,
                "feature": a.feature_name,
                "attribution": a.attribution,
                "share_of_total": a.share_of_total,
                "pushes_toward": a.direction(),
                "peak_timestep": a.peak_timestep,
            }
            for a in self.top_features
        ]


class EnsembleShapExplainer:
    """SHAP explanations for the confidence-weighted ensemble.

    Builds one `shap.GradientExplainer` per branch, then recombines their attributions with the
    fusion layer's own confidence weights (see the module docstring for why).

    Attributes:
        branch_names: branch names in a fixed order, used everywhere downstream.
        feature_names: original feature column names.
        explainers: the per-branch SHAP explainers.
    """

    def __init__(
        self,
        models: dict[str, keras.Model],
        feature_names: list[str],
        background: np.ndarray,
        background_size: int = DEFAULT_BACKGROUND_SIZE,
        gamma: float = CONFIDENCE_SHARPNESS,
        branch_priors: np.ndarray | list[float] | None = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Construct one SHAP explainer per branch.

        Args:
            models: mapping of branch name to trained branch model. All must accept the same
                `(batch, sequence_length, n_features)` input.
            feature_names: original column names, length `n_features`. Passing indices instead
                would defeat T3.1's whole purpose.
            background: reference windows summarising "normal", shape
                `(n, sequence_length, n_features)`. Should come from the TRAINING fold -- drawing
                it from test data would leak evaluation data into the explanation.
            background_size: number of background windows to sample.
            gamma: fusion sharpness, kept in step with `src/models/fusion.py`.
            branch_priors: per-branch `alpha_b`, kept in step with the fusion layer.
            random_state: seed for background sampling, so explanations are reproducible.

        Raises:
            ValueError: if `models` is empty, if `feature_names` does not match the background's
                feature count, or if `background` is not 3-D.
        """
        if not models:
            raise ValueError("At least one branch model is required")
        background = np.asarray(background, dtype=np.float32)
        if background.ndim != 3:
            raise ValueError(
                f"background must be (n, sequence_length, n_features); got {background.shape}"
            )
        if len(feature_names) != background.shape[2]:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but background has "
                f"{background.shape[2]} features"
            )

        self.branch_names = list(models.keys())
        self.models = models
        self.feature_names = list(feature_names)
        self.gamma = gamma
        self.branch_priors = branch_priors

        rng = np.random.default_rng(random_state)
        size = min(background_size, len(background))
        sample = background[rng.choice(len(background), size=size, replace=False)]
        logger.info("Building SHAP explainers with %d background windows", size)

        self.explainers = {
            name: shap.GradientExplainer(model, sample) for name, model in models.items()
        }

    def _branch_attributions(self, window: np.ndarray, class_index: int) -> dict[str, np.ndarray]:
        """Compute each branch's SHAP attributions toward one class.

        Args:
            window: a single window, shape `(1, sequence_length, n_features)`.
            class_index: output class to attribute toward. 1 = Attack.

        Returns:
            Mapping of branch name to a `(sequence_length, n_features)` attribution map.
        """
        attributions: dict[str, np.ndarray] = {}
        for name, explainer in self.explainers.items():
            values = np.asarray(explainer.shap_values(window))
            # GradientExplainer returns (n_samples, sequence_length, n_features, n_classes).
            attributions[name] = values[0, :, :, class_index]
        return attributions

    def explain(
        self,
        window: np.ndarray,
        top_n: int = DEFAULT_TOP_N,
        class_index: int | None = None,
    ) -> Explanation:
        """Explain one detection.

        Args:
            window: a single window, shape `(sequence_length, n_features)` or
                `(1, sequence_length, n_features)`.
            top_n: how many features to list.
            class_index: class to attribute toward. Defaults to the ensemble's own predicted
                class, which is what an operator is asking about when they open an alert.

        Returns:
            An `Explanation`.

        Raises:
            ValueError: if `window` does not describe exactly one window of the right width.
        """
        started = time.perf_counter()

        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 2:
            window = window[np.newaxis, ...]
        if window.ndim != 3 or window.shape[0] != 1:
            raise ValueError(
                f"explain() takes exactly one window; got shape {window.shape}"
            )
        if window.shape[2] != len(self.feature_names):
            raise ValueError(
                f"window has {window.shape[2]} features but the explainer was built for "
                f"{len(self.feature_names)}"
            )

        # 1. Branch predictions, then the SAME fusion the deployed model uses.
        branch_probabilities = [
            self.models[name].predict(window, verbose=0) for name in self.branch_names
        ]
        fusion = confidence_weighted_fusion(
            branch_probabilities,
            self.branch_names,
            branch_priors=self.branch_priors,
            gamma=self.gamma,
        )
        predicted = int(fusion.predictions[0])
        target_class = predicted if class_index is None else class_index

        # 2. Per-branch attributions, recombined with the fusion's own weights.
        weights = fusion.branch_weights[0]
        per_branch = self._branch_attributions(window, target_class)
        combined = np.zeros_like(next(iter(per_branch.values())))
        for weight, name in zip(weights, self.branch_names):
            combined += weight * per_branch[name]

        # 3. Collapse timesteps by SUMMING (preserves SHAP's additivity).
        per_feature = combined.sum(axis=0)
        magnitude_total = np.abs(per_feature).sum() + ATTRIBUTION_EPSILON
        order = np.argsort(np.abs(per_feature))[::-1][:top_n]

        top_features = [
            FeatureAttribution(
                rank=position + 1,
                feature_name=self.feature_names[int(index)],
                attribution=float(per_feature[index]),
                share_of_total=float(abs(per_feature[index]) / magnitude_total),
                peak_timestep=int(np.argmax(np.abs(combined[:, index]))),
            )
            for position, index in enumerate(order)
        ]

        explanation = Explanation(
            predicted_label=predicted,
            confidence=float(fusion.confidence[0]),
            top_features=top_features,
            branch_weights=weights,
            branch_names=list(self.branch_names),
            timestep_attributions=combined,
            feature_names=list(self.feature_names),
            method="shap",
            elapsed_seconds=time.perf_counter() - started,
        )
        logger.info(
            "SHAP explanation for a %s detection in %.2fs; top feature %s",
            explanation.predicted_class_name(),
            explanation.elapsed_seconds,
            top_features[0].feature_name if top_features else "n/a",
        )
        return explanation

    def explain_flagged(
        self,
        windows: np.ndarray,
        predictions: np.ndarray,
        top_n: int = DEFAULT_TOP_N,
        limit: int | None = None,
    ) -> list[Explanation]:
        """Explain only the windows flagged as Attack.

        `docs/xai_survey.md` §5 item 1: explanations are generated **only for flagged flows**, never
        for all traffic, so the detection path stays fast regardless of SHAP's cost.

        Args:
            windows: `(n, sequence_length, n_features)`.
            predictions: predicted labels for those windows.
            top_n: features per explanation.
            limit: cap on how many flagged windows to explain. None explains all of them.

        Returns:
            One `Explanation` per explained window, in input order.
        """
        flagged = np.flatnonzero(np.asarray(predictions).ravel() == POSITIVE_LABEL)
        if limit is not None:
            flagged = flagged[:limit]
        logger.info("Explaining %d flagged (%s) windows", len(flagged), POSITIVE_CLASS_NAME)
        return [self.explain(windows[index], top_n=top_n) for index in flagged]
