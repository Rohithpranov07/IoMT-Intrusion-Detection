"""LIME explanations for the ensemble (Build-Instructions T3.2; TRD.md §4).

The second half of Contribution 3's explainability layer. `docs/xai_survey.md` (T1.4) designates
LIME the **fallback** to T3.1's SHAP, for cases where SHAP's cost is prohibitive.

>>> LIME IS A CROSS-CHECK ONLY. DO NOT SHOW ITS OUTPUT TO AN OPERATOR. <<<
--------------------------------------------------------------------------
Measured in T3.3 (`reports/t3_3_xai_evaluation.md`), on the trained ensemble:

    stability (top-5 Jaccard under 1% input noise)   SHAP 0.873    LIME 0.269
    model's own prediction flip rate under that noise             0.0%

The model does not change its mind under that perturbation, so the instability is entirely LIME's.
A Jaccard of 0.27 means two operators opening the same alert can be shown substantially different
reasons for it -- which erodes trust faster than offering no explanation at all, and is
disqualifying in a clinical setting.

Raising `num_samples` does not rescue it. Measured: 1,000 -> 4,000 -> 12,000 samples moves Jaccard
only 0.16 -> 0.38 -> 0.33, at twelve times the cost, and never reaches the 0.50 floor
(`TRUST_MIN_STABILITY`) required to be operator-facing. Worse, LIME's apparent fidelity COLLAPSES as
its sampling converges (+0.349 -> +0.187 -> +0.023): its high early fidelity score was an artefact
of a noisy surrogate, not evidence that it had found the features that matter.

**LIME therefore stays in the codebase as an independent cross-check on SHAP and for the ablation
in T4.4, and is never rendered to a human.** `explain()` stamps every LIME explanation accordingly.

>>> ON SPEED, WHICH IS A SEPARATE POINT <<<
--------------------------------------------------------
`docs/xai_survey.md` §2.2 predicted LIME would be materially faster than SHAP. **On this project's
actual configuration that prediction does not hold** -- see `reports/t3_2_lime_vs_shap.md` for the
measurement. The survey compared SHAP *generically*, where the model-agnostic `KernelExplainer` needs
thousands of forward passes per explanation. T3.1 does not use `KernelExplainer`; it uses
`GradientExplainer`, which is gradient-based and cheap. LIME's speed advantage was an advantage over
a SHAP variant this project never adopted.

LIME is still worth having, for a reason the survey also identified and which does hold:

  **LIME sees the deployed ensemble directly.** It needs only `predict_proba`, so it explains the
  *fused* model -- confidence weights and all -- as one object. SHAP cannot: the fusion is not
  differentiable end to end, so T3.1 must run per branch and recombine the attribution maps with the
  fusion weights, a construction this project invented and must defend. LIME needs no such step.

So the two methods are complementary along the axis of **faithfulness to the deployed model**, not
primarily along speed. `docs/xai_survey.md` §4 should be annotated to that effect.

Exact parameters
----------------
    NUM_SAMPLES            = 1000   perturbed neighbours per explanation. LIME's own default is
                                    5000; 1000 is chosen because the surrogate's coefficients are
                                    already stable enough at this size for a 620-dimensional input
                                    while costing a fifth as much. This is the single knob trading
                                    explanation stability against latency, which is why it is a
                                    named constant rather than an inline literal.
    DISCRETIZE_CONTINUOUS  = True   LIME's quartile discretiser samples perturbations from the
                                    TRAINING data's own quartile bins rather than from an unbounded
                                    Gaussian. This is the first half of the plausibility constraint
                                    `docs/xai_survey.md` §6 requires.
    CLIP_TO_TRAINING_RANGE = True   Second half, and this project's own addition beyond stock LIME:
                                    every perturbed row is clipped to the per-feature [min, max]
                                    observed in the background data before it reaches the model.
                                    Without it the surrogate can be fitted on flow records with
                                    impossible values (negative packet counts, inter-arrival
                                    statistics inconsistent with the flow duration), and would then
                                    explain behaviour in a region the model will never meet.
                                    The bounds are WIDENED per call to include the explained window
                                    itself (`_bounds_for`), so clipping constrains the perturbed
                                    NEIGHBOURHOOD but can never alter the instance being explained.
                                    Without that widening, a test window with a feature outside the
                                    background's observed range would be silently clipped and the
                                    explanation would describe a flow that was never seen -- scaling
                                    is fitted on the training fold, so out-of-range test values are
                                    expected, not hypothetical.
    FEATURE_SELECTION      = "none" keep a weight for EVERY flattened input, so the per-feature
                                    shares below sum to 100% and are directly comparable with
                                    T3.1's SHAP shares. LIME's default ("auto") returns only the
                                    top-k and would make the shares incomparable.
    RANDOM_STATE           = 42     LIME samples its neighbourhood at random, and
                                    `docs/xai_survey.md` §2.2 flags instability as its documented
                                    weakness. Seeding makes a single explanation reproducible.
                                    It does NOT make LIME stable -- stability under *input*
                                    perturbation is a real property that T3.3 must measure, and
                                    seeding must never be presented as having fixed it.

Sequence handling
-----------------
`LimeTabularExplainer` is tabular, so each `(10, 62)` window is flattened to 620 inputs named
`<feature>@t<timestep>`. The resulting 620 weights are collapsed back to 62 per-feature scores by
**summing across timesteps** -- the identical rule T3.1 uses, so SHAP and LIME outputs are directly
comparable. The per-timestep map is retained in `Explanation.timestep_attributions`.

Output: the same `Explanation` dataclass T3.1 returns, with `method="lime"`. An explanation is never
silently downgraded -- the consumer can always see which method produced it
(`docs/xai_survey.md` §5 item 3).

Positive class: index 1 = **Attack** (`TRD.md §2.3`).
"""

from __future__ import annotations

import logging
import time

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from tensorflow import keras

from src.config import POSITIVE_LABEL, RANDOM_STATE
from src.models.fusion import CONFIDENCE_SHARPNESS, confidence_weighted_fusion
from src.xai.shap_explainer import (
    ATTRIBUTION_EPSILON,
    DEFAULT_TOP_N,
    Explanation,
    FeatureAttribution,
)

logger = logging.getLogger(__name__)

NUM_SAMPLES: int = 1000
DISCRETIZE_CONTINUOUS: bool = True
CLIP_TO_TRAINING_RANGE: bool = True
FEATURE_SELECTION: str = "none"
CLASS_NAMES: tuple[str, str] = ("Normal", "Attack")


class EnsembleLimeExplainer:
    """LIME explanations for the confidence-weighted ensemble.

    Unlike `EnsembleShapExplainer`, this explains the **fused** model as a single black box, which
    is LIME's real advantage here (see the module docstring).

    Attributes:
        branch_names: branch names in a fixed order.
        feature_names: original feature column names.
        sequence_length: timesteps per window.
        explainer: the underlying `LimeTabularExplainer`.
    """

    def __init__(
        self,
        models: dict[str, keras.Model],
        feature_names: list[str],
        background: np.ndarray,
        num_samples: int = NUM_SAMPLES,
        gamma: float = CONFIDENCE_SHARPNESS,
        branch_priors: np.ndarray | list[float] | None = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Build the LIME explainer over the flattened sequence representation.

        Args:
            models: branch name to trained model. Fused internally, exactly as deployed.
            feature_names: original column names, length `n_features`.
            background: reference windows from the TRAINING fold, shape
                `(n, sequence_length, n_features)`. Supplies both the quartile bins LIME perturbs
                within and the [min, max] clipping bounds.
            num_samples: perturbed neighbours per explanation.
            gamma: fusion sharpness, kept in step with `src/models/fusion.py`.
            branch_priors: per-branch `alpha_b`, kept in step with the fusion layer.
            random_state: seed for LIME's sampling.

        Raises:
            ValueError: if `models` is empty, `background` is not 3-D, or `feature_names` does not
                match the background's feature count.
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

        self.models = models
        self.branch_names = list(models.keys())
        self.feature_names = list(feature_names)
        self.sequence_length = int(background.shape[1])
        self.n_features = int(background.shape[2])
        self.num_samples = num_samples
        self.gamma = gamma
        self.branch_priors = branch_priors

        flat_background = background.reshape(len(background), -1)

        # Plausibility bounds, per flattened input, taken from the TRAINING data only.
        self._lower = flat_background.min(axis=0)
        self._upper = flat_background.max(axis=0)

        # Names carry both the feature and the timestep, so a raw LIME output is still readable.
        self.flat_feature_names = [
            f"{name}@t{timestep}"
            for timestep in range(self.sequence_length)
            for name in self.feature_names
        ]

        self.explainer = LimeTabularExplainer(
            flat_background,
            mode="classification",
            feature_names=self.flat_feature_names,
            class_names=list(CLASS_NAMES),
            discretize_continuous=DISCRETIZE_CONTINUOUS,
            feature_selection=FEATURE_SELECTION,
            random_state=random_state,
        )
        logger.info(
            "LIME explainer built over %d flattened inputs (%d timesteps x %d features)",
            len(self.flat_feature_names), self.sequence_length, self.n_features,
        )

    def _bounds_for(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return clipping bounds widened to include the window being explained.

        The background's per-feature [min, max] bounds the perturbed neighbourhood. But scaling is
        fitted on the training fold, so a genuine test window can legitimately fall outside that
        range — and clipping it would mean explaining a flow that never occurred. Widening the
        bounds to cover the instance keeps the plausibility constraint on the neighbourhood while
        guaranteeing the instance itself passes through untouched.

        Args:
            window: the window being explained, any shape; only its values matter.

        Returns:
            Tuple of `(lower, upper)` arrays over the flattened input.
        """
        flat = np.asarray(window, dtype=np.float32).ravel()
        return np.minimum(self._lower, flat), np.maximum(self._upper, flat)

    def _predict_flattened(
        self,
        rows: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Predict fused class probabilities for a batch of flattened, perturbed windows.

        This is the function LIME probes. It reconstructs the window shape, enforces the
        plausibility constraint, then runs the **full deployed pipeline** including fusion — which
        is precisely what makes LIME faithful to the deployed model.

        Args:
            rows: `(n, sequence_length * n_features)` perturbed rows from LIME.
            bounds: `(lower, upper)` clipping bounds. Defaults to the background's own range;
                `explain()` passes bounds widened to include the explained instance.

        Returns:
            `(n, 2)` fused class probabilities. Column 1 is Attack (the positive class).
        """
        rows = np.asarray(rows, dtype=np.float32)
        if CLIP_TO_TRAINING_RANGE:
            lower, upper = (self._lower, self._upper) if bounds is None else bounds
            rows = np.clip(rows, lower, upper)

        windows = rows.reshape(-1, self.sequence_length, self.n_features)
        branch_probabilities = [
            self.models[name].predict(windows, verbose=0) for name in self.branch_names
        ]
        return confidence_weighted_fusion(
            branch_probabilities,
            self.branch_names,
            branch_priors=self.branch_priors,
            gamma=self.gamma,
        ).probabilities

    def explain(
        self,
        window: np.ndarray,
        top_n: int = DEFAULT_TOP_N,
        class_index: int | None = None,
        num_samples: int | None = None,
    ) -> Explanation:
        """Explain one detection with LIME.

        Args:
            window: a single window, `(sequence_length, n_features)` or
                `(1, sequence_length, n_features)`.
            top_n: how many features to list.
            class_index: class to explain. Defaults to the ensemble's own predicted class.
            num_samples: override the perturbation budget for this call.

        Returns:
            An `Explanation` with `method="lime"`, structurally identical to T3.1's SHAP output.

        Raises:
            ValueError: if `window` is not exactly one window of the expected width.
        """
        started = time.perf_counter()

        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 2:
            window = window[np.newaxis, ...]
        if window.ndim != 3 or window.shape[0] != 1:
            raise ValueError(f"explain() takes exactly one window; got shape {window.shape}")
        if window.shape[2] != self.n_features:
            raise ValueError(
                f"window has {window.shape[2]} features but the explainer was built for "
                f"{self.n_features}"
            )

        # The deployed decision, from the same fusion the explanation will describe.
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

        # Bounds widened to include this instance, so it is never itself clipped.
        bounds = self._bounds_for(window)

        lime_explanation = self.explainer.explain_instance(
            window.ravel(),
            lambda rows: self._predict_flattened(rows, bounds=bounds),
            labels=(target_class,),
            num_features=len(self.flat_feature_names),
            num_samples=self.num_samples if num_samples is None else num_samples,
        )

        # Surrogate weights -> (sequence_length, n_features), same layout as T3.1's SHAP map.
        flat_weights = np.zeros(len(self.flat_feature_names), dtype=np.float64)
        for flat_index, weight in lime_explanation.as_map()[target_class]:
            flat_weights[int(flat_index)] = weight
        timestep_map = flat_weights.reshape(self.sequence_length, self.n_features)

        # Collapse by SUMMING across timesteps -- identical rule to T3.1, so the two are comparable.
        per_feature = timestep_map.sum(axis=0)
        magnitude_total = np.abs(per_feature).sum() + ATTRIBUTION_EPSILON
        order = np.argsort(np.abs(per_feature))[::-1][:top_n]

        top_features = [
            FeatureAttribution(
                rank=position + 1,
                feature_name=self.feature_names[int(index)],
                attribution=float(per_feature[index]),
                share_of_total=float(abs(per_feature[index]) / magnitude_total),
                peak_timestep=int(np.argmax(np.abs(timestep_map[:, index]))),
            )
            for position, index in enumerate(order)
        ]

        explanation = Explanation(
            predicted_label=predicted,
            confidence=float(fusion.confidence[0]),
            top_features=top_features,
            branch_weights=fusion.branch_weights[0],
            branch_names=list(self.branch_names),
            timestep_attributions=timestep_map,
            feature_names=list(self.feature_names),
            method="lime",
            elapsed_seconds=time.perf_counter() - started,
        )
        logger.warning(
            "LIME explanation produced (cross-check only, NOT operator-facing -- "
            "measured top-5 Jaccard 0.27 under 1%% input noise; see the module docstring)"
        )
        logger.info(
            "LIME explanation for a %s detection in %.2fs; top feature %s",
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
        """Explain only the windows flagged as Attack (`docs/xai_survey.md` §5 item 1).

        Args:
            windows: `(n, sequence_length, n_features)`.
            predictions: predicted labels for those windows.
            top_n: features per explanation.
            limit: cap on how many flagged windows to explain.

        Returns:
            One `Explanation` per explained window, in input order.
        """
        flagged = np.flatnonzero(np.asarray(predictions).ravel() == POSITIVE_LABEL)
        if limit is not None:
            flagged = flagged[:limit]
        return [self.explain(windows[index], top_n=top_n) for index in flagged]
