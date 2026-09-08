"""Explainability evaluation metrics (Build-Instructions T3.3; TRD.md §4).

Why this module is the deciding measurement, not a box to tick
---------------------------------------------------------------
T3.2 found that SHAP and LIME agree on only ~36% of their top-5 features for the SAME decision on
the SAME model (`reports/t3_2_lime_vs_shap.md`). Agreement is a consistency check, not a
correctness one — two methods can agree and both be wrong, or disagree with one of them right. So
the disagreement cannot be resolved by inspecting explanations. It needs a measurement that asks the
*model* whether an explanation is true of it.

That is what fidelity does here, and it is why this module is the arbiter for the whole
explainability layer: whichever method's attributions actually move the model's decision when
ablated is the one describing the model faithfully.

The three metrics `TRD.md §4` requires
---------------------------------------
1. **Fidelity** — "does the explanation's top features, when ablated, actually change the model's
   prediction in the expected direction?" Implemented as two complementary halves, following the
   standard formulation used in the explainability literature (ERASER):

   - `comprehensiveness`: remove the top-k attributed features. A faithful explanation causes a
     LARGE drop in the predicted-class probability. Higher is better.

     >>> MEASURE THIS OVER A CURVE, NOT AT A SINGLE k. <<<
     A single-k comprehensiveness number was this project's own first attempt and it produced a
     WRONG conclusion. Measured only at k=5 on this ensemble it rated SHAP merely "moderate";
     measured at k=1 the same attributions score a fidelity gain of +0.46 at six times its
     standard error. The reason is saturation: the model predicts at P about 0.99 and its 62
     features are highly redundant, so ablating SHAP's single top feature already costs 0.49 of
     probability, leaving features 2-5 almost no headroom to demonstrate anything, while the
     random control climbs steadily with k (0.02 at k=1, 0.14 at k=10). The gain is therefore
     compressed toward zero as k grows, for reasons that have nothing to do with explanation
     quality. `fidelity_curve` and `AOPC` exist so this cannot happen again.
   - `sufficiency`: keep ONLY the top-k features, ablating everything else. A faithful explanation
     causes a SMALL drop, because it retained what mattered. Lower is better.

   Both are reported **against a random-feature control**. An absolute comprehensiveness number is
   close to meaningless on its own — ablating any k of 62 features moves the prediction somewhat.
   What matters is whether the explanation beats picking k features at random. `fidelity_gain` is
   that difference, and it is the number to quote.

2. **Stability** — "do near-identical inputs produce near-identical explanations?" Measured by
   perturbing the input with small Gaussian noise and comparing the resulting top-k sets
   (Jaccard) and full rankings (Spearman). `docs/xai_survey.md` §2.2 predicts LIME is weak here;
   this measures it rather than assuming it.

3. **Comprehensibility** — a lightweight proxy for the clinician user study that `PRD.md §5.3`
   puts out of scope: how many features a reader must hold in mind, and how concentrated the
   evidence is. It is explicitly a PROXY and must never be reported as a usability result.

Exact parameters
----------------
    FIDELITY_CURVE_KS      = (1, 3, 5, 10, 20)  ks at which the curve is measured. k=1 is the
                                    headline discriminator, for the saturation reason above.
    AOPC_KS                = (1, 3, 5)          ks averaged into the AOPC summary statistic.
    TRUST_MIN_TOP1_GAIN    = 0.05   minimum top-1 fidelity gain for an explanation to be certified
                                    trustworthy. Set from measurement, not taste: the random
                                    control at k=1 averages 0.023, so a gain below 0.05 is not
                                    distinguishable from citing a feature at random.
    TRUST_MIN_STABILITY    = 0.50   minimum top-k Jaccard for a METHOD to be operator-facing.
                                    Below this, two operators can see different reasons for the
                                    same alert, which erodes trust faster than no explanation.
    ABLATION_STRATEGY      = "background_mean"  ablated features are replaced, at every timestep,
                                    by that feature's mean over the background (training) windows.
                                    Zeroing would be wrong: features are min-max scaled, so 0 is
                                    the observed MINIMUM, a meaningful extreme value rather than a
                                    neutral one, and zero-ablation would measure sensitivity to an
                                    extreme instead of to removal.
    DEFAULT_K              = 5      features ablated, matching the top-N an operator is shown.
    N_RANDOM_CONTROLS      = 5      random feature subsets averaged for the control baseline.
    STABILITY_NOISE_SIGMA  = 0.01   Gaussian sigma in min-max-scaled units, i.e. 1% of each
                                    feature's observed range. Small enough that the prediction
                                    should not change; large enough to expose an unstable explainer.
    STABILITY_REPEATS      = 5      perturbed copies compared against the original explanation.
    RANDOM_STATE           = 42     seeds controls and perturbations, so every number here is
                                    reproducible run to run.

Positive class: index 1 = **Attack** (`TRD.md §2.3`). Fidelity is always measured on the
probability of the class the model actually predicted for that window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.stats import spearmanr

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

ABLATION_STRATEGY: str = "background_mean"
DEFAULT_K: int = 5
N_RANDOM_CONTROLS: int = 5
STABILITY_NOISE_SIGMA: float = 0.01
STABILITY_REPEATS: int = 5

#: A function mapping `(n, sequence_length, n_features)` windows to `(n, n_classes)` probabilities.
PredictFn = Callable[[np.ndarray], np.ndarray]


class ExplainerProtocol(Protocol):
    """The minimal interface both `EnsembleShapExplainer` and `EnsembleLimeExplainer` satisfy."""

    feature_names: list[str]

    def explain(self, window: np.ndarray, top_n: int = ...) -> object:
        """Explain a single window."""
        ...


@dataclass
class FidelityResult:
    """Fidelity of one explanation method, measured against a random control.

    Attributes:
        method: "shap" or "lime".
        k: number of features ablated.
        comprehensiveness: mean drop in predicted-class probability when the top-k features are
            removed. Higher is better.
        sufficiency: mean drop when only the top-k are KEPT. Lower is better.
        random_comprehensiveness: the same ablation with k features chosen at random.
        fidelity_gain: `comprehensiveness - random_comprehensiveness`. **This is the number to
            quote.** At or below zero means the explanation is no better than picking features at
            random, i.e. it is not describing the model.
        n_windows: how many windows were measured.
    """

    method: str
    k: int
    comprehensiveness: float
    sufficiency: float
    random_comprehensiveness: float
    fidelity_gain: float
    n_windows: int

    def verdict(self) -> str:
        """Return a plain-language reading of `fidelity_gain`."""
        if self.fidelity_gain <= 0.0:
            return "NOT FAITHFUL — no better than ablating random features"
        if self.fidelity_gain < 0.05:
            return "WEAK — barely distinguishable from random"
        if self.fidelity_gain < 0.20:
            return "MODERATE — measurably better than random"
        return "STRONG — the attributed features clearly drive the decision"

    def summary(self) -> str:
        """Render a printable block."""
        return (
            f"Fidelity — {self.method.upper()} (top-{self.k}, {self.n_windows} windows)\n"
            f"  comprehensiveness (ablate top-k, higher better) : {self.comprehensiveness:+.4f}\n"
            f"  random control (ablate k at random)             : "
            f"{self.random_comprehensiveness:+.4f}\n"
            f"  FIDELITY GAIN (explanation - random)            : {self.fidelity_gain:+.4f}\n"
            f"  sufficiency (keep only top-k, lower better)     : {self.sufficiency:+.4f}\n"
            f"  VERDICT: {self.verdict()}"
        )


@dataclass
class StabilityResult:
    """Stability of one explanation method under small input perturbations.

    Attributes:
        method: "shap" or "lime".
        k: top-k set size compared.
        jaccard: mean Jaccard overlap between the original and perturbed top-k sets. 1.0 means the
            same features are always cited.
        spearman: mean Spearman rank correlation across all features' attributions.
        noise_sigma: the perturbation size used.
        prediction_flip_rate: how often the perturbation changed the model's own prediction. A
            non-zero rate means some instability is the MODEL's, not the explainer's, and the
            explainer must not be blamed for it.
        n_windows: how many windows were measured.
    """

    method: str
    k: int
    jaccard: float
    spearman: float
    noise_sigma: float
    prediction_flip_rate: float
    n_windows: int

    def verdict(self) -> str:
        """Return a plain-language reading of the Jaccard overlap."""
        if self.jaccard >= 0.8:
            return "STABLE — near-identical inputs give near-identical explanations"
        if self.jaccard >= 0.5:
            return "MODERATE — explanations shift noticeably under 1% input noise"
        return "UNSTABLE — two operators could see different reasons for the same alert"

    def summary(self) -> str:
        """Render a printable block."""
        return (
            f"Stability — {self.method.upper()} "
            f"(top-{self.k}, sigma={self.noise_sigma}, {self.n_windows} windows)\n"
            f"  top-k Jaccard overlap        : {self.jaccard:.3f}\n"
            f"  Spearman rank correlation    : {self.spearman:.3f}\n"
            f"  model prediction flip rate   : {self.prediction_flip_rate:.1%}\n"
            f"  VERDICT: {self.verdict()}"
        )


@dataclass
class ComprehensibilityResult:
    """Proxy for the out-of-scope clinician user study (`PRD.md §5.3`).

    Attributes:
        method: "shap" or "lime".
        n_features_cited: how many features an explanation asks the reader to hold in mind.
        top_feature_share: mean share of total evidence carried by the single top feature.
        top_k_share: mean share carried by the whole cited list. A list accounting for only a small
            share of the evidence is not really an explanation of the decision.
        mixed_direction_rate: share of explanations whose top-ranked feature argues AGAINST the
            verdict. Flagged in `reports/t3_1_shap_examples.md` as a possible source of confusion.
        n_windows: how many explanations were measured.
    """

    method: str
    n_features_cited: int
    top_feature_share: float
    top_k_share: float
    mixed_direction_rate: float
    n_windows: int

    def summary(self) -> str:
        """Render a printable block."""
        return (
            f"Comprehensibility proxy — {self.method.upper()} ({self.n_windows} windows)\n"
            f"  features cited per explanation        : {self.n_features_cited}\n"
            f"  evidence share of the top feature     : {self.top_feature_share:.1%}\n"
            f"  evidence share of all cited features  : {self.top_k_share:.1%}\n"
            f"  top feature argues against the verdict: {self.mixed_direction_rate:.0%} "
            f"of explanations\n"
            f"  NOTE: a proxy for readability, NOT a usability result. PRD.md §5.3 puts the "
            f"clinician user study out of scope; this stands in for it and must not be reported "
            f"as though a person had read these."
        )


def ablate_features(
    window: np.ndarray,
    feature_indices: np.ndarray | list[int],
    background_mean: np.ndarray,
    keep_only: bool = False,
) -> np.ndarray:
    """Replace features with their background mean at every timestep.

    Args:
        window: `(sequence_length, n_features)`.
        feature_indices: features to ablate, or — with `keep_only` — the features to RETAIN.
        background_mean: per-feature mean over the background windows, shape `(n_features,)`.
        keep_only: invert the selection, ablating everything except `feature_indices`. This is what
            `sufficiency` needs.

    Returns:
        A new window with the selected features neutralised.
    """
    ablated = np.array(window, dtype=np.float32, copy=True)
    n_features = ablated.shape[1]

    selected = np.zeros(n_features, dtype=bool)
    selected[np.asarray(feature_indices, dtype=int)] = True
    if keep_only:
        selected = ~selected

    ablated[:, selected] = background_mean[selected].astype(np.float32)
    return ablated


def measure_fidelity(
    explainer: ExplainerProtocol,
    predict_fn: PredictFn,
    windows: np.ndarray,
    background: np.ndarray,
    method: str,
    k: int = DEFAULT_K,
    n_random_controls: int = N_RANDOM_CONTROLS,
    random_state: int = RANDOM_STATE,
) -> FidelityResult:
    """Measure whether an explanation's top features actually drive the model's decision.

    For each window: explain it, ablate the top-k attributed features, and record how far the
    predicted-class probability falls. Compare against ablating k features chosen at random — the
    control is essential, because ablating any k of 62 features moves the prediction somewhat.

    Args:
        explainer: a SHAP or LIME explainer exposing `explain()` and `feature_names`.
        predict_fn: maps `(n, sequence_length, n_features)` to `(n, n_classes)` probabilities.
        windows: windows to measure, `(n, sequence_length, n_features)`.
        background: training-fold windows, used for the ablation values.
        method: "shap" or "lime", recorded in the result.
        k: how many features to ablate.
        n_random_controls: random subsets averaged per window for the control.
        random_state: seed for the random controls.

    Returns:
        A `FidelityResult`.
    """
    background_mean = np.asarray(background, dtype=np.float32).mean(axis=(0, 1))
    n_features = len(background_mean)
    rng = np.random.default_rng(random_state)

    comprehensiveness_scores: list[float] = []
    sufficiency_scores: list[float] = []
    random_scores: list[float] = []

    for window in windows:
        original = predict_fn(window[np.newaxis, ...])[0]
        predicted_class = int(np.argmax(original))
        original_probability = float(original[predicted_class])

        explanation = explainer.explain(window, top_n=k)
        top_indices = [
            explanation.feature_names.index(f.feature_name) for f in explanation.top_features
        ]

        # Comprehensiveness: remove the cited features; a faithful explanation drops the score a lot.
        removed = ablate_features(window, top_indices, background_mean)
        comprehensiveness_scores.append(
            original_probability - float(predict_fn(removed[np.newaxis, ...])[0][predicted_class])
        )

        # Sufficiency: keep ONLY the cited features; a faithful explanation drops the score little.
        kept = ablate_features(window, top_indices, background_mean, keep_only=True)
        sufficiency_scores.append(
            original_probability - float(predict_fn(kept[np.newaxis, ...])[0][predicted_class])
        )

        # Control: the same ablation on randomly chosen features.
        control = []
        for _ in range(n_random_controls):
            random_indices = rng.choice(n_features, size=k, replace=False)
            random_window = ablate_features(window, random_indices, background_mean)
            control.append(
                original_probability
                - float(predict_fn(random_window[np.newaxis, ...])[0][predicted_class])
            )
        random_scores.append(float(np.mean(control)))

    comprehensiveness = float(np.mean(comprehensiveness_scores))
    random_comprehensiveness = float(np.mean(random_scores))

    result = FidelityResult(
        method=method,
        k=k,
        comprehensiveness=comprehensiveness,
        sufficiency=float(np.mean(sufficiency_scores)),
        random_comprehensiveness=random_comprehensiveness,
        fidelity_gain=comprehensiveness - random_comprehensiveness,
        n_windows=len(windows),
    )
    logger.info("%s", result.summary())
    return result


def measure_stability(
    explainer: ExplainerProtocol,
    predict_fn: PredictFn,
    windows: np.ndarray,
    method: str,
    k: int = DEFAULT_K,
    noise_sigma: float = STABILITY_NOISE_SIGMA,
    repeats: int = STABILITY_REPEATS,
    random_state: int = RANDOM_STATE,
) -> StabilityResult:
    """Measure whether near-identical inputs produce near-identical explanations.

    `docs/xai_survey.md` §2.2 predicts LIME is the weak method here. This measures it.

    The model's own prediction flip rate is tracked alongside: if a 1% perturbation changes what the
    model decides, an explanation that changes with it is behaving CORRECTLY, and the explainer must
    not be blamed for the model's own sensitivity.

    Args:
        explainer: a SHAP or LIME explainer.
        predict_fn: maps windows to class probabilities.
        windows: windows to measure.
        method: "shap" or "lime".
        k: top-k set size to compare.
        noise_sigma: Gaussian sigma in min-max-scaled units.
        repeats: perturbed copies per window.
        random_state: seed for the perturbations.

    Returns:
        A `StabilityResult`.
    """
    rng = np.random.default_rng(random_state)
    jaccards: list[float] = []
    spearmans: list[float] = []
    flips: list[bool] = []

    for window in windows:
        base_explanation = explainer.explain(window, top_n=k)
        base_top = {f.feature_name for f in base_explanation.top_features}
        base_full = explainer.explain(window, top_n=len(base_explanation.feature_names))
        base_vector = np.array([f.attribution for f in sorted(
            base_full.top_features, key=lambda f: f.feature_name
        )])
        base_prediction = base_explanation.predicted_label

        for _ in range(repeats):
            noisy = (window + rng.normal(0.0, noise_sigma, window.shape)).astype(np.float32)

            flips.append(
                int(np.argmax(predict_fn(noisy[np.newaxis, ...])[0])) != base_prediction
            )

            noisy_explanation = explainer.explain(noisy, top_n=k)
            noisy_top = {f.feature_name for f in noisy_explanation.top_features}
            union = base_top | noisy_top
            jaccards.append(len(base_top & noisy_top) / len(union) if union else 1.0)

            noisy_full = explainer.explain(noisy, top_n=len(noisy_explanation.feature_names))
            noisy_vector = np.array([f.attribution for f in sorted(
                noisy_full.top_features, key=lambda f: f.feature_name
            )])
            correlation = spearmanr(base_vector, noisy_vector).statistic
            spearmans.append(0.0 if np.isnan(correlation) else float(correlation))

    result = StabilityResult(
        method=method,
        k=k,
        jaccard=float(np.mean(jaccards)),
        spearman=float(np.mean(spearmans)),
        noise_sigma=noise_sigma,
        prediction_flip_rate=float(np.mean(flips)),
        n_windows=len(windows),
    )
    logger.info("%s", result.summary())
    return result


def measure_comprehensibility(
    explainer: ExplainerProtocol,
    windows: np.ndarray,
    method: str,
    k: int = DEFAULT_K,
) -> ComprehensibilityResult:
    """Compute the readability proxy standing in for the out-of-scope user study.

    Args:
        explainer: a SHAP or LIME explainer.
        windows: windows to explain.
        method: "shap" or "lime".
        k: features cited per explanation.

    Returns:
        A `ComprehensibilityResult`.
    """
    top_shares: list[float] = []
    list_shares: list[float] = []
    mixed: list[bool] = []

    for window in windows:
        explanation = explainer.explain(window, top_n=k)
        features = explanation.top_features
        top_shares.append(features[0].share_of_total)
        list_shares.append(sum(f.share_of_total for f in features))

        # Does the single most influential feature argue AGAINST the verdict shown?
        verdict_is_attack = explanation.predicted_label == 1
        top_supports = (features[0].attribution > 0) == verdict_is_attack
        mixed.append(not top_supports)

    result = ComprehensibilityResult(
        method=method,
        n_features_cited=k,
        top_feature_share=float(np.mean(top_shares)),
        top_k_share=float(np.mean(list_shares)),
        mixed_direction_rate=float(np.mean(mixed)),
        n_windows=len(windows),
    )
    logger.info("%s", result.summary())
    return result


# ---------------------------------------------------------------------------
# Fidelity curve, AOPC, and per-explanation certification
# ---------------------------------------------------------------------------
# Everything below exists because measuring fidelity at a single k gave this project the WRONG
# answer about its own explainability layer. See the module docstring's comprehensiveness note.

FIDELITY_CURVE_KS: tuple[int, ...] = (1, 3, 5, 10, 20)
AOPC_KS: tuple[int, ...] = (1, 3, 5)
TRUST_MIN_TOP1_GAIN: float = 0.05
TRUST_MIN_STABILITY: float = 0.50


@dataclass
class FidelityCurve:
    """Comprehensiveness measured across several k, with a random control at each k.

    Attributes:
        method: "shap" or "lime".
        ks: the k values measured.
        top_drops: mean probability drop from ablating the top-k attributed features, per k.
        random_drops: mean drop from ablating k features at random, per k.
        gains: `top_drops - random_drops`, per k.
        gain_standard_errors: standard error of each gain, so significance is visible rather than
            asserted. A gain smaller than about twice its standard error means nothing.
        n_windows: windows measured.
    """

    method: str
    ks: tuple[int, ...]
    top_drops: dict[int, float]
    random_drops: dict[int, float]
    gains: dict[int, float]
    gain_standard_errors: dict[int, float]
    n_windows: int

    def aopc(self) -> float:
        """Return the Area Over the Perturbation Curve: the mean gain across `AOPC_KS`.

        A single summary that cannot be gamed by choosing a favourable k, which is exactly the
        failure mode that produced this project's first, wrong fidelity conclusion.

        Returns:
            Mean fidelity gain over the `AOPC_KS` values present in this curve.
        """
        available = [k for k in AOPC_KS if k in self.gains]
        return float(np.mean([self.gains[k] for k in available])) if available else 0.0

    def is_significant(self, k: int = 1) -> bool:
        """Return whether the gain at `k` exceeds twice its standard error.

        Args:
            k: the k to test. Defaults to 1, the headline discriminator.

        Returns:
            True when the gain is statistically distinguishable from zero.
        """
        if k not in self.gains:
            return False
        return self.gains[k] > 2 * self.gain_standard_errors[k]

    def summary(self) -> str:
        """Render the curve as a printable table."""
        lines = [
            f"Fidelity curve — {self.method.upper()} ({self.n_windows} windows)",
            f"  {'k':>4} {'top-k drop':>12} {'random-k':>10} {'gain':>9} {'± sem':>8} {'sig?':>6}",
        ]
        for k in self.ks:
            lines.append(
                f"  {k:>4} {self.top_drops[k]:>12.4f} {self.random_drops[k]:>10.4f} "
                f"{self.gains[k]:>+9.4f} {self.gain_standard_errors[k]:>8.4f} "
                f"{'yes' if self.is_significant(k) else 'no':>6}"
            )
        lines.append(f"  AOPC (mean gain over k={AOPC_KS}) : {self.aopc():+.4f}")
        return "\n".join(lines)


def measure_fidelity_curve(
    explainer: ExplainerProtocol,
    predict_fn: PredictFn,
    windows: np.ndarray,
    background: np.ndarray,
    method: str,
    ks: tuple[int, ...] = FIDELITY_CURVE_KS,
    n_random_controls: int = N_RANDOM_CONTROLS,
    random_state: int = RANDOM_STATE,
) -> FidelityCurve:
    """Measure comprehensiveness across several k, each against its own random control.

    Each window is explained ONCE, with the full ranking retained, so the curve costs one
    explanation per window regardless of how many k are measured.

    Args:
        explainer: a SHAP or LIME explainer.
        predict_fn: maps windows to class probabilities.
        windows: windows to measure.
        background: training-fold windows supplying the ablation values.
        method: "shap" or "lime".
        ks: k values to measure.
        n_random_controls: random subsets averaged per window per k.
        random_state: seed for the controls.

    Returns:
        A `FidelityCurve`.
    """
    background_mean = np.asarray(background, dtype=np.float32).mean(axis=(0, 1))
    n_features = len(background_mean)
    rng = np.random.default_rng(random_state)
    ks = tuple(k for k in ks if k <= n_features)

    per_k_top: dict[int, list[float]] = {k: [] for k in ks}
    per_k_random: dict[int, list[float]] = {k: [] for k in ks}

    for window in windows:
        original = predict_fn(window[np.newaxis, ...])[0]
        predicted_class = int(np.argmax(original))
        original_probability = float(original[predicted_class])

        # One explanation per window; the full ranking serves every k.
        explanation = explainer.explain(window, top_n=n_features)
        ranking = [
            explanation.feature_names.index(f.feature_name) for f in explanation.top_features
        ]

        for k in ks:
            ablated = ablate_features(window, ranking[:k], background_mean)
            per_k_top[k].append(
                original_probability
                - float(predict_fn(ablated[np.newaxis, ...])[0][predicted_class])
            )

            control = []
            for _ in range(n_random_controls):
                random_indices = rng.choice(n_features, size=k, replace=False)
                random_window = ablate_features(window, random_indices, background_mean)
                control.append(
                    original_probability
                    - float(predict_fn(random_window[np.newaxis, ...])[0][predicted_class])
                )
            per_k_random[k].append(float(np.mean(control)))

    gains = {
        k: float(np.mean(np.array(per_k_top[k]) - np.array(per_k_random[k]))) for k in ks
    }
    standard_errors = {}
    for k in ks:
        differences = np.array(per_k_top[k]) - np.array(per_k_random[k])
        standard_errors[k] = (
            float(differences.std(ddof=1) / np.sqrt(len(differences)))
            if len(differences) > 1
            else 0.0
        )

    curve = FidelityCurve(
        method=method,
        ks=ks,
        top_drops={k: float(np.mean(per_k_top[k])) for k in ks},
        random_drops={k: float(np.mean(per_k_random[k])) for k in ks},
        gains=gains,
        gain_standard_errors=standard_errors,
        n_windows=len(windows),
    )
    logger.info("%s", curve.summary())
    return curve


CERTIFY_KS: tuple[int, ...] = (1, 2, 3, 5)


@dataclass
class Certification:
    """Per-explanation trust verdict, measured at the moment the explanation is produced.

    This is how this project makes its explainability layer trustworthy in the only way that
    actually means anything: not by asserting that a *method* is reliable in general, but by
    checking THIS explanation against THIS model before showing it to anyone. An explanation that
    fails is still returned — it is labelled, not hidden — so an operator is never silently handed
    a rationale the model does not support.

    WHY CERTIFICATION IS ADAPTIVE OVER k
    ------------------------------------
    The first version tested the top-1 feature only, and certified just 52% of SHAP explanations.
    That was too strict, and for a measurable reason rather than a matter of taste: with 62 highly
    redundant features, plenty of genuine decisions are driven by a small *set* of measurements
    rather than a single dominant one. The T3.3 fidelity curve shows the top-3 gain is +0.19 and
    statistically significant, so those explanations are faithful — top-1 alone simply could not
    see it.

    Certification therefore tries increasing k and certifies at the SMALLEST k that clears the
    threshold, reporting it as `certified_k`. The explanation shown is then honest about its own
    scope: "these `certified_k` measurements together drive this decision." When no k clears the
    threshold, the explanation is marked UNVERIFIED.

    Attributes:
        trusted: whether some `k` in `CERTIFY_KS` cleared `min_gain`.
        certified_k: the smallest k that cleared it, or None when the explanation failed.
        top1_gain: gain from ablating the single top-cited feature, minus the random control.
        certified_gain: gain at `certified_k`.
        certified_drop: raw probability drop at `certified_k`, before subtracting the control.
        random_drop: the control at `certified_k`.
        gains_by_k: measured gain at every k tried, so a failure is diagnosable.
        method: which method produced the explanation.
        reasons: why it failed; empty when trusted.
        elapsed_seconds: cost of certifying, so the overhead is visible rather than assumed.
    """

    trusted: bool
    certified_k: int | None
    top1_gain: float
    certified_gain: float
    certified_drop: float
    random_drop: float
    gains_by_k: dict[int, float]
    method: str
    reasons: list[str]
    elapsed_seconds: float

    def banner(self) -> str:
        """Return a one-line label to display above the explanation itself."""
        if self.trusted:
            noun = "measurement" if self.certified_k == 1 else "measurements"
            return (
                f"[VERIFIED] Removing the top {self.certified_k} cited {noun} changes this "
                f"decision by {self.certified_drop:.0%}, against {self.random_drop:.0%} for "
                f"unrelated ones."
            )
        return "[UNVERIFIED] " + "; ".join(self.reasons)


def certify_explanation(
    explanation: object,
    predict_fn: PredictFn,
    window: np.ndarray,
    background: np.ndarray,
    n_random_controls: int = N_RANDOM_CONTROLS,
    min_gain: float = TRUST_MIN_TOP1_GAIN,
    ks: tuple[int, ...] = CERTIFY_KS,
    random_state: int = RANDOM_STATE,
) -> Certification:
    """Check one explanation against the model before it is shown to anyone.

    Ablates the top-k cited features for increasing k and certifies at the smallest k whose
    probability drop exceeds a random-feature control by `min_gain`.

    Every ablation is evaluated in a SINGLE batched `predict_fn` call. The first implementation
    issued one call per ablation and cost 0.865s per alert, which would have been a real obstacle
    to running certification on every detection; batching removes that without changing any number
    it produces.

    Args:
        explanation: an `Explanation` from either explainer.
        predict_fn: maps windows to class probabilities.
        window: the window that was explained.
        background: training-fold windows supplying the ablation values.
        n_random_controls: random feature subsets averaged for the control at each k.
        min_gain: threshold a k must clear to certify.
        ks: k values to try, smallest first.
        random_state: seed for the controls.

    Returns:
        A `Certification`.
    """
    import time as _time

    started = _time.perf_counter()

    window = np.asarray(window, dtype=np.float32)
    if window.ndim == 3:
        window = window[0]
    background_mean = np.asarray(background, dtype=np.float32).mean(axis=(0, 1))
    n_features = len(background_mean)
    rng = np.random.default_rng(random_state)

    if not explanation.top_features:
        return Certification(
            trusted=False, certified_k=None, top1_gain=0.0, certified_gain=0.0,
            certified_drop=0.0, random_drop=0.0, gains_by_k={}, method=explanation.method,
            reasons=["the explanation cites no features"],
            elapsed_seconds=_time.perf_counter() - started,
        )

    ranking = [
        explanation.feature_names.index(f.feature_name) for f in explanation.top_features
    ]
    usable_ks = tuple(k for k in ks if k <= len(ranking))

    # Assemble every window to evaluate, then make ONE batched prediction.
    batch = [window]
    layout: list[tuple[str, int]] = [("original", 0)]
    for k in usable_ks:
        batch.append(ablate_features(window, ranking[:k], background_mean))
        layout.append(("top", k))
        for _ in range(n_random_controls):
            batch.append(
                ablate_features(
                    window, rng.choice(n_features, size=k, replace=False), background_mean
                )
            )
            layout.append(("random", k))

    probabilities = predict_fn(np.stack(batch))
    predicted_class = int(np.argmax(probabilities[0]))
    original_probability = float(probabilities[0][predicted_class])

    top_drops: dict[int, float] = {}
    random_drops: dict[int, float] = {k: 0.0 for k in usable_ks}
    random_counts: dict[int, int] = {k: 0 for k in usable_ks}
    for position, (kind, k) in enumerate(layout):
        if kind == "original":
            continue
        drop = original_probability - float(probabilities[position][predicted_class])
        if kind == "top":
            top_drops[k] = drop
        else:
            random_drops[k] += drop
            random_counts[k] += 1

    gains_by_k = {
        k: top_drops[k] - (random_drops[k] / max(random_counts[k], 1)) for k in usable_ks
    }

    certified_k = next((k for k in usable_ks if gains_by_k[k] >= min_gain), None)
    reasons: list[str] = []
    if certified_k is None:
        best_k = max(gains_by_k, key=gains_by_k.get) if gains_by_k else None
        reasons.append(
            "removing the cited measurements barely changes this decision "
            f"(best gain {gains_by_k[best_k]:+.1%} at k={best_k}, threshold {min_gain:.0%}) — "
            "the model's decision here is spread across many measurements, so no short list "
            "explains it"
        )

    resolved_k = certified_k if certified_k is not None else (usable_ks[0] if usable_ks else 1)
    return Certification(
        trusted=certified_k is not None,
        certified_k=certified_k,
        top1_gain=gains_by_k.get(1, 0.0),
        certified_gain=gains_by_k.get(resolved_k, 0.0),
        certified_drop=top_drops.get(resolved_k, 0.0),
        random_drop=random_drops.get(resolved_k, 0.0) / max(random_counts.get(resolved_k, 1), 1),
        gains_by_k=gains_by_k,
        method=explanation.method,
        reasons=reasons,
        elapsed_seconds=_time.perf_counter() - started,
    )
