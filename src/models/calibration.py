"""Per-branch probability calibration by temperature scaling.

>>> MEASURED, AND **NOT ADOPTED**. DO NOT WIRE THIS INTO THE PIPELINE. <<<
--------------------------------------------------------------------------
`docs/calibration_decision.md` rejects calibration on the evidence this module produced. The
branches turn out to be well calibrated already (ECE 0.012-0.023; the BiLSTM is mildly
UNDER-confident at T = 0.79), and calibrating them changes fusion F1 by 0.0001 and the adaptive
threshold's false-positive reduction from -6.8% to -6.9%.

The module is kept on purpose: it is the instrument that produced that refutation, T3.6 must re-run
it on Edge-IIoTset before assuming the conclusion transfers, and a reader of the Review 3 report
should be able to reproduce the rejection rather than take it on trust. Nothing in
`src/models/fusion.py`, the notebooks, or the deployment path calls it.

The hypothesis it was built to test (below) is stated as it was BEFORE measurement, so the
reasoning stays auditable.

Why this module was written
---------------------------
Two of this project's four contributions are measurably limited by ONE defect, and it is not in
either of them — it is in the branches' probability estimates:

  * **Contribution 2 (fusion).** `reports/phase2_results.md` §6: median branch confidence 0.9999,
    so the confidence-weighted fusion weights came out 0.337 / 0.330 / 0.334 — indistinguishable
    from a uniform 1/3. The formula collapsed into the simple average it was chosen to beat, and a
    gamma sweep to 64 could not recover it. Weighting by a confidence that is always ~1 conveys
    nothing.
  * **Contribution 4 (adaptive threshold).** `reports/t3_4_adaptive_threshold.md`: the best
    false-positive reduction measured was -10.8% against `PRD.md §3`'s >30% target. A threshold
    only reclassifies detections sitting NEAR it, and at P=0.9999 almost none do. Moving the bar
    from 0.50 to 0.65 leaves such a detection exactly where it was.

Both symptoms have the same cause: over-parameterised softmax classifiers are systematically
overconfident, a well-documented property of modern networks. The probabilities rank correctly but
their *magnitudes* are not usable as confidences.

Temperature scaling (Guo et al., 2017) is the standard remedy: divide the logits by a single
learned scalar `T` before the softmax. `T > 1` softens the distribution.

    p_calibrated = softmax(logits / T)

WHY THIS IS SAFE TO ADOPT
-------------------------
Temperature scaling is **accuracy-preserving by construction**. Dividing every logit by the same
positive scalar is a monotonic transform, so `argmax` is unchanged: every branch makes exactly the
same predictions before and after. It can only change the *magnitudes* the fusion and threshold
consume. That is what makes it a low-risk change to a frozen architecture — it cannot degrade a
branch's own accuracy, only alter the confidences downstream components read.

    Method             Changes predictions?   Changes architecture?   Fitted on
    temperature scale  no                     no (post-hoc)           validation fold
    Platt scaling      possible               no                      validation fold
    retrain w/ label   yes                    yes (loss function)     training fold
      smoothing

TWO-CLASS IMPLEMENTATION NOTE
-----------------------------
The trained branches emit probabilities, not logits. For a two-class softmax the logit difference
is recoverable exactly:

    d = log(p_attack / p_normal)          then   p_attack(T) = sigmoid(d / T)

so calibration needs no model surgery, no re-export, and nothing added to the graph that must
later reach the Raspberry Pi (T4.1). It is a scalar applied to saved outputs.

LEAKAGE RULE — NON-NEGOTIABLE
-----------------------------
`T` is fitted on the **validation fold only**, never on test. Fitting a calibration parameter on the
evaluation fold is exactly the class of leakage Contribution 1 exists to expose, and doing it here
would invalidate every number this project reports.

Exact parameters
----------------
    TEMPERATURE_SEARCH_LO   = 0.05   lower bound of the search. Below 1 sharpens rather than
                                     softens; the range is kept two-sided so the fit can report an
                                     UNDER-confident branch rather than silently assuming
                                     overconfidence.
    TEMPERATURE_SEARCH_HI   = 100.0  upper bound. A branch needing T > 100 is not miscalibrated,
                                     it is broken, and should surface as such.
    TEMPERATURE_SEARCH_STEPS = 500   log-spaced grid points. A grid rather than a gradient
                                     optimiser: the objective is one-dimensional and smooth, so a
                                     grid is deterministic, has no initialisation or convergence
                                     behaviour to document, and costs milliseconds.
    PROBABILITY_EPSILON     = 1e-7   clamp before log(), since a saturated branch emits exact 0.0
                                     and 1.0, whose logit difference is infinite.
    ECE_N_BINS              = 15     equal-width bins for expected calibration error, the standard
                                     choice in the calibration literature.

Positive class: index 1 = **Attack** (`TRD.md §2.3`).
Determinism: the grid search is exhaustive and deterministic; no seed is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.config import POSITIVE_LABEL

logger = logging.getLogger(__name__)

TEMPERATURE_SEARCH_LO: float = 0.05
TEMPERATURE_SEARCH_HI: float = 100.0
TEMPERATURE_SEARCH_STEPS: int = 500
PROBABILITY_EPSILON: float = 1e-7
ECE_N_BINS: int = 15


def _logit_difference(probabilities: np.ndarray) -> np.ndarray:
    """Recover the two-class logit difference from softmax probabilities.

    Args:
        probabilities: `(n, 2)` softmax outputs.

    Returns:
        `(n,)` array of `log(p_attack / p_normal)`.
    """
    clipped = np.clip(
        np.asarray(probabilities, dtype=np.float64), PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    return np.log(clipped[:, POSITIVE_LABEL]) - np.log(clipped[:, 1 - POSITIVE_LABEL])


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale two-class probabilities by a temperature.

    Args:
        probabilities: `(n, 2)` softmax outputs.
        temperature: the scalar `T`. `T > 1` softens, `T < 1` sharpens, `T == 1` is identity.

    Returns:
        `(n, 2)` calibrated probabilities, in the same column order.

    Raises:
        ValueError: if `temperature` is not positive. A non-positive temperature would invert or
            destroy the ordering, which is not calibration.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}")

    scaled = _logit_difference(probabilities) / temperature
    attack = 1.0 / (1.0 + np.exp(-scaled))
    return np.stack([1.0 - attack, attack], axis=1)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean NLL of `probabilities` against `labels`.

    The objective temperature scaling minimises. NLL is a *proper scoring rule*: it is minimised
    only by the true probabilities, which is why it is the right target here — accuracy is not,
    since temperature scaling cannot change accuracy at all.

    Args:
        probabilities: `(n, 2)` predicted probabilities.
        labels: `(n,)` integer labels.

    Returns:
        Mean negative log-likelihood.
    """
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), PROBABILITY_EPSILON, 1.0)
    labels = np.asarray(labels).ravel().astype(int)
    return float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = ECE_N_BINS
) -> float:
    """Return the expected calibration error.

    Bins predictions by confidence and measures the gap between mean confidence and observed
    accuracy in each bin, weighted by bin size. A perfectly calibrated model scores 0: among
    predictions made with 70% confidence, exactly 70% are correct.

    Args:
        probabilities: `(n, 2)` predicted probabilities.
        labels: `(n,)` integer labels.
        n_bins: number of equal-width confidence bins.

    Returns:
        ECE in [0, 1]. Lower is better.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).ravel().astype(int)

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


@dataclass
class CalibrationResult:
    """Fitted temperature for one branch, with before/after evidence.

    Attributes:
        branch_name: the branch calibrated.
        temperature: the fitted `T`. Above 1 means the branch was overconfident.
        nll_before / nll_after: validation NLL, the fitted objective.
        ece_before / ece_after: validation expected calibration error.
        mean_confidence_before / mean_confidence_after: how saturated the outputs were.
        accuracy_before / accuracy_after: MUST be identical — temperature scaling is a monotonic
            transform, so any difference indicates a bug, not a trade-off.
        n_validation: validation samples the fit used.
    """

    branch_name: str
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    mean_confidence_before: float
    mean_confidence_after: float
    accuracy_before: float
    accuracy_after: float
    n_validation: int

    def summary(self) -> str:
        """Render a printable before/after block."""
        verdict = (
            "overconfident" if self.temperature > 1.05
            else "under-confident" if self.temperature < 0.95
            else "already well calibrated"
        )
        return (
            f"Calibration — {self.branch_name} (fitted on {self.n_validation:,} validation windows)\n"
            f"  temperature T          : {self.temperature:.3f}  ({verdict})\n"
            f"  NLL                    : {self.nll_before:.4f} -> {self.nll_after:.4f}\n"
            f"  ECE                    : {self.ece_before:.4f} -> {self.ece_after:.4f}\n"
            f"  mean confidence        : {self.mean_confidence_before:.4f} -> "
            f"{self.mean_confidence_after:.4f}\n"
            f"  accuracy (must not move): {self.accuracy_before:.6f} -> {self.accuracy_after:.6f}"
        )


def fit_temperature(
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    branch_name: str = "branch",
) -> CalibrationResult:
    """Fit a temperature on the VALIDATION fold by minimising NLL.

    Args:
        validation_probabilities: `(n, 2)` branch outputs on the validation fold.
        validation_labels: `(n,)` validation labels.
        branch_name: name recorded in the result.

    Returns:
        A `CalibrationResult`.

    Raises:
        ValueError: if the inputs disagree in length, or if calibration changed accuracy — which
            is impossible for a correct implementation and therefore indicates a bug.
    """
    probabilities = np.asarray(validation_probabilities, dtype=np.float64)
    labels = np.asarray(validation_labels).ravel().astype(int)
    if len(probabilities) != len(labels):
        raise ValueError(
            f"got {len(probabilities)} probabilities for {len(labels)} labels"
        )

    grid = np.geomspace(TEMPERATURE_SEARCH_LO, TEMPERATURE_SEARCH_HI, TEMPERATURE_SEARCH_STEPS)
    losses = [
        negative_log_likelihood(apply_temperature(probabilities, t), labels) for t in grid
    ]
    temperature = float(grid[int(np.argmin(losses))])
    calibrated = apply_temperature(probabilities, temperature)

    accuracy_before = float((probabilities.argmax(axis=1) == labels).mean())
    accuracy_after = float((calibrated.argmax(axis=1) == labels).mean())
    if not np.isclose(accuracy_before, accuracy_after, atol=1e-9):
        raise ValueError(
            f"Temperature scaling changed {branch_name}'s accuracy "
            f"({accuracy_before} -> {accuracy_after}). This is impossible for a monotonic "
            "transform and means the implementation is wrong."
        )

    result = CalibrationResult(
        branch_name=branch_name,
        temperature=temperature,
        nll_before=negative_log_likelihood(probabilities, labels),
        nll_after=negative_log_likelihood(calibrated, labels),
        ece_before=expected_calibration_error(probabilities, labels),
        ece_after=expected_calibration_error(calibrated, labels),
        mean_confidence_before=float(probabilities.max(axis=1).mean()),
        mean_confidence_after=float(calibrated.max(axis=1).mean()),
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        n_validation=len(labels),
    )
    logger.info("%s", result.summary())
    return result


def fit_all_branches(
    validation_probabilities: dict[str, np.ndarray],
    validation_labels: np.ndarray,
) -> dict[str, CalibrationResult]:
    """Fit an independent temperature per branch.

    Per branch, not one shared scalar: the branches are different architectures trained separately,
    and there is no reason their overconfidence should coincide. Measurement bears this out.

    Args:
        validation_probabilities: branch name to `(n, 2)` validation outputs.
        validation_labels: `(n,)` validation labels.

    Returns:
        Mapping of branch name to `CalibrationResult`.
    """
    return {
        name: fit_temperature(probabilities, validation_labels, name)
        for name, probabilities in validation_probabilities.items()
    }
