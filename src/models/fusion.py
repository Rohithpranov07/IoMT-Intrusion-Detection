"""Confidence-weighted voting fusion (Build-Instructions T2.5; TRD.md §3.3).

`TRD.md §3.3` rules out both a simple average and a majority vote: each branch's vote must be
weighted by *its own* predicted-class confidence. This module implements that, and records the
exact formula rather than leaving it implicit.

THE EXACT FUSION FORMULA (frozen in `docs/architecture_decision.md` §3.1)
-------------------------------------------------------------------------
For each branch `b` with softmax output `p_b` over 2 classes:

    confidence      c_b = max_k p_b[k]                  # the branch's own certainty
    unnormalised    u_b = alpha_b * (c_b ** gamma)
    weight          w_b = u_b / (sum_b' u_b' + EPSILON) # weights sum to 1
    fused           P   = sum_b (w_b * p_b)             # elementwise over the 2 classes
    prediction          = argmax_k P[k]                 # 1 = Attack = POSITIVE (TRD.md §2.3)
    ensemble conf.      = max_k P[k]

Exact parameter values
----------------------
    CONFIDENCE_SHARPNESS (gamma) = 1.0    Weight proportional to raw confidence. gamma > 1 sharpens
                                          toward winner-take-all; 1.0 is the neutral starting point
                                          and the value T4.4's ablation must beat to justify a change.
    DEFAULT_BRANCH_PRIOR (alpha) = 1.0    Per branch. No branch is privileged a priori. Kept as an
                                          explicit vector rather than hardcoded 1s because it is the
                                          ONLY adjustable parameter of this fusion layer, and T3.5's
                                          incremental learning ("fine-tune only the fusion layer")
                                          updates exactly these.
    EPSILON                      = 1e-9   Guards the degenerate all-zero-confidence case.

With gamma = 1 and equal priors this reduces to a confidence-weighted mean: a branch that is 95%
sure outvotes two branches that are 55% sure. A simple average would follow the majority — that
difference is what `tests/test_fusion.py` asserts (T2.5's VERIFY block).

Class ordering: index 0 = Normal, index 1 = **Attack = POSITIVE class** (`TRD.md §2.3`). Every
branch in `src/models/` emits this ordering, so fusion never reorders.

GNN BRANCH: absent by design. `TRD.md §3.4` and `PRD.md §9` make the GNN a gated stretch goal, and
`Build-Instructions.md` §A.1 rule 4 forbids wiring it in before T2.7's sparsity go/no-go check
returns "go". This module takes an arbitrary list of branch outputs, so a fourth branch needs no
code change here — but no GNN is referenced anywhere in it until that gate passes.

Determinism: fusion is a pure function of its inputs; no randomness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.config import POSITIVE_LABEL

logger = logging.getLogger(__name__)

# --- Frozen fusion constants (docs/architecture_decision.md §3.1) --------------------------
CONFIDENCE_SHARPNESS: float = 1.0
DEFAULT_BRANCH_PRIOR: float = 1.0
EPSILON: float = 1e-9


@dataclass
class FusionResult:
    """Output of one fusion pass.

    Attributes:
        probabilities: fused class probabilities, shape `(n_samples, n_classes)`.
        predictions: `argmax` over classes, shape `(n_samples,)`. 1 = Attack = positive.
        confidence: fused max probability per sample, shape `(n_samples,)`. Consumed by the
            adaptive threshold (T3.4).
        branch_weights: the weight each branch received per sample, shape
            `(n_samples, n_branches)`. Retained because it is the ensemble's own account of which
            branch drove each decision — useful directly in the XAI output (T3.1).
        branch_names: branch names, in the column order of `branch_weights`.
    """

    probabilities: np.ndarray
    predictions: np.ndarray
    confidence: np.ndarray
    branch_weights: np.ndarray
    branch_names: list[str]

    def attack_rate(self) -> float:
        """Return the fraction of samples predicted as Attack (the positive class)."""
        return float((self.predictions == POSITIVE_LABEL).mean())

    def dominant_branch(self) -> np.ndarray:
        """Return the index of the highest-weighted branch for each sample."""
        return np.argmax(self.branch_weights, axis=1)


def confidence_weighted_fusion(
    branch_probabilities: list[np.ndarray],
    branch_names: list[str] | None = None,
    branch_priors: np.ndarray | list[float] | None = None,
    gamma: float = CONFIDENCE_SHARPNESS,
) -> FusionResult:
    """Fuse per-branch softmax outputs by each branch's own confidence.

    Implements the formula in this module's docstring verbatim.

    Args:
        branch_probabilities: one `(n_samples, n_classes)` softmax array per branch. All must
            share a shape.
        branch_names: names in the same order. Defaults to `branch_0`, `branch_1`, ...
        branch_priors: per-branch `alpha_b`. Defaults to `DEFAULT_BRANCH_PRIOR` for every branch.
            This is the vector T3.5's incremental learning updates.
        gamma: `CONFIDENCE_SHARPNESS`. 1.0 weights proportionally to raw confidence.

    Returns:
        A `FusionResult`.

    Raises:
        ValueError: if no branches are given, if their shapes disagree, if `branch_priors` has the
            wrong length or any negative entry, or if `gamma` is negative.
    """
    if not branch_probabilities:
        raise ValueError("At least one branch output is required")

    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in branch_probabilities])
    if stacked.ndim != 3:
        raise ValueError(
            "Each branch output must be 2-D (n_samples, n_classes); "
            f"stacked shape was {stacked.shape}"
        )

    n_branches, n_samples, _ = stacked.shape

    if branch_names is None:
        branch_names = [f"branch_{i}" for i in range(n_branches)]
    if len(branch_names) != n_branches:
        raise ValueError(f"Got {len(branch_names)} names for {n_branches} branches")

    if branch_priors is None:
        priors = np.full(n_branches, DEFAULT_BRANCH_PRIOR, dtype=np.float64)
    else:
        priors = np.asarray(branch_priors, dtype=np.float64)
        if priors.shape != (n_branches,):
            raise ValueError(
                f"branch_priors must have shape ({n_branches},), got {priors.shape}"
            )
        if (priors < 0).any():
            raise ValueError("branch_priors must be non-negative")

    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    # c_b = max_k p_b[k]  ->  shape (n_branches, n_samples)
    confidences = stacked.max(axis=2)

    # u_b = alpha_b * c_b ** gamma
    unnormalised = priors[:, np.newaxis] * np.power(confidences, gamma)

    # w_b = u_b / sum_b' u_b'
    weights = unnormalised / (unnormalised.sum(axis=0, keepdims=True) + EPSILON)

    # P = sum_b w_b * p_b
    fused = (weights[:, :, np.newaxis] * stacked).sum(axis=0)

    result = FusionResult(
        probabilities=fused,
        predictions=np.argmax(fused, axis=1).astype(np.int8),
        confidence=fused.max(axis=1),
        branch_weights=weights.T,  # -> (n_samples, n_branches)
        branch_names=list(branch_names),
    )
    logger.info(
        "Fused %d branches over %d samples (gamma=%.2f); predicted Attack rate %.2f%%",
        n_branches, n_samples, gamma, 100 * result.attack_rate(),
    )
    return result


def simple_average_fusion(branch_probabilities: list[np.ndarray]) -> np.ndarray:
    """Unweighted mean of branch probabilities — the baseline `TRD.md §3.3` rules out.

    Provided ONLY so T4.4's ablation can quantify what confidence weighting actually buys, and so
    `tests/test_fusion.py` can assert the two disagree in the intended direction. It must never be
    used as the deployed fusion rule.

    Args:
        branch_probabilities: one `(n_samples, n_classes)` softmax array per branch.

    Returns:
        Mean probabilities, shape `(n_samples, n_classes)`.
    """
    return np.mean(np.stack([np.asarray(p, dtype=np.float64) for p in branch_probabilities]), axis=0)


def majority_vote_fusion(branch_probabilities: list[np.ndarray]) -> np.ndarray:
    """Per-branch `argmax` then majority vote — the other baseline `TRD.md §3.3` rules out.

    Ties are broken toward the POSITIVE class (Attack), because in an intrusion-detection setting a
    missed attack costs more than a false alarm. Ablation-only, like `simple_average_fusion`.

    Args:
        branch_probabilities: one `(n_samples, n_classes)` softmax array per branch.

    Returns:
        Predicted labels, shape `(n_samples,)`.
    """
    votes = np.stack([np.argmax(np.asarray(p), axis=1) for p in branch_probabilities])
    attack_votes = (votes == POSITIVE_LABEL).sum(axis=0)
    return (attack_votes >= (len(branch_probabilities) / 2)).astype(np.int8)


def describe_formula() -> str:
    """Return the fusion formula and its parameter values as plain text.

    Returns:
        A human-readable description, quotable directly into reports and slides.
    """
    return (
        "Confidence-weighted voting fusion (TRD.md §3.3):\n"
        "  c_b = max_k p_b[k]                     branch b's own confidence\n"
        "  u_b = alpha_b * c_b ** gamma\n"
        "  w_b = u_b / sum_b'(u_b')\n"
        "  P   = sum_b w_b * p_b                  fused class probabilities\n"
        f"with gamma (CONFIDENCE_SHARPNESS) = {CONFIDENCE_SHARPNESS}, "
        f"alpha_b (branch prior) = {DEFAULT_BRANCH_PRIOR} for every branch, "
        f"epsilon = {EPSILON}. "
        "Class index 1 = Attack = the positive class (TRD.md §2.3). "
        "This is neither a simple average nor a majority vote, both of which TRD.md §3.3 excludes."
    )
