"""Unit tests for confidence-weighted fusion (Build-Instructions T2.5 VERIFY block).

The VERIFY condition is: *"Given synthetic branch outputs with known confidences, the fusion output
favors the higher-confidence branch's prediction in a unit test."* That is
`test_confident_minority_branch_overrides_majority`, which also shows the two rules `TRD.md §3.3`
excludes — simple average and majority vote — giving the opposite answer on the same input.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import NEGATIVE_LABEL, POSITIVE_LABEL
from src.models.fusion import (
    CONFIDENCE_SHARPNESS,
    DEFAULT_BRANCH_PRIOR,
    confidence_weighted_fusion,
    majority_vote_fusion,
    simple_average_fusion,
)

BRANCH_NAMES = ["cnn", "bilstm", "transformer"]


def test_confident_minority_branch_overrides_majority() -> None:
    """T2.5 VERIFY: the confident branch wins, even outvoted two-to-one.

    Constructed so the three rules disagree:
      - cnn         is 97% sure the flow is an ATTACK
      - bilstm      is 75% sure it is NORMAL
      - transformer is 74% sure it is NORMAL

    Majority vote says Normal (2 votes to 1). A simple average says Normal (0.507 vs 0.493).
    Confidence-weighted fusion says Attack (0.463 vs 0.537), because the one branch that is
    nearly certain outweighs two that are merely leaning.
    """
    cnn = np.array([[0.03, 0.97]])          # confident Attack
    bilstm = np.array([[0.75, 0.25]])       # moderately sure Normal
    transformer = np.array([[0.74, 0.26]])  # moderately sure Normal
    branches = [cnn, bilstm, transformer]

    result = confidence_weighted_fusion(branches, BRANCH_NAMES)

    assert result.predictions[0] == POSITIVE_LABEL, (
        "Confidence-weighted fusion must follow the 97%-confident Attack branch"
    )
    # The two rules TRD.md §3.3 excludes disagree — which is the point of excluding them.
    assert np.argmax(simple_average_fusion(branches)[0]) == NEGATIVE_LABEL
    assert majority_vote_fusion(branches)[0] == NEGATIVE_LABEL

    # The confident branch must also carry the largest weight.
    assert result.dominant_branch()[0] == 0
    assert result.branch_weights[0, 0] > result.branch_weights[0, 1]
    assert result.branch_weights[0, 0] > result.branch_weights[0, 2]


def test_weights_sum_to_one() -> None:
    """Weights are a normalised distribution over branches, for every sample."""
    rng = np.random.default_rng(0)
    branches = []
    for _ in range(3):
        raw = rng.random((50, 2))
        branches.append(raw / raw.sum(axis=1, keepdims=True))

    result = confidence_weighted_fusion(branches, BRANCH_NAMES)
    np.testing.assert_allclose(result.branch_weights.sum(axis=1), 1.0, atol=1e-9)


def test_fused_probabilities_are_a_valid_distribution() -> None:
    """A convex combination of softmax outputs must itself sum to 1 and stay in [0, 1]."""
    rng = np.random.default_rng(1)
    branches = []
    for _ in range(3):
        raw = rng.random((30, 2))
        branches.append(raw / raw.sum(axis=1, keepdims=True))

    result = confidence_weighted_fusion(branches, BRANCH_NAMES)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert (result.probabilities >= 0).all() and (result.probabilities <= 1).all()
    np.testing.assert_allclose(result.confidence, result.probabilities.max(axis=1))


def test_equal_confidences_reduce_to_simple_average() -> None:
    """When every branch is equally confident, weighting degenerates to the mean."""
    a = np.array([[0.3, 0.7]])
    b = np.array([[0.7, 0.3]])  # same max confidence, opposite prediction

    result = confidence_weighted_fusion([a, b], ["a", "b"])
    np.testing.assert_allclose(result.probabilities, simple_average_fusion([a, b]), atol=1e-9)


def test_gamma_sharpens_toward_the_confident_branch() -> None:
    """Raising gamma must increase the confident branch's share of the weight."""
    branches = [np.array([[0.05, 0.95]]), np.array([[0.6, 0.4]])]

    neutral = confidence_weighted_fusion(branches, ["a", "b"], gamma=1.0)
    sharp = confidence_weighted_fusion(branches, ["a", "b"], gamma=8.0)
    assert sharp.branch_weights[0, 0] > neutral.branch_weights[0, 0]


def test_branch_prior_can_downweight_a_branch() -> None:
    """alpha_b is the fusion layer's only adjustable parameter (T3.5 updates it)."""
    branches = [np.array([[0.05, 0.95]]), np.array([[0.9, 0.1]])]

    equal = confidence_weighted_fusion(branches, ["a", "b"])
    suppressed = confidence_weighted_fusion(branches, ["a", "b"], branch_priors=[0.01, 1.0])
    assert suppressed.branch_weights[0, 0] < equal.branch_weights[0, 0]
    assert suppressed.predictions[0] == NEGATIVE_LABEL
    assert equal.predictions[0] == POSITIVE_LABEL


def test_frozen_defaults_match_architecture_decision() -> None:
    """Defaults must equal the values frozen in docs/architecture_decision.md §3.1."""
    assert CONFIDENCE_SHARPNESS == 1.0
    assert DEFAULT_BRANCH_PRIOR == 1.0


def test_majority_vote_breaks_ties_toward_attack() -> None:
    """A missed attack costs more than a false alarm, so ties go to the positive class."""
    a = np.array([[0.4, 0.6]])  # Attack
    b = np.array([[0.6, 0.4]])  # Normal
    assert majority_vote_fusion([a, b])[0] == POSITIVE_LABEL


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"branch_priors": [1.0]}, "shape"),
        ({"branch_priors": [-1.0, 1.0, 1.0]}, "non-negative"),
        ({"gamma": -1.0}, "non-negative"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict, message: str) -> None:
    """Bad fusion parameters must raise rather than silently produce a wrong weighting."""
    branches = [np.array([[0.5, 0.5]])] * 3
    with pytest.raises(ValueError, match=message):
        confidence_weighted_fusion(branches, BRANCH_NAMES, **kwargs)


def test_empty_branch_list_is_rejected() -> None:
    """Fusing nothing is a programming error, not an empty result."""
    with pytest.raises(ValueError, match="At least one branch"):
        confidence_weighted_fusion([])


def test_no_gnn_branch_is_wired_in() -> None:
    """Build-Instructions §A.1 rule 4: no GNN until T2.7's go/no-go gate returns 'go'.

    T2.7 has not run, so `fusion.py` must contain no GNN wiring. This test is the automated guard
    against the branch quietly reappearing.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "models" / "fusion.py"
    text = source.read_text(encoding="utf-8")
    code_lines = [
        line for line in text.splitlines()
        if not line.lstrip().startswith("#") and "GNN BRANCH: absent by design" not in line
    ]
    code = "\n".join(code_lines)
    # The docstring explains WHY there is no GNN; no executable reference may exist.
    assert "gnn_branch" not in code.lower().replace("gnn_branch.py", "")
    assert not (source.parent / "gnn_branch.py").exists(), (
        "gnn_branch.py exists but reports/gnn_go_nogo.md has not recorded a 'go' decision"
    )
