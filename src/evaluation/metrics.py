"""Classification metrics with an enforced, explicit positive-class convention.

NOTE ON REPO LAYOUT: `Build-Instructions.md` §B.3 does not list this file. It was added rather
than duplicating metric code across notebooks 02/03/04, because Objection #3 is a
*consistency* failure and the surest way to avoid repeating it is to have exactly one place in the
repo where precision and recall are computed. See §D of the run notes.

THIS PROJECT'S POSITIVE CLASS IS **ATTACK** (label 1). TRD.md §2.3.
------------------------------------------------------------------
The base paper defines the positive class as *Normal*, then labels both its Equations 7 and 8
"Recall" -- one of them is the precision formula -- which is why its reported precision and recall
are swapped (PRD.md §2.1.3, §10). This module exists so that mistake cannot be repeated here:

  * `POSITIVE_CLASS_NAME` and `POSITIVE_LABEL` come from `src/config.py`; nothing here hardcodes
    a different convention.
  * Every returned/printed table carries the positive class in its own text. There is no code path
    that emits a metric without stating what "positive" means.
  * `hand_verify_metrics()` recomputes each metric straight from the four confusion-matrix cells
    using the literal formulas below, and asserts agreement with scikit-learn. This is the exact
    check that would have caught the base paper's bug before publication, and TRD.md §2.3 requires
    it be run against THIS project's own results too, not just theirs.

Formulas (TRD.md §2.3), with Attack as positive:
    TP = actual Attack  predicted Attack        FN = actual Attack  predicted Normal
    FP = actual Normal  predicted Attack        TN = actual Normal  predicted Normal

    Accuracy  = (TP + TN) / (TP + FP + TN + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)

Fixed parameters:
    HAND_VERIFY_TOLERANCE = 1e-9   max allowed |hand-computed - sklearn| before an AssertionError.
Determinism: no randomised operation in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    NEGATIVE_CLASS_NAME,
    NEGATIVE_LABEL,
    POSITIVE_CLASS_NAME,
    POSITIVE_LABEL,
)

logger = logging.getLogger(__name__)

HAND_VERIFY_TOLERANCE: float = 1e-9

#: Printed above every metrics table produced by this repo. Required by Build-Instructions §A.1.
POSITIVE_CLASS_STATEMENT: str = (
    f"POSITIVE CLASS = {POSITIVE_CLASS_NAME} (label {POSITIVE_LABEL}); "
    f"negative class = {NEGATIVE_CLASS_NAME} (label {NEGATIVE_LABEL}). "
    "This is TRD.md §2.3's convention and is the OPPOSITE of the base paper's, "
    "which uses Normal as positive (PRD.md §2.1.3)."
)


@dataclass
class MetricResult:
    """Metrics for one evaluated model, with the positive class carried alongside them.

    Attributes:
        model_name: label for the evaluated pipeline/model.
        tp, fp, tn, fn: confusion-matrix cells under the Attack-positive convention.
        accuracy, precision, recall, f1: metrics computed by scikit-learn.
        positive_class: always `POSITIVE_CLASS_NAME` ("Attack"); recorded so no downstream
            consumer can display a metric without knowing its convention.
        hand_verified: True once `hand_verify_metrics` has confirmed the numbers from the raw
            confusion-matrix cells.
    """

    model_name: str
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    positive_class: str = POSITIVE_CLASS_NAME
    hand_verified: bool = False

    def as_row(self) -> dict[str, object]:
        """Return a flat dict suitable for a results DataFrame."""
        return {
            "model": self.model_name,
            "positive_class": self.positive_class,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "hand_verified": self.hand_verified,
        }

    def report(self) -> str:
        """Render a printable block that always leads with the positive-class statement."""
        return (
            f"{self.model_name}\n"
            f"  {POSITIVE_CLASS_STATEMENT}\n"
            f"  Confusion matrix (Attack = positive):\n"
            f"      TP (Attack  -> Attack) = {self.tp:,}\n"
            f"      FN (Attack  -> Normal) = {self.fn:,}   <- missed attacks\n"
            f"      FP (Normal  -> Attack) = {self.fp:,}   <- false alarms\n"
            f"      TN (Normal  -> Normal) = {self.tn:,}\n"
            f"  Accuracy  = (TP+TN)/(TP+FP+TN+FN) = {self.accuracy:.6f}\n"
            f"  Precision = TP/(TP+FP)            = {self.precision:.6f}\n"
            f"  Recall    = TP/(TP+FN)            = {self.recall:.6f}\n"
            f"  F1        = 2PR/(P+R)             = {self.f1:.6f}\n"
            f"  hand-verified against the confusion matrix: {self.hand_verified}"
        )


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str,
) -> MetricResult:
    """Compute accuracy/precision/recall/F1 with **Attack (label 1) as the positive class**.

    Args:
        y_true: ground-truth binary labels (1 = Attack).
        y_pred: predicted binary labels (1 = Attack).
        model_name: label for the evaluated model.

    Returns:
        A `MetricResult` (not yet hand-verified -- call `hand_verify_metrics` for that).
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    # labels=[0, 1] pins the cell order so `ravel()` is tn, fp, fn, tp under Attack-positive.
    tn, fp, fn, tp = confusion_matrix(
        y_true_arr, y_pred_arr, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]
    ).ravel()

    # pos_label is passed explicitly on every call -- never left to scikit-learn's default.
    return MetricResult(
        model_name=model_name,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        accuracy=float(accuracy_score(y_true_arr, y_pred_arr)),
        precision=float(precision_score(y_true_arr, y_pred_arr, pos_label=POSITIVE_LABEL, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred_arr, pos_label=POSITIVE_LABEL, zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred_arr, pos_label=POSITIVE_LABEL, zero_division=0)),
    )


def hand_verify_metrics(result: MetricResult, verbose: bool = True) -> MetricResult:
    """Recompute every metric from the four confusion-matrix cells and assert agreement.

    This is TRD.md §2.3's mandatory pre-publication check, applied to this project's own numbers.
    It is the check that would have exposed the base paper's swapped precision/recall labels.

    Args:
        result: a `MetricResult` from `compute_metrics`.
        verbose: print the worked arithmetic (useful in notebooks as visible evidence).

    Returns:
        The same result with `hand_verified=True`.

    Raises:
        AssertionError: if any hand-computed value differs from scikit-learn's by more than
            `HAND_VERIFY_TOLERANCE`.
    """
    tp, fp, tn, fn = result.tp, result.fp, result.tn, result.fn

    hand_accuracy = (tp + tn) / (tp + fp + tn + fn)
    hand_precision = tp / (tp + fp) if (tp + fp) else 0.0
    hand_recall = tp / (tp + fn) if (tp + fn) else 0.0
    hand_f1 = (
        2 * hand_precision * hand_recall / (hand_precision + hand_recall)
        if (hand_precision + hand_recall)
        else 0.0
    )

    if verbose:
        print(f"Hand-verification of: {result.model_name}")
        print(f"  {POSITIVE_CLASS_STATEMENT}")
        print(f"  Accuracy  = ({tp:,} + {tn:,}) / {tp + fp + tn + fn:,} = {hand_accuracy:.9f}")
        print(f"  Precision = {tp:,} / ({tp:,} + {fp:,}) = {hand_precision:.9f}")
        print(f"  Recall    = {tp:,} / ({tp:,} + {fn:,}) = {hand_recall:.9f}")
        print(f"  F1        = 2*{hand_precision:.6f}*{hand_recall:.6f} / "
              f"({hand_precision:.6f}+{hand_recall:.6f}) = {hand_f1:.9f}")

    for name, hand_value, library_value in (
        ("accuracy", hand_accuracy, result.accuracy),
        ("precision", hand_precision, result.precision),
        ("recall", hand_recall, result.recall),
        ("f1", hand_f1, result.f1),
    ):
        delta = abs(hand_value - library_value)
        assert delta <= HAND_VERIFY_TOLERANCE, (
            f"{name} mismatch for {result.model_name}: hand-computed {hand_value!r} vs "
            f"scikit-learn {library_value!r} (delta {delta:g}). If this fires, the positive-class "
            "convention has drifted somewhere -- that is Objection #3 happening in OUR code."
        )
        if verbose:
            print(f"    OK {name}: hand {hand_value:.9f} == sklearn {library_value:.9f}")

    result.hand_verified = True
    return result


def results_table(results: list[MetricResult]) -> pd.DataFrame:
    """Assemble multiple `MetricResult`s into one comparison table.

    The returned frame always carries a `positive_class` column, so no table can be copied out of
    this repo without its convention attached (Build-Instructions §A.1).

    Args:
        results: the results to tabulate.

    Returns:
        DataFrame, one row per result.
    """
    return pd.DataFrame([r.as_row() for r in results])
