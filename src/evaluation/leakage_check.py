"""Empirical train/test leakage detection (Build-Instructions T1.1, extended in T1.2).

What this module proves
-----------------------
Objection #1 (PRD.md §2.1.1) is that the base paper applies SMOTE to the FULL dataset and only
then splits 80/10/10. Its own Table 5 totals (936,548 / 117,069 / 117,069 = 1,170,684) equal its
post-SMOTE instance count exactly, which is only possible if synthetic minority samples -- each
interpolated between two REAL training points -- were themselves dealt into the test set.

That argument is textual. This module makes it **measurable**:

  For every minority-class (Normal) test sample, compute the Euclidean distance to its nearest
  TRAINING-set neighbour, in scaled feature space.

  - Leaky pipeline (SMOTE before split): a large fraction of these distances sit at or near ZERO,
    because the test sample is a synthetic point lying on the segment between two training points
    (or an exact duplicate of one).
  - Honest pipeline (split before SMOTE): the same distribution has essentially no near-zero mass;
    test points are genuinely unseen traffic.

The gap between those two distributions is the concrete evidence Build-Instructions T1.1 asks for.

Metric conventions
------------------
This module reports DISTANCES, not classification metrics, so no positive-class convention applies
here. Classification metrics live in `src/evaluation/metrics.py`, where **Attack is the positive
class** (TRD.md §2.3).

Fixed parameters
----------------
    NEAR_ZERO_DISTANCE_THRESHOLD = 1e-6   a distance below this in min-max-scaled space means the
                                          test point is numerically indistinguishable from a
                                          training point -- an outright duplicate.
    LEAKY_FRACTION_ALARM         = 0.01   >1% of minority test points at near-zero distance is
                                          treated as a positive leakage finding.
    DISTANCE_PERCENTILES         = (0, 1, 5, 25, 50, 75, 100)  reported percentile grid.
    N_NEIGHBORS                  = 1      nearest neighbour only.
    NN_ALGORITHM                 = "brute"  exact search. Tree indices degrade to worse-than-brute
                                          in 62 dimensions, so brute force is both faster and
                                          exactly correct here.
    QUERY_CHUNK_SIZE             = 500    query rows per batch, to bound peak memory of the
                                          n_query x n_train distance block.
    MAX_TEST_SAMPLES             = 5000   cap on examined test points. An exact 1-NN search in 62
                                          dimensions is O(n_test * n_train); 5,000 test points
                                          against the full training fold is enough to estimate the
                                          near-zero FRACTION to within well under a percentage
                                          point, and the alarm threshold is 1%. The subsample is
                                          drawn with `RANDOM_STATE`, so it is reproducible.
Determinism: `NearestNeighbors` with `algorithm="auto"` is exact (not approximate), so results are
reproducible without a seed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

NEAR_ZERO_DISTANCE_THRESHOLD: float = 1e-6
LEAKY_FRACTION_ALARM: float = 0.01
DISTANCE_PERCENTILES: tuple[int, ...] = (0, 1, 5, 25, 50, 75, 100)
N_NEIGHBORS: int = 1
MAX_TEST_SAMPLES: int = 5000
NN_ALGORITHM: str = "brute"
QUERY_CHUNK_SIZE: int = 500


@dataclass
class LeakageReport:
    """Result of one nearest-neighbour leakage check.

    Attributes:
        pipeline_name: label for the pipeline being checked (e.g. "leaky (SMOTE before split)").
        n_test_samples: number of test rows examined.
        n_train_samples: number of training rows searched against.
        class_label: the class whose test samples were examined (minority class by default).
        distances: nearest-training-neighbour distance for each examined test sample.
        near_zero_count: how many of those distances are below NEAR_ZERO_DISTANCE_THRESHOLD.
        near_zero_fraction: `near_zero_count / n_test_samples`.
        leakage_detected: True when `near_zero_fraction` exceeds LEAKY_FRACTION_ALARM.
        percentiles: distance percentiles keyed by DISTANCE_PERCENTILES.
        n_test_class_total: how many test rows of `class_label` existed before subsampling.
    """

    pipeline_name: str
    n_test_samples: int
    n_train_samples: int
    class_label: int
    distances: np.ndarray = field(repr=False)
    near_zero_count: int
    near_zero_fraction: float
    leakage_detected: bool
    percentiles: dict[int, float]
    n_test_class_total: int

    def summary(self) -> str:
        """Render a human-readable block suitable for pasting into a notebook or report."""
        verdict = (
            "LEAKAGE DETECTED" if self.leakage_detected else "no near-duplicate leakage signature"
        )
        pct_lines = "\n".join(
            f"      p{p:<3d} : {self.percentiles[p]:.6g}" for p in DISTANCE_PERCENTILES
        )
        return (
            f"Nearest-neighbour leakage check -- {self.pipeline_name}\n"
            f"  test samples examined (class {self.class_label}) : {self.n_test_samples:,}"
            f" of {self.n_test_class_total:,}\n"
            f"  training samples searched                        : {self.n_train_samples:,}\n"
            f"  distances < {NEAR_ZERO_DISTANCE_THRESHOLD:g}"
            f"                          : {self.near_zero_count:,} "
            f"({self.near_zero_fraction:.2%})\n"
            f"  distance percentiles:\n{pct_lines}\n"
            f"  VERDICT: {verdict}"
        )


def nearest_neighbour_distances(
    X_train: np.ndarray | pd.DataFrame,
    X_test: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Euclidean distance from each test row to its nearest training row.

    Args:
        X_train: training feature matrix (already scaled).
        X_test: test feature matrix (scaled with the SAME fitted scaler).

    Returns:
        1-D array of length `len(X_test)`.
    """
    train = np.asarray(X_train, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm=NN_ALGORITHM)
    nn.fit(train)

    # Chunked so the n_query x n_train distance block never materialises in full.
    out = np.empty(len(test), dtype=np.float64)
    for start in range(0, len(test), QUERY_CHUNK_SIZE):
        stop = min(start + QUERY_CHUNK_SIZE, len(test))
        distances, _ = nn.kneighbors(test[start:stop], n_neighbors=N_NEIGHBORS)
        out[start:stop] = distances[:, 0]
    return out


def check_leakage(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    pipeline_name: str,
    class_label: int | None = None,
    max_test_samples: int = MAX_TEST_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> LeakageReport:
    """Run the nearest-neighbour leakage check for one class of test samples.

    The check is restricted to a single class -- by default the MINORITY class of the training
    fold, because that is the class SMOTE synthesises and therefore the only class whose test
    samples can be synthetic near-duplicates.

    Args:
        X_train: scaled training features.
        y_train: training labels.
        X_test: scaled test features.
        y_test: test labels.
        pipeline_name: label used in the report (e.g. "leaky (SMOTE before split)").
        class_label: class to examine. Defaults to the training fold's minority class.
        max_test_samples: cap on examined test rows (see MAX_TEST_SAMPLES). Pass 0 for no cap.
        random_state: seed for the subsample, so the check is reproducible.

    Returns:
        A `LeakageReport`.

    Raises:
        ValueError: if the chosen class has no test samples.
    """
    y_train_arr = np.asarray(y_train).ravel()
    y_test_arr = np.asarray(y_test).ravel()

    if class_label is None:
        values, counts = np.unique(y_train_arr, return_counts=True)
        class_label = int(values[np.argmin(counts)])
        logger.info("Minority class of the training fold: %d", class_label)

    mask = y_test_arr == class_label
    if not mask.any():
        raise ValueError(f"No test samples with class label {class_label}")

    X_test_arr = np.asarray(X_test, dtype=np.float64)[mask]
    n_test_class_total = len(X_test_arr)

    if max_test_samples and n_test_class_total > max_test_samples:
        rng = np.random.default_rng(random_state)
        picked = rng.choice(n_test_class_total, size=max_test_samples, replace=False)
        X_test_arr = X_test_arr[picked]
        logger.info(
            "Subsampled %d of %d class-%d test rows for the 1-NN search (seed %d)",
            max_test_samples, n_test_class_total, class_label, random_state,
        )

    distances = nearest_neighbour_distances(X_train, X_test_arr)

    near_zero_count = int((distances < NEAR_ZERO_DISTANCE_THRESHOLD).sum())
    near_zero_fraction = near_zero_count / len(distances)

    report = LeakageReport(
        pipeline_name=pipeline_name,
        n_test_samples=len(distances),
        n_train_samples=len(np.asarray(X_train)),
        class_label=class_label,
        distances=distances,
        near_zero_count=near_zero_count,
        near_zero_fraction=near_zero_fraction,
        leakage_detected=near_zero_fraction > LEAKY_FRACTION_ALARM,
        percentiles={p: float(np.percentile(distances, p)) for p in DISTANCE_PERCENTILES},
        n_test_class_total=n_test_class_total,
    )
    logger.info("%s", report.summary())
    return report


def compare_reports(leaky: LeakageReport, honest: LeakageReport) -> pd.DataFrame:
    """Build the side-by-side leaky-vs-honest table for the T1.1/T1.2 notebooks.

    Args:
        leaky: report from the pre-split-SMOTE pipeline (T1.1).
        honest: report from the split-first pipeline (T1.2).

    Returns:
        DataFrame with one row per reported statistic and one column per pipeline.
    """
    rows = {
        "test samples examined": (leaky.n_test_samples, honest.n_test_samples),
        "test samples of that class (total)": (
            leaky.n_test_class_total,
            honest.n_test_class_total,
        ),
        f"count distance < {NEAR_ZERO_DISTANCE_THRESHOLD:g}": (
            leaky.near_zero_count,
            honest.near_zero_count,
        ),
        "fraction near-zero": (leaky.near_zero_fraction, honest.near_zero_fraction),
        **{
            f"distance p{p}": (leaky.percentiles[p], honest.percentiles[p])
            for p in DISTANCE_PERCENTILES
        },
        "leakage detected": (leaky.leakage_detected, honest.leakage_detected),
    }
    return pd.DataFrame(rows, index=[leaky.pipeline_name, honest.pipeline_name]).T


# ---------------------------------------------------------------------------
# Direct synthetic-row contamination check
# ---------------------------------------------------------------------------
# The nearest-neighbour distances above are *statistical* evidence, and on IoTID20 they are
# partly confounded: after cleaning, 363,884 of IoTID20's 625,415 rows (58.2%) are exact duplicates
# in the 69-feature space, so some near-zero
# train/test distances occur even in a correctly-ordered pipeline. The check below has no such
# confound -- it is a direct count of rows that SMOTE fabricated and that then landed in the test
# fold. A synthetic row is, by construction, an interpolation between two real minority rows that
# the model trains on, so its presence in the test fold is leakage by definition, not by inference.


@dataclass
class ContaminationReport:
    """How much of an evaluation fold was fabricated by SMOTE rather than observed.

    Attributes:
        pipeline_name: label for the pipeline being checked.
        fold_name: which fold was examined ("test" or "validation").
        n_rows: size of that fold.
        n_synthetic: rows in it that SMOTE generated.
        fraction_synthetic: `n_synthetic / n_rows`.
        n_minority_rows: rows of the resampled (minority) class in that fold.
        fraction_of_minority_synthetic: share of those minority rows that are synthetic.
    """

    pipeline_name: str
    fold_name: str
    n_rows: int
    n_synthetic: int
    fraction_synthetic: float
    n_minority_rows: int
    fraction_of_minority_synthetic: float

    def summary(self) -> str:
        """Render a printable verdict block."""
        verdict = (
            f"LEAKAGE: {self.fraction_synthetic:.2%} of the {self.fold_name} fold never existed "
            "in the real capture -- it was interpolated from rows the model trained on."
            if self.n_synthetic
            else f"CLEAN: every row of the {self.fold_name} fold is real observed traffic."
        )
        return (
            f"Synthetic-contamination check -- {self.pipeline_name}\n"
            f"  {self.fold_name} fold rows                 : {self.n_rows:,}\n"
            f"  of which SMOTE-generated              : {self.n_synthetic:,} "
            f"({self.fraction_synthetic:.2%})\n"
            f"  minority-class rows in the fold       : {self.n_minority_rows:,}\n"
            f"  of those, SMOTE-generated             : "
            f"{self.fraction_of_minority_synthetic:.2%}\n"
            f"  VERDICT: {verdict}"
        )


def synthetic_contamination(
    y_fold: np.ndarray | pd.Series,
    synthetic_mask: np.ndarray | None,
    pipeline_name: str,
    fold_name: str = "test",
    minority_label: int = 0,
) -> ContaminationReport:
    """Count how many rows of an evaluation fold were fabricated by SMOTE.

    Args:
        y_fold: labels of the fold being examined.
        synthetic_mask: the fold's `synthetic_*` mask from `DataSplit`. None is treated as
            "no resampling applied", i.e. an all-real fold.
        pipeline_name: label used in the report.
        fold_name: "test" or "validation", for the report text.
        minority_label: the class SMOTE oversamples. Defaults to 0 (Normal) -- note this is the
            NEGATIVE class; the POSITIVE class for metrics is Attack=1 (TRD.md §2.3).

    Returns:
        A `ContaminationReport`.

    Raises:
        ValueError: if `synthetic_mask` length does not match `y_fold`.
    """
    y_arr = np.asarray(y_fold).ravel()
    mask = (
        np.zeros(len(y_arr), dtype=bool)
        if synthetic_mask is None
        else np.asarray(synthetic_mask, dtype=bool)
    )
    if len(mask) != len(y_arr):
        raise ValueError(f"synthetic_mask has length {len(mask)} but fold has {len(y_arr)} rows")

    minority = y_arr == minority_label
    n_minority = int(minority.sum())

    report = ContaminationReport(
        pipeline_name=pipeline_name,
        fold_name=fold_name,
        n_rows=len(y_arr),
        n_synthetic=int(mask.sum()),
        fraction_synthetic=float(mask.mean()),
        n_minority_rows=n_minority,
        fraction_of_minority_synthetic=(
            float(mask[minority].mean()) if n_minority else 0.0
        ),
    )
    logger.info("%s", report.summary())
    return report
