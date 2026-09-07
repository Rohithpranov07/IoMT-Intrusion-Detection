"""Shared Random Forest baseline classifier (Build-Instructions T1.1 / T1.2).

NOTE ON REPO LAYOUT: `Build-Instructions.md` §B.3 does not list this file. It exists because T1.2
requires "the same baseline classifier as T1.1 for a fair before/after comparison" -- if the
leaky and honest notebooks each defined their own classifier, any drift between them would
contaminate the one comparison Contribution 1 rests on. Defining it once makes that impossible.

This is NOT the final architecture. The committed architecture is the CNN + BiLSTM + Transformer
ensemble of Phase 2 (TRD.md §3). This baseline is a fast, well-understood classifier whose only
job is to hold everything except the pipeline ORDER constant between T1.1 and T1.2.

Exact hyperparameters (plain numbers, per Build-Instructions §A.3)
------------------------------------------------------------------
    N_ESTIMATORS     = 100     trees
    MAX_DEPTH        = 20      capped so a 936k-row leaky fold and a 500k-row honest fold train in
                               comparable wall-clock time; also stops the forest from memorising
                               every row, which would blur the leaky-vs-honest gap
    MIN_SAMPLES_LEAF = 5       same rationale as the feature selector's
    CLASS_WEIGHT     = None    the training fold is already balanced by SMOTE; adding class
                               weights on top would double-correct
    N_JOBS           = -1      wall-clock only; no effect on the seeded result
    RANDOM_STATE     = 42      project-wide seed

Positive class for every metric computed from this model's predictions: **Attack (label 1)**
(TRD.md §2.3, enforced in `src/evaluation/metrics.py`).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

N_ESTIMATORS: int = 100
MAX_DEPTH: int = 20
MIN_SAMPLES_LEAF: int = 5
CLASS_WEIGHT: str | None = None
N_JOBS: int = -1


def build_baseline_classifier(random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """Construct the baseline classifier with the hyperparameters fixed in this module.

    Args:
        random_state: seed; defaults to the project-wide 42.

    Returns:
        An unfitted `RandomForestClassifier`.
    """
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS,
        random_state=random_state,
    )


def fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[RandomForestClassifier, np.ndarray]:
    """Fit the baseline on the training fold and predict on an evaluation fold.

    Args:
        X_train: scaled training features.
        y_train: training labels (1 = Attack).
        X_eval: scaled features of the fold to predict (test or validation).
        random_state: seed; defaults to the project-wide 42.

    Returns:
        Tuple of the fitted model and its predicted labels for `X_eval`.
    """
    model = build_baseline_classifier(random_state=random_state)
    logger.info("Fitting baseline RF on %s", (X_train.shape,))
    model.fit(X_train, y_train)
    return model, model.predict(X_eval)
