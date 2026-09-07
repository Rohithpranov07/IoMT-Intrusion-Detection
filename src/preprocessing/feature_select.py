"""Random Forest importance-based feature selection (Build-Instructions T1.1, T1.3).

Why Random Forest and not PSO
-----------------------------
The base paper reduces IoTID20 83 -> 62 and Edge-IIoTset 61 -> 46 features with PSO, but states
NO particle count, NO iteration count, NO inertia weight `w`, NO acceleration constants `c1`/`c2`,
and NO fitness function (TRD.md §6.1). That step is therefore not reproducible. Rather than invent
those values and present them as the paper's -- explicitly forbidden by Build-Instructions §A.1 --
this project uses a fully-specified alternative that the base paper also mentions: Random Forest
impurity-based feature importance. The full rationale and the trade-off table are in
`docs/feature_selection_decision.md` (T1.3).

**This is THIS PROJECT'S OWN method. It does not reproduce, approximate, or claim equivalence to
the base paper's PSO, and the 62 features it keeps are almost certainly a different subset.**

Exact hyperparameters (every one a named constant; no magic numbers)
--------------------------------------------------------------------
    N_ESTIMATORS              = 200      trees; top-k ranking is stable past ~200
    MAX_DEPTH                 = None     grow fully; depth caps bias importance toward early splits
    MIN_SAMPLES_LEAF          = 5        stops near-unique columns (ports) faking high importance
    CLASS_WEIGHT              = "balanced_subsample"   IoTID20 is 93.6% Attack / 6.4% Normal
    N_JOBS                    = -1       wall-clock only; no effect on the seeded result
    RANDOM_STATE              = 42       from src/config.py (Build-Instructions §A.3)
    IOTID20_N_FEATURES        = 62       matches the base paper's post-selection count
    EDGE_IIOTSET_N_FEATURES   = 46       matches the base paper's post-selection count

Selection rule: rank by `feature_importances_` descending, keep the top k. Ties break by original
column order, which is deterministic.

Leakage warning
---------------
`fit_select` learns the feature ranking from whatever data it is given. In the LEAKAGE-FREE
pipeline (T1.2) it must be fitted on the TRAINING FOLD ONLY; fitting it on the full dataset leaks
test-fold information into the feature choice. The T1.1 leaky reproduction deliberately fits on the
full dataset, because that is the broken pipeline order being demonstrated.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    EDGE_IIOTSET_SELECTED_FEATURE_COUNT,
    IOTID20_SELECTED_FEATURE_COUNT,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# --- Selector hyperparameters (docs/feature_selection_decision.md §4) ----------------------
N_ESTIMATORS: int = 200
MAX_DEPTH: int | None = None
MIN_SAMPLES_LEAF: int = 5
CLASS_WEIGHT: str = "balanced_subsample"
N_JOBS: int = -1

# --- Target feature counts, matched to the base paper for comparability (TRD.md §6.1) ------
IOTID20_N_FEATURES: int = IOTID20_SELECTED_FEATURE_COUNT  # 62
EDGE_IIOTSET_N_FEATURES: int = EDGE_IIOTSET_SELECTED_FEATURE_COUNT  # 46


class RandomForestFeatureSelector:
    """Rank features by Random Forest impurity importance and keep the top `n_features`.

    Attributes:
        n_features: how many features to keep.
        ranking_: DataFrame of every feature with its importance, rank, and kept/dropped flag.
            Written out by the notebooks so the selection is auditable.
        selected_features_: names of the kept features, in importance order.
    """

    def __init__(
        self,
        n_features: int = IOTID20_N_FEATURES,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_features = n_features
        self.random_state = random_state
        self.ranking_: pd.DataFrame | None = None
        self.selected_features_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestFeatureSelector":
        """Fit the forest and record the full importance ranking.

        Args:
            X: numeric feature matrix.
            y: binary labels (1 = Attack = positive class, per TRD.md §2.3).

        Returns:
            self.

        Raises:
            ValueError: if `n_features` exceeds the number of available columns.
        """
        if self.n_features > X.shape[1]:
            raise ValueError(
                f"Requested {self.n_features} features but X has only {X.shape[1]} columns. "
                "Check the feature funnel in src/preprocessing/clean.py."
            )

        forest = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight=CLASS_WEIGHT,
            n_jobs=N_JOBS,
            random_state=self.random_state,
        )
        logger.info("Fitting RF selector on %s to rank %d features", X.shape, X.shape[1])
        forest.fit(X, y)

        ranking = pd.DataFrame(
            {"feature": X.columns, "importance": forest.feature_importances_}
        )
        # Stable sort keeps original column order as the deterministic tie-break.
        ranking = ranking.sort_values(
            "importance", ascending=False, kind="mergesort"
        ).reset_index(drop=True)
        ranking["rank"] = ranking.index + 1
        ranking["kept"] = ranking["rank"] <= self.n_features

        self.ranking_ = ranking
        self.selected_features_ = ranking.loc[ranking["kept"], "feature"].tolist()
        logger.info(
            "Selected %d/%d features; top 5: %s",
            len(self.selected_features_),
            X.shape[1],
            self.selected_features_[:5],
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subset `X` to the selected features, in importance order.

        Args:
            X: feature matrix containing at least the selected columns.

        Returns:
            `X` restricted to `selected_features_`.

        Raises:
            RuntimeError: if called before `fit`.
            KeyError: if `X` is missing a selected column.
        """
        if self.selected_features_ is None:
            raise RuntimeError("transform() called before fit()")
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise KeyError(f"X is missing selected features: {missing}")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Convenience wrapper: `fit` then `transform` on the same data."""
        return self.fit(X, y).transform(X)
