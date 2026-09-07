"""Split-then-resample: the fix for Objection #1 (Build-Instructions T1.2, TRD.md §2.2).

The bug this module exists to prevent
-------------------------------------
The base paper's described pipeline is  clean -> select -> **SMOTE** -> **split**.
SMOTE creates each synthetic minority sample by interpolating between two REAL minority samples.
If that happens before the split, synthetic points derived from training rows land in the test
fold, so the test fold is no longer unseen and the reported accuracy is inflated (PRD.md §2.1.1).

The correct order, enforced here, is  clean -> select -> **split** -> **SMOTE (train fold only)**
-> **scale (scaler fitted on the train fold only)**  (TRD.md §2.2).

`resample_training_fold()` will raise if handed anything but a training fold, and
`split_dataset()` is the only supported way to produce folds, so the broken order cannot be
reached by accident. The T1.1 leaky reproduction deliberately bypasses both, by calling
`leaky_resample_then_split()` -- a function whose name makes the mistake impossible to commit
silently.

Exact parameters
----------------
    TRAIN_RATIO / TEST_RATIO / VAL_RATIO = 0.80 / 0.10 / 0.10   (src/config.py, TRD.md §6.2;
        matches the base paper's stated split so our numbers stay comparable to theirs)
    RANDOM_STATE   = 42        project-wide seed (Build-Instructions §A.3)
    SMOTE_K_NEIGHBORS = 5      imbalanced-learn's default; stated here rather than left implicit
    SMOTE_SAMPLING_STRATEGY = "auto"   oversample the minority (Normal) up to the majority
        (Attack) count -- i.e. a balanced training fold, which is what the base paper describes
    STRATIFY = True            both splits are stratified on the label, so the 93.6/6.4 class
        balance is preserved in every fold before resampling
Scaling: MinMaxScaler, matching the base paper's stated min-max normalisation (PRD.md FR-1).

Positive-class note: this module does not compute metrics. Where it refers to the minority class
it means **Normal (label 0)**; the POSITIVE class for all metrics is **Attack (label 1)**
(TRD.md §2.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.config import RANDOM_STATE, TEST_RATIO, TRAIN_RATIO, VAL_RATIO

logger = logging.getLogger(__name__)

SMOTE_K_NEIGHBORS: int = 5
SMOTE_SAMPLING_STRATEGY: str = "auto"
STRATIFY: bool = True


@dataclass
class DataSplit:
    """A train/test/validation split, with a flag recording which pipeline order produced it.

    Attributes:
        X_train, y_train: training fold.
        X_test, y_test: test fold.
        X_val, y_val: validation fold.
        resampled_before_split: True only for the deliberately-leaky T1.1 reproduction. Every
            report generated from a split must state this value.
        scaler: the fitted `MinMaxScaler`, or None if scaling has not been applied yet.
        synthetic_train / synthetic_test / synthetic_val: boolean masks marking which rows of each
            fold were SYNTHESISED by SMOTE rather than observed in the raw capture. In the honest
            pipeline `synthetic_test` and `synthetic_val` are all-False by construction. In the
            leaky pipeline they are not -- and that is the direct, countable proof of Objection #1
            (see `src/evaluation/leakage_check.py::synthetic_contamination`). None when no
            resampling has been applied.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    resampled_before_split: bool
    scaler: MinMaxScaler | None = None
    synthetic_train: np.ndarray | None = None
    synthetic_test: np.ndarray | None = None
    synthetic_val: np.ndarray | None = None

    def summary(self) -> str:
        """Render fold sizes and class balances, with the pipeline order stated up front."""
        order = (
            "LEAKY  : resample BEFORE split (the base paper's order -- T1.1)"
            if self.resampled_before_split
            else "HONEST : split BEFORE resample (TRD.md §2.2 -- T1.2)"
        )

        def line(name: str, y: pd.Series) -> str:
            y_arr = np.asarray(y).ravel()
            n_attack = int((y_arr == 1).sum())
            n_normal = int((y_arr == 0).sum())
            return (
                f"  {name:<11s} n={len(y_arr):>9,}  "
                f"Attack(1)={n_attack:>9,}  Normal(0)={n_normal:>9,}"
            )

        return (
            f"Pipeline order: {order}\n"
            f"{line('train', self.y_train)}\n"
            f"{line('test', self.y_test)}\n"
            f"{line('validation', self.y_val)}"
        )


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
    resampled_before_split: bool = False,
    is_synthetic: np.ndarray | None = None,
) -> DataSplit:
    """Split into 80/10/10 train/test/validation, stratified on the label.

    Performed as two successive `train_test_split` calls: first 80/20, then that 20 split in half
    to give 10/10. Both are stratified, so every fold keeps the original class balance.

    Args:
        X: feature matrix.
        y: binary labels (1 = Attack).
        random_state: seed; defaults to the project-wide 42.
        resampled_before_split: set True ONLY by `leaky_resample_then_split`, to stamp the
            resulting split as leaky.
        is_synthetic: per-row boolean marking SMOTE-generated rows, split alongside X and y so the
            test fold's synthetic contamination can be counted exactly.

    Returns:
        A `DataSplit` with no resampling or scaling applied yet.
    """
    holdout_ratio = TEST_RATIO + VAL_RATIO
    stratify_full = y if STRATIFY else None

    # `synth` rides along through both splits so each fold keeps its own synthetic-row mask.
    synth = (
        np.zeros(len(X), dtype=bool) if is_synthetic is None else np.asarray(is_synthetic, dtype=bool)
    )
    if len(synth) != len(X):
        raise ValueError(f"is_synthetic has length {len(synth)} but X has {len(X)} rows")

    X_train, X_holdout, y_train, y_holdout, synth_train, synth_holdout = train_test_split(
        X,
        y,
        synth,
        test_size=holdout_ratio,
        random_state=random_state,
        stratify=stratify_full,
    )
    # Half of the 20% holdout -> test, half -> validation, giving 10/10.
    X_test, X_val, y_test, y_val, synth_test, synth_val = train_test_split(
        X_holdout,
        y_holdout,
        synth_holdout,
        test_size=VAL_RATIO / holdout_ratio,
        random_state=random_state,
        stratify=y_holdout if STRATIFY else None,
    )

    split = DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        resampled_before_split=resampled_before_split,
        synthetic_train=synth_train,
        synthetic_test=synth_test,
        synthetic_val=synth_val,
    )
    logger.info(
        "Split %d rows into %.0f/%.0f/%.0f train/test/val\n%s",
        len(X),
        TRAIN_RATIO * 100,
        TEST_RATIO * 100,
        VAL_RATIO * 100,
        split.summary(),
    )
    return split


def resample_training_fold(split: DataSplit, random_state: int = RANDOM_STATE) -> DataSplit:
    """Apply SMOTE to the TRAINING FOLD ONLY. This is the fix (TRD.md §2.2 step 3).

    Args:
        split: an un-resampled split from `split_dataset`.
        random_state: seed; defaults to the project-wide 42.

    Returns:
        A new `DataSplit` whose training fold is class-balanced. Test and validation folds are
        returned untouched -- they must remain a sample of the real class distribution.

    Raises:
        ValueError: if `split.resampled_before_split` is True, i.e. someone is trying to resample
            a fold that was already contaminated by the leaky pipeline.
    """
    if split.resampled_before_split:
        raise ValueError(
            "Refusing to resample a split that was already produced by the leaky "
            "resample-then-split pipeline. See TRD.md §2.2."
        )

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(split.X_train, split.y_train)

    # imbalanced-learn returns the original rows first, in order, then the synthesised rows.
    # Asserted rather than assumed, because the whole leakage argument depends on this tagging.
    n_original = len(split.X_train)
    assert np.array_equal(
        np.asarray(X_res)[:n_original], np.asarray(split.X_train)
    ), "SMOTE did not return the original rows first; the synthetic-row tagging is invalid."
    synth_train = np.zeros(len(X_res), dtype=bool)
    synth_train[n_original:] = True

    logger.info(
        "SMOTE on TRAINING FOLD ONLY: %d -> %d rows (test/val untouched at %d/%d)",
        len(split.X_train),
        len(X_res),
        len(split.X_test),
        len(split.X_val),
    )
    return DataSplit(
        X_train=pd.DataFrame(X_res, columns=split.X_train.columns),
        y_train=pd.Series(y_res, name=split.y_train.name),
        X_test=split.X_test,
        y_test=split.y_test,
        X_val=split.X_val,
        y_val=split.y_val,
        resampled_before_split=False,
        scaler=split.scaler,
        synthetic_train=synth_train,
        # Test and validation folds are untouched real traffic -- no synthetic rows, by
        # construction. This is exactly what the leaky pipeline cannot say.
        synthetic_test=np.zeros(len(split.X_test), dtype=bool),
        synthetic_val=np.zeros(len(split.X_val), dtype=bool),
    )


def leaky_resample_then_split(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE
) -> DataSplit:
    """Reproduce the BASE PAPER'S BROKEN ORDER: SMOTE the full dataset, then split (T1.1).

    **Do not use this for any reported result.** It exists solely to produce the leaky number and
    the near-duplicate test samples that `src/evaluation/leakage_check.py` then detects, so that
    Objection #1 rests on measurement rather than assertion (PRD.md §2.1.1, TRD.md §2.1).

    Args:
        X: full feature matrix.
        y: full binary label vector.
        random_state: seed; defaults to the project-wide 42.

    Returns:
        A `DataSplit` stamped `resampled_before_split=True`.
    """
    logger.warning(
        "Running the LEAKY pipeline (SMOTE on the FULL dataset, then split). "
        "T1.1 reproduction only -- never cite these numbers as a result."
    )
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X, y)

    n_original = len(X)
    assert np.array_equal(
        np.asarray(X_res)[:n_original], np.asarray(X)
    ), "SMOTE did not return the original rows first; the synthetic-row tagging is invalid."
    is_synthetic = np.zeros(len(X_res), dtype=bool)
    is_synthetic[n_original:] = True
    logger.info(
        "SMOTE on FULL dataset: %d -> %d rows (%d synthetic, now eligible for the TEST fold)",
        len(X), len(X_res), int(is_synthetic.sum()),
    )

    return split_dataset(
        pd.DataFrame(X_res, columns=X.columns),
        pd.Series(y_res, name=y.name),
        random_state=random_state,
        resampled_before_split=True,
        is_synthetic=is_synthetic,
    )


def scale_split(split: DataSplit) -> DataSplit:
    """Min-max scale all folds using a scaler FITTED ON THE TRAINING FOLD ONLY (TRD.md §2.2 step 4).

    Fitting the scaler on the full dataset would leak test-fold min/max into training -- a milder
    version of the same bug this project exposes.

    Args:
        split: the split to scale. For the honest pipeline this should already be resampled.

    Returns:
        A new `DataSplit` with all three folds scaled and `scaler` set to the fitted instance.
    """
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(split.X_train), columns=split.X_train.columns
    )
    X_test_s = pd.DataFrame(scaler.transform(split.X_test), columns=split.X_test.columns)
    X_val_s = pd.DataFrame(scaler.transform(split.X_val), columns=split.X_val.columns)
    logger.info("MinMaxScaler fitted on the training fold (%d rows) and applied to all folds",
                len(split.X_train))

    return DataSplit(
        X_train=X_train_s,
        y_train=split.y_train.reset_index(drop=True),
        X_test=X_test_s,
        y_test=split.y_test.reset_index(drop=True),
        X_val=X_val_s,
        y_val=split.y_val.reset_index(drop=True),
        resampled_before_split=split.resampled_before_split,
        scaler=scaler,
        synthetic_train=split.synthetic_train,
        synthetic_test=split.synthetic_test,
        synthetic_val=split.synthetic_val,
    )
