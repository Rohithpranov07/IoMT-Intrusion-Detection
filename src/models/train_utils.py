"""Shared training helpers for the ensemble branches (Build-Instructions T2.6).

NOTE ON REPO LAYOUT: `Build-Instructions.md` §B.3 does not list this file. It exists so that the
three branches in `notebooks/03_train_ensemble_iotid20.ipynb` (and later
`04_train_edge_iiotset.ipynb`, T3.6) are trained by *identical* code. If each notebook re-derived
its own training loop, a difference between them would silently confound the branch comparison and
the cross-dataset comparison -- the same reasoning that put `baseline.py` and `metrics.py` in
`src/`.

Training configuration (frozen in `docs/architecture_decision.md` §2.4)
------------------------------------------------------------------------
    OPTIMIZER        = Adam, learning rate 1e-3   matches the base paper's stated optimizer
    LOSS             = categorical cross-entropy  (2-class softmax output)
    BATCH_SIZE       = 256
    MAX_EPOCHS       = 50                         with early stopping, rarely reached
    EARLY_STOPPING   = patience 5 on validation POSITIVE-CLASS F1, restoring best weights
                       Not val-accuracy: at ~89% Attack, accuracy is a poor stopping signal.
    SEED             = 42                         (src/config.py)

IMBALANCE HANDLING -- A DELIBERATE DEVIATION FROM THE ROW-LEVEL PIPELINE
------------------------------------------------------------------------
`TRD.md §2.2` step 3 applies SMOTE to the training fold. That is right for the row-level baseline
and is what `src/preprocessing/resample.py` does. It is **not** applied to sequences here, and the
reason must be stated wherever these results are reported:

  SMOTE interpolates between two neighbouring samples. Applied to a flattened flow *sequence*
  (10 x 62 = 620 dimensions) it would fabricate sessions that are a blend of two different
  devices' traffic over time -- objects that could not occur on a real network, and whose
  timestep-to-timestep dynamics are exactly what the BiLSTM and attention branches are supposed to
  learn. Generating physically impossible sequences to fix an imbalance would trade a measurable
  problem for an unmeasurable one.

Instead the training fold is balanced with **inverse-frequency class weights** in the loss
(`compute_class_weights`). No synthetic samples are created, so the test fold's 0.00% synthetic
contamination is preserved, and the imbalance is still corrected. `reports/phase2_results.md` must
state this difference when comparing the ensemble against the row-level baseline, because the two
do not handle imbalance the same way.

Positive class: index 1 = **Attack** (`TRD.md §2.3`) in every metric this module computes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import NEGATIVE_LABEL, POSITIVE_LABEL, RANDOM_STATE

logger = logging.getLogger(__name__)

# --- Frozen training constants (docs/architecture_decision.md §2.4) ------------------------
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 256
MAX_EPOCHS: int = 50
EARLY_STOPPING_PATIENCE: int = 5
LOSS: str = "categorical_crossentropy"


def set_global_seeds(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and TensorFlow so a training run is reproducible.

    Args:
        seed: the seed to apply. Defaults to the project-wide 42.
    """
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    logger.info("Global seeds set to %d with op determinism enabled", seed)


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights for the training fold.

    Weight for class `c` is `n_samples / (n_classes * count_c)`, so the two classes contribute
    equally to the loss without any sample being duplicated or synthesised.

    Args:
        y: integer training labels (0 = Normal, 1 = Attack).

    Returns:
        Mapping of class label to loss weight.

    Raises:
        ValueError: if either class is absent from `y`.
    """
    y = np.asarray(y).ravel()
    counts = {int(c): int((y == c).sum()) for c in (NEGATIVE_LABEL, POSITIVE_LABEL)}
    if min(counts.values()) == 0:
        raise ValueError(f"Both classes must be present in the training fold; got {counts}")

    n_samples = len(y)
    weights = {c: n_samples / (len(counts) * n) for c, n in counts.items()}
    logger.info("Class counts %s -> weights %s", counts, {k: round(v, 4) for k, v in weights.items()})
    return weights


@keras.utils.register_keras_serializable(package="iomt_ids")
class PositiveClassF1(keras.metrics.Metric):
    """F1 for the POSITIVE class (Attack, index 1) -- `TRD.md §2.3`'s convention.

    Keras's built-in `F1Score` returns either a per-class vector (which `EarlyStopping` cannot
    monitor) or an average across classes (which is not the number this project reports). This
    metric returns the single scalar we actually care about, so training stops on the right signal.
    """

    def __init__(self, name: str = "positive_f1", **kwargs: object) -> None:
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None
    ) -> None:
        """Accumulate TP/FP/FN for the positive class from one batch."""
        true_labels = tf.argmax(y_true, axis=-1)
        pred_labels = tf.argmax(y_pred, axis=-1)

        is_true_positive = tf.logical_and(
            tf.equal(true_labels, POSITIVE_LABEL), tf.equal(pred_labels, POSITIVE_LABEL)
        )
        is_false_positive = tf.logical_and(
            tf.equal(true_labels, NEGATIVE_LABEL), tf.equal(pred_labels, POSITIVE_LABEL)
        )
        is_false_negative = tf.logical_and(
            tf.equal(true_labels, POSITIVE_LABEL), tf.equal(pred_labels, NEGATIVE_LABEL)
        )

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(is_true_positive, tf.float32)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(is_false_positive, tf.float32)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(is_false_negative, tf.float32)))

    def result(self) -> tf.Tensor:
        """Return `2TP / (2TP + FP + FN)`, the algebraic simplification of the F1 formula."""
        numerator = 2.0 * self.true_positives
        denominator = numerator + self.false_positives + self.false_negatives
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_state(self) -> None:
        """Zero the accumulators between epochs."""
        for variable in (self.true_positives, self.false_positives, self.false_negatives):
            variable.assign(0.0)


def load_branch(path: str | Path, compile_model: bool = False) -> keras.Model:
    """Load a saved branch, resolving this project's custom layers and metrics.

    Phase 3 and Phase 4 consume the *trained* branches rather than retraining them, so loading has
    to work without the caller knowing which custom objects a branch contains. Both custom classes
    carry `@keras.utils.register_keras_serializable`, AND are passed explicitly in `custom_objects`
    here. The belt-and-braces is deliberate: the registry is keyed by the `registered_name` stored
    in the file, so a model saved before a class was registered (or under a different package
    prefix) will not resolve through the registry alone. `custom_objects` matches on the bare class
    name and therefore loads both old and new checkpoints.

    Args:
        path: path to a `.keras` file written by `notebooks/03_train_ensemble_iotid20.ipynb`.
        compile_model: restore the training configuration too. Defaults to False -- inference and
            explanation need no optimizer or metrics, and skipping it makes loading independent of
            the training setup.

    Returns:
        The loaded `keras.Model`.
    """
    from src.models.transformer_branch import AddPositionalEncoding

    return keras.models.load_model(
        path,
        compile=compile_model,
        custom_objects={
            "AddPositionalEncoding": AddPositionalEncoding,
            "PositiveClassF1": PositiveClassF1,
        },
    )


def scale_sequences(
    X_train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Min-max scale sequence tensors with statistics from the TRAINING fold only.

    The scaler is fitted per feature across all training timesteps, then applied unchanged to the
    other folds. Fitting on the full dataset would leak test-fold ranges into training -- the
    milder cousin of the bug this project exposes (`TRD.md §2.2` step 4).

    Args:
        X_train: training tensor, shape `(n_windows, sequence_length, n_features)`.
        *others: further tensors (test, validation) to transform with the same statistics.

    Returns:
        Tuple of scaled tensors in the order given, starting with `X_train`.
    """
    n_features = X_train.shape[2]
    flat = X_train.reshape(-1, n_features)
    minimum = flat.min(axis=0)
    maximum = flat.max(axis=0)
    spread = np.where(maximum > minimum, maximum - minimum, 1.0)

    def apply(tensor: np.ndarray) -> np.ndarray:
        return ((tensor - minimum) / spread).astype(np.float32)

    logger.info("Min-max scaling fitted on %d training windows", len(X_train))
    return (apply(X_train), *(apply(t) for t in others))


def train_branch(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float] | None = None,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Compile and train one branch with the frozen §2.4 configuration.

    Args:
        model: an uncompiled branch from `build_*_branch`.
        X_train: training sequences, `(n, sequence_length, n_features)`.
        y_train: integer training labels.
        X_val: validation sequences.
        y_val: integer validation labels.
        class_weights: loss weights per class. Computed from `y_train` when None.
        max_epochs: epoch cap; early stopping usually triggers first.
        batch_size: minibatch size.
        verbose: Keras verbosity.

    Returns:
        The Keras `History` of the run.
    """
    if class_weights is None:
        class_weights = compute_class_weights(y_train)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=["accuracy", PositiveClassF1()],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_positive_f1",
        mode="max",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=verbose,
    )

    logger.info("Training %s on %d windows for up to %d epochs", model.name, len(X_train), max_epochs)
    return model.fit(
        X_train,
        keras.utils.to_categorical(y_train, num_classes=2),
        validation_data=(X_val, keras.utils.to_categorical(y_val, num_classes=2)),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=verbose,
    )


def describe_training_configuration() -> str:
    """Return the training configuration as plain text for reports and slides.

    Returns:
        A human-readable description of every training hyperparameter.
    """
    return (
        f"Adam(lr={LEARNING_RATE}), loss={LOSS}, batch_size={BATCH_SIZE}, "
        f"max_epochs={MAX_EPOCHS}, EarlyStopping(monitor='val_positive_f1', mode='max', "
        f"patience={EARLY_STOPPING_PATIENCE}, restore_best_weights=True), seed={RANDOM_STATE}. "
        "Class imbalance handled by inverse-frequency class weights, NOT SMOTE -- see this "
        "module's docstring for why SMOTE is inappropriate for flow sequences."
    )
