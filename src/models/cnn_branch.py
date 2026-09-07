"""CNN branch of the ensemble (Build-Instructions T2.2; TRD.md §3.2.1).

Role: spatial / local feature extraction across the per-timestep feature vector, and detection of
short bursts within the window.

EXACT ARCHITECTURE (frozen in `docs/architecture_decision.md` §2.1 — these numbers are the
contract; changing them here without changing that document is a spec violation)
--------------------------------------------------------------------------------------------
    Input                (batch, 10, 62)   = (batch, SEQUENCE_LENGTH, n_features)

    CONV_LAYER_COUNT   = 3            conv layers  <-- matches the base paper's stated Conv1D
                                      count, so the ensemble comparison isolates the sequence and
                                      fusion changes rather than confounding them with depth
    CONV_FILTERS       = (64, 128, 64)  filters per layer, in order
    CONV_KERNEL_SIZE   = 3            all three layers; a 3-layer stack of kernel-3 convolutions
                                      has a receptive field of 7 of the 10 timesteps
    CONV_STRIDE        = 1
    CONV_PADDING       = "same"       preserves length so branches stay dimensionally comparable
    CONV_ACTIVATION    = "relu"       matches the base paper's stated activation
    USE_BATCH_NORM     = True         one BatchNormalization after each conv layer
    DROPOUT_RATE       = 0.3
    POOLING            = GlobalMaxPooling1D   max, NOT average: attack evidence is typically a
                                      spike in one or two timesteps, not a shift in the window mean
    EMBEDDING_DIM      = 64           output width of the pooled embedding
    HEAD_UNITS         = 2            Dense classification head, softmax

Total: 3 Conv1D + 3 BatchNorm + 1 Dropout + 1 GlobalMaxPooling1D + 1 Dense(2, softmax).

Positive class: output index 1 is **Attack**, the POSITIVE class (`TRD.md §2.3`). Index 0 is
Normal. Every branch in this package uses the same ordering so fusion never has to reorder.

Determinism: weight initialisation is seeded by the caller via `set_seed`/`RANDOM_STATE`.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.config import RANDOM_STATE

# --- Frozen architecture constants (docs/architecture_decision.md §2.1) --------------------
CONV_LAYER_COUNT: int = 3
CONV_FILTERS: tuple[int, int, int] = (64, 128, 64)
CONV_KERNEL_SIZE: int = 3
CONV_STRIDE: int = 1
CONV_PADDING: str = "same"
CONV_ACTIVATION: str = "relu"
USE_BATCH_NORM: bool = True
DROPOUT_RATE: float = 0.3
EMBEDDING_DIM: int = 64
HEAD_UNITS: int = 2
BRANCH_NAME: str = "cnn_branch"

assert len(CONV_FILTERS) == CONV_LAYER_COUNT, (
    "CONV_FILTERS must give one filter count per conv layer; "
    "docs/architecture_decision.md §2.1 fixes both at 3."
)
assert CONV_FILTERS[-1] == EMBEDDING_DIM, (
    "The last conv layer's filter count is the pooled embedding width, so it must equal "
    "EMBEDDING_DIM for the three branches to share a 64-d embedding."
)


def build_cnn_branch(
    sequence_length: int,
    n_features: int,
    n_classes: int = HEAD_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    seed: int = RANDOM_STATE,
    name: str = BRANCH_NAME,
) -> keras.Model:
    """Build the CNN branch exactly as specified in `docs/architecture_decision.md` §2.1.

    Args:
        sequence_length: window length. Must be > 1 (`TRD.md §9`); a length-1 input would make
            this a per-record classifier, which is Objection #2.
        n_features: number of features per timestep (62 for IoTID20).
        n_classes: output units. 2 for the binary task; index 1 = Attack = positive.
        dropout_rate: dropout applied after the conv stack.
        seed: weight-initialisation seed.
        name: model name.

    Returns:
        An uncompiled `keras.Model` mapping `(batch, sequence_length, n_features)` to
        `(batch, n_classes)` softmax probabilities.

    Raises:
        ValueError: if `sequence_length` < 2.
    """
    if sequence_length < 2:
        raise ValueError(
            f"sequence_length={sequence_length} makes this a single-record classifier, "
            "which is exactly Objection #2 (PRD.md §2.1.2). TRD.md §9 requires > 1."
        )

    initializer = keras.initializers.GlorotUniform(seed=seed)

    inputs = keras.Input(shape=(sequence_length, n_features), name=f"{name}_input")
    x = inputs
    for index, filters in enumerate(CONV_FILTERS, start=1):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=CONV_KERNEL_SIZE,
            strides=CONV_STRIDE,
            padding=CONV_PADDING,
            activation=CONV_ACTIVATION,
            kernel_initializer=initializer,
            name=f"{name}_conv{index}",
        )(x)
        if USE_BATCH_NORM:
            x = layers.BatchNormalization(name=f"{name}_bn{index}")(x)

    x = layers.Dropout(dropout_rate, seed=seed, name=f"{name}_dropout")(x)
    embedding = layers.GlobalMaxPooling1D(name=f"{name}_embedding")(x)
    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        kernel_initializer=initializer,
        name=f"{name}_head",
    )(embedding)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_cnn_embedding_model(branch: keras.Model, name: str = BRANCH_NAME) -> keras.Model:
    """Expose the 64-d embedding of a built branch, without its classification head.

    Used by the XAI modules (T3.1/T3.2) and by any later experiment that fuses embeddings rather
    than probabilities.

    Args:
        branch: a model from `build_cnn_branch`.
        name: name prefix used when the branch was built.

    Returns:
        A `keras.Model` mapping the branch input to its `EMBEDDING_DIM`-wide embedding.
    """
    return keras.Model(
        inputs=branch.input,
        outputs=branch.get_layer(f"{name}_embedding").output,
        name=f"{name}_embedding_model",
    )
