"""BiLSTM branch of the ensemble (Build-Instructions T2.3; TRD.md §3.2.2).

>>> THIS MODULE IS THE DIRECT, LITERAL FIX FOR OBJECTION #2 <<<
--------------------------------------------------------------
`PRD.md §2.1.2`: the base paper's Table 7 lists three Conv1D layers, ReLU, Adam, dropout, epochs
and batch size — and states **nothing whatsoever** about its LSTM: no unit count, no layer count.
Its recurrent component is therefore unreproducible. `TRD.md §3.2.2` requires this project to
document both numbers explicitly, and `Build-Instructions.md` §A.1 forbids leaving them vague
anywhere in this repo.

    >>> LAYER COUNT = 2 <<<
    >>> UNITS = 64 PER DIRECTION (128 concatenated per timestep) <<<

Those two numbers are the content of the fix. They are stated here, in
`docs/architecture_decision.md` §2.2, and in `reports/phase2_results.md`. No description of this
branch anywhere in the project may say "a couple of layers", "several units", or similar.

EXACT ARCHITECTURE (frozen in `docs/architecture_decision.md` §2.2)
-------------------------------------------------------------------
    Input                (batch, 10, 62)   = (batch, SEQUENCE_LENGTH, n_features)

    LSTM_LAYER_COUNT   = 2       Bidirectional LSTM layers
    LSTM_UNITS         = 64      units PER DIRECTION, both layers
                                 -> 128-d concatenated state per timestep
    MERGE_MODE         = "concat"  forward and backward states are concatenated, not summed
    Layer 1            return_sequences=True   (feeds layer 2)
    Layer 2            return_sequences=False  (emits one vector per window)
    DROPOUT_RATE       = 0.3     applied after each BiLSTM layer
    RECURRENT_DROPOUT  = 0.0     DELIBERATELY ZERO: any non-zero value disables the cuDNN fast
                                 path and blocks TFLite conversion for the Raspberry Pi
                                 deployment (T4.1)
    EMBEDDING_DIM      = 64      Dense(64, relu) projects the 128-d state down to the shared
                                 embedding width used by all three branches
    HEAD_UNITS         = 2       Dense classification head, softmax

Why these values: 2 layers is the shallowest depth that is meaningfully deep while remaining
trainable on 10-timestep windows — a third layer over 10 timesteps overfits with no
receptive-field gain. 64 units per direction gives a 128-d recurrent state, matching the CNN
branch's widest layer so neither branch is capacity-starved relative to the other.

Positive class: output index 1 is **Attack**, the POSITIVE class (`TRD.md §2.3`).
Determinism: weight initialisation is seeded via `seed`/`RANDOM_STATE`.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.config import RANDOM_STATE

# --- Frozen architecture constants (docs/architecture_decision.md §2.2) --------------------
#: Number of Bidirectional LSTM layers. The base paper states no equivalent number (Objection #2).
LSTM_LAYER_COUNT: int = 2
#: Units PER DIRECTION in every BiLSTM layer. The base paper states no equivalent number.
LSTM_UNITS: int = 64
MERGE_MODE: str = "concat"
DROPOUT_RATE: float = 0.3
#: Must stay 0.0 — see the module docstring's TFLite/cuDNN note.
RECURRENT_DROPOUT: float = 0.0
EMBEDDING_DIM: int = 64
HEAD_UNITS: int = 2
BRANCH_NAME: str = "bilstm_branch"

assert LSTM_LAYER_COUNT >= 2, (
    "docs/architecture_decision.md §2.2 fixes the BiLSTM at 2 layers; a 1-layer configuration "
    "would need that document updated first."
)
assert RECURRENT_DROPOUT == 0.0, (
    "Non-zero recurrent_dropout disables cuDNN and blocks the TFLite export path needed for "
    "the Raspberry Pi deployment (T4.1)."
)


def build_bilstm_branch(
    sequence_length: int,
    n_features: int,
    n_classes: int = HEAD_UNITS,
    lstm_units: int = LSTM_UNITS,
    layer_count: int = LSTM_LAYER_COUNT,
    dropout_rate: float = DROPOUT_RATE,
    seed: int = RANDOM_STATE,
    name: str = BRANCH_NAME,
) -> keras.Model:
    """Build the BiLSTM branch: **2 layers, 64 units per direction** (§2.2).

    Args:
        sequence_length: window length. Must be > 1 — a length-1 sequence gives a bidirectional
            LSTM nothing to be bidirectional *over*, which is the heart of Objection #2.
        n_features: features per timestep (62 for IoTID20).
        n_classes: output units. 2 for the binary task; index 1 = Attack = positive.
        lstm_units: units per direction. Defaults to the frozen 64.
        layer_count: number of BiLSTM layers. Defaults to the frozen 2.
        dropout_rate: dropout after each BiLSTM layer.
        seed: weight-initialisation seed.
        name: model name.

    Returns:
        An uncompiled `keras.Model` mapping `(batch, sequence_length, n_features)` to
        `(batch, n_classes)` softmax probabilities.

    Raises:
        ValueError: if `sequence_length` < 2 or `layer_count` < 1.
    """
    if sequence_length < 2:
        raise ValueError(
            f"sequence_length={sequence_length} gives the BiLSTM a single timestep, which is "
            "Objection #2 (PRD.md §2.1.2) exactly. TRD.md §9 requires > 1."
        )
    if layer_count < 1:
        raise ValueError(f"layer_count must be >= 1, got {layer_count}")

    initializer = keras.initializers.GlorotUniform(seed=seed)

    inputs = keras.Input(shape=(sequence_length, n_features), name=f"{name}_input")
    x = inputs
    for index in range(1, layer_count + 1):
        is_last = index == layer_count
        x = layers.Bidirectional(
            layers.LSTM(
                units=lstm_units,
                return_sequences=not is_last,
                recurrent_dropout=RECURRENT_DROPOUT,
                kernel_initializer=initializer,
                name=f"{name}_lstm{index}",
            ),
            merge_mode=MERGE_MODE,
            name=f"{name}_bilstm{index}",
        )(x)
        x = layers.Dropout(dropout_rate, seed=seed, name=f"{name}_dropout{index}")(x)

    embedding = layers.Dense(
        EMBEDDING_DIM,
        activation="relu",
        kernel_initializer=initializer,
        name=f"{name}_embedding",
    )(x)
    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        kernel_initializer=initializer,
        name=f"{name}_head",
    )(embedding)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def describe_architecture() -> str:
    """Return the branch's exact layer and unit counts as plain text.

    Exists so the Review 2 report, the notebooks, and any slide can quote these numbers from a
    single source rather than restating them by hand — restating by hand is how the base paper's
    Table 7 ended up incomplete.

    Returns:
        A human-readable specification string.
    """
    return (
        f"BiLSTM branch: {LSTM_LAYER_COUNT} Bidirectional LSTM layers, "
        f"{LSTM_UNITS} units per direction "
        f"({2 * LSTM_UNITS}-d concatenated state), merge_mode={MERGE_MODE!r}, "
        f"dropout {DROPOUT_RATE}, recurrent_dropout {RECURRENT_DROPOUT}, "
        f"projected to a {EMBEDDING_DIM}-d embedding, {HEAD_UNITS}-unit softmax head. "
        "The base paper states neither its LSTM layer count nor its unit count (Objection #2, "
        "PRD.md §2.1.2)."
    )
