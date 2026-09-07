"""Transformer-attention branch of the ensemble (Build-Instructions T2.4; TRD.md §3.2.3).

Role: long-range dependency modelling across the constructed sequence — every timestep attends to
every other in one hop, which neither the CNN's local kernels nor the BiLSTM's sequential state
provides directly.

EXACT ARCHITECTURE (frozen in `docs/architecture_decision.md` §2.3)
-------------------------------------------------------------------
    Input                (batch, 10, 62)   = (batch, SEQUENCE_LENGTH, n_features)

    EMBED_DIM          = 64      d_model. Dense linear projection 62 -> 64, then positional
                                 encoding is added
    POSITIONAL_ENCODING = fixed sinusoidal (NOT learned) — no extra parameters, and it
                                 generalises to windows shorter than 10 after pre-padding
    NUM_ENCODER_LAYERS = 2       encoder blocks
    NUM_ATTENTION_HEADS = 4      heads per MultiHeadAttention
    KEY_DIM            = 16      per-head key/query width; 4 x 16 = 64 = EMBED_DIM, the standard
                                 head-splitting relation
    FFN_DIM            = 128     = 2 x EMBED_DIM; Dense(128, relu) -> Dense(64)
    DROPOUT_RATE       = 0.1     deliberately below the CNN/BiLSTM's 0.3: these encoder blocks are
                                 shallow and residual-connected, and 0.3 destabilises them
    LAYER_NORM_EPSILON = 1e-6
    POOLING            = GlobalAveragePooling1D   average, NOT max (unlike the CNN branch), so
                                 this branch is deliberately biased toward window-wide evidence.
                                 Ensemble diversity is the point of running three branches.
    HEAD_UNITS         = 2       Dense classification head, softmax

Each encoder block is:
    x -> MultiHeadAttention(4 heads, key_dim 16) -> Dropout -> +residual -> LayerNorm
      -> Dense(128, relu) -> Dense(64) -> Dropout -> +residual -> LayerNorm

Why 2 layers: with only 10 timesteps, one attention layer already reaches every position; a second
allows composition of attended features. Beyond that, depth buys nothing at this window length and
costs Raspberry Pi inference time (T4.3).

Positive class: output index 1 is **Attack**, the POSITIVE class (`TRD.md §2.3`).
Determinism: weight initialisation is seeded via `seed`/`RANDOM_STATE`; the positional encoding is
a fixed deterministic function of position.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import RANDOM_STATE

# --- Frozen architecture constants (docs/architecture_decision.md §2.3) --------------------
EMBED_DIM: int = 64
NUM_ENCODER_LAYERS: int = 2
NUM_ATTENTION_HEADS: int = 4
KEY_DIM: int = 16
FFN_DIM: int = 128
DROPOUT_RATE: float = 0.1
LAYER_NORM_EPSILON: float = 1e-6
HEAD_UNITS: int = 2
BRANCH_NAME: str = "transformer_branch"

assert NUM_ATTENTION_HEADS * KEY_DIM == EMBED_DIM, (
    f"{NUM_ATTENTION_HEADS} heads x key_dim {KEY_DIM} must equal EMBED_DIM {EMBED_DIM} "
    "(docs/architecture_decision.md §2.3's head-splitting relation)."
)
assert FFN_DIM == 2 * EMBED_DIM, (
    "docs/architecture_decision.md §2.3 fixes the feed-forward expansion at 2x EMBED_DIM."
)


def sinusoidal_positional_encoding(sequence_length: int, embed_dim: int) -> np.ndarray:
    """Build the fixed sinusoidal positional encoding of Vaswani et al. (2017).

    Even dimensions use sine, odd dimensions cosine, with geometrically increasing wavelengths.
    Deterministic and parameter-free, so it adds nothing to the model size that must be exported
    to the Raspberry Pi (T4.1).

    Args:
        sequence_length: number of positions to encode.
        embed_dim: encoding width; must match `EMBED_DIM`.

    Returns:
        Array of shape `(sequence_length, embed_dim)`.

    Raises:
        ValueError: if `embed_dim` is not even.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for sine/cosine pairing, got {embed_dim}")

    positions = np.arange(sequence_length, dtype=np.float32)[:, np.newaxis]
    dimension_pairs = np.arange(0, embed_dim, 2, dtype=np.float32)
    angle_rates = 1.0 / np.power(10_000.0, dimension_pairs / embed_dim)

    encoding = np.zeros((sequence_length, embed_dim), dtype=np.float32)
    encoding[:, 0::2] = np.sin(positions * angle_rates)
    encoding[:, 1::2] = np.cos(positions * angle_rates)
    return encoding


@keras.utils.register_keras_serializable(package="iomt_ids")
class AddPositionalEncoding(layers.Layer):
    """Add a fixed sinusoidal positional encoding to the projected input.

    A layer rather than a lambda so the model serialises cleanly. Registered as serializable so a
    saved branch reloads without the caller having to pass `custom_objects` -- needed by the XAI
    modules (T3.1/T3.2), which load the trained ensemble, and by the TFLite export (T4.1).
    """

    def __init__(self, sequence_length: int, embed_dim: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.encoding = tf.constant(
            sinusoidal_positional_encoding(sequence_length, embed_dim), dtype=tf.float32
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Add the encoding, broadcasting over the batch dimension."""
        return inputs + self.encoding

    def get_config(self) -> dict[str, object]:
        """Return the config needed to reconstruct this layer on load."""
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "embed_dim": self.embed_dim})
        return config


def _encoder_block(
    x: tf.Tensor,
    index: int,
    dropout_rate: float,
    seed: int,
    name: str,
) -> tf.Tensor:
    """Apply one pre-residual encoder block: attention, then feed-forward.

    Args:
        x: input tensor of shape `(batch, sequence_length, EMBED_DIM)`.
        index: 1-based block index, used in layer names.
        dropout_rate: dropout applied to both sub-layers.
        seed: dropout seed.
        name: branch name prefix.

    Returns:
        Tensor of the same shape as `x`.
    """
    attention = layers.MultiHeadAttention(
        num_heads=NUM_ATTENTION_HEADS,
        key_dim=KEY_DIM,
        dropout=dropout_rate,
        name=f"{name}_mha{index}",
    )(x, x)
    attention = layers.Dropout(dropout_rate, seed=seed, name=f"{name}_attn_drop{index}")(attention)
    x = layers.LayerNormalization(
        epsilon=LAYER_NORM_EPSILON, name=f"{name}_attn_norm{index}"
    )(layers.Add(name=f"{name}_attn_residual{index}")([x, attention]))

    ffn = layers.Dense(FFN_DIM, activation="relu", name=f"{name}_ffn{index}_expand")(x)
    ffn = layers.Dense(EMBED_DIM, name=f"{name}_ffn{index}_project")(ffn)
    ffn = layers.Dropout(dropout_rate, seed=seed, name=f"{name}_ffn_drop{index}")(ffn)
    return layers.LayerNormalization(
        epsilon=LAYER_NORM_EPSILON, name=f"{name}_ffn_norm{index}"
    )(layers.Add(name=f"{name}_ffn_residual{index}")([x, ffn]))


def build_transformer_branch(
    sequence_length: int,
    n_features: int,
    n_classes: int = HEAD_UNITS,
    num_layers: int = NUM_ENCODER_LAYERS,
    num_heads: int = NUM_ATTENTION_HEADS,
    embed_dim: int = EMBED_DIM,
    dropout_rate: float = DROPOUT_RATE,
    seed: int = RANDOM_STATE,
    name: str = BRANCH_NAME,
) -> keras.Model:
    """Build the Transformer branch: **2 encoder layers, 4 heads, embedding dim 64** (§2.3).

    Args:
        sequence_length: window length. Must be > 1 — self-attention over one timestep is the
            identity, which is Objection #2 again.
        n_features: features per timestep (62 for IoTID20).
        n_classes: output units. 2 for the binary task; index 1 = Attack = positive.
        num_layers: encoder blocks. Defaults to the frozen 2.
        num_heads: attention heads. Defaults to the frozen 4.
        embed_dim: model width. Defaults to the frozen 64.
        dropout_rate: dropout inside each encoder block.
        seed: weight-initialisation seed.
        name: model name.

    Returns:
        An uncompiled `keras.Model` mapping `(batch, sequence_length, n_features)` to
        `(batch, n_classes)` softmax probabilities.

    Raises:
        ValueError: if `sequence_length` < 2, or if `embed_dim` is not divisible by `num_heads`.
    """
    if sequence_length < 2:
        raise ValueError(
            f"sequence_length={sequence_length} makes self-attention degenerate; "
            "TRD.md §9 requires > 1 (PRD.md §2.1.2, Objection #2)."
        )
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        )

    inputs = keras.Input(shape=(sequence_length, n_features), name=f"{name}_input")
    x = layers.Dense(
        embed_dim,
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
        name=f"{name}_project",
    )(inputs)
    x = AddPositionalEncoding(sequence_length, embed_dim, name=f"{name}_positional")(x)

    for index in range(1, num_layers + 1):
        x = _encoder_block(x, index, dropout_rate, seed, name)

    embedding = layers.GlobalAveragePooling1D(name=f"{name}_embedding")(x)
    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
        name=f"{name}_head",
    )(embedding)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def describe_architecture() -> str:
    """Return the branch's exact head, layer, and dimension counts as plain text.

    Returns:
        A human-readable specification string, quotable directly into reports and slides.
    """
    return (
        f"Transformer branch: {NUM_ENCODER_LAYERS} encoder layers, "
        f"{NUM_ATTENTION_HEADS} attention heads, key_dim {KEY_DIM}, "
        f"embedding dimension {EMBED_DIM}, feed-forward dimension {FFN_DIM}, "
        f"dropout {DROPOUT_RATE}, fixed sinusoidal positional encoding, "
        f"GlobalAveragePooling1D, {HEAD_UNITS}-unit softmax head."
    )
