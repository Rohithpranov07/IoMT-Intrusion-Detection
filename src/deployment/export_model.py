"""Export the trained ensemble for Raspberry Pi inference (Build-Instructions T4.1; TRD.md §7).

`TRD.md §7` is explicit: training stays on Colab or a lab GPU, and **only the inference graph** goes
to the Pi. This module produces that graph — three TensorFlow Lite files plus a manifest carrying
everything a standalone script needs to reproduce the deployed decision without TensorFlow, Keras,
scikit-learn, or any of this repository's training code.

>>> THE BiLSTM DOES NOT CONVERT TO TFLITE AS TRAINED. THIS MODULE FIXES THAT. <<<
---------------------------------------------------------------------------------
Measured, not assumed (`reports/t4_1_export.md`):

    branch        as trained                                    fix applied
    cnn           converts, max output diff 2.1e-07             none needed
    transformer   converts, max output diff 6.0e-08             none needed
    bilstm        ConverterError: TensorListReserve             unroll the recurrence

Keras 3 exports a `Bidirectional(LSTM(...))` as a dynamic loop over a `TensorList`, which TFLite's
built-in op set cannot represent. Two standard escapes were tried and **both rejected on
measurement**:

  * `SELECT_TF_OPS` (the Flex delegate) converted at 718 KB but the produced model would not even
    run in this TensorFlow build's own interpreter, and it would require a Flex-enabled runtime on
    the Pi — a much heavier dependency than `tflite-runtime`.
  * Converting from a fixed-batch concrete function appeared to work and produced a 16 KB file.
    **It silently dropped the weights**: outputs were NaN and argmax agreement with the Keras model
    was 0%. It is recorded here because it is exactly the kind of export bug that ships unnoticed
    if conversion success is treated as verification.

The adopted fix is to **unroll the recurrence** (`unroll=True`) before export. This is available
only because `docs/architecture_decision.md` §1.2 fixes `SEQUENCE_LENGTH = 10`: a fixed, short
window can be unrolled into 10 explicit steps, which removes the dynamic loop entirely and converts
to pure TFLite builtins.

**Unrolling is an export-time graph transformation, not an architecture change.** The layer counts,
unit counts, and every trained weight are identical — `export_bilstm_for_tflite` transfers the
weights and asserts the Keras outputs match to 0.0 before conversion. `docs/architecture_decision.md`
§2.2 is untouched.

Exact parameters
----------------
    DEFAULT_PRECISION   = "float32"   no quantisation. float16 is offered and measured, but the
                                      default is the faithful one: T4.3 has not yet measured
                                      whether the Pi is fast enough to need the trade, and
                                      `Build-Instructions.md` §A.1 rule 3 forbids pre-emptively
                                      optimising against an unmeasured latency claim.
    PROBABILITY_TOLERANCE = {"float32": 1e-4, "float16": 5e-3}
                                      max permitted |TFLite - Keras| per output probability, BY
                                      PRECISION. A single threshold was the first attempt and it
                                      was wrong: float16 carries about three decimal digits, so its
                                      measured error (1.7e-4 to 1.1e-03) exceeded a 1e-4 bound
                                      purely by construction, and the exporter rejected a
                                      perfectly usable graph. The bound now matches the format's
                                      actual precision, so it still catches real breakage -- the
                                      discarded concrete-function export produced NaN -- without
                                      failing a format for being the format it is.
    MIN_ARGMAX_AGREEMENT = 1.0        the exported model must reproduce EVERY prediction, at BOTH
                                      precisions. This, not the probability tolerance, is the hard
                                      guarantee: a branch that disagrees on 1% of windows changes
                                      the fusion, and the deployed system would then differ from
                                      the evaluated one in a way no report would capture. float16
                                      meets it at 100% on all three branches.
    VERIFY_SAMPLE_SIZE  = 128         windows checked during verification.

Positive class: output index 1 = **Attack** (`TRD.md §2.3`). The manifest records this so the Pi
script cannot silently adopt a different convention.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    POSITIVE_LABEL,
)
from src.models import bilstm_branch
from src.models.fusion import CONFIDENCE_SHARPNESS, DEFAULT_BRANCH_PRIOR, EPSILON

logger = logging.getLogger(__name__)

DEFAULT_PRECISION: str = "float32"
#: Probability-difference bound per precision; see the module docstring for why it is not one value.
PROBABILITY_TOLERANCE: dict[str, float] = {"float32": 1e-4, "float16": 5e-3}
MIN_ARGMAX_AGREEMENT: float = 1.0
VERIFY_SAMPLE_SIZE: int = 128


@dataclass
class ExportResult:
    """Outcome of exporting and verifying one branch.

    Attributes:
        branch_name: the branch exported.
        path: the written `.tflite` file.
        size_bytes: its size on disk.
        precision: "float32" or "float16".
        max_output_difference: largest |TFLite − Keras| over the verification sample.
        argmax_agreement: fraction of verification windows where the prediction matches.
        unrolled: whether the recurrence was unrolled before conversion.
        verified: whether both tolerances were met.
    """

    branch_name: str
    path: Path
    size_bytes: int
    precision: str
    max_output_difference: float
    argmax_agreement: float
    unrolled: bool
    verified: bool

    def summary(self) -> str:
        """Render a printable line."""
        return (
            f"{self.branch_name:<12} {self.precision:<8} "
            f"{self.size_bytes / 1024:>7.0f} KB  "
            f"max diff {self.max_output_difference:.2e}  "
            f"agreement {self.argmax_agreement:.1%}  "
            f"{'unrolled  ' if self.unrolled else '          '}"
            f"{'VERIFIED' if self.verified else 'FAILED'}"
        )


@dataclass
class EnsembleExport:
    """Everything written for a deployment.

    Attributes:
        output_dir: directory containing the `.tflite` files and `manifest.json`.
        branches: per-branch export results.
        manifest_path: the written manifest.
        total_bytes: combined size of all exported graphs.
    """

    output_dir: Path
    branches: list[ExportResult] = field(default_factory=list)
    manifest_path: Path | None = None
    total_bytes: int = 0

    def all_verified(self) -> bool:
        """Return whether every branch reproduced its Keras predictions exactly."""
        return bool(self.branches) and all(b.verified for b in self.branches)

    def summary(self) -> str:
        """Render the full export report."""
        lines = [f"Exported to {self.output_dir}"]
        lines += [f"  {b.summary()}" for b in self.branches]
        lines.append(f"  total {self.total_bytes / 1024:.0f} KB")
        lines.append(
            f"  T4.1 VERIFY: {'PASS' if self.all_verified() else 'FAIL'}"
        )
        return "\n".join(lines)


def export_bilstm_for_tflite(trained: keras.Model) -> keras.Model:
    """Rebuild the BiLSTM with `unroll=True` and transfer the trained weights.

    Removes the dynamic `TensorList` loop that blocks TFLite conversion. Valid because
    `SEQUENCE_LENGTH` is fixed at 10 (`docs/architecture_decision.md` §1.2); an unrolled graph is
    only expressible for a known, short sequence length.

    Args:
        trained: the trained BiLSTM branch.

    Returns:
        An architecturally identical, unrolled model carrying the same weights.

    Raises:
        ValueError: if the transferred model does not reproduce the original's outputs exactly.
    """
    sequence_length, n_features = trained.input_shape[1], trained.input_shape[2]
    name = bilstm_branch.BRANCH_NAME

    inputs = keras.Input(shape=(sequence_length, n_features), name=f"{name}_input")
    x = inputs
    for index in range(1, bilstm_branch.LSTM_LAYER_COUNT + 1):
        is_last = index == bilstm_branch.LSTM_LAYER_COUNT
        x = layers.Bidirectional(
            layers.LSTM(
                units=bilstm_branch.LSTM_UNITS,
                return_sequences=not is_last,
                recurrent_dropout=bilstm_branch.RECURRENT_DROPOUT,
                unroll=True,  # the whole point of this function
                name=f"{name}_lstm{index}",
            ),
            merge_mode=bilstm_branch.MERGE_MODE,
            name=f"{name}_bilstm{index}",
        )(x)
        x = layers.Dropout(bilstm_branch.DROPOUT_RATE, name=f"{name}_dropout{index}")(x)

    embedding = layers.Dense(
        bilstm_branch.EMBEDDING_DIM, activation="relu", name=f"{name}_embedding"
    )(x)
    outputs = layers.Dense(
        bilstm_branch.HEAD_UNITS, activation="softmax", name=f"{name}_head"
    )(embedding)

    unrolled = keras.Model(inputs=inputs, outputs=outputs, name=name)
    unrolled.set_weights(trained.get_weights())

    # Unrolling must be mathematically identical, not merely similar.
    probe = np.random.default_rng(0).random(
        (8, sequence_length, n_features)
    ).astype(np.float32)
    difference = float(
        np.abs(unrolled.predict(probe, verbose=0) - trained.predict(probe, verbose=0)).max()
    )
    if difference != 0.0:
        raise ValueError(
            f"Unrolled BiLSTM differs from the trained model by {difference:.3e}; "
            "weight transfer is wrong."
        )
    logger.info("BiLSTM unrolled for TFLite export; outputs identical to the trained model")
    return unrolled


def _convert(model: keras.Model, precision: str) -> bytes:
    """Convert a Keras model to a TFLite flatbuffer via the SavedModel path.

    The SavedModel route is used rather than `from_keras_model` or a concrete function: the
    concrete-function route silently produced a 16 KB weightless model with NaN outputs (see the
    module docstring), which conversion success alone would not have revealed.

    Args:
        model: the model to convert.
        precision: "float32" or "float16".

    Returns:
        The TFLite flatbuffer.

    Raises:
        ValueError: on an unknown precision.
    """
    if precision not in ("float32", "float16"):
        raise ValueError(f"precision must be 'float32' or 'float16'; got {precision!r}")

    with tempfile.TemporaryDirectory() as staging:
        # TensorFlow's exporter is extremely chatty and writes progress to stdout/stderr.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.export(staging)
            converter = tf.lite.TFLiteConverter.from_saved_model(staging)
            if precision == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            return converter.convert()


def verify_tflite(
    blob: bytes, reference: keras.Model, sample: np.ndarray
) -> tuple[float, float]:
    """Run the exported graph and compare it against the Keras model.

    This is T4.1's VERIFY block. Conversion succeeding is **not** verification: an earlier attempt
    converted cleanly and produced NaN outputs.

    Args:
        blob: the TFLite flatbuffer.
        reference: the Keras model it was exported from.
        sample: windows to compare on, `(n, sequence_length, n_features)`.

    Returns:
        Tuple of `(max_output_difference, argmax_agreement)`.
    """
    interpreter = tf.lite.Interpreter(model_content=blob)
    input_detail = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(
        input_detail["index"], [1, sample.shape[1], sample.shape[2]]
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    predictions = []
    for window in sample:
        interpreter.set_tensor(input_detail["index"], window[np.newaxis].astype(np.float32))
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_detail["index"])[0])

    tflite_output = np.asarray(predictions)
    keras_output = reference.predict(sample, verbose=0)
    return (
        float(np.abs(tflite_output - keras_output).max()),
        float((tflite_output.argmax(axis=1) == keras_output.argmax(axis=1)).mean()),
    )


def export_ensemble(
    models: dict[str, keras.Model],
    output_dir: str | Path,
    verification_sample: np.ndarray,
    feature_names: list[str],
    precision: str = DEFAULT_PRECISION,
) -> EnsembleExport:
    """Export every branch to TFLite, verify each, and write the deployment manifest.

    Args:
        models: branch name to trained Keras model.
        output_dir: directory to write into; created if absent.
        verification_sample: real windows to verify against, `(n, sequence_length, n_features)`.
        feature_names: selected feature names, in input-column order. Recorded in the manifest so
            the Pi script builds its input in the same order the model was trained on — a silent
            column reordering would be undetectable at inference time.
        precision: "float32" (default) or "float16".

    Returns:
        An `EnsembleExport`.

    Raises:
        ValueError: if any branch fails verification. A deployment that does not reproduce the
            evaluated model must not be written out as if it were usable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = np.asarray(verification_sample, dtype=np.float32)[:VERIFY_SAMPLE_SIZE]
    export = EnsembleExport(output_dir=output_dir)

    for name, model in models.items():
        unrolled = name == bilstm_branch.BRANCH_NAME.replace("_branch", "") or "bilstm" in name
        to_convert = export_bilstm_for_tflite(model) if unrolled else model

        blob = _convert(to_convert, precision)
        path = output_dir / f"{name}.tflite"
        path.write_bytes(blob)

        difference, agreement = verify_tflite(blob, model, sample)
        result = ExportResult(
            branch_name=name,
            path=path,
            size_bytes=len(blob),
            precision=precision,
            max_output_difference=difference,
            argmax_agreement=agreement,
            unrolled=unrolled,
            verified=(
                difference <= PROBABILITY_TOLERANCE[precision]
                and agreement >= MIN_ARGMAX_AGREEMENT
            ),
        )
        export.branches.append(result)
        export.total_bytes += result.size_bytes
        logger.info("%s", result.summary())

    failed = [b.branch_name for b in export.branches if not b.verified]
    if failed:
        raise ValueError(
            f"Export verification failed for {failed}. The exported graph does not reproduce the "
            "evaluated model, so it must not be deployed."
        )

    manifest = {
        "sequence_length": int(sample.shape[1]),
        "n_features": int(sample.shape[2]),
        "feature_names": list(feature_names),
        "branches": [
            {
                "name": b.branch_name,
                "file": b.path.name,
                "size_bytes": b.size_bytes,
                "unrolled": b.unrolled,
            }
            for b in export.branches
        ],
        "precision": precision,
        "fusion": {
            "rule": "confidence_weighted",
            "formula": "w_b = alpha_b * (max_k p_b[k]) ** gamma / sum_b'; P = sum_b w_b * p_b",
            "gamma": CONFIDENCE_SHARPNESS,
            "branch_priors": {b.branch_name: DEFAULT_BRANCH_PRIOR for b in export.branches},
            "epsilon": EPSILON,
        },
        "classes": {
            "0": NEGATIVE_CLASS_NAME,
            "1": POSITIVE_CLASS_NAME,
            "positive_label": POSITIVE_LABEL,
            "note": (
                "Attack is the positive class (TRD.md §2.3), deliberately the opposite of the "
                "base paper's convention. Any metric computed on the Pi must state this."
            ),
        },
        "verification": {
            "sample_size": int(len(sample)),
            "probability_tolerance": PROBABILITY_TOLERANCE[precision],
            "min_argmax_agreement": MIN_ARGMAX_AGREEMENT,
            "results": {
                b.branch_name: {
                    "max_output_difference": b.max_output_difference,
                    "argmax_agreement": b.argmax_agreement,
                }
                for b in export.branches
            },
        },
    }
    export.manifest_path = output_dir / "manifest.json"
    export.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("%s", export.summary())
    return export
