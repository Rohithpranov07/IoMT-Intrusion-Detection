"""Standalone ensemble inference for the Raspberry Pi 4B (Build-Instructions T4.2; TRD.md §7).

>>> THIS FILE MUST RUN WITH NOTHING BUT NUMPY AND A TFLITE INTERPRETER. <<<
---------------------------------------------------------------------------
It deliberately imports **nothing from `src/`** — not `config`, not `fusion`, not `metrics`. The
whole point of `TRD.md §7` is that only the inference graph reaches the Pi, so this script is copied
to the device alongside `artifacts/deployment/<precision>/` and nothing else. Any import from this
repository would quietly reintroduce TensorFlow, scikit-learn, pandas, and the training pipeline as
deployment dependencies.

That constraint is enforced by a test (`tests/test_pi_inference.py::
test_pi_inference_imports_nothing_from_the_repo`), not left to discipline.

Everything the script needs comes from `manifest.json`, written by T4.1: the branch files, the
feature order, the sequence length, the fusion formula and its constants, and the positive-class
convention. The fusion is pure arithmetic and is reimplemented here in NumPy — see
`confidence_weighted_fusion` below, which mirrors `src/models/fusion.py` exactly and is checked
against it by test.

RUNTIME
-------
Prefers `tflite_runtime.interpreter` (the ~2 MB standalone wheel, which is what should be installed
on the Pi) and falls back to `tensorflow.lite` so the script is testable on a development machine.
Which one was used is reported in the output, because a benchmark taken through the TensorFlow
fallback is NOT a `tflite-runtime` measurement and must never be quoted as one.

HARDWARE HONESTY
----------------
`describe_host()` reads the real platform — CPU model, core count, RAM, OS — from the machine it is
running on, and `is_raspberry_pi()` reports whether that machine is actually a Pi. Nothing here
guesses or accepts a hardware string from a config file. `Build-Instructions.md` §A.1 rule 3 forbids
any latency claim not traceable to a real run on the actual Pi 4B, and T4.3 relies on these
functions to refuse to write a deployment report on the wrong hardware.

Usage on the Pi:
    python3 pi_inference.py --model-dir deployment/float32 --self-test
    python3 pi_inference.py --model-dir deployment/float32 --input windows.npy
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Set by `_load_interpreter_class` to "tflite_runtime" or "tensorflow".
INTERPRETER_BACKEND: str = "unknown"


def _load_interpreter_class():
    """Return a TFLite `Interpreter` class, preferring the standalone runtime.

    Returns:
        The `Interpreter` class.

    Raises:
        ImportError: if neither `tflite_runtime` nor `tensorflow` is available.
    """
    global INTERPRETER_BACKEND

    # 1. The standalone runtime -- what should be installed on the Pi (~2 MB, no TensorFlow).
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        INTERPRETER_BACKEND = "tflite_runtime"
        return Interpreter
    except ImportError:
        pass

    # 2. LiteRT, the renamed successor to tflite-runtime, for newer Pi images.
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        INTERPRETER_BACKEND = "ai_edge_litert"
        return Interpreter
    except ImportError:
        pass

    # 3. Full TensorFlow, so this script is runnable on a development machine. `tf.lite.Interpreter`
    #    is an ATTRIBUTE of the lazily-loaded `tf.lite` module, not an importable submodule symbol:
    #    `from tensorflow.lite import Interpreter` raises ImportError on TF 2.21 and would have
    #    made the fallback dead code.
    try:
        import tensorflow as tf  # type: ignore

        INTERPRETER_BACKEND = "tensorflow"
        return tf.lite.Interpreter
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "No TFLite interpreter found. On the Raspberry Pi install the standalone runtime:\n"
            "    pip install tflite-runtime\n"
            "On a development machine, TensorFlow provides a fallback."
        ) from error


# --- Hardware identification ------------------------------------------------------------------


@dataclass
class HostInfo:
    """What machine this actually is.

    Attributes:
        model: the board or machine model, read from the device tree on a Pi.
        machine: the CPU architecture, e.g. "aarch64".
        processor_count: logical cores.
        total_ram_gb: total physical RAM.
        os_description: kernel and release.
        python_version: interpreter version.
        is_raspberry_pi: whether `model` identifies a Raspberry Pi.
    """

    model: str
    machine: str
    processor_count: int
    total_ram_gb: float
    os_description: str
    python_version: str
    is_raspberry_pi: bool

    def summary(self) -> str:
        """Render the hardware block that must accompany any timing number."""
        return (
            f"Host: {self.model}\n"
            f"  architecture : {self.machine}\n"
            f"  cores        : {self.processor_count}\n"
            f"  RAM          : {self.total_ram_gb:.1f} GB\n"
            f"  OS           : {self.os_description}\n"
            f"  Python       : {self.python_version}\n"
            f"  Raspberry Pi : {self.is_raspberry_pi}"
        )


def _read_pi_model() -> str | None:
    """Return the Raspberry Pi model string from the device tree, if present.

    Returns:
        The model string, or None when not running on a Pi.
    """
    for candidate in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            if candidate.exists():
                return candidate.read_text(errors="ignore").strip("\x00").strip()
        except OSError:
            continue
    return None


def _total_ram_gb() -> float:
    """Return total physical RAM in GB, or 0.0 if it cannot be determined."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo.read_text(errors="ignore"))
        if match:
            return int(match.group(1)) / (1024 * 1024)
    try:  # macOS / BSD development machines
        output = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if output.returncode == 0:
            return int(output.stdout.strip()) / (1024 ** 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0.0


def describe_host() -> HostInfo:
    """Identify the machine this is actually running on.

    Read from the OS, never from configuration. A deployment report's hardware line has to be a
    measurement like any other (`Build-Instructions.md` §A.1 rule 3).

    Returns:
        A `HostInfo`.
    """
    pi_model = _read_pi_model()
    return HostInfo(
        model=pi_model or f"{platform.system()} {platform.machine()}",
        machine=platform.machine(),
        processor_count=len(getattr(__import__("os"), "sched_getaffinity", lambda _: [])(0))
        or (__import__("os").cpu_count() or 0),
        total_ram_gb=_total_ram_gb(),
        os_description=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        is_raspberry_pi=bool(pi_model and "raspberry pi" in pi_model.lower()),
    )


def is_raspberry_pi() -> bool:
    """Return whether this machine is a Raspberry Pi.

    Returns:
        True only when the device tree identifies a Raspberry Pi.
    """
    return describe_host().is_raspberry_pi


# --- The ensemble ------------------------------------------------------------------------------


def confidence_weighted_fusion(
    branch_probabilities: list[np.ndarray],
    priors: list[float],
    gamma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse branch outputs. Mirrors `src/models/fusion.py` exactly, in plain NumPy.

    Reimplemented rather than imported so this file has no dependency on the training repository.
    `tests/test_pi_inference.py` asserts the two produce identical results, so the duplication
    cannot drift silently.

        c_b = max_k p_b[k];  u_b = alpha_b * c_b ** gamma;  w_b = u_b / sum(u);  P = sum(w_b * p_b)

    Args:
        branch_probabilities: one `(n, 2)` array per branch.
        priors: per-branch alpha, in the same order.
        gamma: confidence sharpness.
        epsilon: denominator guard.

    Returns:
        Tuple of `(fused_probabilities, predictions, branch_weights)`. Prediction 1 = Attack.
    """
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in branch_probabilities])
    confidences = stacked.max(axis=2)
    unnormalised = np.asarray(priors, dtype=np.float64)[:, None] * np.power(confidences, gamma)
    weights = unnormalised / (unnormalised.sum(axis=0, keepdims=True) + epsilon)
    fused = (weights[:, :, None] * stacked).sum(axis=0)
    return fused, np.argmax(fused, axis=1).astype(np.int8), weights.T


class EnsembleRunner:
    """Loads the exported branches and reproduces the deployed decision.

    Attributes:
        manifest: the parsed `manifest.json`.
        branch_names: branch order, as exported.
        sequence_length: expected timesteps per window.
        n_features: expected features per timestep.
        feature_names: expected feature order.
        backend: which interpreter is in use.
    """

    def __init__(self, model_dir: str | Path) -> None:
        """Load every branch described by the manifest.

        Args:
            model_dir: directory holding the `.tflite` files and `manifest.json`.

        Raises:
            FileNotFoundError: if the manifest or any branch file is missing.
        """
        self.model_dir = Path(model_dir)
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Copy the whole exported directory to the device."
            )

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.sequence_length = int(self.manifest["sequence_length"])
        self.n_features = int(self.manifest["n_features"])
        self.feature_names = list(self.manifest["feature_names"])
        self.branch_names = [b["name"] for b in self.manifest["branches"]]

        interpreter_class = _load_interpreter_class()
        self.backend = INTERPRETER_BACKEND
        self._interpreters = {}
        for branch in self.manifest["branches"]:
            path = self.model_dir / branch["file"]
            if not path.exists():
                raise FileNotFoundError(f"branch graph {path} is missing")
            interpreter = interpreter_class(model_path=str(path))
            interpreter.resize_tensor_input(
                interpreter.get_input_details()[0]["index"],
                [1, self.sequence_length, self.n_features],
            )
            interpreter.allocate_tensors()
            self._interpreters[branch["name"]] = interpreter

    def predict_branch(self, name: str, window: np.ndarray) -> np.ndarray:
        """Run one branch on one window.

        Args:
            name: branch name.
            window: `(sequence_length, n_features)`.

        Returns:
            `(2,)` class probabilities.
        """
        interpreter = self._interpreters[name]
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        interpreter.set_tensor(
            input_detail["index"], window[np.newaxis].astype(np.float32)
        )
        interpreter.invoke()
        return interpreter.get_tensor(output_detail["index"])[0]

    def predict(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        """Run the full ensemble.

        Args:
            windows: `(n, sequence_length, n_features)`, or a single window.

        Returns:
            Dict with `probabilities`, `predictions`, `confidence`, and `branch_weights`.

        Raises:
            ValueError: if the input shape does not match the manifest.
        """
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim == 2:
            windows = windows[np.newaxis, ...]
        if windows.shape[1:] != (self.sequence_length, self.n_features):
            raise ValueError(
                f"expected windows of shape (n, {self.sequence_length}, {self.n_features}); "
                f"got {windows.shape}. Feature order must match manifest['feature_names']."
            )

        per_branch = [
            np.stack([self.predict_branch(name, w) for w in windows])
            for name in self.branch_names
        ]
        fusion = self.manifest["fusion"]
        probabilities, predictions, weights = confidence_weighted_fusion(
            per_branch,
            [fusion["branch_priors"][name] for name in self.branch_names],
            float(fusion["gamma"]),
            float(fusion["epsilon"]),
        )
        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "confidence": probabilities.max(axis=1),
            "branch_weights": weights,
            "branch_probabilities": {
                name: per_branch[i] for i, name in enumerate(self.branch_names)
            },
        }

    def positive_class_name(self) -> str:
        """Return the positive class recorded in the manifest (`TRD.md §2.3`)."""
        return self.manifest["classes"]["1"]

    def self_test(self, n: int = 8, seed: int = 42) -> bool:
        """Run deterministic synthetic windows through the ensemble as a smoke check.

        Confirms the graphs load, accept the manifest's shape, and produce valid distributions.
        It does **not** check accuracy — that needs real labelled windows, which is what
        `--input` and `--expected` are for.

        Args:
            n: windows to generate.
            seed: RNG seed, so the check is reproducible.

        Returns:
            True if every output is a valid probability distribution.
        """
        rng = np.random.default_rng(seed)
        windows = rng.random((n, self.sequence_length, self.n_features)).astype(np.float32)
        result = self.predict(windows)

        probabilities = result["probabilities"]
        ok = (
            probabilities.shape == (n, 2)
            and np.all(np.isfinite(probabilities))
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
            and np.allclose(result["branch_weights"].sum(axis=1), 1.0, atol=1e-6)
        )
        print(f"Self-test on {n} synthetic windows: {'PASS' if ok else 'FAIL'}")
        print(f"  backend            : {self.backend}")
        print(f"  branches           : {', '.join(self.branch_names)}")
        print(f"  input shape        : (n, {self.sequence_length}, {self.n_features})")
        print(f"  positive class     : {self.positive_class_name()} (label 1)")
        print(f"  mean confidence    : {result['confidence'].mean():.4f}")
        return bool(ok)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: argument list, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit code; 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", required=True, help="exported deployment directory")
    parser.add_argument("--input", help=".npy of shape (n, sequence_length, n_features)")
    parser.add_argument("--expected", help="optional .npy of true labels, to report agreement")
    parser.add_argument("--self-test", action="store_true", help="run a synthetic smoke check")
    parser.add_argument("--limit", type=int, default=0, help="cap windows processed")
    arguments = parser.parse_args(argv)

    host = describe_host()
    print(host.summary())
    print()

    runner = EnsembleRunner(arguments.model_dir)

    if arguments.self_test:
        return 0 if runner.self_test() else 1

    if not arguments.input:
        parser.error("provide --input or --self-test")

    windows = np.load(arguments.input)
    if arguments.limit:
        windows = windows[: arguments.limit]

    result = runner.predict(windows)
    attack = int((result["predictions"] == 1).sum())
    print(f"Processed {len(windows):,} windows with the {runner.backend} backend")
    print(f"  flagged as {runner.positive_class_name()}: {attack:,} ({attack / len(windows):.1%})")
    print(f"  mean confidence: {result['confidence'].mean():.4f}")

    if arguments.expected:
        expected = np.load(arguments.expected)[: len(windows)]
        agreement = float((result["predictions"] == expected) .mean())
        print(f"  agreement with supplied labels: {agreement:.4%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
