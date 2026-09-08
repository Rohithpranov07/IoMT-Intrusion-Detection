"""Unit tests for the Pi inference runner and benchmark (Build-Instructions T4.2, T4.3).

T4.2's and T4.3's VERIFY blocks require the actual Raspberry Pi 4B, which this project does not yet
have. These tests cover everything that CAN be established without it, so that when the hardware
arrives the only open question is the hardware itself:

  * the runner reproduces the trained ensemble's predictions exactly;
  * it is genuinely standalone — no import from this repository;
  * the fusion reimplemented in `pi_inference` matches `src/models/fusion.py` bit for bit;
  * and the benchmark REFUSES to emit deployment numbers off Pi hardware.

That last one is the point of `test_benchmark_refuses_to_write_a_report_off_pi`. `PRD.md §2.2`
identifies the senior's prior work's central failure as a real-time claim with no hardware behind
it; a rule that lives only in a document gets broken, so it is enforced in code and pinned here.
"""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.deployment.benchmark import (
    NotOnTargetHardwareError,
    run_benchmark,
    write_deployment_report,
)
from src.deployment.pi_inference import (
    EnsembleRunner,
    confidence_weighted_fusion,
    describe_host,
    is_raspberry_pi,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "artifacts" / "deployment" / "float32"
requires_export = pytest.mark.skipif(
    not (DEPLOYMENT_DIR / "manifest.json").exists(),
    reason="run scripts/export_for_pi.py first",
)


# --- The standalone constraint (TRD.md §7) -----------------------------------------------------


def test_pi_inference_imports_nothing_from_the_repo() -> None:
    """The Pi gets the inference graph and nothing else.

    A single `from src...` import would drag TensorFlow, scikit-learn, pandas and the whole
    training pipeline onto the device. Enforced here rather than left to discipline.
    """
    source = (REPO_ROOT / "src" / "deployment" / "pi_inference.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    for module in imported:
        assert not module.startswith("src"), (
            f"pi_inference.py imports {module!r} from this repository; it must be standalone"
        )
        assert module.split(".")[0] not in {"sklearn", "pandas", "shap", "lime", "imblearn"}, (
            f"pi_inference.py pulls in a training-only dependency: {module!r}"
        )


def test_pi_inference_does_not_require_tensorflow_at_import_time() -> None:
    """TensorFlow is a fallback, not a dependency: the top-level imports must not include it."""
    source = (REPO_ROOT / "src" / "deployment" / "pi_inference.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:  # module level only -- function-local fallbacks are fine
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "tensorflow" for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "tensorflow"


# --- The fusion must not drift from the training implementation --------------------------------


def test_pi_fusion_matches_the_training_implementation() -> None:
    """`pi_inference` reimplements fusion in NumPy; the duplicate must not drift.

    It is duplicated deliberately (the Pi cannot import `src/`), so a test is the only thing
    keeping the deployed decision identical to the evaluated one.
    """
    from src.models.fusion import (
        CONFIDENCE_SHARPNESS,
        DEFAULT_BRANCH_PRIOR,
        EPSILON,
        confidence_weighted_fusion as training_fusion,
    )

    rng = np.random.default_rng(0)
    branches = []
    for _ in range(3):
        raw = rng.random((64, 2))
        branches.append(raw / raw.sum(axis=1, keepdims=True))

    reference = training_fusion(branches, ["a", "b", "c"])
    probabilities, predictions, weights = confidence_weighted_fusion(
        branches, [DEFAULT_BRANCH_PRIOR] * 3, CONFIDENCE_SHARPNESS, EPSILON
    )

    np.testing.assert_allclose(probabilities, reference.probabilities, atol=1e-12)
    np.testing.assert_array_equal(predictions, reference.predictions)
    np.testing.assert_allclose(weights, reference.branch_weights, atol=1e-12)


# --- Hardware identification -------------------------------------------------------------------


def test_host_description_is_read_not_configured() -> None:
    """The hardware line in a report must be a measurement like any other."""
    host = describe_host()
    assert host.machine and host.os_description and host.python_version
    assert host.processor_count > 0
    assert isinstance(host.is_raspberry_pi, bool)
    assert host.is_raspberry_pi == is_raspberry_pi()


def test_this_machine_is_not_mistaken_for_a_pi() -> None:
    """A development machine must never satisfy the Pi check."""
    host = describe_host()
    if "raspberry" not in host.model.lower():
        assert host.is_raspberry_pi is False


# --- The runner ---------------------------------------------------------------------------------


@requires_export
def test_runner_reproduces_the_trained_ensemble() -> None:
    """The deployed decision must equal the evaluated one on real windows.

    This is the substance of T4.2's VERIFY block, minus the hardware.
    """
    warnings.filterwarnings("ignore")
    from src.config import ARTIFACTS_DIR
    from src.models.fusion import confidence_weighted_fusion as training_fusion
    from src.models.train_utils import load_branch

    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    windows = data["X_test"][:64].astype(np.float32)

    runner = EnsembleRunner(DEPLOYMENT_DIR)
    deployed = runner.predict(windows)

    models = {n: load_branch(ARTIFACTS_DIR / "models" / f"{n}.keras") for n in runner.branch_names}
    reference = training_fusion(
        [models[n].predict(windows, verbose=0) for n in runner.branch_names], runner.branch_names
    )

    agreement = float((deployed["predictions"] == reference.predictions).mean())
    assert agreement == 1.0, f"deployed model disagrees with the evaluated one on {1 - agreement:.2%}"


@requires_export
def test_runner_rejects_a_wrongly_shaped_window() -> None:
    """A silently reshaped or reordered input would corrupt every prediction."""
    runner = EnsembleRunner(DEPLOYMENT_DIR)
    with pytest.raises(ValueError, match="expected windows of shape"):
        runner.predict(np.zeros((4, runner.sequence_length, runner.n_features + 1), np.float32))


@requires_export
def test_runner_reports_the_positive_class_convention() -> None:
    """`TRD.md §2.3` must travel with the model onto the device."""
    assert EnsembleRunner(DEPLOYMENT_DIR).positive_class_name() == "Attack"


@requires_export
def test_self_test_passes_on_the_exported_bundle() -> None:
    """The smoke check the Pi README tells the operator to run first."""
    assert EnsembleRunner(DEPLOYMENT_DIR).self_test(n=4) is True


def test_missing_manifest_fails_with_a_useful_message(tmp_path: Path) -> None:
    """Copying only the .tflite files is an easy mistake; it must say so."""
    with pytest.raises(FileNotFoundError, match="Copy the whole exported directory"):
        EnsembleRunner(tmp_path)


# --- The hardware gate (TRD.md §9, Build-Instructions §A.1 rule 3) ------------------------------


@requires_export
def test_benchmark_refuses_to_write_a_report_off_pi(tmp_path: Path) -> None:
    """THE gate. A deployment report must not exist unless a Pi produced it.

    `PRD.md §2.2`'s objection to the prior work is a real-time claim with no hardware behind it.
    Presenting a desktop measurement as a Pi measurement would repeat exactly that.
    """
    result = run_benchmark(DEPLOYMENT_DIR, iterations=5, warmup=2)

    if result.host.is_raspberry_pi:
        pytest.skip("running on a Pi; the refusal path cannot be exercised here")

    assert result.is_deployment_evidence() is False
    with pytest.raises(NotOnTargetHardwareError, match="not a Raspberry Pi"):
        write_deployment_report(result, tmp_path / "deployment_benchmark.md")


@requires_export
def test_non_pi_draft_is_labelled_and_renamed(tmp_path: Path) -> None:
    """An override must produce something that cannot be mistaken for the real report."""
    result = run_benchmark(DEPLOYMENT_DIR, iterations=5, warmup=2)
    if result.host.is_raspberry_pi:
        pytest.skip("running on a Pi")

    path = write_deployment_report(
        result, tmp_path / "deployment_benchmark.md", allow_non_pi=True
    )
    assert path.name.startswith("DRAFT_")

    text = path.read_text(encoding="utf-8")
    assert "NOT deployment evidence" in text
    assert "not a Raspberry Pi" in text
    # The machine that produced them must appear alongside the numbers.
    assert result.host.model in text


@requires_export
def test_benchmark_records_the_interpreter_backend(tmp_path: Path) -> None:
    """A measurement through full TensorFlow is not a tflite-runtime measurement."""
    result = run_benchmark(DEPLOYMENT_DIR, iterations=5, warmup=2)
    assert result.backend in {"tflite_runtime", "ai_edge_litert", "tensorflow"}

    if result.backend == "tensorflow" and not result.host.is_raspberry_pi:
        path = write_deployment_report(
            result, tmp_path / "deployment_benchmark.md", allow_non_pi=True
        )
        assert "not a `tflite-runtime` measurement" in path.read_text(encoding="utf-8")


@requires_export
def test_benchmark_reports_percentiles_not_just_a_mean(tmp_path: Path) -> None:
    """A fog node's worst case decides whether an alert is late; a mean hides it."""
    result = run_benchmark(DEPLOYMENT_DIR, iterations=10, warmup=2)
    percentiles = result.percentiles()
    assert set(percentiles) == {50, 90, 95, 99}
    assert percentiles[50] <= percentiles[95] <= percentiles[99]
    assert result.throughput_per_second > 0


# --- The bundle ------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not (REPO_ROOT / "artifacts" / "pi_bundle" / "README.md").exists(),
    reason="run scripts/make_pi_bundle.py first",
)
def test_bundle_is_self_contained() -> None:
    """Everything the Pi needs, and nothing that would drag in the training stack."""
    bundle = REPO_ROOT / "artifacts" / "pi_bundle"
    for required in (
        "pi_inference.py", "benchmark.py", "benchmark_pi.py", "requirements-pi.txt",
        "README.md", "sample_windows.npy", "expected.npy", "deployment/manifest.json",
    ):
        assert (bundle / required).exists(), f"bundle is missing {required}"

    # The copied modules must not reach back into the repository.
    for name in ("pi_inference.py", "benchmark.py", "benchmark_pi.py"):
        assert "from src." not in (bundle / name).read_text(encoding="utf-8"), (
            f"{name} in the bundle still imports from the repository"
        )

    # Check the actual requirement lines, not the comments. The file's own comment says
    # "no TensorFlow", so a naive substring search would flag the line that states the rule.
    requirement_lines = [
        line.split("#")[0].strip()
        for line in (bundle / "requirements-pi.txt").read_text(encoding="utf-8").splitlines()
    ]
    packages = [line for line in requirement_lines if line]
    assert packages, "requirements-pi.txt lists no packages"
    assert not any("tensorflow" in line.lower() for line in packages), (
        f"the Pi bundle must not depend on TensorFlow; got {packages}"
    )

    manifest = json.loads((bundle / "deployment" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["classes"]["1"] == "Attack"
