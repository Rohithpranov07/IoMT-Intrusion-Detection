"""Build the self-contained bundle to copy onto the Raspberry Pi (T4.2 support).

Produces `artifacts/pi_bundle/` holding everything the Pi needs and nothing it does not:

    pi_inference.py     the standalone runner -- imports only numpy and a TFLite interpreter
    benchmark.py        the timing harness
    benchmark_pi.py     its command-line entry point
    deployment/         the exported .tflite graphs and manifest.json
    sample_windows.npy  real test windows, so correctness can be checked on-device
    expected.npy        this machine's predictions for those windows, as the reference
    requirements-pi.txt the two dependencies
    README.md           the run-through, start to finish

The bundle deliberately contains no training code, no datasets, and no TensorFlow — `TRD.md §7`
puts only the inference graph on the device.

Usage:
    .venv/bin/python scripts/make_pi_bundle.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import ARTIFACTS_DIR, REPO_ROOT  # noqa: E402
from src.deployment.pi_inference import EnsembleRunner  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

#: Real test windows shipped for the on-device correctness check.
N_SAMPLE_WINDOWS: int = 256
PRECISION: str = "float32"


def main() -> None:
    """Assemble the bundle."""
    source = ARTIFACTS_DIR / "deployment" / PRECISION
    if not (source / "manifest.json").exists():
        raise SystemExit("Run scripts/export_for_pi.py first.")

    bundle = ARTIFACTS_DIR / "pi_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    shutil.copytree(source, bundle / "deployment")
    for relative in (
        "src/deployment/pi_inference.py",
        "src/deployment/benchmark.py",
        "scripts/benchmark_pi.py",
    ):
        shutil.copy(REPO_ROOT / relative, bundle / Path(relative).name)

    # benchmark.py imports pi_inference via the package path; on the Pi they are siblings.
    benchmark = bundle / "benchmark.py"
    benchmark.write_text(
        benchmark.read_text(encoding="utf-8").replace(
            "from src.deployment.pi_inference import", "from pi_inference import"
        ),
        encoding="utf-8",
    )
    runner_script = bundle / "benchmark_pi.py"
    text = runner_script.read_text(encoding="utf-8")
    text = text.replace(
        'sys.path.insert(0, str(Path(__file__).resolve().parents[1]))',
        'sys.path.insert(0, str(Path(__file__).resolve().parent))',
    ).replace("from src.deployment.benchmark import", "from benchmark import")
    runner_script.write_text(text, encoding="utf-8")

    # Real windows plus this machine's predictions, so the Pi can prove it agrees.
    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    windows = data["X_test"][:N_SAMPLE_WINDOWS].astype(np.float32)
    np.save(bundle / "sample_windows.npy", windows)

    reference = EnsembleRunner(source).predict(windows)
    np.save(bundle / "expected.npy", reference["predictions"])
    np.save(bundle / "expected_probabilities.npy", reference["probabilities"])

    (bundle / "requirements-pi.txt").write_text(
        "# Raspberry Pi 4B (64-bit Raspberry Pi OS). Inference only -- no TensorFlow.\n"
        "numpy>=1.24\n"
        "tflite-runtime>=2.14   # if unavailable for your Python, use: ai-edge-litert\n",
        encoding="utf-8",
    )

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    (bundle / "README.md").write_text(
        f"""# Raspberry Pi deployment bundle

Everything needed to run the IoMT intrusion-detection ensemble on a Raspberry Pi 4B, and nothing
else. No TensorFlow, no training code, no datasets — `TRD.md §7` puts only the inference graph on
the device.

## 1. Copy this directory to the Pi

```bash
scp -r pi_bundle/ pi@raspberrypi.local:~/iomt-ids
```

## 2. Install the two dependencies

```bash
cd ~/iomt-ids
python3 -m pip install -r requirements-pi.txt
```

If `tflite-runtime` has no wheel for your Python version, use `ai-edge-litert` instead — the
runner accepts either, and reports which one it used.

## 3. Check it loads

```bash
python3 pi_inference.py --model-dir deployment --self-test
```

Expected: `Self-test ... PASS`, `Raspberry Pi : True`, and `backend : tflite_runtime`.
If the backend says `tensorflow`, the standalone runtime is not installed and any timing you take
will not be a `tflite-runtime` measurement.

## 4. Prove it agrees with the evaluated model

```bash
python3 pi_inference.py --model-dir deployment \\
    --input sample_windows.npy --expected expected.npy
```

`agreement with supplied labels` must read **100.0000%**. `expected.npy` holds the predictions the
development machine produced for the same {N_SAMPLE_WINDOWS} real test windows, so anything below
100% means the Pi is not reproducing the model this project evaluated — investigate before
believing any result from the device.

## 5. Benchmark (this is T4.3's deliverable)

```bash
python3 benchmark_pi.py --model-dir deployment --windows sample_windows.npy
```

Writes `reports/deployment_benchmark.md` with the measured latency percentiles, per-branch timings,
throughput, and the hardware read from the device tree. **On non-Pi hardware this refuses to run**
unless `--allow-non-pi` is passed, which produces a `DRAFT_` file stamped as not deployment
evidence.

Copy the resulting report and its `.json` back into the repository's `reports/`.

## What is in here

| File | Purpose |
|---|---|
| `pi_inference.py` | Standalone runner. Imports only numpy and a TFLite interpreter. |
| `benchmark.py` | Timing harness; refuses to write a deployment report off a Pi. |
| `benchmark_pi.py` | Command-line entry point for the benchmark. |
| `deployment/` | {len(manifest['branches'])} `.tflite` graphs plus `manifest.json`. |
| `sample_windows.npy` | {N_SAMPLE_WINDOWS} real test windows, shape (n, {manifest['sequence_length']}, {manifest['n_features']}). |
| `expected.npy` | Reference predictions for those windows. |

## Model contract

- Input: `({manifest['sequence_length']}, {manifest['n_features']})` — {manifest['sequence_length']} consecutive flow records, features in the exact
  order given by `manifest.json`'s `feature_names`. A reordered column would corrupt every
  prediction undetectably.
- Output: 2 class probabilities. **Index 1 is `{manifest['classes']['1']}` — the positive class**
  (`TRD.md §2.3`, deliberately the opposite of the base paper's convention). Any metric computed on
  the Pi must state this.
- Fusion: `{manifest['fusion']['formula']}` with gamma = {manifest['fusion']['gamma']}.

## Not included, on purpose

Flow capture, feature extraction, and sessionisation. The bundle takes pre-built windows. An
end-to-end real-time claim needs the capture path measured too, which is outside this project's
scope — and the benchmark report says so rather than implying otherwise.
""",
        encoding="utf-8",
    )

    total = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
    print(f"Bundle: {bundle}  ({total / 1024:.0f} KB)")
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(bundle)}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
