"""Latency and throughput benchmark for the deployed ensemble (T4.3; TRD.md §7).

>>> THIS MODULE REFUSES TO PRODUCE A DEPLOYMENT REPORT OFF THE RASPBERRY PI. <<<
--------------------------------------------------------------------------------
`Build-Instructions.md` §A.1 rule 3 and `TRD.md §9`'s "Deployment numbers are real" gate require
that any latency or throughput claim trace to an actual run on the actual Pi 4B. `PRD.md §2.2`
identifies the senior's prior work's central failure as a real-time claim with **no** hardware,
latency, or throughput numbers behind it — this project's whole objection to it.

A rule that lives only in a document gets broken. So it is enforced in code:

  * `run_benchmark()` will happily measure on any machine — that is useful during development.
  * `write_deployment_report()` **raises** unless `pi_inference.is_raspberry_pi()` returns True,
    or the caller passes `allow_non_pi=True`, which stamps every page of the output as
    NOT-DEPLOYMENT-EVIDENCE and refuses to write to the canonical report filename.

The hardware description is read from the device tree at run time (`pi_inference.describe_host`),
never taken from a flag or a config file, so the hardware line in a report is itself a measurement.

WHAT IS MEASURED
----------------
    per-window latency     the end-to-end cost of one decision: three branch invocations plus the
                           fusion. Reported as median and p95, not just the mean — a fog node's
                           worst case matters more than its average, and the mean hides it.
    per-branch latency     so a slow branch is attributable. The BiLSTM is unrolled for TFLite
                           (T4.1) and is the one to watch.
    throughput             windows per second under sustained load.
    cold start             time to load the graphs and allocate tensors, paid once at boot.

Fixed parameters
----------------
    WARMUP_ITERATIONS = 20    discarded. The first invocations pay one-off allocation and cache
                              costs that are not part of steady-state latency; including them
                              would overstate the median.
    MIN_ITERATIONS    = 200   timed windows. Enough for a stable p95.
    LATENCY_PERCENTILES = (50, 90, 95, 99)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.deployment.pi_inference import EnsembleRunner, HostInfo, describe_host

logger = logging.getLogger(__name__)

WARMUP_ITERATIONS: int = 20
MIN_ITERATIONS: int = 200
LATENCY_PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)


class NotOnTargetHardwareError(RuntimeError):
    """Raised when a deployment report is requested off the Raspberry Pi.

    Exists so `Build-Instructions.md` §A.1 rule 3 is enforced by the code rather than by the
    author's memory.
    """


@dataclass
class BenchmarkResult:
    """Timing measurements, inseparable from the hardware that produced them.

    Attributes:
        host: the machine measured. Carried with the numbers so they cannot be quoted without it.
        backend: "tflite_runtime", "ai_edge_litert", or "tensorflow".
        precision: the exported precision benchmarked.
        n_windows: timed windows, excluding warm-up.
        cold_start_seconds: graph load and tensor allocation, paid once.
        latencies_ms: per-window end-to-end latency.
        branch_latencies_ms: per-branch invocation latency.
        throughput_per_second: sustained windows per second.
    """

    host: HostInfo
    backend: str
    precision: str
    n_windows: int
    cold_start_seconds: float
    latencies_ms: np.ndarray = field(repr=False)
    branch_latencies_ms: dict[str, np.ndarray] = field(repr=False)
    throughput_per_second: float

    def percentiles(self) -> dict[int, float]:
        """Return end-to-end latency percentiles in milliseconds."""
        return {p: float(np.percentile(self.latencies_ms, p)) for p in LATENCY_PERCENTILES}

    def is_deployment_evidence(self) -> bool:
        """Return whether these numbers may be cited as deployment evidence.

        True only on real Pi hardware. `TRD.md §9`'s gate is about the machine, not the code.
        """
        return self.host.is_raspberry_pi

    def summary(self) -> str:
        """Render a printable block, leading with the hardware."""
        percentiles = self.percentiles()
        branch_lines = "\n".join(
            f"      {name:<12} median {np.median(values):7.2f} ms"
            for name, values in self.branch_latencies_ms.items()
        )
        banner = (
            "DEPLOYMENT EVIDENCE (measured on Raspberry Pi hardware)"
            if self.is_deployment_evidence()
            else "NOT DEPLOYMENT EVIDENCE -- this is not a Raspberry Pi"
        )
        return (
            f"{banner}\n"
            f"{self.host.summary()}\n"
            f"  backend      : {self.backend}\n"
            f"  precision    : {self.precision}\n"
            f"  windows timed: {self.n_windows:,} (after {WARMUP_ITERATIONS} warm-up)\n"
            f"  cold start   : {self.cold_start_seconds:.2f} s\n"
            f"  latency ms   : "
            + "  ".join(f"p{p}={percentiles[p]:.2f}" for p in LATENCY_PERCENTILES)
            + f"\n  per branch   :\n{branch_lines}\n"
            f"  throughput   : {self.throughput_per_second:.1f} windows/s"
        )


def run_benchmark(
    model_dir: str | Path,
    windows: np.ndarray | None = None,
    iterations: int = MIN_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> BenchmarkResult:
    """Measure latency and throughput on whatever machine this is.

    Runs anywhere; `write_deployment_report` is what enforces the hardware requirement.

    Args:
        model_dir: exported deployment directory.
        windows: real windows to time on. Synthetic windows of the manifest's shape are generated
            when None — acceptable for timing, since TFLite's cost does not depend on input values.
        iterations: timed windows.
        warmup: discarded leading iterations.

    Returns:
        A `BenchmarkResult`.
    """
    host = describe_host()

    started = time.perf_counter()
    runner = EnsembleRunner(model_dir)
    cold_start = time.perf_counter() - started

    if windows is None:
        rng = np.random.default_rng(42)
        windows = rng.random(
            (iterations + warmup, runner.sequence_length, runner.n_features)
        ).astype(np.float32)
    windows = np.asarray(windows, dtype=np.float32)
    if len(windows) < iterations + warmup:
        repeats = int(np.ceil((iterations + warmup) / len(windows)))
        windows = np.tile(windows, (repeats, 1, 1))
    windows = windows[: iterations + warmup]

    for window in windows[:warmup]:
        runner.predict(window)

    latencies: list[float] = []
    branch_latencies: dict[str, list[float]] = {n: [] for n in runner.branch_names}

    throughput_started = time.perf_counter()
    for window in windows[warmup:]:
        window_started = time.perf_counter()
        for name in runner.branch_names:
            branch_started = time.perf_counter()
            runner.predict_branch(name, window)
            branch_latencies[name].append((time.perf_counter() - branch_started) * 1000)
        latencies.append((time.perf_counter() - window_started) * 1000)
    elapsed = time.perf_counter() - throughput_started

    result = BenchmarkResult(
        host=host,
        backend=runner.backend,
        precision=runner.manifest.get("precision", "unknown"),
        n_windows=len(latencies),
        cold_start_seconds=cold_start,
        latencies_ms=np.asarray(latencies),
        branch_latencies_ms={n: np.asarray(v) for n, v in branch_latencies.items()},
        throughput_per_second=len(latencies) / elapsed if elapsed > 0 else 0.0,
    )
    logger.info("%s", result.summary())
    return result


def write_deployment_report(
    result: BenchmarkResult,
    output_path: str | Path,
    allow_non_pi: bool = False,
) -> Path:
    """Write the T4.3 deployment report, refusing to do so off the Pi.

    Args:
        result: the benchmark to report.
        output_path: where to write.
        allow_non_pi: permit writing off a Pi. The output is then stamped as not deployment
            evidence and the filename is prefixed `DRAFT_`, so it cannot be mistaken for, or
            silently substituted for, the real report.

    Returns:
        The path written.

    Raises:
        NotOnTargetHardwareError: when not on a Pi and `allow_non_pi` is False.
    """
    output_path = Path(output_path)

    if not result.is_deployment_evidence():
        if not allow_non_pi:
            raise NotOnTargetHardwareError(
                f"Refusing to write a deployment report: this is {result.host.model!r}, not a "
                "Raspberry Pi. TRD.md §9's 'Deployment numbers are real' gate and "
                "Build-Instructions.md §A.1 rule 3 require the actual Pi 4B. "
                "Pass allow_non_pi=True to write a clearly-labelled draft instead."
            )
        output_path = output_path.with_name(f"DRAFT_{output_path.name}")

    percentiles = result.percentiles()
    header = (
        "# T4.3 — Deployment latency and throughput on the Raspberry Pi 4B"
        if result.is_deployment_evidence()
        else "# DRAFT — benchmark harness output (NOT deployment evidence)"
    )
    lines = [header, ""]

    if not result.is_deployment_evidence():
        lines += [
            "> ## ⚠️ These numbers are NOT deployment evidence and must not be cited as such.",
            ">",
            f"> They were measured on **{result.host.model}**, which is not a Raspberry Pi. This",
            "> file exists to show the harness works end to end and to fix the report's format",
            "> before the hardware arrives. `PRD.md §2.2` identifies the senior's prior work's",
            "> central failure as a real-time claim with no hardware behind it; presenting a",
            "> desktop measurement as a Pi measurement would be the same failure.",
            ">",
            "> Re-run `scripts/benchmark_pi.py` **on the Pi** to produce the real report.",
            "",
        ]

    lines += [
        "## Hardware (read from the device at run time, not configured)",
        "",
        "```",
        result.host.summary(),
        "```",
        "",
        f"Interpreter backend: **{result.backend}**  ·  model precision: **{result.precision}**",
        "",
    ]
    if result.backend == "tensorflow":
        lines += [
            "> The **tensorflow** backend was used, not `tflite-runtime`. A measurement taken",
            "> through full TensorFlow is not a `tflite-runtime` measurement and must not be quoted",
            "> as one. On the Pi, install `tflite-runtime` so this line reads `tflite_runtime`.",
            "",
        ]

    lines += [
        "## Results",
        "",
        f"{result.n_windows:,} windows timed after {WARMUP_ITERATIONS} discarded warm-up "
        "iterations.",
        "",
        "| Measurement | Value |",
        "|---|---:|",
        f"| Median latency (p50) | **{percentiles[50]:.2f} ms** |",
        f"| p90 latency | {percentiles[90]:.2f} ms |",
        f"| p95 latency | **{percentiles[95]:.2f} ms** |",
        f"| p99 latency | {percentiles[99]:.2f} ms |",
        f"| Throughput | **{result.throughput_per_second:.1f} windows/s** |",
        f"| Cold start | {result.cold_start_seconds:.2f} s |",
        "",
        "Percentiles rather than a mean alone: a fog node's worst case is what determines whether "
        "an alert is late, and a mean hides it.",
        "",
        "### Per branch",
        "",
        "| Branch | Median | p95 |",
        "|---|---:|---:|",
    ]
    for name, values in result.branch_latencies_ms.items():
        lines.append(
            f"| {name} | {np.median(values):.2f} ms | {np.percentile(values, 95):.2f} ms |"
        )

    lines += [
        "",
        "## What these numbers do and do not support",
        "",
        "- A window is **10 flow records**, so the per-window latency is the cost of one decision,",
        "  not one packet.",
        "- Latency here excludes flow capture, feature extraction, and sessionisation. It is",
        "  **model inference only**. An end-to-end real-time claim needs the capture path measured",
        "  too, which is not in this project's scope.",
        "- Explanation cost is **not** included. SHAP adds ~0.21 s per flagged detection on a",
        "  desktop (`reports/t3_2_lime_vs_shap.md`) and has not been measured here; certification",
        "  adds ~0.07 s. Both apply only to flagged windows.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python3 scripts/benchmark_pi.py --model-dir deployment/float32",
        "```",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "is_deployment_evidence": result.is_deployment_evidence(),
        "host": result.host.__dict__,
        "backend": result.backend,
        "precision": result.precision,
        "n_windows": result.n_windows,
        "cold_start_seconds": result.cold_start_seconds,
        "latency_ms": percentiles,
        "throughput_per_second": result.throughput_per_second,
        "branch_median_ms": {
            n: float(np.median(v)) for n, v in result.branch_latencies_ms.items()
        },
    }
    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
