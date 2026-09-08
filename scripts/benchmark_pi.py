"""Run the deployment benchmark (Build-Instructions T4.3).

On the Raspberry Pi this writes `reports/deployment_benchmark.md` — the real T4.3 deliverable.
Anywhere else it refuses, unless `--allow-non-pi` is passed, in which case it writes a clearly
stamped `DRAFT_` file that cannot be mistaken for deployment evidence.

Usage on the Pi:
    python3 scripts/benchmark_pi.py --model-dir deployment/float32

Usage on a development machine (harness check only):
    .venv/bin/python scripts/benchmark_pi.py --model-dir artifacts/deployment/float32 --allow-non-pi
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.deployment.benchmark import (  # noqa: E402
    MIN_ITERATIONS,
    NotOnTargetHardwareError,
    run_benchmark,
    write_deployment_report,
)

logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")


def main(argv: list[str] | None = None) -> int:
    """Benchmark the exported ensemble and write the report.

    Args:
        argv: argument list, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--windows", help="optional .npy of real windows to time on")
    parser.add_argument("--iterations", type=int, default=MIN_ITERATIONS)
    parser.add_argument("--output", default="reports/deployment_benchmark.md")
    parser.add_argument(
        "--allow-non-pi",
        action="store_true",
        help="write a clearly-labelled DRAFT when not on Raspberry Pi hardware",
    )
    arguments = parser.parse_args(argv)

    windows = np.load(arguments.windows) if arguments.windows else None
    result = run_benchmark(arguments.model_dir, windows=windows, iterations=arguments.iterations)
    print(result.summary())

    try:
        path = write_deployment_report(
            result, arguments.output, allow_non_pi=arguments.allow_non_pi
        )
    except NotOnTargetHardwareError as error:
        print(f"\n{error}", file=sys.stderr)
        return 2

    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
