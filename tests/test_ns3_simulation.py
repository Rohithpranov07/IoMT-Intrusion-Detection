"""Tests for the NS-3 validation layer (NS3-Simulation.md T-N1 – T-N7).

These do not run NS-3 — a compiled simulator is not available in every checkout, and the raw runs
are committed precisely so the reports stay re-derivable without one. What they pin is the part
that can rot silently:

  1. The criticality rule exists in TWO languages. `src/adaptive/threshold.py` holds the frozen
     T3.4 constants and `scratch/hids-iomt-adaptive.cc` re-declares them in C++. Nothing but a test
     stops the Python from being retuned while the C++ keeps simulating the old rule and the report
     keeps saying they are the same rule.
  2. NS3-Simulation.md §A.1's provenance discipline, which is a rule about text and therefore
     exactly the kind that gets forgotten when the number is convenient.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.adaptive.threshold import (
    CONTEXT_SENSITIVITY,
    Criticality,
    INTER_MESSAGE_BASE_MS,
    INTER_MESSAGE_CEILING_MS,
    INTER_MESSAGE_FLOOR_MS,
    LOAD_NEUTRAL,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = REPO_ROOT / "scratch" / "hids-iomt-adaptive.cc"
RUNS_DIR = REPO_ROOT / "reports" / "ns3_runs"
NS3_REPORT = REPO_ROOT / "reports" / "ns3_simulation_results.md"


def cc_constant(name: str) -> float:
    """Read a `static const double` out of the simulation source.

    Args:
        name: the C++ identifier.

    Returns:
        Its literal value.

    Raises:
        AssertionError: if the constant is missing, which is itself the drift this guards.
    """
    text = SCRATCH.read_text(encoding="utf-8")
    match = re.search(rf"static const double {name} *= *([\d.]+)", text)
    assert match is not None, f"{name} is no longer declared in {SCRATCH.name}"
    return float(match.group(1))


@pytest.mark.parametrize(
    ("cc_name", "python_value"),
    [
        ("kInterMessageBaseMs", INTER_MESSAGE_BASE_MS),
        ("kInterMessageFloorMs", INTER_MESSAGE_FLOOR_MS),
        ("kInterMessageCeilingMs", INTER_MESSAGE_CEILING_MS),
        ("kContextSensitivity", CONTEXT_SENSITIVITY),
        ("kLoadNeutral", LOAD_NEUTRAL),
    ],
)
def test_cc_threshold_constants_match_the_frozen_python_ones(cc_name: str, python_value: float) -> None:
    """The C++ and the Python must be the same rule, or the report's comparison is false."""
    assert cc_constant(cc_name) == pytest.approx(python_value), (
        f"{cc_name} in {SCRATCH.name} has drifted from src/adaptive/threshold.py. "
        "reports/ns3_simulation_results.md claims these are the same rule; fix one or the other."
    )


def test_cc_carries_every_criticality_weight() -> None:
    """All four T3.4 weights must appear in the simulation's weight table."""
    text = SCRATCH.read_text(encoding="utf-8")
    match = re.search(r"const double weights\[4\] = \{([^}]+)\}", text)
    assert match is not None, "the criticality weight table is gone from the simulation"

    found = sorted(float(value) for value in match.group(1).split(","))
    expected = sorted(member.weight for member in Criticality)
    assert found == pytest.approx(expected), (
        f"simulation weights {found} do not match src/adaptive/threshold.py's {expected}"
    )


def test_the_load_term_cannot_annihilate_the_criticality_threshold() -> None:
    """The clamp is what makes this rule immune to T-N2's failure; it must stay in the source.

    docs/zero_detection_investigation.md §5 rests on the criticality threshold being clamped, so a
    silent removal of the clamp would turn a documented immunity into a false claim.
    """
    text = SCRATCH.read_text(encoding="utf-8")
    assert "std::min(kInterMessageCeilingMs, std::max(kInterMessageFloorMs, raw))" in text, (
        "the criticality rule's clamp is gone. docs/zero_detection_investigation.md claims the "
        "rule cannot be driven to a zero threshold; without the clamp that claim is false."
    )


def test_no_ns3_document_claims_a_measured_latency() -> None:
    """§A.1 rule 4: a simulated number may never be presented as a physical measurement."""
    if not NS3_REPORT.exists():
        pytest.skip("NS-3 report not generated in this checkout")

    text = NS3_REPORT.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "simulation output" in lowered, "the report must say its numbers are simulated"
    for phrase in ("measured on the pi", "measured on the raspberry", "on real hardware:"):
        assert phrase not in lowered, (
            f"reports/ns3_simulation_results.md contains {phrase!r}; no Raspberry Pi measurement "
            "exists in this repository."
        )


@pytest.mark.parametrize("run", sorted(RUNS_DIR.glob("*.txt")) if RUNS_DIR.exists() else [])
def test_every_saved_run_stamps_its_latency_provenance(run: Path) -> None:
    """A raw run must not be quotable without the origin of its latency travelling with it."""
    text = run.read_text(encoding="utf-8")
    match = re.search(r"latency provenance *: *(\S+)", text)
    assert match is not None, f"{run.name} has no latency provenance line"
    assert match.group(1) != "UNLABELLED-SWEEP", (
        f"{run.name} was run without --latencySource, so its latency has no recorded origin"
    )
