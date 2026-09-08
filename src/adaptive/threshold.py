"""Adaptive, criticality-aware detection thresholds (Build-Instructions T3.4; TRD.md §5.1).

The defect this replaces
------------------------
`PRD.md §2.1.4` (Objection #4, second half): HIDS-IoMT fixes its adjacency-table threshold at
**tau = 500 ms for every device class regardless of criticality**, and never justifies the value
beyond a citation. An insulin pump and a lobby kiosk are held to the same standard, which is wrong
in both directions at once — too slack for the pump, needlessly twitchy for the kiosk.

`TRD.md §5.1` requires a threshold that is a function of **device/patient criticality** and
**network context**, with the exact function and its weights documented rather than left implicit
"the way the base paper leaves tau=500ms unjustified".

THE EXACT FORMULA (this project's own choice, per `TRD.md §5.1`)
----------------------------------------------------------------
    effective = clamp(base * criticality_weight * context_factor, floor, ceiling)

    context_factor = 1 + CONTEXT_SENSITIVITY * (load - LOAD_NEUTRAL)

Two thresholds are produced from the same multipliers, because Objection #4 names two different
constants and they move in OPPOSITE directions:

  1. `adaptive_detection_threshold` — the confidence a detection must reach before an alert is
     raised. For a critical device the bar is LOWERED (weight < 1), so borderline evidence still
     raises an alert. This is the threshold that integrates with this project's ensemble.
  2. `adaptive_inter_message_threshold` — the literal replacement for the base paper's tau, in
     milliseconds. For a critical device the window is TIGHTENED (weight < 1), so a smaller timing
     anomaly is enough to be suspicious.

Both use `criticality_weight < 1` for critical devices, and in both cases that means "more
sensitive". They are not the same quantity and must not be swapped: one is a probability in [0, 1],
the other a duration in milliseconds.

EXACT WEIGHTS, AND WHY THESE VALUES
-----------------------------------
    Criticality              weight   effect on a 0.50 detection base
    LIFE_CRITICAL   (0.60)   0.60     -> 0.30   implanted//life-sustaining: infusion and insulin
                                               pumps, ventilators, pacemakers. A missed attack is
                                               potentially fatal; a false alarm costs a nurse's
                                               attention. The asymmetry is not close, so this is
                                               the largest step in the scale.
    HIGH            (0.80)   0.80     -> 0.40   continuous vital-signs monitoring: cardiac
                                               monitors, anaesthesia machines. Harm is serious but
                                               usually not immediate.
    STANDARD        (1.00)   1.00     -> 0.50   general clinical equipment. Weight 1.0 by
                                               definition: this is the reference class, and the
                                               base threshold is calibrated for it.
    NON_CLINICAL    (1.30)   1.30     -> 0.65   kiosks, digital signage, guest devices. No patient
                                               is harmed by a missed detection here, and these are
                                               numerous, so they dominate the false-alarm budget.

    CONTEXT_SENSITIVITY = 0.30   fog-node load moves the threshold by at most +/-15% (load 0.0 ->
                                 factor 0.85, load 1.0 -> factor 1.15). Deliberately much smaller
                                 than the criticality range: clinical criticality is a stable
                                 property of the device, whereas load is a transient condition, and
                                 a transient condition must never be able to override the clinical
                                 one. Under congestion the bar RISES, because benign traffic looks
                                 more anomalous when the network is saturated and an operator
                                 flooded with alerts during an incident is worse off than one shown
                                 fewer, better ones.
    LOAD_NEUTRAL        = 0.50   the load at which context has no effect.

    DETECTION_FLOOR     = 0.05   even LIFE_CRITICAL at zero load cannot demand less than this, or
    DETECTION_CEILING   = 0.95   the detector degenerates into "always alert"; likewise the ceiling
                                 prevents NON_CLINICAL under load from becoming "never alert".
                                 Without clamping, multiplying two factors can leave the [0, 1]
                                 range entirely and silently produce a threshold no probability can
                                 ever cross.
    INTER_MESSAGE_BASE_MS = 500  the base paper's own value, kept as the STANDARD-class base so
                                 the comparison to it is like-for-like. It is a reference point,
                                 not an endorsement: the paper never justifies it.
    INTER_MESSAGE_FLOOR_MS   = 50
    INTER_MESSAGE_CEILING_MS = 2000

NOT IN SCOPE HERE: the base paper's 30-second check cycle is the OTHER half of Objection #4, and
`TRD.md §5.2` assigns it to incremental learning (T3.5). It is deliberately not touched by this
module.

Determinism: every function here is a pure function of its arguments. No randomness, no state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class Criticality(Enum):
    """Clinical criticality of the device a flow is addressed to.

    The value is the multiplier applied to the base threshold. A weight below 1 makes detection
    MORE sensitive; above 1, less. See the module docstring for the justification of each value.
    """

    LIFE_CRITICAL = 0.60
    HIGH = 0.80
    STANDARD = 1.00
    NON_CLINICAL = 1.30

    @property
    def weight(self) -> float:
        """Return the criticality multiplier."""
        return float(self.value)

    @property
    def description(self) -> str:
        """Return a plain-language description for operator-facing output."""
        return {
            Criticality.LIFE_CRITICAL: "life-sustaining device (e.g. infusion pump, ventilator)",
            Criticality.HIGH: "continuous vital-signs monitoring (e.g. cardiac monitor)",
            Criticality.STANDARD: "general clinical equipment",
            Criticality.NON_CLINICAL: "non-clinical device (e.g. kiosk, signage)",
        }[self]


# --- Frozen constants (see the module docstring for the justification of each) --------------
CONTEXT_SENSITIVITY: float = 0.30
LOAD_NEUTRAL: float = 0.50

DETECTION_BASE: float = 0.50
DETECTION_FLOOR: float = 0.05
DETECTION_CEILING: float = 0.95

INTER_MESSAGE_BASE_MS: float = 500.0
INTER_MESSAGE_FLOOR_MS: float = 50.0
INTER_MESSAGE_CEILING_MS: float = 2000.0

#: The base paper's flat value, kept for explicit comparison in reports (`PRD.md §2.1.4`).
BASE_PAPER_FIXED_THRESHOLD_MS: float = 500.0


@dataclass
class ThresholdDecision:
    """An effective threshold together with everything that produced it.

    Every field is retained so a threshold can be explained to an operator as readily as a
    detection can (`PRD.md §4`), and so a disputed alert can be reconstructed exactly.

    Attributes:
        effective: the threshold actually applied.
        base: the base value before any adjustment.
        criticality: the device's criticality class.
        criticality_weight: the multiplier that class contributed.
        load: the fog-node load in [0, 1] used for the context factor.
        context_factor: the multiplier the load contributed.
        clamped: whether the floor or ceiling bound the result. True means the formula's raw output
            was out of range, which is worth surfacing rather than hiding.
        units: "probability" or "milliseconds".
    """

    effective: float
    base: float
    criticality: Criticality
    criticality_weight: float
    load: float
    context_factor: float
    clamped: bool
    units: str

    def explain(self) -> str:
        """Render a plain-language account of how this threshold was reached."""
        direction = "more sensitive" if self.effective < self.base else "less sensitive"
        if self.effective == self.base:
            direction = "unchanged"
        suffix = "" if self.units == "probability" else " ms"
        return (
            f"Threshold {self.effective:.3g}{suffix} ({direction} than the "
            f"{self.base:.3g}{suffix} baseline).\n"
            f"  device criticality : {self.criticality.name} "
            f"({self.criticality.description}) -> x{self.criticality_weight:.2f}\n"
            f"  fog-node load      : {self.load:.0%} -> x{self.context_factor:.3f}\n"
            f"  clamped to range   : {self.clamped}"
        )


def _context_factor(load: float) -> float:
    """Return the load-driven multiplier.

    Args:
        load: normalised fog-node load in [0, 1]. 0 is idle, 1 is saturated.

    Returns:
        A multiplier in `[1 - CONTEXT_SENSITIVITY/2, 1 + CONTEXT_SENSITIVITY/2]`.

    Raises:
        ValueError: if `load` is outside [0, 1]. Out-of-range load is a caller bug, and silently
            clamping it would hide a broken telemetry feed.
    """
    if not 0.0 <= load <= 1.0:
        raise ValueError(f"load must be in [0, 1]; got {load}")
    return 1.0 + CONTEXT_SENSITIVITY * (load - LOAD_NEUTRAL)


def _apply(
    base: float,
    criticality: Criticality,
    load: float,
    floor: float,
    ceiling: float,
    units: str,
) -> ThresholdDecision:
    """Apply the shared formula and clamp the result.

    Args:
        base: baseline threshold for the STANDARD class.
        criticality: the device's criticality class.
        load: normalised fog-node load in [0, 1].
        floor: minimum permitted effective value.
        ceiling: maximum permitted effective value.
        units: "probability" or "milliseconds", recorded on the result.

    Returns:
        A `ThresholdDecision`.

    Raises:
        ValueError: if `base` is not positive or `load` is out of range.
    """
    if base <= 0:
        raise ValueError(f"base threshold must be positive; got {base}")

    factor = _context_factor(load)
    raw = base * criticality.weight * factor
    effective = float(np.clip(raw, floor, ceiling))

    return ThresholdDecision(
        effective=effective,
        base=base,
        criticality=criticality,
        criticality_weight=criticality.weight,
        load=load,
        context_factor=factor,
        clamped=not np.isclose(raw, effective),
        units=units,
    )


def adaptive_detection_threshold(
    criticality: Criticality,
    load: float = LOAD_NEUTRAL,
    base: float = DETECTION_BASE,
) -> ThresholdDecision:
    """Return the confidence a detection must reach before an alert is raised.

    This is the threshold that integrates with this project's ensemble: it is compared against the
    fused Attack probability from `src.models.fusion`. A LOWER value means MORE sensitive.

    Args:
        criticality: the destination device's criticality class.
        load: normalised fog-node load in [0, 1]. Defaults to neutral.
        base: baseline threshold for the STANDARD class. Defaults to 0.50, the ordinary decision
            boundary of a two-class softmax, so `STANDARD` at neutral load reproduces `argmax`.

    Returns:
        A `ThresholdDecision` in probability units.
    """
    decision = _apply(
        base, criticality, load, DETECTION_FLOOR, DETECTION_CEILING, "probability"
    )
    logger.debug("Detection threshold: %s", decision.effective)
    return decision


def adaptive_inter_message_threshold(
    criticality: Criticality,
    load: float = LOAD_NEUTRAL,
    base_ms: float = INTER_MESSAGE_BASE_MS,
) -> ThresholdDecision:
    """Return the inter-message timing threshold, in milliseconds.

    This is the literal replacement for the base paper's flat tau = 500 ms (`PRD.md §2.1.4`). A
    SMALLER value means a smaller timing anomaly is enough to be treated as suspicious.

    Args:
        criticality: the destination device's criticality class.
        load: normalised fog-node load in [0, 1].
        base_ms: baseline for the STANDARD class. Defaults to the base paper's own 500 ms so the
            comparison is like-for-like — a reference point, not an endorsement.

    Returns:
        A `ThresholdDecision` in milliseconds.
    """
    return _apply(
        base_ms,
        criticality,
        load,
        INTER_MESSAGE_FLOOR_MS,
        INTER_MESSAGE_CEILING_MS,
        "milliseconds",
    )


def apply_threshold(
    attack_probabilities: np.ndarray,
    criticalities: list[Criticality] | np.ndarray,
    loads: np.ndarray | float = LOAD_NEUTRAL,
    base: float = DETECTION_BASE,
) -> np.ndarray:
    """Flag detections using a per-sample adaptive threshold.

    Args:
        attack_probabilities: fused P(Attack) per sample, shape `(n,)`.
        criticalities: the criticality class of each sample's destination device.
        loads: fog-node load per sample, or one value for all.
        base: baseline threshold for the STANDARD class.

    Returns:
        Integer array of predictions; 1 = Attack = the positive class (`TRD.md §2.3`).

    Raises:
        ValueError: if `criticalities` or `loads` do not match the number of samples.
    """
    probabilities = np.asarray(attack_probabilities, dtype=np.float64).ravel()
    if len(criticalities) != len(probabilities):
        raise ValueError(
            f"got {len(criticalities)} criticalities for {len(probabilities)} samples"
        )

    load_array = (
        np.full(len(probabilities), float(loads))
        if np.isscalar(loads)
        else np.asarray(loads, dtype=np.float64).ravel()
    )
    if len(load_array) != len(probabilities):
        raise ValueError(f"got {len(load_array)} loads for {len(probabilities)} samples")

    thresholds = np.array([
        adaptive_detection_threshold(criticality, float(load), base).effective
        for criticality, load in zip(criticalities, load_array)
    ])
    return (probabilities >= thresholds).astype(np.int8)


def threshold_table(load: float = LOAD_NEUTRAL) -> str:
    """Render every criticality class's effective thresholds, for reports and slides.

    Args:
        load: the fog-node load to tabulate at.

    Returns:
        A printable table contrasting this project's thresholds with the base paper's flat rule.
    """
    lines = [
        f"Adaptive thresholds at fog-node load {load:.0%} "
        f"(base paper: a flat {BASE_PAPER_FIXED_THRESHOLD_MS:.0f} ms for EVERY device)",
        f"  {'criticality':<15}{'detection':>11}{'inter-message':>16}",
    ]
    for criticality in Criticality:
        detection = adaptive_detection_threshold(criticality, load)
        timing = adaptive_inter_message_threshold(criticality, load)
        lines.append(
            f"  {criticality.name:<15}{detection.effective:>11.3f}"
            f"{timing.effective:>13.0f} ms"
        )
    return "\n".join(lines)
