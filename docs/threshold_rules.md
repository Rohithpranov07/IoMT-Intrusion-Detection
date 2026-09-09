# The three threshold rules, side by side (T-N4)

**Task:** T-N4 · **Implementation:** `scratch/hids-iomt-adaptive.cc`, `FogIds::EffectiveThresholdMs`
**Spec:** `TRD.md §5.1`, `PRD.md §2.1.4`

All three answer one question: **is this source's inter-message gap too short?** That is the
quantity the base paper fixes flat at τ = 500 ms for every device, which is the half of Objection #4
this contribution exists to fix.

## The rules

| | `fixed` | `ewma` | `criticality` |
|---|---|---|---|
| Threshold | `500 ms` | `500·(1−load)` or `500·load` | `clamp(500 · w · (1 + 0.3·(load − 0.5)), 50, 2000)` |
| Observed quantity | raw gap | EWMA of gaps (α = 0.30) | raw gap |
| Varies by device? | **no** | no | **yes** — clinical criticality |
| Varies by load? | no | **yes, unboundedly** | yes, bounded to ±15% |
| Can reach 0 ms? | no | **yes** — see `docs/zero_detection_investigation.md` | **no** — clamped at 50 ms |
| Origin | `BASE_PAPER_FIXED_THRESHOLD_MS`, `src/adaptive/threshold.py` | this project's reconstruction | `TRD.md §5.1`, T3.4 |

`w` is the criticality weight, identical to the frozen constants in `src/adaptive/threshold.py`:
`LIFE_CRITICAL` 0.60, `HIGH` 0.80, `STANDARD` 1.00, `NON_CLINICAL` 1.30, with
`CONTEXT_SENSITIVITY` 0.30, `LOAD_NEUTRAL` 0.50, floor 50 ms, ceiling 2000 ms. The C++ and the
Python are numerically the same rule; if one changes, both must.

## Measured behaviour (T-N4's VERIFY)

Same scenario throughout: `idsLatencyUs=500` (swept), 60 attacker-bots, 20 wearables, 4 fog nodes,
30 simulated seconds, seed 42.

| Rule | Alerts | True positives | **False positives** | Attackers blocked | **Wearables blocked** |
|---|---:|---:|---:|---:|---:|
| `fixed` | 327 | 267 | **60** | 60/60 | **20/20** |
| `ewma` | 303 | 267 | **36** | 60/60 | **8/20** |
| `criticality` | 296 | 267 | **29** | 60/60 | **9/20** |

Three distinct outcomes, so the new rule is demonstrably not a no-op.

## What the numbers say

**All three catch every attacker here.** At a 5 ms flood against a 500 ms threshold, detection is
not the hard part — the flood is two orders of magnitude outside any of these thresholds. The rules
separate on the other axis.

**The flat rule blocks every legitimate wearable in the simulation.** 20 of 20, from telemetry
jittering around a 600 ms period against a 500 ms constant. This is the concrete cost of one
threshold for every device: a rule that quarantines the entire patient-monitoring estate has not
prevented an incident, it has caused one. It is also why `PRD.md §2.1.4` calls the base paper's
unjustified τ a defect rather than a simplification.

**The criticality rule halves the false alerts** (60 → 29) and blocks 9 wearables instead of 20. It
does not eliminate them, and the residual is structured rather than random: `NON_CLINICAL` devices
carry weight 1.30, so their threshold is 650 ms and jittery telemetry crosses it more often than for
a `LIFE_CRITICAL` device at 300 ms. **That is the design working as intended, not a leak.** The
weighting deliberately spends false alarms where they are cheapest — a flagged kiosk costs someone a
glance; a missed infusion pump does not.

**The EWMA rule looks competitive here and is not**, which is the trap this table would set if read
alone. Its 36 false positives at this latency sit between the other two, but
`docs/zero_detection_investigation.md` shows it dropping to 0/60 attackers blocked once the fog node
saturates. A rule that looks mid-table at one operating point and blind at another is worse than
either number suggests, and no single row of this table would tell you that.
