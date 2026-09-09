# The congestion-coupled detector goes blind — mechanism confirmed (T-N2)

**Task:** T-N2 (blocking) · **Evidence:** `reports/ns3_runs/tn2_ewma-*.txt` · **Spec:** `NS3-Simulation.md §D`

## 1. What was asked, and what could actually be done

T-N2 asks why a prior `C_batched` run (57.9 µs/flow) produced 0 alerts and 0/60 blocked while
`C_batch1` (14542.7 µs/flow) produced 1,928 alerts and 8/60 blocked.

**Neither run, nor the script that produced them, exists in this repository** — see
`docs/ns3_architecture_inventory.md §0`. Nothing below reproduces, confirms, or refutes them. What
follows establishes the mechanism from first principles on this project's own reconstruction of a
congestion-adaptive rule, which is the most that can honestly be claimed.

## 2. The hypothesis, and the measurement

`§B.2`'s leading hypothesis: *a fog node fast enough to avoid queue buildup never produces the
timing signature the rule reacts to.*

The rule's threshold is a product of a base and a load term, and there are exactly two ways to
couple them. Both are implemented (`--ewmaCoupling`), because which one you pick decides which end
of the load range goes blind — and picking one silently would have buried the finding.

Measured, 30 simulated seconds, seed 42, run 1, 20 wearables / 60 attacker-bots / 4 fog nodes:

| Coupling | τ(load) | IDS latency | max load | min τ reached | Alerts | Attackers blocked |
|---|---|---:|---:|---:|---:|---:|
| `inverse` | `500·(1−load)` | 57.9 µs | 0.0150 | 492.5 ms | 216 | **60/60** |
| `inverse` | `500·(1−load)` | 14542.7 µs | 0.9950 | **2.5 ms** | **8** | **0/60** |
| `direct` | `500·load` | 57.9 µs | 0.0150 | **0.0 ms** | 156 | **45/60** |
| `direct` | `500·load` | 14542.7 µs | 0.9950 | 0.0 ms | 894 | 60/60 |

## 3. The confirmed mechanism

**A threshold multiplied by a load term collapses to zero at one end of that term's range, and no
inter-message gap can be shorter than zero.** The `min τ reached` column is the mechanism made
visible: where the effective threshold falls to 2.5 ms, an attacker sending every 5 ms is *compliant*
by the rule's own arithmetic.

This is **not a bug**. There is no off-by-one to fix and no missing EWMA update. It is what
"let the queue state drive the threshold" means when written down.

The direction differs from the hypothesis, and that matters:

- **`inverse` goes blind when congested** — 8 alerts, 0/60 blocked, at 99.5% queue occupancy. The
  detector fails *exactly when the network is under attack hard enough to saturate it*. This is the
  worse of the two failures and it is not the one §B.2 predicted.
- **`direct` under-detects when fast** — 45/60 blocked against 60/60 when congested. This *is* the
  direction §B.2 predicted, though in this model it degrades rather than reaching zero: brief load
  spikes (max load 0.015 > 0) still let some detections through. **The hypothesis is directionally
  confirmed, not exactly confirmed**, and the difference is recorded rather than rounded away.

## 3b. The failure has a tipping point, and crossing it is irreversible

§3 says the rule goes blind under load. The full cross-product (`reports/ns3_simulation_results.md`
§5) sharpens that, and the sharper version is the more useful one.

**The blind spot is not at a latency.** At 2000 µs the `ewma` rule blocks 100% of attacker-bots at
20 and 40 bots, and 0% at 60 and 80 — same rule, same latency, same formula. What matters is offered
load, the *product* of inspection cost and attack scale. A sweep along either axis alone finds the
collapse and misattributes its cause.

**And the collapse is a race.** `blockAfter` — how many violations the rule waits for before
quarantining a source — is the only variable changed below (`ewma`, 2000 µs, 60 attacker-bots):

| `blockAfter` | Alerts | Attackers blocked | Queue overflow |
|---:|---:|---:|---:|
| 1 | 511 | **60/60** | **0** |
| 2 | 883 | **60/60** | **0** |
| 3 | 102 | **0/60** | 280,128 |
| 4 | 102 | **0/60** | 280,128 |

One violation of patience decides the outcome. Blocking removes a bot's load, so blocking early
keeps the queue empty, which keeps the threshold high, which keeps blocking possible. Miss that
window and the loop runs backwards: the queue saturates, the threshold collapses toward zero, no
source can violate a threshold of zero, so nothing further is ever blocked and the load never comes
down.

**A congestion-coupled detector does not degrade under load. It latches off.** That is worse than
gradual degradation in the way that matters operationally: there is no partial service to notice, no
warning band, and no recovery once the tipping point is behind you — the system that would have to
act is the one that has stopped acting.

Where that tipping point sits depends on the **defender's own configuration**, not only on the
attack. Two deployments running identical code with different `blockAfter` values sit on opposite
sides of it under the same attack.

## 4. Why this is a finding and not a defect report

A rule of this shape is a **congestion detector wearing an intrusion detector's label**. Its output
tracks queue occupancy, and its agreement with actual hostility is an assumption nobody has
measured. `docs/ns3_architecture_inventory.md §3` records that none of the three rules inspects
packet contents — so for the timing-only family, "is this traffic hostile" is only ever inferred
from "is this traffic fast", and the congestion coupling then makes that inference conditional on
how busy the *defender* is.

That is a fair **fifth objection** to the base paper's family of adjacency-table timing rules, in
the same spirit as the four in `PRD.md §2.1`. Stated carefully: it is a finding about the *shape* of
congestion-adaptive timing rules, demonstrated on this project's own reconstruction. It is **not**
evidence about the base paper's unpublished implementation, and must not be cited as though it were.

## 5. What this does not license

The criticality rule (`TRD.md §5.1`) also carries a load term — `1 + 0.30·(load − 0.5)` — so the
obvious question is whether it shares the defect. It does not, and the reason is structural rather
than lucky: its load term is **bounded to ±15% and clamped to [50 ms, 2000 ms]**, so it can never
reach zero. Load *modulates* that threshold; it cannot *annihilate* it. `reports/ns3_simulation_results.md §5`
shows it blocking 60/60 at every swept latency, including the one where the EWMA rule blocks none.

That clamp was written into `src/adaptive/threshold.py` for a different reason (T3.4's docstring:
"without clamping, multiplying two factors can leave the [0, 1] range entirely"). This simulation is
the first evidence that it prevents a concrete failure rather than a hypothetical one.
