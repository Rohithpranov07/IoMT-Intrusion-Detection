# Incremental learning as a network-simulation stand-in (T-N5)

**Task:** T-N5 · **Implementation:** `--novelAttackAt`, `--incrementalAt`, `--incrementalBaseMs`
**Evidence:** `reports/ns3_runs/tn5_*.txt` · **Spec:** `TRD.md §5.2`, `Build-Instructions.md` T3.5

## This is a stand-in, and here is exactly what it stands in for

**NS-3 cannot retrain a model.** No CNN, BiLSTM, or Transformer runs inside this simulation; the
detector is a timing rule. So "the model was updated incrementally" has to be *represented* by
something, and what represents it must be stated plainly or the result is theatre.

| Real thing (`src/adaptive/incremental.py`, T3.5) | Stand-in here |
|---|---|
| A novel attack type appears that the trained model has not seen | A **slow-and-low wave**: half the attacker-bots switch to a 450 ms period at `t = 10 s` |
| Replay-buffer fine-tuning absorbs the new type | The rule's **base threshold changes** at `t = 20 s`, from 500 ms to 900 ms |
| Before/after evaluation on prior classes | Alert and false-positive counts reported **separately** for the two phases |

The novel wave's 450 ms period is chosen deliberately: it sits **just under the base paper's flat
500 ms**, in the narrow band where a slow attacker is indistinguishable from chatty telemetry to any
rule using one constant. That is the band an incremental update has to open up.

**What the stand-in does not model.** A real incremental update changes what the model *knows*; this
one changes one number. It cannot show catastrophic forgetting, it cannot show a decision boundary
moving, and it cannot show the replay buffer earning its place — all three of which are what
`reports/t3_5_incremental_learning.md` actually measures, on the real model. This scenario shows
only the *network-level consequence* of a detector becoming more sensitive partway through an
attack.

## Measured

30 simulated seconds; novel wave at `t = 10 s`; update at `t = 20 s`; seed 42.

| Rule | Update | Alerts before | Alerts after | FP before | FP after | Attackers blocked |
|---|---|---:|---:|---:|---:|---:|
| `fixed` | none | 282 | 0 | 60 | 0 | 60/60 |
| `fixed` | applied | 282 | 0 | 60 | 0 | 60/60 |
| `criticality` | none | 160 | 1 | 28 | 1 | **30/60** |
| `criticality` | applied | 160 | 121 | 28 | 31 | **60/60** |

## What happened

**The criticality rule's update works, and it is not free.** Attacker blocking goes from 30/60 to
60/60: without the update, the slow-and-low half of the botnet is never caught, because a 450 ms gap
clears a threshold that criticality weighting has *tightened* to 300–650 ms. Loosening the base to
900 ms opens the band and the second wave is caught.

The cost is in the same table: after-phase false positives rise from **1 to 31**. A threshold
loosened enough to catch a 450 ms attacker also catches more jittery 600 ms telemetry. Reporting the
detection gain without the false-positive cost would misrepresent the trade, and the trade is the
point — this is the same tension `reports/t3_4_adaptive_threshold.md` measures on the real model,
appearing again at network level.

**The flat rule shows no before/after difference at all**, and that is not a null result. Its single
500 ms constant already blocks all 60 bots during the before-phase — including the 450 ms wave, by a
50 ms margin — so by `t = 20 s` there is nothing left to improve and the update has no surface to
act on. **A rule with one constant has nothing to adapt**, which is the structural point
`TRD.md §5.2` makes about the base paper's fixed 30-second check cycle, showing up here as a row of
unchanging numbers.

Note also what the flat rule paid for that coverage: 60 false-positive alerts in the before-phase
against the criticality rule's 28. It catches the slow wave by being indiscriminate, not by being
discerning.
