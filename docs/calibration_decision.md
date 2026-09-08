# Calibration Decision — **REJECTED on measured evidence**

**Status:** Decided (rejected) · **Decides:** whether per-branch probability calibration enters the pipeline
**Evidence:** `reports/calibration_evidence.md`, produced by `scripts/evaluate_calibration.py`
**Spec refs:** `docs/architecture_decision.md` §3.1, `TRD.md §3.3`, `PRD.md §3`

> ## This document corrects two earlier claims of this project's own
>
> `reports/phase2_results.md` §6 and `reports/t3_4_adaptive_threshold.md` both asserted that
> per-branch **miscalibration** was the cause of two measured shortfalls, and that temperature
> scaling was "the highest-value remaining Phase 3 work".
>
> **That diagnosis was wrong.** It was a plausible inference from a real symptom, never a
> measurement, and this document exists because measuring it refuted it. The affected sections of
> both reports have been corrected rather than quietly left standing.

---

## 1. The two symptoms this was supposed to explain

| Symptom | Where measured |
|---|---|
| Confidence-weighted fusion collapsed into a simple average — weights 0.337 / 0.330 / 0.334, and the best single branch beat every fusion rule | `reports/phase2_results.md` §6 |
| The adaptive threshold reduced false positives by at most 10.8%, against `PRD.md §3`'s >30% target | `reports/t3_4_adaptive_threshold.md` |

Both were traced to the same observation: median branch confidence **0.9999**. The inference was
that the branches were badly miscalibrated, and that softening their probabilities would give both
mechanisms something to work with.

## 2. What was measured

Temperature scaling (Guo et al., 2017), fitted per branch on the **validation** fold by minimising
NLL, evaluated on the **held-out test** fold. Implementation: `src/models/calibration.py`.

### 2.1 The branches were barely miscalibrated at all

| Branch | Fitted T | ECE before → after | Mean confidence before → after |
|---|---:|---|---|
| cnn | 1.34 | 0.0118 → 0.0039 | 0.9753 → 0.9639 |
| bilstm | **0.79** | 0.0181 → 0.0083 | 0.9552 → 0.9672 |
| transformer | 1.34 | 0.0234 → 0.0148 | 0.9571 → 0.9417 |

An expected calibration error of **0.012–0.023 is already good**. These are not the temperatures of
a pathologically overconfident model. The BiLSTM's `T = 0.79` is the clearest refutation: it was
mildly **under**-confident, the opposite of the assumed defect.

The high median confidence is therefore not miscalibration — it is **justified confidence**. The
branches say 99% and are right about 99% of the time. On a dataset whose sessions are 100%
label-pure (`reports/phase2_results.md` §9), most windows genuinely are unambiguous.

### 2.2 Calibration changed nothing downstream

| | Uncalibrated | Calibrated |
|---|---|---|
| Mean fusion weight spread | 0.0258 | 0.0290 |
| Fusion F1 | 0.9661 | 0.9662 |
| Best single branch F1 | 0.9707 | 0.9707 |
| Adaptive threshold, Δ false positives | −6.8% | −6.9% |
| Detections in the actionable band 0.05–0.95 | 22.0% | 24.2% |

Every downstream number moves in the third decimal place or not at all.

## 3. The actual cause

Diagnosed after calibration was refuted, by measuring where fusion can act at all:

- **The three branches agree on 90.9% of test windows.** Fusion can only change the outcome on the
  remaining **9.1%** (445 windows).
- **On those contested windows, fusion is *worse* than its best branch**: 62.0% correct versus the
  transformer's 71.7%. The two weaker branches jointly outvote the stronger one.
- **Ceiling analysis:** even an oracle that resolved every disagreement perfectly would gain only
  **+0.0257** accuracy over the best single branch. There is very little for any fusion rule to win.

This is why calibration could not help. Confidence weighting fails here not because confidences are
*mis-scaled* but because the CNN and BiLSTM are **confidently wrong on precisely the windows that
are contested** — and they are well calibrated, so their confidence carries no signal that this
particular sample is one they get wrong. ECE is an average property over a fold; it says nothing
about which individual predictions are mistaken. No monotonic rescaling of a per-sample confidence
can encode information the confidence does not contain.

## 4. The alternative fix was also tested, and also fails

`docs/architecture_decision.md` §3.1 makes `BRANCH_PRIORS` (`alpha_b`) the fusion layer's only
adjustable parameter. Fitting it to upweight the transformer is the natural response to §3's
diagnosis, needs no architectural change, and is already what T3.5 was designated to update.

Fitted by grid search on the validation fold, maximising positive-class F1:

| | Validation F1 | Test F1 |
|---|---:|---:|
| Equal priors (α = 1, 1, 1) | 0.9851 | **0.9661** |
| Fitted priors (α = 0.0, 0.2, 0.1) | **0.9868** | 0.9658 |
| Transformer alone | — | **0.9707** |

**The fitted priors overfit the validation fold.** They improve it by 0.0017 and cost 0.0003 on
test, and the fit zeroes the CNN entirely — a solution that does not generalise. Reported rather
than quoted from validation only, which would have looked like a gain.

## 5. Decision

**Calibration is NOT adopted into the pipeline.**

1. It does not fix either symptom (§2.2).
2. It addresses a defect the branches do not have (§2.1).
3. Adding an unnecessary transform to every branch would complicate the TFLite export (T4.1) and
   the Pi inference path (T4.2) for no measured benefit.

`src/models/calibration.py` is **kept**, and this is deliberate. It is the instrument that produced
the refutation, T3.6 must re-run it on Edge-IIoTset before assuming this conclusion transfers, and a
reader of the Review 3 report should be able to reproduce the rejection rather than take it on
trust. Its docstring states that it is measured-but-not-adopted, so no one wires it in by accident.

**`docs/architecture_decision.md` is unchanged.** Nothing here alters a frozen number — which is the
point of having frozen them.

## 6. What this means for Contribution 2, stated plainly

**On IoTID20, the three-branch ensemble does not beat its best single branch, and no weighting
scheme recovers it.** Fusion F1 0.9661 against the transformer's 0.9707; the ceiling on any fusion
rule is +0.0257 accuracy; both candidate remedies are now measured and rejected.

That is a negative result about *this dataset*, and it should be reported as one rather than
smoothed over. `PRD.md §3` defines success for Contribution 2 as a fully specified architecture
trained on genuine sequences with confidence-weighted fusion, evaluated honestly against the
leakage-free baseline — **not** as beating a number. That criterion is met. Whether the ensemble
*earns its complexity* is a separate question, and on IoTID20 the current answer is no.

### Recommended next steps, in order

1. **T3.6 (Edge-IIoTset) is now a decision point, not just an extension.** IoTID20's 90.9% branch
   agreement and 100% label-pure sessions are what leave fusion no room. A dataset with genuinely
   diverse branch behaviour is where the ensemble either earns its place or does not. Re-run the
   agreement and ceiling analysis there first — it is cheap and it decides what the rest is worth.
2. **T4.4's ablation must lead with this.** The honest headline is that the best single branch beats
   the ensemble on IoTID20, and `reports/ablation_study.md` has to say so in its first paragraph.
3. **Do not tune the branches to rescue the result.** The CNN and BiLSTM being weaker is a finding.
   Hyperparameter-searching until fusion wins would be fitting the conclusion.
4. **If the team wants the ensemble to earn its place**, the evidence points at branch *diversity*,
   not weighting: the branches currently make the same mistakes. That is an architecture question
   for a new decision record, not a parameter to nudge.

## 7. Reproducing

```bash
.venv/bin/python scripts/evaluate_calibration.py    # writes reports/calibration_evidence.md
```

Deterministic. Temperatures fitted on 4,700 validation windows, evaluated on 4,894 test windows.
