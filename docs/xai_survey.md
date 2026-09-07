# XAI Method Survey and Recommendation (T1.4)

**Status:** Decided · **Drives:** T3.1 (`src/xai/shap_explainer.py`), T3.2 (`src/xai/lime_explainer.py`), T3.3 (`src/xai/metrics.py`)
**Spec refs:** `TRD.md §4`, `PRD.md §4` (personas), `PRD.md §3` Contribution 3
**Documentation task only — no code.**

---

## 1. What this project actually needs from an explanation

Objection #4 (`PRD.md §2.1.4`) is that HIDS-IoMT is a black box in a life-critical setting. The
fix is not "add SHAP because reviewers like SHAP" — it is an explanation that satisfies four
concrete constraints drawn from the PRD and TRD:

| # | Constraint | Source |
|---|---|---|
| C1 | Readable by a **hospital IT lead or clinician**, not an ML engineer. A ranked list of named features plus a short natural-language summary. | `PRD.md §4`, `TRD.md §4` output contract |
| C2 | **Per-detection**, not global. "Which features made *this* flow suspicious", not "which features matter on average". | `TRD.md §4` |
| C3 | Fast enough that explaining a flagged flow does not defeat the near-real-time claim. The adaptive threshold (`TRD.md §5.1`) targets sub-30-second response; an explanation taking minutes is useless. | `PRD.md` NFR "Timeliness" |
| C4 | Works on the **ensemble** — a confidence-weighted fusion of a CNN, a BiLSTM, and a Transformer over `(batch, sequence_length, features)` input. Not a single differentiable tower, and not a tree model. | `TRD.md §3.2`, `§3.3` |

C4 is the constraint that does most of the work below, and it is easy to overlook: the input is a
**sequence**, so an attribution is naturally a `sequence_length × n_features` matrix, not a vector
of `n_features`. Any method chosen has to have an answer for how that gets collapsed into something
a clinician can read.

## 2. The three candidates

### 2.1 SHAP (SHapley Additive exPlanations)

Attributes a prediction to features by averaging each feature's marginal contribution over
coalitions of the others — a Shapley value from cooperative game theory.

**For:**
- The only one of the three with **additive consistency**: attributions sum to
  `prediction − baseline`, so "these five features account for 80% of the score" is a literally true
  statement rather than a rhetorical one. That property is what makes a per-detection number
  defensible when a clinician asks how much a feature mattered.
- Strong fidelity in practice, which is directly one of `TRD.md §4`'s three evaluation metrics.
- Widely recognised by reviewers; the comparison is expected in this literature.

**Against:**
- **Slow.** `KernelSHAP` is model-agnostic but needs thousands of perturbed forward passes per
  explanation. `DeepSHAP`/`GradientExplainer` are far faster but assume a single differentiable
  graph — which the confidence-weighted fusion is *not*, because the fusion weights depend on each
  branch's own softmax output.
- Background-distribution sensitivity: the attributions are relative to a reference set, and a
  poorly chosen one silently changes the answer.

**On C4:** the practical route is to apply `GradientExplainer` **per branch** and combine the three
attribution maps using the same confidence weights the fusion layer uses. This is coherent — the
fusion is a weighted sum of branch outputs, so a weighted sum of branch attributions is the
consistent thing to do — but it must be documented as a project choice, not presented as
off-the-shelf behaviour.

### 2.2 LIME (Local Interpretable Model-agnostic Explanations)

Fits a sparse linear surrogate to the model's behaviour in a small neighbourhood of the instance.

**For:**
- **Fully model-agnostic** — it only needs `predict_proba`, so it sees the fused ensemble exactly as
  deployed, fusion weights and all. No per-branch decomposition, no assumption violated.
- **Fast and tunable**: cost is `num_samples` forward passes, and `num_samples` is a dial. Hundreds
  of samples give a usable explanation where KernelSHAP needs thousands.
- Its output — a handful of signed feature weights — is already close to C1's target format.

**Against:**
- **No additivity guarantee.** The weights are surrogate coefficients; they do not sum to anything
  meaningful about the prediction. "Feature X contributed 0.3" is not a claim that survives scrutiny.
- **Instability** is LIME's documented weakness: repeated runs on the same input give different
  explanations because the neighbourhood is sampled randomly. That collides head-on with
  `TRD.md §4`'s *stability* metric — so T3.3 must measure it rather than assume it, and T3.2 must
  fix the sampling seed.
- Perturbing a flow-sequence neighbourhood requires care: naive Gaussian noise produces
  network-flow records that could not physically occur (negative packet counts, IAT statistics
  inconsistent with the flow duration), and the surrogate then explains behaviour in a region the
  model will never actually meet.

### 2.3 Integrated Gradients

Integrates the model's gradient along a straight path from a baseline input to the actual input.

**For:**
- **Fast** — cost is the number of interpolation steps (typically 20–100 forward+backward passes),
  bounded and predictable, which is attractive for C3.
- Satisfies completeness (attributions sum to the prediction difference), like SHAP.
- Handles sequence input natively, producing a clean `sequence_length × n_features` attribution map.

**Against:**
- **Requires gradients through the whole model.** Same problem as DeepSHAP, and worse: the
  confidence-weighted fusion's `argmax`/softmax-weighting step is not cleanly differentiable, so IG
  would explain the branches rather than the deployed decision. Applying it per branch and
  recombining lands us back at exactly the SHAP compromise, with weaker recognition value.
- **Baseline choice is arbitrary and consequential.** An all-zeros baseline is a *meaningful* flow
  record in this feature space, not a neutral one, so attributions inherit that arbitrary choice.
- Output is a raw gradient-derived attribution — furthest of the three from C1's plain-language
  target without additional post-processing.

## 3. Comparison

| Criterion | SHAP | LIME | Integrated Gradients |
|---|---|---|---|
| Additive / complete attributions (C1 defensibility) | **Yes** | No | Yes |
| Works on the fused ensemble as deployed (C4) | Per-branch + reweight | **Directly** | Per-branch + reweight |
| Per-detection local explanation (C2) | Yes | Yes | Yes |
| Speed (C3) | Slowest | **Fast, tunable** | Fast |
| Stability across repeated runs | Good | **Poor** (must be measured, T3.3) | Good |
| Output shape for a non-expert (C1) | Good | **Best** | Needs post-processing |
| Handles sequence input naturally | Yes | Needs a flattening convention | Yes |
| Reviewer expectation in this literature | **High** | High | Low |

## 4. Recommendation

**Primary: SHAP.** **Fallback: LIME.** **Integrated Gradients: not implemented.**

**Why SHAP is primary.** C1 is the binding constraint — the entire point of Contribution 3 is an
explanation a clinician can *act on*, and additive attributions are what make "these features
account for most of this alert" a true statement rather than a suggestive one. Fidelity, the first
of `TRD.md §4`'s three metrics, is where SHAP is strongest. Its cost is acceptable because
explanations are generated **only for flagged flows**, which are a small minority of traffic — the
detection path stays fast regardless.

**Why LIME is the fallback rather than a second primary.** It is genuinely faster and it sees the
deployed ensemble directly, which is a real advantage over SHAP's per-branch decomposition. But its
instability is a poor fit for a clinical setting, where two operators pulling up the same alert and
seeing different reasons would destroy trust in the system faster than no explanation at all. It is
the right tool when SHAP's latency is prohibitive, and T3.2's brief is exactly that.

**Why Integrated Gradients is dropped.** It is dominated: it shares SHAP's fusion problem while
offering less interpretable output and less reviewer recognition, and its speed advantage is already
covered by LIME. Implementing all three would spend Phase 3 time that `PRD.md §5.1` allocates to the
adaptive-threshold and incremental-learning work. Recorded here as a considered rejection.

## 5. The `TRD.md §4` tradeoff this document is required to record

> *"computed per detection (or per representative batch, if full per-instance SHAP is too slow for
> the deployed model — document whichever tradeoff is chosen)"*

**Chosen: per-detection SHAP for flagged flows, with LIME as an automatic fallback above a latency
budget.** Concretely, for T3.1/T3.2 to implement:

1. Explanations are generated **only for flows the model flags as Attack**, never for all traffic.
2. SHAP runs first, under an explicit per-explanation latency budget (a named constant in
   `shap_explainer.py`, tuned in Phase 3 against the Pi measurements from T4.3 — **not** guessed
   now, per `Build-Instructions.md` §A.1's rule against unmeasured latency claims).
3. If SHAP exceeds that budget, `lime_explainer.py` produces the explanation instead, and the
   output records **which method was used** — an explanation must never be silently downgraded.
4. Sequence attributions (`sequence_length × n_features`) are collapsed to per-feature scores by
   summing across timesteps, with the per-timestep map retained so "*when* in the sequence" remains
   answerable. The collapse rule goes in the T3.1 module docstring.

## 6. Handoff to T3.1 / T3.2 / T3.3

- **T3.1** — `shap_explainer.py`: per-branch `GradientExplainer`, recombined with the fusion's own
  confidence weights; top-N features by **original feature name**, never index (T3.1's VERIFY block);
  plus the short natural-language summary `TRD.md §4`'s output contract requires.
- **T3.2** — `lime_explainer.py`: `LimeTabularExplainer` over the flattened sequence, `num_samples`
  as a named constant, sampling seed fixed to `RANDOM_STATE`, and constrained perturbation so
  generated neighbours remain physically plausible flow records.
- **T3.3** — `metrics.py`: fidelity, stability, comprehensibility. **Stability must be run against
  LIME specifically** — §2.2 predicts it is the weak point, and if the measurement confirms that,
  it belongs in the Review 3 report as a finding, not as something quietly omitted.

---

# AMENDMENT (added at T3.2, after measurement)

> **Scope note:** `Build-Instructions.md` §A.2 confines each task to its declared **Files** line, and
> T3.2's is `src/xai/lime_explainer.py` alone. This amendment is appended anyway, because §4 above
> makes a factual claim that measurement has since falsified, and leaving it uncorrected would be
> the same defect this project criticises the base paper for. The original §1–§6 text is left
> **unedited** so the reasoning as it stood at T1.4 remains auditable; everything below is the
> correction.

## What was measured

`scripts/benchmark_xai.py`, 10 flagged detections, identical inputs to both methods, warm-up
excluded. Full output: `reports/t3_2_lime_vs_shap.md`.

| | SHAP (`GradientExplainer`) | LIME (`num_samples=1000`) |
|---|---:|---:|
| Median seconds per explanation | **0.212** | **0.401** |

**SHAP is ~1.9× faster than LIME.**

## Where §2 went wrong

§2.2 listed speed as LIME's headline advantage and §4 designated it "the faster fallback". That
comparison was made against SHAP **generically**, and the slow SHAP variant is `KernelExplainer`,
which needs thousands of forward passes per explanation.

**T3.1 never adopted `KernelExplainer`.** §2.1's own "On C4" paragraph chose `GradientExplainer`
per branch — a gradient-based method costing a handful of passes. The survey then carried §2.2's
speed comparison forward without noticing that §2.1 had already eliminated the SHAP variant that
comparison depended on. That is an internal inconsistency in this document, not a surprise in the
data.

LIME's cost here is structural rather than a tuning error: every explanation pushes 1,000 perturbed
windows through all three branches *and* the fusion. Cutting `num_samples` would buy speed directly
out of surrogate stability — already LIME's documented weakness (§2.2) and the exact property T3.3
must measure. Trading stability away to win a benchmark would be the wrong call.

## Corrected recommendation

**Primary: SHAP. Secondary: LIME — kept for faithfulness, not speed.**

§4's ranking of the two methods is unchanged, and SHAP's case is now *stronger*: it is both faster
and additively consistent. What changes is LIME's justification.

LIME earns its place through a property §2.2 identified and which does hold: **it explains the
deployed ensemble directly.** It needs only `predict_proba`, so it sees the fused model — confidence
weights and all — as one object. SHAP cannot: the fusion is not differentiable end to end, so T3.1
runs per branch and recombines the attribution maps with the fusion weights, a construction this
project invented and must defend. LIME provides an independent check that needs no such step.

## Consequence for §5's latency-cutover design

§5's items 2 and 3 specify a latency budget above which SHAP hands over to LIME. **That mechanism is
now pointless and should not be built**: the primary method is already the cheaper one, so the
fallback would never fire, and if it did it would make things slower. T3.1 correctly hardcodes no
budget. `Explanation.method` still records which method produced each explanation, which remains
worth keeping.

## A second finding, for T3.3

**Mean top-5 feature agreement between SHAP and LIME is 38%** (min 20%, max 40%). The two methods
largely disagree about which measurements drove the same decision on the same model.

Agreement is a consistency check, not a correctness one — both could be wrong together. But 38%
means at least one of them is not describing the model faithfully, and that cannot be resolved by
inspection. **T3.3's fidelity metric is now the deciding measurement**, not a box to tick: ablating
each method's top-attributed features and checking whether the prediction actually moves in the
predicted direction is what will say which explanation a clinician should be shown. If fidelity
comes out low for both, this project should report that its explainability layer is not yet
trustworthy — which would be an honest and publishable result given Objection #4.
