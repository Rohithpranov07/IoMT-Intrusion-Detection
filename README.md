# Explainable Hybrid Deep Ensemble for IoMT Intrusion Detection

A critique-driven reimplementation of **HIDS-IoMT** (Berguiga et al., *IEEE Access* 13, 2025),
targeting four verifiable defects in that paper rather than trying to beat its headline number.

> ### Positive-class convention
> **Attack is the positive class** (label 1) in every metric this repository reports
> (`TRD.md §2.3`). This is deliberately the *opposite* of the base paper's choice, and it is stated
> in every table, every report, every metrics code path, and the deployment manifest. A test
> (`tests/test_report_integrity.py`) fails the build if any generated document reports
> precision/recall without stating it.

**Course:** Computer Networks Project · **Team:** Sanhit, Malika, Nehaa, Rakshan, Rohith
**Specifications:** [`PRD.md`](PRD.md) (what/why) · [`TRD.md`](TRD.md) (how) ·
[`Build-Instructions.md`](Build-Instructions.md) (task list) · [`OPERATING_CONTRACT.md`](OPERATING_CONTRACT.md) (working rules)

---

## 1. Status at a glance

| Phase | Tasks | Status |
|---|---|---|
| 1 — Foundation | T1.1 – T1.5 | **Complete** |
| 2 — Implementation | T2.1 – T2.7 | **Complete** |
| 3 — Explainability & adaptive | T3.1 – T3.6 | **Complete**, except T3.3's human reader test (needs a person) |
| 4 — Deployment | T4.1 – T4.4 | **T4.1 and T4.4 complete.** T4.2/T4.3 built and verified but **need Raspberry Pi hardware** |
| 4b — NS-3 validation | T-N1 – T-N7 | **Complete except T-N3**, which is blocked on the same Pi as T4.2/T4.3 |

**393 tests: 373 pass, 20 skip** (those 14 are gated on the Raspberry Pi or on `artifacts/` being
built — they are skips, not failures). Environment pinned exactly in [`requirements.txt`](requirements.txt) and asserted
by `tests/test_environment.py`.

**Two things are outstanding.** They are listed in full in [§7](#7-what-is-missing), and neither is
"unwritten code" — one needs hardware, one needs a person.

---

## 2. What this project is, in one paragraph

HIDS-IoMT reports 99.92% accuracy, 99.91% precision, 99.99% recall and 99.95% F1 on IoTID20. This
project does **not** try to exceed those figures, and they must not be quoted without two caveats.
First, its precision and recall labels are very likely **swapped**: the paper takes *Normal* as its
positive class and labels both Equations 7 and 8 "Recall", one of which is the precision formula;
recomputing from its own Fig. 11c confusion matrix reproduces the reported accuracy exactly but
shows the reported "precision" is the true recall and vice versa (`PRD.md §10`). Second, its
evaluation fold was **contaminated** — §4 below reproduces that empirically. Instead this project
(1) reproduces and fixes the
data leakage that inflates it, (2) replaces an underspecified CNN-LSTM with a fully documented
CNN + BiLSTM + Transformer ensemble over genuine flow *sequences*, (3) adds SHAP explanations that
are **certified per alert** before being shown, and (4) replaces a flat 500 ms threshold with a
criticality-aware one. Along the way it reports every result that came out negative, of which there
are several.

---

## 3. Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/download_data.py                  # IoTID20  (~300 MB)
.venv/bin/python scripts/download_data.py --edge-iiotset   # Edge-IIoTset ML CSV (78 MB)

.venv/bin/python -m pytest tests/ -q                       # 393 tests
```

Then, in order (each depends on the previous):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_reproduce_leakage.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_leakage_free_baseline.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_ensemble_iotid20.ipynb  # writes artifacts/
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/04_train_edge_iiotset.ipynb
.venv/bin/python scripts/evaluate_edge_iiotset.py          # T3.6 record level
.venv/bin/python scripts/evaluate_xai.py                   # T3.3
.venv/bin/python scripts/ablation_study.py                 # T4.4
.venv/bin/python scripts/export_for_pi.py                  # T4.1
.venv/bin/python scripts/make_pi_bundle.py                 # T4.2 bundle
```

The NS-3 validation layer is independent of the above and needs a built `ns-3-dev`:

```bash
scripts/ns3_experiments/run_matrix.sh [path-to-ns-3-dev]   # 29 runs -> reports/ns3_runs/
.venv/bin/python scripts/ns3_experiments/aggregate_results.py
```

`artifacts/` and `data/` are gitignored build outputs — a fresh clone must run notebook 03 before
anything in `src/xai/` or `src/deployment/` will work.

---

## 4. The four objections, and what happened to each

| # | Objection to HIDS-IoMT | Outcome | Evidence |
|---|---|---|---|
| 1 | Pre-split SMOTE inflates the headline number | **Confirmed, more strongly than the PRD claimed. Fixed.** | [`notebooks/01`](notebooks/01_reproduce_leakage.ipynb), [`02`](notebooks/02_leakage_free_baseline.ipynb) |
| 2 | The LSTM is never specified; input is a single record | **Fixed** | [`docs/architecture_decision.md`](docs/architecture_decision.md) |
| 3 | Precision and recall labels are swapped | **Confirmed by re-derivation** | `PRD.md §10`, enforced repo-wide by test |
| 4 | Black box; flat 500 ms threshold for every device | **Partly fixed** | [`t3_3_xai_evaluation.md`](reports/t3_3_xai_evaluation.md), [`t3_4_adaptive_threshold.md`](reports/t3_4_adaptive_threshold.md) |

### Objection 1, quantified

IoTID20's cleaned majority class is 585,342 rows, and **2 × 585,342 = 1,170,684 reproduces the base
paper's stated post-SMOTE total exactly**. Its folds therefore partition synthetic data.
Reproducing that order put **46.61%** of the test fold into rows that never existed in the capture.

One correction *against* this project's own case, recorded because we do not get to round in our
favour: `PRD.md §2.1.1` says the paper's split totals match "exactly". They sum to 1,170,68**6** —
two more. Trivial, but the PRD's wording should read "within 2 rows of".

---

## 5. Headline results

All metrics: **Attack = positive**, hand-verified against their confusion matrices.

### 5.1 Leakage-free baseline (IoTID20, record level)

| Pipeline | Accuracy | Precision | Recall | F1 | Test fold synthetic | Near-duplicate |
|---|---:|---:|---:|---:|---:|---:|
| A. Leaky (base paper's order) | 0.9986 | 0.9975 | 0.9996 | 0.9986 | 46.61% | 43.98% |
| B. Honest order, duplicates kept | 0.9988 | 0.9991 | 0.9996 | 0.9994 | 0.00% | 42.18% |
| **C. Honest order + deduplicated** | **0.9964** | **0.9971** | **0.9990** | **0.9980** | **0.00%** | **5.70%** |

**Pipeline C is the project's baseline.** Pipeline B is why: fixing the SMOTE ordering *alone did
not lower the number*, because **58.2% of cleaned IoTID20 rows are exact duplicates** — a larger
leakage vector that neither the base paper nor the prior work mentions.

### 5.2 The ensemble (IoTID20, window level)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| bilstm + transformer | 0.9487 | 0.9563 | 0.9862 | **0.9710** |
| **transformer alone** | 0.9487 | 0.9678 | 0.9735 | **0.9707** |
| **Full three-branch ensemble** | 0.9399 | 0.9497 | 0.9831 | 0.9661 |
| bilstm alone | 0.9271 | 0.9295 | 0.9916 | 0.9595 |
| cnn alone | 0.9230 | 0.9408 | 0.9728 | 0.9565 |

**The best single branch beats the ensemble, and removing the CNN improves it (+0.0049).**
§5.1's and §5.2's numbers are **not comparable** — record-level vs window-level, different folds,
different imbalance handling.

### 5.3 The ensemble (Edge-IIoTset, window level)

The second dataset, over the 13 attack types whose timestamps survive
([`t3_6_ensemble.md`](reports/t3_6_ensemble.md)).

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **cnn alone** | 0.9798 | 0.9845 | 0.9929 | **0.9887** |
| **Full three-branch ensemble** | 0.9214 | 0.9979 | 0.9133 | 0.9537 |
| bilstm alone | 0.9201 | 0.9984 | 0.9114 | 0.9529 |
| transformer alone | 0.8945 | 0.9936 | 0.8868 | 0.9372 |

**The best single branch beats the ensemble again**, by −0.0349 F1 — the same verdict as §5.2 on a
second dataset, and by a wider margin. The winning branch is different (CNN here, Transformer
there), which is itself the point: no branch is reliably best, and the fusion does not exploit
that. Read §7.3 before treating this as decisive — 83.56% of these test windows appear verbatim in
the training fold, a property of the published CSV that no split can fix.

### 5.4 The threshold rules at network scale (NS-3 — **simulated, not measured**)

Every number here is NS-3 output. No Raspberry Pi number exists anywhere in this repository, and
none is implied by these ([`ns3_simulation_results.md`](reports/ns3_simulation_results.md)).

| Rule | Alerts | True positives | **False positives** | Attackers blocked | **Wearables blocked** |
|---|---:|---:|---:|---:|---:|
| `fixed` (the base paper's flat 500 ms) | 327 | 267 | **60** | 60/60 | **20/20** |
| `ewma` (congestion-adaptive) | 303 | 267 | 36 | 60/60 | 8/20 |
| **`criticality` (this project, T3.4)** | 296 | 267 | **29** | 60/60 | 9/20 |

**The flat rule blocks every legitimate wearable in the simulation.** A rule that quarantines the
whole patient-monitoring estate has not prevented an incident, it has caused one — which is
Objection #4 with a number attached. The criticality weighting halves the false alerts.

**This is not the `PRD.md §3.1` target being met.** That target is the *ensemble's* false-positive
rate on unseen attack types, measured at **−10.8%** in
[`t3_4_adaptive_threshold.md`](reports/t3_4_adaptive_threshold.md) and **still unmet**. The −51.7%
above is a different quantity that happens to share a name: how many legitimate devices a *timing
rule* quarantines in a simulated network. §5.2/5.3 and §5.4 must not be read across.

---

## 6. What is complete

<details>
<summary><b>Phase 1 — Foundation (T1.1 – T1.5)</b></summary>

| Task | Deliverable |
|---|---|
| T1.1 | [`clean.py`](src/preprocessing/clean.py), [`feature_select.py`](src/preprocessing/feature_select.py), [`leakage_check.py`](src/evaluation/leakage_check.py), [notebook 01](notebooks/01_reproduce_leakage.ipynb) |
| T1.2 | [`resample.py`](src/preprocessing/resample.py), [notebook 02](notebooks/02_leakage_free_baseline.ipynb) |
| T1.3 | [`docs/feature_selection_decision.md`](docs/feature_selection_decision.md) — **Random Forest importance, not PSO.** The base paper states no PSO hyperparameters, so they cannot be reproduced; guessing them was forbidden by the operating contract. |
| T1.4 | [`docs/xai_survey.md`](docs/xai_survey.md) — SHAP primary, LIME fallback. **Carries an AMENDMENT**: measurement later overturned its speed reasoning. |
| T1.5 | [`docs/architecture_decision.md`](docs/architecture_decision.md) — every layer, unit, head and window count frozen. |

</details>

<details>
<summary><b>Phase 2 — Implementation (T2.1 – T2.7)</b></summary>

| Task | Deliverable |
|---|---|
| T2.1 | [`sequence_builder.py`](src/preprocessing/sequence_builder.py) — sessions by `Dst_IP` + 1 s gap, window 10, stride 5, **session-level splitting** so overlapping windows cannot leak |
| T2.2 | [`cnn_branch.py`](src/models/cnn_branch.py) — 3 Conv1D (64/128/64), kernel 3, ReLU, dropout 0.3 |
| T2.3 | [`bilstm_branch.py`](src/models/bilstm_branch.py) — **2 layers, 64 units per direction** (the literal fix for Objection #2) |
| T2.4 | [`transformer_branch.py`](src/models/transformer_branch.py) — 2 encoder layers, 4 heads, key_dim 16, embed 64 |
| T2.5 | [`fusion.py`](src/models/fusion.py) — `w_b = α_b·c_b^γ / Σ`, γ = 1.0 |
| T2.6 | [notebook 03](notebooks/03_train_ensemble_iotid20.ipynb), [`phase2_results.md`](reports/phase2_results.md) — **the Review 2 deliverable** |
| T2.7 | [`gnn_go_nogo.md`](reports/gnn_go_nogo.md) — **NO-GO.** 99.68% of nodes have degree ≤ 1; the branch was never built |

</details>

<details>
<summary><b>Phase 3 — Explainability, adaptive, incremental (T3.1 – T3.6)</b></summary>

| Task | Deliverable |
|---|---|
| T3.1 | [`shap_explainer.py`](src/xai/shap_explainer.py) — per-branch `GradientExplainer` recombined with the fusion's own weights |
| T3.2 | [`lime_explainer.py`](src/xai/lime_explainer.py) — **demoted to cross-check, never operator-facing** |
| T3.3 | [`metrics.py`](src/xai/metrics.py), [`feature_glossary.py`](src/xai/feature_glossary.py), [`t3_3_xai_evaluation.md`](reports/t3_3_xai_evaluation.md) |
| T3.4 | [`threshold.py`](src/adaptive/threshold.py) — `effective = clamp(base × criticality × context)` |
| T3.5 | [`incremental.py`](src/adaptive/incremental.py) — replay-buffer fine-tuning |
| T3.6 | [`t3_6_edge_iiotset.md`](reports/t3_6_edge_iiotset.md) (record level, all 15 attack types) + [`t3_6_ensemble.md`](reports/t3_6_ensemble.md) via [notebook 04](notebooks/04_train_edge_iiotset.ipynb) (windowed ensemble, the 13 types carrying a clock) |

</details>

<details>
<summary><b>Phase 4 — Deployment (T4.1 – T4.4)</b></summary>

| Task | Deliverable |
|---|---|
| T4.1 | [`export_model.py`](src/deployment/export_model.py) — TFLite, 100% prediction agreement. **The BiLSTM does not convert as trained**; fixed by unrolling |
| T4.2 | [`pi_inference.py`](src/deployment/pi_inference.py) — standalone, imports nothing from `src/`. **Needs the Pi** |
| T4.3 | [`benchmark.py`](src/deployment/benchmark.py) — **refuses to write a report off Pi hardware.** Needs the Pi |
| T4.4 | [`ablation_study.md`](reports/ablation_study.md), [`final_comparison.md`](reports/final_comparison.md) |

</details>

<details>
<summary><b>Phase 4b — NS-3 network-level validation (T-N1 – T-N7)</b></summary>

Governed by [`NS3-Simulation.md`](NS3-Simulation.md), which folds a network simulation into Phase 4.
**Its §B.1 lists a `scratch/hids-iomt-adaptive.cc` and two completed runs as already existing. None
of it existed** — not in the tree, not in git history, not in `~/ns-3-dev/scratch/` — so T-N1 had
nothing to inventory and T-N2 no run to root-cause. The layer is built from scratch and nothing in
it is a reproduction of those runs.

| Task | Deliverable |
|---|---|
| T-N1 | [`ns3_architecture_inventory.md`](docs/ns3_architecture_inventory.md) — three **disjoint** node sets (20 wearables / 60 attacker-bots / 4 fog); capacity printed per-node *and* aggregate |
| T-N2 | [`zero_detection_investigation.md`](docs/zero_detection_investigation.md) — **mechanism confirmed**; see §8.11 |
| T-N3 | [`latency_provenance.md`](docs/latency_provenance.md) — **deferred, not approximated.** No Pi number exists and the base paper publishes none |
| T-N4 | [`threshold_rules.md`](docs/threshold_rules.md) — `fixed` / `ewma` / `criticality`, three distinct outcomes |
| T-N5 | [`incremental_scenario.md`](docs/incremental_scenario.md) — a threshold change standing in for a retrain, labelled as such |
| T-N6 | **60 raw runs** in `reports/ns3_runs/`: the full 3x4x4 cross-product plus the T-N2 and T-N5 probes, saved before any aggregation |
| T-N7 | [`ns3_simulation_results.md`](reports/ns3_simulation_results.md) — generated from those runs |

**One §F checkbox cannot be ticked**, for the same reason as T3.6's: T-N1–T-N5 declare single
**Files** lines, and the work also touched `docs/`, `scripts/ns3_experiments/`, `TRD.md §8` and
`Build-Instructions.md §B.3` — the last two because §F itself requires them updated.

</details>

### Extra decision records, produced because measurement demanded them

- [`docs/calibration_decision.md`](docs/calibration_decision.md) — calibration **measured and rejected**
- [`docs/raspberry_pi_setup.md`](docs/raspberry_pi_setup.md) — six-step run-through for when the Pi arrives

---

## 7. What is missing

Three items. **None is unwritten code.**

### 7.1 T3.3's human reader test — needs a person

`TRD.md §9`'s "XAI output is usable" gate requires **a team member unfamiliar with the model** to
read one alert and state why the flow was flagged. That cannot be self-assessed: anyone who has seen
the internals is no longer the reader the gate is about.

**To close it:** hand [`reports/t3_3_reader_test.md`](reports/t3_3_reader_test.md) to a teammate. It
is self-contained, free of model internals and metrics, and contains one deliberately **UNVERIFIED**
alert — if they do not notice it, the trust labelling has failed, which matters more than any
wording problem. Record the result in the report's table.

### 7.2 T4.2 / T4.3 Pi latency and throughput — needs hardware

**No latency or throughput number for the Raspberry Pi exists anywhere in this repository**, and
`src/deployment/benchmark.py` raises `NotOnTargetHardwareError` rather than writing one off-device.
`PRD.md §2.2`'s objection to the prior work is a real-time claim with no hardware behind it; the
prohibition is enforced in code, not left to discipline.

Everything else is done and verified: 1.4 MB of TFLite graphs (798 KB at float16), 100% prediction
agreement, a standalone runner, and a ~2 MB bundle. **To close it:** follow
[`docs/raspberry_pi_setup.md`](docs/raspberry_pi_setup.md). Step 4 is the one that matters — if
agreement is not 100.0000%, the Pi is not running the model this project evaluated.

### 7.3 A dataset the ensemble question can actually be settled on

**T3.6 itself is now complete** — see §8.9 for the correction that unblocked it and
[`t3_6_ensemble.md`](reports/t3_6_ensemble.md) for the result. What remains open is the *question*
[`docs/calibration_decision.md`](docs/calibration_decision.md) §6 asked T3.6 to answer: **does the
ensemble beat its best branch on a dataset where the branches genuinely differ?**

Edge-IIoTset now answers it the same way IoTID20 did — it does not, by −0.0349 F1 — but it is a
poor referee for the question, because 83.56% of its test windows appear verbatim in the training
fold and only 1,545 distinct feature vectors underlie its 142,088 timed rows. Two datasets agreeing
is worth more than one, and neither has the record diversity to make the test decisive.
**To close it:** a dataset with intact timestamps *and* genuine record diversity, or Edge-IIoTset's
raw pcaps (a 1.63 GB download this project did not take).

### 7.4 Also not done, and honestly so

- **CICIDS2017 cross-dataset validation** (`PRD.md` FR-11, a *Should*) — not attempted.
- **Repeated runs / significance testing** — single seed, single split throughout. This project
  criticises the prior work for exactly this omission and inherits it.
- **`PRD.md §3`'s >30% false-positive reduction target** — **not met**; best measured is −10.8%.

---

## 8. Findings worth reading even if you skip the rest

### 8.1 IoTID20 is 58.2% duplicate rows
Random splitting puts identical records on both sides regardless of when SMOTE runs. Fixing the
SMOTE ordering alone did not lower the accuracy; **the duplicates were the larger leakage vector**,
and neither the base paper nor the prior work mentions deduplication.

### 8.2 The ensemble does not earn its complexity on IoTID20
Best single branch 0.9707 F1 vs the ensemble's 0.9661. The branches agree on **90.9%** of windows,
so fusion can act on 9.1% — and on those it is *worse* than its best branch. An oracle resolving
every disagreement perfectly would gain only **+0.026** accuracy.
[`ablation_study.md`](reports/ablation_study.md)

### 8.3 Calibration was the obvious fix, and it does nothing
Two reports had blamed miscalibration and called temperature scaling the highest-value remaining
work. **That was an inference, and measurement refuted it**: the branches are already well
calibrated (ECE 0.012–0.023, and the BiLSTM is mildly *under*-confident at T = 0.79). Fitting the
fusion's `BRANCH_PRIORS` also fails — it overfits the validation fold. Both corrections were applied
at source with the original reasoning left visible.
[`calibration_decision.md`](docs/calibration_decision.md)

### 8.4 Edge-IIoTset has three defects, one severe
- **98.95% duplicates.** 157,800 rows contain **1,661 distinct feature vectors**. `DDoS_UDP`'s
  14,498 rows are **one** vector; `DDoS_ICMP`'s 14,090 are one; `DDoS_TCP`'s 10,247 are two. A random
  split places identical rows on both sides with certainty.
- **Placeholder spelling leaks the label.** Normal rows spell an absent field `"0"`, attack rows
  `"0.0"`. Three columns with three distinct values each reach **100% accuracy alone**.
- **Timestamps are damaged** — but not fatally; see §8.9, which corrects this entry.

After removing the leakage vectors only **30** features remain — so the base paper's stated
46-of-61 selection **necessarily includes leaking columns**.
[`t3_6_edge_iiotset.md`](reports/t3_6_edge_iiotset.md)

### 8.5 LIME is unusable here, and its apparent quality was noise
0.27 top-5 Jaccard under 1% input noise (SHAP: 0.87), with the model's own prediction flip rate at
0%. Raising `num_samples` does not rescue it, and its fidelity *collapses* as sampling converges
(+0.349 → +0.023). 88% of its alerts have a top feature arguing against their own verdict. Retained
as a cross-check; never shown to a human.

### 8.6 Explanations are certified per alert, and 18% fail
A method being reliable on average is not the same as *this* alert being sound.
`certify_explanation()` ablates the top-cited features and requires the decision to move more than
for random ones — **82% certify**, at 0.068 s each. The rest are **labelled, not hidden**.

### 8.7 The BiLSTM does not convert to TFLite
Keras 3 exports it as a dynamic `TensorList` loop. Two standard escapes were rejected on
measurement — one produced a 16 KB file that loaded, emitted **NaN**, and agreed with the Keras
model on **0%** of windows. Fixed by unrolling, valid only because the window length is frozen at 10.

### 8.8 An unpinned install silently changed the environment
A `pip install scipy` in Phase 3 upgraded numpy 1.26 → 2.4 and scikit-learn 1.3.2 → 1.9.0.
`requirements.txt` already named the right versions; pip moved past them anyway. It surfaced only
when `imbalanced-learn` stopped importing, several tasks later. Every package is now pinned exactly
and `tests/test_environment.py` asserts the live environment matches.

### 8.9 A blocker this project reported was its own reasoning error
An earlier `t3_6_edge_iiotset.md` cut T3.6's ensemble half on the grounds that Edge-IIoTset's
records "cannot be ordered in time" and that "row order is not capture order". The first is half
true and the second is false. `frame.time` loses its **date** to an unquoted comma inside
`Dec 26, 2021 22:14:30.939803000 IST`, but the **clock survives on 90.04% of rows**, and it
*confirms* the file's row order rather than contradicting it: across all 13 timed capture blocks,
0.9992–1.0000 of adjacent pairs agree, with 13 backward steps in 142,088 rows, worst −0.024 s.

An ordering existed the whole time. The error was inferring a property of the data from a corrupted
column instead of parsing the column — the same species of mistake as §8.3, caught the same way, by
measuring the thing that was assumed. Both halves of T3.6 now run.
[`t3_6_ensemble.md`](reports/t3_6_ensemble.md)

### 8.10 A fold-balancing proxy that is safe on one dataset and wrong on the next
`split_sessions` balances folds by counting each session's entries and documents them as *windows*;
notebook 03 passes *records*. On IoTID20 that is a fair proxy (81.2%/9.6%/9.2% of windows, matched
class balance). On Edge-IIoTset it is not — attack sessions run to 7,050 records against a Normal
median of 2 — and the first run of notebook 04 produced 93.7%/3.2%/3.2% folds whose **test fold
held no Normal windows at all**. Every model scored precision 1.0000 against zero negatives, which
reads as a triumph and is an empty set. Notebook 04 splits window-level ids; notebook 03's folds
were re-checked and stand.

---

### 8.11 A congestion-adaptive detector goes blind — at whichever end you couple it to
A threshold multiplied by a queue-load term collapses to zero at one end of that term's range, and
no inter-message gap is shorter than zero. Coupled **inversely**, the detector dies when
**congested** — 8 alerts and 0/60 attackers blocked at 99.5% queue occupancy, i.e. exactly when the
network is under attack hard enough to saturate it. Coupled **directly**, it under-detects when
**fast** (45/60). Not a bug at either end: it is what "let queue state drive the threshold" means.

**It does not degrade under load — it latches off.** The full 48-run cross-product shows the blind
spot is not at a latency at all: at 2000 µs the rule blocks 100% of attacker-bots at 20 and 40 bots
and 0% at 60 and 80. What decides it is a race. `blockAfter` — how many violations the rule waits
for before quarantining a source — is the *only* difference between blocking 60/60 with zero queue
overflow and blocking 0/60 while dropping 280,128 packets. Blocking removes load, so blocking early
keeps blocking possible; miss the window and the queue saturates, the threshold collapses toward
zero, nothing can violate zero, and the loop never comes back.

A rule of that shape is a **congestion detector wearing an intrusion detector's label**, and it is a
fair fifth objection to timing-only adjacency rules — scoped, carefully, to *this project's own
reconstruction*, since no published implementation was available to test.

This project's own criticality rule cannot fail this way: its load term is bounded to ±15% and
clamped to [50, 2000] ms, so load can modulate the threshold but never annihilate it. That clamp was
written into T3.4 for an unrelated reason; this is the first evidence it prevents a real failure.
[`zero_detection_investigation.md`](docs/zero_detection_investigation.md)

## 9. Repository layout

```
├── PRD.md · TRD.md · Build-Instructions.md · OPERATING_CONTRACT.md   the specifications
├── docs/            5 decision records (feature selection, XAI, architecture, calibration, Pi setup)
├── notebooks/       01 leakage · 02 baseline · 03 ensemble · 04 Edge-IIoTset  (executed)
├── src/
│   ├── preprocessing/   clean · feature_select · resample · sequence_builder
│   ├── models/          cnn · bilstm · transformer · fusion · baseline · train_utils · calibration
│   ├── evaluation/      leakage_check · metrics
│   ├── xai/             shap_explainer · lime_explainer · metrics · feature_glossary
│   ├── adaptive/        threshold · incremental
│   └── deployment/      export_model · pi_inference · benchmark
├── scratch/         hids-iomt-adaptive.cc  (NS-3, C++ only; copied into ns-3's scratch/)
├── scripts/         15 entry points + ns3_experiments/; every report is generated, never hand-written
├── tests/           16 files, 393 tests (373 pass, 20 hardware/artifact-gated skips)
└── reports/         17 generated reports + ns3_runs/ (60 raw NS-3 runs)
```

**No `src/models/gnn_branch.py`** — T2.7 returned no-go, and a test fails the build if it reappears.

### Deviations from `Build-Instructions.md` §B.3's layout

Added, each because a notebook cannot be unit-tested and duplicated logic drifts:
`src/config.py`, `evaluation/metrics.py`, `models/baseline.py`, `models/train_utils.py`,
`models/calibration.py`, `xai/feature_glossary.py`.

**T3.6 spans two files rather than its declared one.** T3.6's declared **Files** line is
`notebooks/04_train_edge_iiotset.ipynb` alone. That notebook exists and holds the ensemble half, but
the record-level half and the dataset-defect findings live in
[`scripts/evaluate_edge_iiotset.py`](scripts/evaluate_edge_iiotset.py), and the timestamp recovery
both depend on is in `src/preprocessing/clean.py`. Recorded because `Build-Instructions.md` §A.2
forbids touching files outside a task's declared line: a notebook cannot be unit-tested, and the
timeline reconstruction is the kind of code that must be. §F's "No task touched files outside its
declared **Files** line" checkbox is therefore **still not checkable**, and saying so is cheaper
than pretending otherwise.

---

## 10. How honesty is enforced mechanically

Rules that depend on remembering get forgotten exactly when the numbers are inconvenient, so the
operating contract's rules are tests:

| Rule | Test |
|---|---|
| Never cite the base paper's figures without the metric-inversion caveat **in the same paragraph** | `test_report_integrity.py` scans every generated `.md` |
| Every metrics table states its positive class | `test_report_integrity.py` |
| The cut GNN branch must not reappear | `test_fusion.py`, `test_report_integrity.py` |
| No Pi timing may be claimed without a Pi | `test_report_integrity.py`, `benchmark.py` raises |
| The Pi runner must import nothing from `src/` | `test_pi_deployment.py` parses its AST |
| Sequence length must exceed 1 | `test_sequence_builder.py` |
| Branch hyperparameters must match the frozen decision record | `test_branches.py` |
| The rejected calibration module must not be wired in | `test_calibration.py` |
| The environment must match `requirements.txt` | `test_environment.py` |
| The NS-3 criticality rule must stay numerically identical to T3.4's Python one | `test_ns3_simulation.py` |
| The clamp that makes that rule immune to §8.11's failure must not be removed | `test_ns3_simulation.py` |
| No saved NS-3 run may be quotable without its latency's origin attached | `test_ns3_simulation.py` |

---

## 11. Reading order for a reviewer

1. [`reports/final_comparison.md`](reports/final_comparison.md) — what was established, and §5 for what was not
2. [`reports/phase2_results.md`](reports/phase2_results.md) — the Review 2 deliverable
3. [`reports/ablation_study.md`](reports/ablation_study.md) — the ensemble does not beat its best branch
4. [`docs/calibration_decision.md`](docs/calibration_decision.md) — a diagnosis this project made, then refuted
5. [`reports/t3_6_edge_iiotset.md`](reports/t3_6_edge_iiotset.md) — 98.95% duplicates
6. [`reports/t3_3_xai_evaluation.md`](reports/t3_3_xai_evaluation.md) — is the explainability layer trustworthy?
7. [`reports/ns3_simulation_results.md`](reports/ns3_simulation_results.md) — the threshold rules at network scale, and §2 for a detector that goes blind under load
