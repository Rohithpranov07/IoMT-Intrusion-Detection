# PRD — Explainable Hybrid Deep Ensemble for IoMT Intrusion Detection with Adaptive Threat Intelligence
### Product/Project Requirements Document

> Companion to `TRD.md` (technical spec). This document answers *what* to build and *why*; the TRD answers *exactly how*. If this PRD and the TRD ever disagree on a technical detail, **the TRD wins** — update this document to match, not the other way around.

**Course:** Computer Networks Project · **Team:** Sanhit, Malika, Nehaa, Rakshan, Rohith
**Base Paper (the system being extended and critiqued):** Berguiga, A., Harchay, A., Massaoudi, A., *"HIDS-IoMT: A Deep Learning-Based Intelligent Intrusion Detection System for the Internet of Medical Things,"* IEEE Access, vol. 13, 2025. CNN-LSTM hybrid, fog-deployed on Raspberry Pi, evaluated on IoTID20 and Edge-IIoTset.
**Prior Work Reviewed (a senior's project, not part of this codebase's target architecture):** Badiyani, V., *"An Intelligent Intrusion Detection System Using Machine Learning on Network Flow Features with an Optimized Stacking Ensemble Framework."* Stacking ensemble (LR+DT+RF, meta-classifier LR) on CICIDS2017.
**Reference Slides:** `PPT.pdf` (Review 1 presentation) — the source of the objections, contribution plan, timeline, and risk register this PRD formalizes.
**Milestones:** Review 2 — 7 September · Review 3 — after 2 November

---

## 0. What This Project Actually Is (read this before anything else)

Two other artifacts exist in this repo's history and must not be confused with the actual scope:

- **The base paper (HIDS-IoMT)** is not a system to reproduce faithfully — it is the **target of a critique**. Four specific, evidence-backed objections against it drive this project's contribution list (§2.1). Its headline numbers (99.92% accuracy, etc.) are **not** treated as a ceiling to approach; §2.1.3 shows they are very likely miscomputed.
- **The senior's stacking-ensemble repo (`ids-main`, Badiyani)** is not the architecture this project builds on top of — it is **prior work with its own identified gaps** (§2.2), reviewed for context and lessons, not extended in place. Its live-dashboard engineering (FastAPI/Scapy/React) is a useful reference for what "real-time deployment evidence" looks like, since the base paper provides none — but its model (classical stacking on CICIDS2017) is explicitly the wrong domain for this project's IoMT target and is not the deployed model here.
- **This project** is the four-contribution extension described in `Date_24-07-26.docx` and `PPT.pdf`: a leakage-free baseline, a deep ensemble (CNN-BiLSTM + Transformer, GNN as stretch goal), SHAP/LIME explainability, and adaptive threshold + incremental learning — evaluated on **IoTID20 and Edge-IIoTset** (the base paper's own datasets, for a fair comparison), with **CICIDS2017 held out only for cross-dataset validation**, not as a primary training set.

## 1. Executive Summary

The base paper (HIDS-IoMT) claims near-perfect IoMT intrusion detection but has four verifiable defects: likely data leakage from pre-split SMOTE, an underspecified LSTM component, swapped precision/recall labels, and no real explainability or adaptive behavior. A senior's prior stacking-ensemble project addressed none of these — it targeted a different dataset/domain (CICIDS2017, general network traffic) entirely and reported no deployment evidence. This project targets the base paper's actual weaknesses directly: it re-establishes a leakage-free performance baseline on the base paper's own datasets, replaces the underspecified CNN-LSTM with a properly sequenced CNN-BiLSTM+Transformer ensemble, adds SHAP/LIME so a clinician or IT lead can see *why* traffic was flagged, and adds an adaptive, criticality-aware threshold plus incremental learning so the system can absorb new attack types without full retraining.

## 2. Problem Statement

### 2.1 Four Objections to the Base Paper (HIDS-IoMT) — the actual problem this project solves

1. **Data leakage inflates the headline number.** SMOTE appears to run before the train/test split — Table 5's reported training/testing/validation totals (936,548 / 117,069 / 117,069) match the paper's own post-SMOTE instance count (1,170,684) exactly, meaning synthetic samples interpolated from training points also landed in the test set. A model evaluated this way cannot be trusted to generalize to genuinely unseen traffic.
2. **The LSTM is never actually specified.** Table 7 (hyperparameters) lists three Conv1D layers, ReLU, Adam, dropout, epochs, batch size — and nothing about the LSTM: no unit count, no layer count. Worse, each input is a single flow record, not a sequence, which undermines the stated rationale for using an LSTM (capturing long-term temporal dependencies) in the first place.
3. **The reported metrics are very likely inverted.** The paper defines the positive class as "Normal" (not "Attack"), then labels both Equations 7 and 8 as "Recall" — one of which is actually the precision formula (`TP/(FP+TP)`). Recomputing directly from the paper's own Fig. 11c confusion matrix (87,987 / 77 / 9 / 28,996) reproduces the reported accuracy (99.927%, matches Table 8) but shows the reported "precision" (99.913%) is actually the true recall, and the reported "recall" (99.99%) is actually the true precision. The same swap appears to hold on the Edge-IIoTset results (Fig. 12c). This is a paper-reading finding, not something that needs new experiments to confirm — see §9 (Appendix) for the full worked calculation.
4. **It is a black box, and not truly real-time for a life-critical context.** No detection comes with an explanation a clinician or IT operator could act on. The adjacency-table threshold τ is fixed at 500ms for every device class regardless of criticality, and the periodic malicious-node check only wakes once every 30 seconds — a long window when the alert concerns a patient's vital signs.

### 2.2 Gaps in the Senior's Prior Work (Badiyani stacking ensemble) — reviewed, not extended

- **Below the base paper on its own (disputed) numbers**: 98.83% vs. the base paper's claimed 99.92%, and only 0.13 points above a plain Decision Tree, with no repeated runs or significance testing to show the stacking ensemble's improvement is real rather than noise.
- **Wrong domain**: trained and evaluated on CICIDS2017 (general enterprise campus traffic), which contains no IoMT or medical-device traffic — the clinical context that motivates this whole project is absent from that model.
- **No temporal modeling**: Logistic Regression, Decision Tree, and Random Forest are all stateless per-flow classifiers; whatever sequential structure the base paper's LSTM was reaching for (however underspecified) is entirely dropped.
- **No deployment evidence despite a real-time claim**: the codebase includes a genuinely working FastAPI/Scapy/React live-capture dashboard (useful as an engineering reference), but no hardware, latency, or throughput numbers are reported anywhere in the paper itself.
- **IQR outlier removal is risky for this problem**: attacks are frequently statistical outliers by nature; capping/removing them before training can delete the rare attack samples the model most needs to learn from.

## 3. Goals and Success Metrics

This project does **not** target beating a raw accuracy number — the base paper's claimed 99.92% is both already near-saturated and, per §2.1.3, likely mismeasured. Success is defined by fixing the four objections directly.

| Contribution | Fixes Objection | Success Criterion |
|---|---|---|
| 1. Leakage-free baseline | #1 | SMOTE (or equivalent resampling) applied only inside the training fold; a single honest accuracy/precision/recall/F1 number published, with precision/recall correctly labeled (not swapped, per §2.1.3) |
| 2. Deep ensemble (CNN-BiLSTM + Transformer, GNN stretch) | #2 | A fully specified architecture (documented layer counts, units, sequence construction) trained on genuine flow *sequences*, not single records; fusion via confidence-weighted voting, evaluated against the leakage-free baseline |
| 3. SHAP/LIME explanations | #4 (explainability half) | Every flagged detection carries a per-feature attribution a non-ML-expert can read; measured via fidelity, stability, and comprehensibility (quantitative substitute for a user study — see §7) |
| 4. Adaptive threshold + incremental learning | #4 (real-time half) | Detection threshold weighted by patient criticality and network context (not a flat 500ms for every device); new attack types absorbed without a full retrain; false positives reduced by >30% on unseen attack types vs. the leakage-free baseline |

### 3.1 Measurement Plan
| Category | Metrics |
|---|---|
| Detection | Leakage-free accuracy, precision, recall, F1-score |
| Explainability | Fidelity, stability, comprehensibility |
| Efficiency | Inference latency and throughput on the Raspberry Pi 4B |
| Adaptability | False-positive rate reduction (target: >30%) on unseen/novel attack types |

## 4. Users / Personas

| User | Need |
|---|---|
| Fog-node operator / hospital IT lead | A detection that comes with a reason, not just a label — SHAP/LIME output should be readable without ML background |
| Clinical engineering staff | Confidence that alert sensitivity scales with how critical the affected device is (an insulin pump vs. a lobby kiosk should not share one flat threshold) |
| Course evaluators (Review 2 / Review 3) | A clearly falsifiable, evidence-based critique of the base paper and prior work, plus a working, honestly-measured system that addresses the identified gaps |
| The team itself | A scoped, deliverable plan that fits the semester timeline without overcommitting to the GNN stretch goal at the expense of the committed core |

## 5. Scope

### 5.1 In Scope — Committed Core
- Reproducing the base paper's pipeline faithfully enough to demonstrate the leakage bug, then fixing it (split-then-resample).
- CNN-BiLSTM + Transformer-attention ensemble over properly constructed flow *sequences* (not single records), trained and validated on IoTID20, later extended to Edge-IIoTset.
- Confidence-weighted voting fusion across ensemble branches.
- SHAP and LIME integration for per-detection explanations.
- Adaptive threshold logic weighted by a patient/device criticality signal and network context.
- Incremental learning mechanism to absorb new attack types without a full retrain.
- Deployment of the trained **inference graph only** to a Raspberry Pi 4B (8GB), with latency/throughput measurement under load. Training itself happens on Colab / lab GPU, not on the Pi.
- Cross-dataset validation using CICIDS2017 as a held-out check (not primary training data).

### 5.2 In Scope — Explicit Stretch Goal (may be cut)
- **GNN branch over network topology**, constructed from source/destination IP pairs in the IoTID20/Edge-IIoTset flow records. If the induced graph is too sparse to be informative, this branch is cut — this is a pre-agreed decision, not a failure if it happens (see `PPT.pdf` risk register, §9).
- If Phase 2 (§8) slips, the Transformer-fusion work is still delivered; the GNN is the first thing dropped.

### 5.3 Out of Scope (this project, this semester)
- **A user study with real healthcare IT professionals** — access is uncertain; quantitative explainability metrics (fidelity, stability, sparsity) are the substitute, per the risk register.
- **Training on the Raspberry Pi itself** — impractical given the hardware; only inference is deployed there.
- **Re-extending the senior's stacking-ensemble codebase** — it is reviewed as prior work, not built upon architecturally (see §0).
- **Any claim of beating the base paper's raw accuracy number** — that number is disputed (§2.1.3) and is not this project's target metric.
- **Automated response/remediation** (blocking, quarantining) — detection and explanation only.

## 6. Functional Requirements

| ID | Requirement | Priority | TRD Reference |
|---|---|---|---|
| FR-1 | System shall reproduce the base paper's preprocessing pipeline (clean → label-encode → PSO feature selection → SMOTE → min-max scale) closely enough to demonstrate the pre-split-SMOTE leakage empirically. | Must | TRD §2.1 |
| FR-2 | System shall re-run the same pipeline with resampling applied only inside the training fold, and report the resulting (lower, honest) detection metrics. | Must | TRD §2.2 |
| FR-3 | System shall construct genuine flow *sequences* (multiple time-ordered records per flow/session) as ensemble input, not single flow records. | Must | TRD §3.1 |
| FR-4 | System shall implement a CNN branch, a BiLSTM branch, and a Transformer-attention branch, each fully specified (documented layer counts and units — unlike the base paper's Objection #2). | Must | TRD §3.2 |
| FR-5 | System shall fuse branch outputs via confidence-weighted voting. | Must | TRD §3.3 |
| FR-6 (stretch) | System shall optionally include a GNN branch over a graph constructed from source/destination IP relationships in the flow data. | Could | TRD §3.4 |
| FR-7 | System shall generate a SHAP-based and a LIME-based explanation for every flagged detection. | Must | TRD §4 |
| FR-8 | System shall compute an adaptive detection threshold as a function of device/patient criticality and network context, replacing the base paper's flat 500ms rule. | Must | TRD §5.1 |
| FR-9 | System shall support incorporating new labeled attack samples via an incremental-update mechanism that does not require retraining the full model from scratch. | Must | TRD §5.2 |
| FR-10 | System shall report inference latency and throughput when the trained model is deployed on a Raspberry Pi 4B. | Must | TRD §7 |
| FR-11 | System shall evaluate on IoTID20 and Edge-IIoTset (matching the base paper) and additionally cross-validate on CICIDS2017 as a held-out generalization check. | Should | TRD §6 |
| FR-12 | System shall report all evaluation metrics with correctly labeled precision/recall (explicitly avoiding the base paper's Objection #3 swap). | Must | TRD §6.3 |

## 7. Non-Functional Requirements

| NFR | Requirement |
|---|---|
| Scientific honesty | No metric is reported without stating exactly how it was computed and which class was treated as positive — this directly targets Objection #3. |
| Reproducibility | Every architectural choice (LSTM/BiLSTM units, Transformer heads, PSO parameters if re-run) must be documented in full — directly targets Objection #2's "not stated anywhere" problem. |
| Deployment honesty | Any real-time or deployment claim must be backed by an actual measured number (latency, throughput, hardware used) — directly targets the senior's work's "no deployment evidence" gap. |
| Explainability | SHAP/LIME output must be interpretable by a non-ML-expert reader without additional tooling. |
| Timeliness | Adaptive threshold and incremental learning must plausibly operate faster than a fixed 30-second check cycle for high-criticality contexts. |

## 8. Milestones

| Phase | Dates | Focus | Deliverable |
|---|---|---|---|
| Phase 1 — Foundation | 17–24 Aug | Reproduce base paper end-to-end; build leakage-free baseline; survey SHAP/LIME/Integrated Gradients; freeze ensemble architecture | Documented leakage bug + fix; architecture decision recorded |
| Phase 2 — Implementation | 25 Aug–7 Sept | Build CNN-BiLSTM+Transformer; train/validate on IoTID20; confidence-weighted fusion | **→ Review 2, 7 September** |
| Phase 3 — Explainability | 8 Sept–5 Oct | Integrate SHAP & LIME; adaptive threshold algorithm; incremental learning module; extend training to Edge-IIoTset | Working XAI layer + adaptive/incremental modules |
| Phase 4 — Deployment | 6 Oct–2 Nov | Deploy to Raspberry Pi 4B; latency & throughput under load; ablation + comparative analysis | **→ Review 3** |

## 9. Risks & Mitigations (from `PPT.pdf`, formalized)

| Risk | Mitigation |
|---|---|
| Scope is large for one semester (four novel components) | Contributions 1 (leakage-free baseline) and 3 (SHAP/LIME) are the committed core. Transformer fusion is planned. The GNN is an explicit stretch goal, dropped first if Phase 2 slips. |
| GNN needs topology data the flow-record datasets don't natively provide | Construct the communication graph from source/destination IP pairs in the flow records; if the induced graph is too sparse to be informative, cut the GNN branch — pre-agreed, not a failure. |
| Accuracy headroom is near zero against the base paper's claimed 99.92% | Do not target raw accuracy. Success is a defensible leakage-free baseline plus measurable gains in explainability and false-positive rate (§3.1). |
| Training on a Raspberry Pi is impractical (and the base paper never reports training time/platform either) | Train on Colab / lab GPU; deploy only the inference graph to the Pi; report inference latency/throughput separately from training cost. |
| A user study with healthcare IT professionals may not be feasible | Substitute quantitative explainability metrics — fidelity, stability, sparsity — which are reproducible and don't require external participants. |

## 10. Appendix — The Metric Inversion, Worked (for direct citation in the paper/report)

From the base paper's Fig. 11c confusion matrix (IoTID20, binary, HIDS-IoMT model):

| | Pred. Normal | Pred. Attack |
|---|---|---|
| **Actual Normal** | 87,987 | 77 |
| **Actual Attack** | 9 | 28,996 |

The paper defines TP as "instances correctly classified as normal" — i.e., **Normal is the positive class**, not Attack.

- **Accuracy** = 116,983 / 117,069 = **99.927%** — matches the paper's own Table 8 exactly. The matrix is internally consistent with the paper's accuracy claim.
- **True recall** (with Normal positive) = 87,987 / (87,987 + 77) = **99.913%** — but the paper reports this number as its *precision*.
- **True precision** (with Normal positive) = 87,987 / (87,987 + 9) = **99.990%** — but the paper reports this number as its *recall*.

Equations 7 and 8 in the base paper are both labeled "Recall"; Equation 8 (`TP/(FP+TP)`) is actually the precision formula. The same swap pattern holds on the Edge-IIoTset results (Fig. 12c). There is a further internal inconsistency worth noting in the report: with Normal as the positive class, a "false positive" is technically an *attack misread as normal* (a missed attack) — but the paper's own discussion section describes false positives as false alarms disrupting healthcare operations, which is the opposite framing. Table 5's stated class balance (75% normal / 25% attack in the confusion matrix) also does not match a stratified split of the claimed 50/50 SMOTE-balanced dataset from Section G — Table 5, the SMOTE description, and the confusion matrices cannot all be simultaneously correct as written.
