# TRD — Explainable Hybrid Deep Ensemble for IoMT Intrusion Detection with Adaptive Threat Intelligence
### Technical Requirements Document

> Companion to `PRD.md`. This document is the source of truth for exact data pipelines, architectures, and evaluation formulas. Where the base paper (HIDS-IoMT) leaves a detail unspecified — and it leaves several — this document says so explicitly rather than inventing a plausible-sounding number, per `PRD.md §2.1.2`'s own objection to that exact failure mode.

**Companion Documents:** `PRD.md` (scope and rationale) · `PPT.pdf` (Review 1 slides — source of the architecture sketch and timeline) · Base paper (Berguiga et al., HIDS-IoMT) · Senior's prior repo (`ids-main`, Badiyani) — reviewed for engineering lessons only, not extended

---

## 1. System Architecture Overview

```
IoTID20 / Edge-IIoTset (raw flow records, per base paper's own feature sets)
        │
        ▼
┌───────────────────────┐
│ Preprocessing (§2)     │  clean → label-encode → PSO feature select → [SPLIT] → resample (train only) → scale
└───────────┬───────────┘
            ▼
┌────────────────────────────────────────────────────────────┐
│ Sequence construction (§3.1): group records into per-flow/  │
│ per-session time-ordered sequences (fixes Objection #2's    │
│ "single flow record, not a sequence" problem)               │
└───────────┬──────────────────────────────────────────────────┘
            ▼
┌─────────────┬─────────────┬──────────────────┬───────────────────┐
│ CNN branch  │ BiLSTM      │ Transformer       │ GNN branch         │
│ (§3.2.1)    │ branch      │ -attention branch │ (§3.4, STRETCH)    │
│             │ (§3.2.2)    │ (§3.2.3)          │                    │
└──────┬──────┴──────┬──────┴─────────┬─────────┴─────────┬─────────┘
       └─────────────┴─────────────────┴───────────────────┘
                              ▼
                 Confidence-weighted voting fusion (§3.3)
                              ▼
              ┌───────────────┴────────────────┐
              ▼                                 ▼
    SHAP / LIME explanation (§4)     Adaptive threshold (§5.1) ── Incremental learning (§5.2)
              │                                 │
              └────────────────┬────────────────┘
                                ▼
                   Detection + explanation + confidence
                                ▼
              Deployment: inference graph on Raspberry Pi 4B (§7)
```

## 2. Leakage-Free Baseline (Contribution 1 — Committed Core)

### 2.1 Reproducing the Leak (must be demonstrated, not just asserted)
To make Objection #1 (`PRD.md §2.1.1`) concrete rather than theoretical, Phase 1 must reproduce the base paper's own pipeline order closely enough to show the leak empirically:
1. Load IoTID20 (or Edge-IIoTset), clean, label-encode (per base paper §II.G.1–2).
2. Apply PSO-based feature selection (base paper reduces IoTID20 from 83→62 features, Edge-IIoTset from 61→46 — see §6.1 for the caveat that PSO's own hyperparameters are unspecified in the base paper).
3. Apply SMOTE to the **full** dataset (as the base paper's own described order implies).
4. **Then** split into train/test/validation (80/10/10, matching the base paper's reported totals).
5. Train and evaluate; confirm that test-set accuracy is implausibly high and/or that synthetic minority samples can be shown to be near-duplicates of training-set points (e.g., via nearest-neighbor distance checks between train and test folds) — this is the concrete leakage evidence to report.

### 2.2 The Fix
1. Load, clean, label-encode, PSO-select features — identical to §2.1 steps 1–2.
2. **Split first**: train/test/validation split (80/10/10, or whatever split ratio the team settles on — document it) **before any resampling**.
3. Apply SMOTE (or the combined under/oversampling strategy) **only to the training fold**.
4. Scale features (fit scaler on training fold only; apply the fitted scaler to test/validation folds — do not fit on the full dataset).
5. Train and evaluate. This is the number reported everywhere else in this project as "the baseline" — never the leaky number from §2.1.

### 2.3 Correct Metric Formulas (fixes Objection #3)
Explicitly state the positive class in every reported table. Standard convention for this project: **Attack is the positive class** (not Normal, unlike the base paper — this project's convention should be stated once, prominently, in the results section to avoid any ambiguity).
```
Accuracy  = (TP + TN) / (TP + FP + TN + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```
Before publishing any table, verify against a hand-computed example from the actual confusion matrix (the way `PRD.md §10` did for the base paper) — this is the single check that would have caught Objection #3 before publication, and it must be applied to this project's own results as a matter of process, not just to the base paper's.

## 3. Deep Ensemble (Contribution 2 — Committed Core, GNN Stretch)

### 3.1 Sequence Construction (fixes Objection #2's "not a sequence" problem)
The base paper's CNN-LSTM operates on a single flow record — this project's ensemble must not repeat that mistake. Construct sequences by grouping consecutive flow records sharing a session/device key (e.g., source device ID + time window) into fixed-length windows (window size to be determined empirically in Phase 1/2 — document whatever value is chosen and why, unlike the base paper's silence on LSTM specifics). Pad or truncate to a fixed sequence length for batched training.

### 3.2 Branches — every layer count and unit count must be documented (this is the direct fix for Objection #2)
| Branch | Role | Requirement |
|---|---|---|
| 3.2.1 CNN | Spatial/local feature extraction across the per-timestep feature vector | Document exact conv layer count, filter sizes, and activation — do not leave this to "three Conv1D layers" without follow-up detail the way the base paper's Table 7 does |
| 3.2.2 BiLSTM | Temporal sequence modeling in both directions | **Must document unit count and layer count explicitly** — this is the single most direct fix for Objection #2, which is precisely the base paper's failure to do this |
| 3.2.3 Transformer (attention) | Long-range dependency modeling across the constructed sequence | Document number of attention heads, layer count, embedding dimension |

### 3.3 Confidence-Weighted Voting Fusion
Combine the three (or four, with GNN) branch outputs by weighting each branch's vote by its own predicted-class confidence (e.g., softmax max probability), rather than simple majority voting. Document the exact fusion formula used (e.g., weighted sum of per-class probabilities, weights = each branch's own confidence) — this is a design decision the team makes and must record, not something to leave implicit.

### 3.4 GNN Branch (Explicit Stretch Goal — see `PRD.md §5.2`)
- **Graph construction**: nodes = unique source/destination IPs observed in the flow records; edges = communication events between them (weighted by flow count or aggregated flow statistics).
- **Sparsity check (go/no-go gate)**: before investing further engineering time, compute the graph's density/connectivity on a sample of IoTID20 and Edge-IIoTset. If the graph is too sparse (e.g., most nodes have degree ≤1, no meaningful community/neighborhood structure), **cut this branch** — this is a pre-agreed decision per `PRD.md §9`, not a project failure.
- If it proceeds: a graph convolution or graph attention layer over this constructed graph, fused into §3.3 alongside the other branches.

## 4. Explainability (Contribution 3 — Committed Core)

- **SHAP**: feature-importance attribution, computed per detection (or per representative batch, if full per-instance SHAP is too slow for the deployed model — document whichever tradeoff is chosen).
- **LIME**: local surrogate-model explanation as a faster/complementary alternative where SHAP computation cost is prohibitive.
- **Output contract**: for every flagged detection, produce a ranked list of the top-N contributing features plus a short natural-language summary — target audience is a hospital IT lead or clinician, not an ML engineer (per `PRD.md §4`).
- **Evaluation** (since a user study is out of scope per `PRD.md §5.3`):
  - *Fidelity*: does the SHAP/LIME explanation's top features, when ablated, actually change the model's prediction in the expected direction?
  - *Stability*: do near-identical inputs produce near-identical explanations (low variance across repeated/perturbed runs)?
  - *Comprehensibility*: a lightweight proxy (e.g., explanation length, number of features cited) standing in for the unavailable clinician user study.

## 5. Adaptive Threshold + Incremental Learning (Contribution 4 — Committed Core)

### 5.1 Adaptive Threshold (fixes the "flat 500ms for every device" half of Objection #4)
Replace the base paper's single fixed τ=500ms inter-message threshold with a threshold that is a function of:
- **Device/patient criticality** (e.g., an insulin pump or cardiac monitor gets a tighter/faster threshold than a non-critical device).
- **Network context** (e.g., current traffic volume/congestion at the fog node).

Document the exact function used (a simple weighted formula is an acceptable starting point — e.g., `threshold = base_threshold * criticality_weight * context_factor` — but the specific weights and their justification must be recorded, not left implicit the way the base paper leaves τ=500ms unjustified beyond a citation).

### 5.2 Incremental Learning (fixes the "30-second-only check cycle" half of Objection #4)
Mechanism to absorb new labeled attack samples without a full retrain — candidate approaches to evaluate in Phase 3:
- Fine-tuning only the fusion layer (§3.3) on new samples while freezing branch weights.
- A small replay buffer of prior samples mixed with new ones to avoid catastrophic forgetting.
- Document whichever approach is chosen and why, along with a before/after check that prior attack-type detection performance does not regress after an incremental update.

## 6. Datasets, Features & Evaluation

### 6.1 Datasets
| Dataset | Role | Feature Count (base paper's PSO selection) |
|---|---|---|
| IoTID20 | Primary training/validation (Phase 2 onward) | 83 raw → 62 selected |
| Edge-IIoTset | Extended training/validation (Phase 3) | 61 raw → 46 selected |
| CICIDS2017 | Cross-dataset generalization check only — **not** primary training data | N/A (different feature schema entirely; used only to check whether the trained model's behavior generalizes qualitatively, not for a like-for-like accuracy comparison) |

**Open item — PSO parameters are unspecified in the base paper.** The base paper's Algorithm 5 (PSO feature selection) does not state particle count, iteration count, inertia weight `w`, acceleration constants `c1`/`c2`, or the fitness function used. This project must either (a) independently choose and document these parameters when re-running PSO, or (b) use a different, fully-specified feature-selection method (e.g., Random Forest feature importance, which the base paper also mentions using) and state clearly that this is a deliberate deviation from an unreproducible step in the base paper — do not silently guess PSO hyperparameters and present them as if they matched the original.

### 6.2 Evaluation Split
Document whatever split ratio is used (base paper claims 80/10/10 training/testing/validation — matching this is reasonable for comparability, but the split must happen **before** resampling per §2.2).

### 6.3 Metrics — see §2.3 for the exact formulas and positive-class convention. This applies to every table produced by this project, not just the leakage-free baseline.

## 7. Deployment & Efficiency Evaluation

- **Training**: Colab or lab GPU. Never on the Raspberry Pi (per `PRD.md §5.3` — explicitly out of scope, and impractical given the hardware).
- **Deployment**: only the trained inference graph (exported/quantized as appropriate for the target framework) is deployed to a **Raspberry Pi 4B (8GB)**.
- **Measurements to report** (this directly fixes the senior's-work gap of "no deployment evidence despite a real-time claim," `PRD.md §2.2`):
  - Inference latency per flow/sequence.
  - Throughput (flows/sequences processed per second) under simulated load.
  - Explicitly state the hardware (Pi 4B, RAM, OS) — do not report a real-time claim without these numbers, unlike both the base paper (which never reports training platform/time) and the senior's prior work (which never reports deployment numbers at all).

## 8. Software & Tools Stack

| Category | Tools |
|---|---|
| Language / runtime | Python 3.11 |
| Deep learning | TensorFlow 2.15 / Keras 3 (matches the base paper's own stack, for comparability) |
| Classical ML / preprocessing | scikit-learn 1.3, Pandas, NumPy |
| Explainability | SHAP, LIME |
| Graph learning (stretch, §3.4) | PyTorch Geometric |
| Compute | Google Colab / lab GPU for training; Raspberry Pi 4B (8GB) for inference-only deployment testing |

## 9. Verification Checklist (Acceptance Gates)

| Gate | How to Verify |
|---|---|
| Leakage demonstrated | §2.1's pre-split-SMOTE run shows implausibly high test accuracy and/or near-duplicate train/test samples via nearest-neighbor check |
| Leakage fixed | §2.2's split-first pipeline produces a lower, and now trustworthy, accuracy/precision/recall/F1 — this is the number cited everywhere else |
| No repeat of the base paper's metric-label bug | Every results table states its positive-class convention explicitly and is hand-verified against at least one confusion matrix per §2.3 |
| Sequences are real sequences | Confirm ensemble input shape is `(batch, sequence_length, features)`, not `(batch, features)` — directly checks that Objection #2's "single flow record" problem was not repeated |
| BiLSTM/Transformer fully specified | Architecture documentation lists exact unit/layer/head counts — checked against `PRD.md`'s NFR on reproducibility |
| GNN go/no-go | Graph density/connectivity check completed and documented before further GNN engineering time is spent |
| XAI output is usable | A non-ML-team-member (e.g., a teammate unfamiliar with the model internals) can read a sample SHAP/LIME explanation and correctly identify why a flagged flow was flagged |
| Adaptive threshold responds to criticality | A test with two synthetic device-criticality levels produces two different effective thresholds, not one shared value |
| Incremental learning doesn't regress old classes | Before/after comparison on prior attack-type detection accuracy after an incremental update shows no significant drop |
| Deployment numbers are real | Latency/throughput numbers on the actual Raspberry Pi 4B are recorded, not estimated or assumed |
