# Build-Instructions — Explainable Hybrid Deep Ensemble for IoMT Intrusion Detection with Adaptive Threat Intelligence

> Companion to `PRD.md` (what/why) and `TRD.md` (exact technical contracts). This is the implementation task list that builds the four committed contributions from scratch: a leakage-free baseline, a CNN-BiLSTM+Transformer ensemble (GNN as a gated stretch goal), SHAP/LIME explainability, and adaptive threshold + incremental learning. Nothing in this repo currently implements any of these — the only prior code reviewed (the senior's `ids-main` stacking-ensemble repo) is explicitly **not** the codebase being extended (`PRD.md §0`), so this document assumes a fresh build against the file layout in §B.3.
>
> If this document and the TRD disagree on a technical detail, **the TRD wins** — update this document, not the code, to match.

**Milestones this document targets:** Review 2 (7 September — Phase 1+2 deliverables) · Review 3 (after 2 November — Phase 4 deliverable)

---

## §A — Operating Contract (paste into `OPERATING_CONTRACT.md`)

### A.1 Anti-hallucination rules
- **Never state a positive-class convention implicitly.** Every task that computes or reports accuracy/precision/recall/F1 must state explicitly which class is "positive" in code comments, docstrings, and any output report. This project's own §2.3 convention (`TRD.md §2.3`: **Attack is positive**) must be followed consistently — repeating the base paper's Objection #3 mistake anywhere in this project's own results is the single worst outcome this document exists to prevent.
- **Never fabricate PSO hyperparameters and present them as reproducing the base paper.** `TRD.md §6.1` is explicit: the base paper never states particle count, `w`, `c1`/`c2`, or fitness function. If a task involves feature selection, it must either (a) choose and document its own PSO parameters plainly labeled as this project's own choice, or (b) use a fully-specified alternative (e.g., Random Forest feature importance) and say so. Never write a PSO call with parameters presented as "the paper's settings."
- **Never claim "real-time" or "deployed" without an actual measured number.** Any latency/throughput claim must trace to a real run on the actual Raspberry Pi 4B (`TRD.md §7`), not an estimate, not a desktop benchmark presented as if it were the Pi's number.
- **Never silently keep the GNN branch if the sparsity go/no-go check says no-go.** `TRD.md §3.4` and `PRD.md §9` pre-agree that a sparse graph means cutting the branch. If T2.7's check comes back no-go, later tasks must not quietly re-add GNN code — that would contradict a decision the team already made and documented.
- **Never let the sequence builder collapse to single-record windows.** A window/sequence length of 1 silently reintroduces exactly the failure this project exists to fix (Objection #2, `PRD.md §2.1.2`). `TRD.md §9`'s "Sequences are real sequences" gate requires shape `(batch, sequence_length, features)` with `sequence_length > 1`, asserted in code, not just assumed.
- **Never compare this project's results to the base paper's raw numbers without the metric-inversion caveat.** Any report or slide citing HIDS-IoMT's 99.92%/99.91%/99.99%/99.95% must note, in the same paragraph, that those precision/recall labels are very likely swapped (`PRD.md §2.1.3` / `§10`) — omitting that caveat when convenient (e.g., because this project's own numbers are lower) is an integrity failure, not a formatting nicety.

### A.2 Anti-drift rules (files each task may touch)
Each task in §D lists an explicit **Files** line. Do not edit files outside that list for that task.

### A.3 Quality gates
- Every model/pipeline file must have a module-level docstring stating its exact hyperparameters (layer counts, unit counts, window sizes, thresholds) in plain numbers — this project's whole premise is that the base paper failed to do this, so it cannot fail to do this itself.
- Deterministic: fixed `random_state`/seed wherever randomness is involved (splits, resampling, weight init), so results are reproducible run-to-run and across team members.
- No bare `except:` — catch and log specific exceptions.
- Type hints on function signatures for new Python code.

### A.4 Working method
1. Read `PRD.md` and `TRD.md` in full before starting any task.
2. Work phase by phase, task by task, in the order given in §D. Phase 2 depends on Phase 1's leakage-free pipeline and frozen architecture decisions; Phase 3 depends on Phase 2's trained ensemble; Phase 4 depends on Phase 3's completed modules.
3. After each task, run whatever the task produces (a script, a notebook cell, a test) and confirm it does what the VERIFY block says before moving to the next task.
4. Commit per task with a conventional-commit message.

### A.5 Definition of done
Every gate in `TRD.md §9` passes, every §F checkbox below is checked, and no task expanded scope beyond its declared **Files** line.

---

## §B — Canonical Spec Index & Confirmed Facts

### B.1 Documents
| Doc | Role |
|---|---|
| `PRD.md` | Scope, the four objections, the four contributions, milestones, risk register |
| `TRD.md` | Exact pipeline, architecture, evaluation formulas, deployment plan |
| `PPT.pdf` | Original Review 1 slides — source of the timeline and risk register |
| Base paper (Berguiga et al., HIDS-IoMT) | The system under critique — reference for dataset/feature counts only, not for its (disputed) results |
| `NS3-Simulation.md` | The network-level validation layer folded into Phase 4 — NS-3 topology, threshold-rule comparison, experiment matrix. Governs its own tasks (T-N1–T-N7); the TRD still wins on any conflict |
| Senior's `ids-main` repo (Badiyani) | Reviewed for engineering lessons (its live FastAPI/Scapy/React dashboard is a good reference for what real deployment evidence looks like) — **not** the codebase extended here |

### B.2 Confirmed facts to build against (don't re-derive differently)
| Fact | Value | Source |
|---|---|---|
| IoTID20 feature counts | 83 raw → 62 selected (base paper's PSO) | `TRD.md §6.1` |
| Edge-IIoTset feature counts | 61 raw → 46 selected (base paper's PSO) | `TRD.md §6.1` |
| Base paper's PSO hyperparameters | **Not stated anywhere in the paper** — do not guess them | `TRD.md §6.1` |
| Base paper's LSTM unit/layer count | **Not stated anywhere in the paper** — this project's BiLSTM must state its own explicitly | `PRD.md §2.1.2` |
| Base paper's fixed threshold | τ = 500ms for every device, IDS check every 30s | `PRD.md §2.1.4` |
| Base paper's positive class | "Normal" (unusual choice) | `PRD.md §2.1.3` |
| This project's positive class | **Attack** (this project's own explicit convention) | `TRD.md §2.3` |
| Metric inversion evidence | Worked calculation from Fig. 11c confusion matrix | `PRD.md §10` |
| Review 2 date | 7 September | `PRD.md` header |
| Review 3 date | After 2 November | `PRD.md` header |

### B.3 Repository Layout to Build
```
repo/
├── data/                          # raw + processed datasets (gitignored)
├── docs/
│   ├── feature_selection_decision.md   (T1.3)
│   ├── xai_survey.md                   (T1.4)
│   └── architecture_decision.md        (T1.5)
├── notebooks/
│   ├── 01_reproduce_leakage.ipynb      (T1.1)
│   ├── 02_leakage_free_baseline.ipynb  (T1.2)
│   ├── 03_train_ensemble_iotid20.ipynb (T2.6)
│   └── 04_train_edge_iiotset.ipynb     (T3.6)
├── src/
│   ├── preprocessing/
│   │   ├── clean.py                    (T1.1)
│   │   ├── feature_select.py           (T1.1, T1.3)
│   │   ├── resample.py                 (T1.2)
│   │   └── sequence_builder.py         (T2.1)
│   ├── models/
│   │   ├── cnn_branch.py               (T2.2)
│   │   ├── bilstm_branch.py            (T2.3)
│   │   ├── transformer_branch.py       (T2.4)
│   │   ├── gnn_branch.py               (T2.7, conditional)
│   │   └── fusion.py                   (T2.5)
│   ├── xai/
│   │   ├── shap_explainer.py           (T3.1)
│   │   ├── lime_explainer.py           (T3.2)
│   │   └── metrics.py                  (T3.3)
│   ├── adaptive/
│   │   ├── threshold.py                (T3.4)
│   │   └── incremental.py              (T3.5)
│   ├── evaluation/
│   │   └── leakage_check.py            (T1.1, T1.2)
│   └── deployment/
│       ├── export_model.py             (T4.1)
│       ├── pi_inference.py             (T4.2)
│       └── benchmark.py                (T4.3)
├── scratch/
│   └── hids-iomt-adaptive.cc           (NS3-Simulation.md T-N1..T-N5; copied into ns-3's scratch/)
├── scripts/ns3_experiments/
│   ├── run_matrix.sh                   (T-N6)
│   └── aggregate_results.py            (T-N7)
└── reports/
    ├── ns3_runs/                       (T-N6, raw saved runs)
    ├── ns3_simulation_results.md       (T-N7)
    ├── phase2_results.md               (T2.6)
    ├── gnn_go_nogo.md                  (T2.7)
    ├── deployment_benchmark.md         (T4.3)
    ├── ablation_study.md               (T4.4)
    └── final_comparison.md             (T4.4)
```

## §C — Failure-Mode Table

| Failure mode | Why it matters | Guarded by |
|---|---|---|
| `resample.py` ever called before the train/test split | Reintroduces the exact leak this project exists to fix | T1.2's split-first ordering, code review against `TRD.md §2.2` |
| A results table doesn't state its positive class | Silently repeats Objection #3 | A.1, every task's VERIFY block that touches metrics |
| GNN branch code left in after a no-go decision | Contradicts a pre-agreed, documented decision; wastes remaining Phase 2/3 time | T2.7's go/no-go gate is binding |
| PSO hyperparameters invented and presented as "the paper's" | Fabricates reproducibility that doesn't exist | A.1, T1.3 |
| Sequence builder defaults to window length 1 | Silently reintroduces Objection #2 | A.1, T2.1's shape assertion |
| A deployment claim ships without a real Pi measurement | Repeats the senior's-work gap this project is supposed to fix | A.1, T4.3 |
| Base paper's numbers cited without the inversion caveat | Misleads reviewers/readers who don't independently re-derive `PRD.md §10` | A.1, T4.4 |

---

## §D — Phased Tasks

### Phase 1 — Foundation (target: complete before/at Review 2 groundwork)

#### T1.1 — Reproduce the leakage empirically
**Files:** `src/preprocessing/clean.py`, `src/preprocessing/feature_select.py`, `src/evaluation/leakage_check.py`, `notebooks/01_reproduce_leakage.ipynb`
**Prompt:**
> Build a pipeline that: (1) loads IoTID20, cleans it (drop NaN/irrelevant indices, per base paper §II.G.1), label-encodes categoricals; (2) applies a feature-selection step (see T1.3 for which method — if T1.3 isn't done yet, default to Random Forest importance and note in a comment that this is a placeholder pending T1.3's decision); (3) applies SMOTE (or the base paper's described combined under/oversampling) to the **full** dataset; (4) **then** splits into train/test/validation. Train a fast baseline classifier (e.g., Random Forest is fine for this reproduction — it doesn't need to be the final architecture). In `leakage_check.py`, implement a nearest-neighbor check: for each test-set minority-class sample, find its nearest training-set neighbor and report the distance distribution — near-zero distances are the leakage signature. Report the (implausibly high) test accuracy alongside this evidence in the notebook.
**VERIFY (→ TRD §9 "Leakage demonstrated"):** Notebook output shows both an implausibly high test accuracy and a nearest-neighbor distance distribution with a cluster of near-zero distances for oversampled classes.

#### T1.2 — Build the leakage-free baseline
**Files:** `src/preprocessing/resample.py`, `src/evaluation/leakage_check.py` (extend), `notebooks/02_leakage_free_baseline.ipynb`
**Prompt:**
> Rebuild the pipeline from T1.1 with the split moved before resampling: clean → label-encode → feature-select → **split** (train/test/validation) → resample (**training fold only**, using `resample.py`) → scale (fit scaler on training fold only, apply to test/validation). Train the same baseline classifier as T1.1 for a fair before/after comparison. Report accuracy/precision/recall/F1 using the exact formulas in `TRD.md §2.3`, explicitly stating **Attack is the positive class** in the notebook's markdown and in a code comment above the metric computation. Hand-verify at least one result against the actual confusion matrix values (print the matrix, manually recompute one metric, assert it matches the library-computed value) — this is the same check that would have caught the base paper's Objection #3.
**VERIFY (→ TRD §9 "Leakage fixed" + "No repeat of the base paper's metric-label bug"):** Reported accuracy is measurably lower than T1.1's leaky number; the notebook contains an explicit positive-class statement and a hand-verified metric.

#### T1.3 — Feature-selection method decision
**Files:** `src/preprocessing/feature_select.py`, `docs/feature_selection_decision.md`
**Prompt:**
> `TRD.md §6.1` flags that the base paper's PSO hyperparameters are unspecified. Decide between (a) implementing PSO with this team's own chosen and documented particle count, `w`, `c1`/`c2`, and fitness function, or (b) using Random Forest feature importance (also mentioned in the base paper) as a fully-specified alternative. Implement the chosen method in `feature_select.py` with every parameter as a named, documented constant (no magic numbers). Write `docs/feature_selection_decision.md` explaining which was chosen and why, explicitly noting this is the team's own choice made necessary by the base paper's silence on this point.
**VERIFY:** `docs/feature_selection_decision.md` exists and states concrete parameter values (if PSO) or the importance-ranking method and threshold (if RF); `feature_select.py` has no unexplained magic numbers.

#### T1.4 — XAI method survey
**Files:** `docs/xai_survey.md`
**Prompt:**
> Write a short comparison of SHAP, LIME, and Integrated Gradients for this project's use case (per-detection explanation for a non-ML-expert reader, on a model that must eventually run inference fast enough for near-real-time flagging). Recommend a primary method and a faster fallback for cases where the primary is too slow, matching `TRD.md §4`'s "document whichever tradeoff is chosen" requirement. This is a documentation task — no code.
**VERIFY:** Doc states a clear primary + fallback recommendation with reasoning, ready to drive T3.1/T3.2.

#### T1.5 — Freeze the ensemble architecture
**Files:** `docs/architecture_decision.md`
**Prompt:**
> Decide and document, in concrete numbers: (1) sequence window length and how flow records are grouped into sequences (`TRD.md §3.1`); (2) CNN layer count, filter sizes, activation (`TRD.md §3.2.1`); (3) BiLSTM unit count and layer count (`TRD.md §3.2.2` — this is the literal, direct fix for Objection #2, so it must not be left vague); (4) Transformer attention head count, layer count, embedding dimension (`TRD.md §3.2.3`); (5) the exact confidence-weighted fusion formula (`TRD.md §3.3`). No "TBD" placeholders — every number must be a specific, justified choice (a brief justification per choice is enough; this doesn't need to be a hyperparameter search yet).
**VERIFY:** Every architectural number referenced in T2.2–T2.5 is traceable to a specific line in this document; no task in Phase 2 should need to invent a number that isn't already decided here.

### Phase 2 — Implementation (target: Review 2, 7 September)

#### T2.1 — Sequence construction
**Files:** `src/preprocessing/sequence_builder.py`
**Prompt:**
> Implement sequence construction per `docs/architecture_decision.md`'s window length and `TRD.md §3.1`: group time-ordered flow records by session/device key into fixed-length windows, padding or truncating as needed. Output shape must be `(batch, sequence_length, num_features)`. Add an assertion (not just a comment) that `sequence_length > 1`, and a unit test constructing a small synthetic example that checks the output shape directly.
**VERIFY (→ TRD §9 "Sequences are real sequences"):** Unit test passes; assertion present and cannot be silently bypassed by a window-length-1 configuration.

#### T2.2 — CNN branch
**Files:** `src/models/cnn_branch.py`
**Prompt:**
> Implement the CNN branch exactly per `docs/architecture_decision.md`. Module docstring must state the exact layer count, filter sizes, and activation function used.
**VERIFY:** Docstring numbers match `docs/architecture_decision.md` exactly.

#### T2.3 — BiLSTM branch
**Files:** `src/models/bilstm_branch.py`
**Prompt:**
> Implement the BiLSTM branch exactly per `docs/architecture_decision.md`. Module docstring must state the exact unit count and layer count — this is the direct, literal fix for Objection #2 (`PRD.md §2.1.2`), so under no circumstances leave this undocumented.
**VERIFY (→ TRD §9 "BiLSTM/Transformer fully specified"):** Docstring states unit and layer counts in plain numbers; no vague language like "a few layers."

#### T2.4 — Transformer branch
**Files:** `src/models/transformer_branch.py`
**Prompt:**
> Implement the Transformer-attention branch per `docs/architecture_decision.md`. Module docstring must state attention head count, layer count, and embedding dimension.
**VERIFY (→ TRD §9 "BiLSTM/Transformer fully specified"):** Docstring numbers present and match the decision doc.

#### T2.5 — Confidence-weighted fusion
**Files:** `src/models/fusion.py`
**Prompt:**
> Implement the fusion layer per the formula recorded in `docs/architecture_decision.md` and `TRD.md §3.3`: combine per-branch class probabilities weighted by each branch's own confidence (e.g., its own max softmax probability), not a simple average or majority vote. Document the exact formula in the module docstring.
**VERIFY:** Given synthetic branch outputs with known confidences, the fusion output favors the higher-confidence branch's prediction in a unit test.

#### T2.6 — Train and validate the ensemble on IoTID20
**Files:** `notebooks/03_train_ensemble_iotid20.ipynb`, `reports/phase2_results.md`
**Prompt:**
> Using the leakage-free pipeline from T1.2 and the sequence builder from T2.1, train the full CNN+BiLSTM+Transformer ensemble (via `fusion.py`) on IoTID20. Evaluate with `TRD.md §2.3`'s metrics, explicit positive-class statement, and a hand-verified confusion-matrix check (same standard as T1.2). Write `reports/phase2_results.md` comparing this result against T1.2's leakage-free baseline — this is the Review 2 deliverable per `PRD.md §8`.
**VERIFY:** Report exists, states positive class explicitly, includes a hand-verified metric, and makes an honest (not inflated) comparison against the T1.2 baseline.

#### T2.7 (stretch, gated) — GNN go/no-go check
**Files:** `src/models/gnn_branch.py` (only created if the check says "go"), `reports/gnn_go_nogo.md`
**Prompt:**
> Per `TRD.md §3.4`: construct a graph from source/destination IP pairs in the IoTID20 flow records (nodes = IPs, edges = communication events, weighted by flow count). Compute density/connectivity statistics (e.g., average degree, fraction of isolated nodes, clustering coefficient). Write `reports/gnn_go_nogo.md` stating the statistics and a clear go/no-go decision. **Only if "go":** implement `gnn_branch.py` and integrate into `fusion.py` as a fourth branch (this second half is optional and time-permitting — do not let it block Phase 3 tasks). **If "no-go":** stop here, do not create `gnn_branch.py`, and do not add GNN references to `fusion.py`.
**VERIFY (→ TRD §9 "GNN go/no-go"):** `reports/gnn_go_nogo.md` exists with a clear decision regardless of outcome; if "no-go," confirm `gnn_branch.py` does not exist and `fusion.py` has no GNN branch wired in.

### Phase 3 — Explainability, Adaptive Threshold, Incremental Learning (8 Sept – 5 Oct)

#### T3.1 — SHAP integration
**Files:** `src/xai/shap_explainer.py`
**Prompt:**
> Implement SHAP-based explanation generation for the trained ensemble from Phase 2, per the method chosen in `docs/xai_survey.md` (T1.4). Output: for a given flagged detection, the top-N contributing features and their attribution values.
**VERIFY:** Given a sample flagged detection, output includes a ranked feature list a reader can map back to the original feature names (not just indices).

#### T3.2 — LIME integration
**Files:** `src/xai/lime_explainer.py`
**Prompt:**
> Implement LIME-based local explanation as the faster fallback identified in `docs/xai_survey.md`, for cases where SHAP is too slow for near-real-time use.
**VERIFY:** Runs measurably faster than the SHAP path from T3.1 on the same sample input.

#### T3.3 — XAI evaluation metrics
**Files:** `src/xai/metrics.py`
**Prompt:**
> Implement the three explainability metrics from `TRD.md §4`: (1) fidelity — ablate the top-attributed features and confirm the prediction changes in the expected direction; (2) stability — perturb an input slightly and confirm the explanation doesn't change drastically; (3) a comprehensibility proxy (e.g., explanation length/feature count) standing in for the out-of-scope user study (`PRD.md §5.3`).
**VERIFY (→ TRD §9 "XAI output is usable"):** Have a team member unfamiliar with the model internals read one sample explanation output and correctly state, in their own words, why that flow was flagged.

#### T3.4 — Adaptive threshold
**Files:** `src/adaptive/threshold.py`
**Prompt:**
> Implement the adaptive threshold function from `TRD.md §5.1`: `threshold = base_threshold * criticality_weight * context_factor` (or whatever exact formula the team settles on — document it in the module docstring, including the specific weight values and why they were chosen). This replaces the base paper's flat 500ms rule (`PRD.md §2.1.4`).
**VERIFY (→ TRD §9 "Adaptive threshold responds to criticality"):** A test with two synthetic criticality levels (e.g., "critical device" vs. "non-critical device") produces two different effective threshold values, not one shared constant.

#### T3.5 — Incremental learning
**Files:** `src/adaptive/incremental.py`
**Prompt:**
> Implement the incremental-update mechanism chosen in `TRD.md §5.2` (fine-tune only the fusion layer on new samples, or a replay-buffer approach — pick one, document the choice and why in the module docstring). Include a before/after evaluation: measure detection performance on prior attack types immediately before and after an incremental update on a new attack type.
**VERIFY (→ TRD §9 "Incremental learning doesn't regress old classes"):** Before/after comparison shows no significant drop in prior-class detection performance after the update.

#### T3.6 — Extend training to Edge-IIoTset
**Files:** `notebooks/04_train_edge_iiotset.ipynb`
**Prompt:**
> Repeat the leakage-free pipeline (T1.2) and ensemble training (T2.6) on Edge-IIoTset (46 selected features per `TRD.md §6.1`), applying the same feature-selection method decided in T1.3. Report results with the same metric-formula and positive-class discipline as T2.6.
**VERIFY:** Report exists, uses the same positive-class convention, and is comparable in format to `reports/phase2_results.md`.

### Phase 4 — Deployment (6 Oct – 2 Nov)

#### T4.1 — Export the inference graph
**Files:** `src/deployment/export_model.py`
**Prompt:**
> Export the trained ensemble (post-Phase 3) into a format suitable for Raspberry Pi inference (e.g., TensorFlow Lite, or whatever the team's chosen framework's lightweight export path is). Training artifacts stay on Colab/GPU per `TRD.md §7` — only the exported inference graph goes to the Pi.
**VERIFY:** Exported model file loads successfully in a standalone inference script without requiring the full training environment.

#### T4.2 — Raspberry Pi inference script
**Files:** `src/deployment/pi_inference.py`
**Prompt:**
> Write a script that loads the exported model from T4.1 and runs inference on sample flow sequences, intended to run directly on a Raspberry Pi 4B (8GB).
**VERIFY:** Runs successfully on the actual Pi hardware (not just simulated locally) and produces predictions matching the training-environment output within floating-point tolerance.

#### T4.3 — Latency & throughput benchmark
**Files:** `src/deployment/benchmark.py`, `reports/deployment_benchmark.md`
**Prompt:**
> Measure per-sequence inference latency and throughput (sequences/sec) running T4.2's script on the actual Raspberry Pi 4B under a simulated load. Write `reports/deployment_benchmark.md` stating the exact hardware (Pi 4B, 8GB RAM, OS version), the measured numbers, and nothing estimated or assumed.
**VERIFY (→ TRD §9 "Deployment numbers are real"):** Report cites specific numbers from an actual run on the Pi, with hardware specs stated — this directly fixes the senior's-work gap (`PRD.md §2.2`) of claiming real-time capability with zero reported numbers.

#### T4.4 — Ablation and comparative analysis
**Files:** `reports/ablation_study.md`, `reports/final_comparison.md`
**Prompt:**
> Ablate each ensemble branch (CNN-only, BiLSTM-only, Transformer-only, full fusion, plus GNN if T2.7 went "go") against the leakage-free baseline from T1.2/T2.6. Write `reports/final_comparison.md` summarizing this project's results against (a) the leakage-free baseline honestly, and (b) the base paper's published numbers **with the metric-inversion caveat stated in the same paragraph** (per A.1 and `PRD.md §10`) — never cite HIDS-IoMT's 99.92%/99.91%/99.99%/99.95% without that caveat attached.
**VERIFY:** `final_comparison.md` contains the metric-inversion caveat wherever base-paper numbers are cited; `ablation_study.md` isolates each branch's contribution with correctly-labeled metrics per `TRD.md §2.3`.

---

## §E — Build Order

| Session block | Tasks | Notes |
|---|---|---|
| 1 | T1.1, T1.2 | Highest-priority pair — the leakage demonstration and fix are the evidentiary core of Contribution 1 |
| 2 | T1.3, T1.4, T1.5 | Decision-record tasks; can run in parallel across team members since they don't depend on each other |
| 3 | T2.1, T2.2, T2.3, T2.4, T2.5 | Sequence builder first (T2.1), then the three branches (can parallelize across team members), then fusion (T2.5) last since it depends on all three |
| 4 | T2.6 | Training run — needs real compute time (Colab/lab GPU); this is the Review 2 deliverable, so timebox it against the 7 September date |
| 5 (optional) | T2.7 | Only if time permits before Review 2 or early in Phase 3; respect the go/no-go gate strictly |
| 6 | T3.1, T3.2, T3.3 | XAI trio — can build in parallel |
| 7 | T3.4, T3.5 | Adaptive/incremental — depends on the trained ensemble from T2.6, not on T3.1–T3.3 |
| 8 | T3.6 | Edge-IIoTset extension — repeat of T1.2+T2.6's pattern on a new dataset |
| 9 | T4.1, T4.2, T4.3 | Deployment trio, strictly in order — each depends on the previous |
| 10 | T4.4 | Final analysis — needs everything else done first |

## §F — Final Acceptance Checklist

- [ ] Leakage demonstrated with concrete evidence (T1.1) and fixed with a lower, honest number (T1.2)
- [ ] Every metrics table in the repo states its positive-class convention explicitly (Attack = positive, per `TRD.md §2.3`)
- [ ] Feature-selection method is fully documented with no fabricated PSO parameters (T1.3)
- [ ] Ensemble architecture (CNN, BiLSTM, Transformer) has every hyperparameter documented in code (T2.2–T2.4) — the literal fix for Objection #2
- [ ] Sequence builder asserts `sequence_length > 1` and cannot silently collapse to single-record input (T2.1)
- [ ] GNN branch either fully justified via a documented go/no-go check, or entirely absent from the codebase (T2.7)
- [ ] SHAP and LIME explanations are readable by a non-ML-expert (T3.1–T3.3)
- [ ] Adaptive threshold demonstrably varies with device criticality, not a flat constant (T3.4)
- [ ] Incremental learning does not regress prior-class performance (T3.5)
- [ ] Deployment latency/throughput numbers come from an actual Raspberry Pi 4B run, hardware stated (T4.3)
- [ ] Any citation of the base paper's numbers includes the metric-inversion caveat in the same breath (T4.4)
- [ ] No task touched files outside its declared **Files** line
- [ ] Review 2 deliverable (`reports/phase2_results.md`) exists and is honest about the base paper's numbers being disputed rather than a target to beat
