# Ensemble Architecture Decision (T1.5) — FROZEN

**Status:** Frozen · **Binds:** T2.1–T2.5 · **Spec refs:** `TRD.md §3.1`–`§3.3`, `PRD.md §2.1.2` (Objection #2)

> **This document exists because the base paper does not have one.** Objection #2 is that
> HIDS-IoMT's Table 7 specifies three Conv1D layers, ReLU, Adam, dropout, epochs and batch size —
> and states **nothing** about its LSTM: no unit count, no layer count. Every number below is
> therefore concrete and justified. There are no "TBD"s, and no task in Phase 2 should need to
> invent a value that is not fixed here.

---

## 1. Sequence construction (`TRD.md §3.1`) — the fix for "single flow record, not a sequence"

### 1.1 Grouping key: **destination IP + a 1-second inactivity gap**

Measured on the cleaned IoTID20 (625,415 rows), comparing candidate grouping keys:

| Grouping key | Groups | Median records/group | Rows in groups ≥ 10 | Label-pure groups |
|---|---:|---:|---:|---:|
| `Src_IP` | 57,985 | 1 | 90.7% | 100% of groups, but only 23% of rows |
| `Src_IP` + `Dst_IP` | 58,538 | 1 | 90.5% | — |
| `Flow_ID` | 64,064 | 1 | 88.9% | — |
| **`Dst_IP` + gap > 1 s** | **4,301** | **5** | **98.5%** | **100% of groups, 100% of rows** |

**Chosen: `Dst_IP`, ordered by `Timestamp`, with a new session started whenever the gap since the
previous record on that device exceeds `SESSION_GAP_SECONDS = 1.0`.**

Three reasons:

1. **It is the only key that produces actual sequences.** Every raw key has a *median group size of
   1* — grouping by `Src_IP` or `Flow_ID` would leave most windows padded from a single record,
   which is Objection #2 reintroduced through the back door. Time-gap sessionisation on `Dst_IP`
   raises the median to 5 and puts 98.5% of rows into sessions of ≥ 10.
2. **It matches the deployment story.** `Dst_IP` is the *device being talked to*. An IoMT fog node
   protecting an insulin pump cares about the traffic arriving at that pump over time — which is
   also exactly the unit the adaptive threshold keys on (`TRD.md §5.1` weights by device
   criticality). `Src_IP` is the wrong unit: IoTID20 has 57,985 distinct source IPs against 478
   destinations, because scanning and spoofing vary the source constantly.
3. **Sessions are label-pure.** All 4,301 sessions carry a single label, so a window never straddles
   the normal/attack boundary and the window label is unambiguous.

**Honest caveat on that third point.** 100% purity is a property of *how IoTID20 was captured* —
each attack scenario is a separate recording, so a contiguous burst on one destination is
necessarily all one class. It is **not** a property that will hold on Edge-IIoTset (T3.6) or in
deployment, and it makes the sequence task easier than reality. Two consequences that must not be
forgotten: the labelling rule in §1.3 is stated as a general rule rather than relying on purity,
and `reports/phase2_results.md` must state that this dataset property inflates sequence-model
performance relative to a live setting.

### 1.2 Window geometry

| Constant | Value | Justification |
|---|---|---|
| `SEQUENCE_LENGTH` | **10** | Covers 98.5% of rows (1,354 sessions have ≥ 10 records); the p75 session is 14 records, so most multi-record sessions yield a full window. Short enough to keep the BiLSTM and attention cost low on a Pi 4B (`TRD.md §7`). |
| `TRAIN_STRIDE` | **5** | 50% overlap. Stride 1 would generate ~604,000 near-identical windows (adjacent windows share 9 of 10 records), inflating the apparent dataset size with redundant examples. |
| `INFERENCE_STRIDE` | **1** | At detection time every arriving record must produce a decision, so windows advance one record at a time. Deliberately different from `TRAIN_STRIDE`; both are named constants. |
| `SESSION_GAP_SECONDS` | **1.0** | 1 s gives 4,301 sessions with a median of 5 records; 5 s and 30 s give coarser sessions (2,696 / 1,498) whose p99 lengths run to thousands of records, which merges distinct behavioural episodes. |
| `MIN_SESSION_LENGTH` | **2** | Sessions of exactly 1 record are dropped rather than padded to 10 from nothing — a 9/10-padding window is a single flow record wearing a sequence costume, i.e. Objection #2 again. This drops ~1.5% of rows, which must be reported. |
| `PAD_VALUE` | **0.0** | Sessions of length 2–9 are **pre-padded** (padding at the front, real records at the end) so the most recent record is always the last timestep. |

### 1.3 Window labelling rule

**A window takes the label of its LAST (most recent) record.**

Causal and deployment-realistic: the model is deciding about *now*, given the preceding nine
records as context. It never requires a label from the future. On IoTID20 the choice is moot
because sessions are label-pure (§1.1), but the rule is stated generally because that purity will
not hold on Edge-IIoTset.

### 1.4 ⚠️ Splitting rule — a new leakage risk this document creates

**Windows must be split by SESSION, never by window.** Overlapping windows from one session share
up to 9 of their 10 records; a random window-level split would put near-identical windows in both
train and test — **the same class of bug this entire project exists to expose**, reintroduced by
our own sequence builder.

Binding rule for T2.1 and T2.6: assign whole **sessions** to train/test/validation (80/10/10,
stratified on session label), then build windows *within* each fold. `sequence_builder.py` must
expose the session id per window so this is checkable, and T2.6 must re-run
`src/evaluation/leakage_check.py` on the sequence representation to confirm it.

### 1.5 Shape contract

Ensemble input is **`(batch, 10, 62)`** — `(batch, SEQUENCE_LENGTH, n_selected_features)`.
`TRD.md §9`'s gate requires `sequence_length > 1`; T2.1 must **assert** this, not merely comment
it, and the assertion must be unreachable-by-configuration (a config of 1 raises, it does not warn).

---

## 2. Branch architectures

All three branches consume the same `(batch, 10, 62)` input, each emits a **64-dimensional
embedding** followed by its **own** `Dense(2, softmax)` classification head. Branches are trained
independently; §3's fusion combines their softmax outputs at inference. Independent heads are what
make "confidence-weighted voting" meaningful — each branch must have a calibrated opinion of its
own to be weighted by.

### 2.1 CNN branch (`TRD.md §3.2.1`, T2.2)

| Layer | Configuration |
|---|---|
| Conv1D-1 | 64 filters, kernel 3, stride 1, padding `same`, activation **ReLU** |
| BatchNorm-1 | — |
| Conv1D-2 | 128 filters, kernel 3, stride 1, padding `same`, activation **ReLU** |
| BatchNorm-2 | — |
| Conv1D-3 | 64 filters, kernel 3, stride 1, padding `same`, activation **ReLU** |
| BatchNorm-3 | — |
| Dropout | rate **0.3** |
| GlobalMaxPooling1D | → 64-d embedding |
| Dense head | 2 units, softmax |

- **3 conv layers** deliberately matches the base paper's stated Conv1D count, so the ensemble
  comparison isolates the *sequence* and *fusion* changes rather than confounding them with depth.
- **64 → 128 → 64** widens then narrows, the standard encoder shape; the final 64 fixes the
  embedding width shared across all three branches.
- **Kernel 3** with `SEQUENCE_LENGTH = 10`: a 3-layer stack has a receptive field of 7 timesteps,
  most of the window, without padding dominating.
- **Padding `same`** preserves length so the three branches stay dimensionally comparable.
- **GlobalMaxPooling** (not average) because attack evidence is typically a *spike* in one or two
  timesteps, not a shift in the window mean.

### 2.2 BiLSTM branch (`TRD.md §3.2.2`, T2.3) — the direct fix for Objection #2

| Layer | Configuration |
|---|---|
| Bi-LSTM-1 | **64 units per direction** (128 concatenated), `return_sequences=True` |
| Dropout | rate **0.3** |
| Bi-LSTM-2 | **64 units per direction** (128 concatenated), `return_sequences=False` |
| Dropout | rate **0.3** |
| Dense | 64 units, ReLU → 64-d embedding |
| Dense head | 2 units, softmax |

**LAYER COUNT = 2. UNIT COUNT = 64 PER DIRECTION.** These two numbers are the literal content of
Objection #2 — the base paper states neither. They are recorded here, must be repeated verbatim in
`bilstm_branch.py`'s module docstring (T2.3's VERIFY block), and must never be described as "a
couple of layers" anywhere in this project.

Justification: 2 layers is the shallowest depth that is meaningfully *deep* while remaining
trainable on 10-timestep windows — a third layer over 10 timesteps overfits with no receptive-field
gain. 64 units per direction gives a 128-d recurrent state, matching the CNN's widest layer so no
branch is capacity-starved relative to the others. `recurrent_dropout` is fixed at **0.0**
deliberately: non-zero values disable the cuDNN fast path and block TFLite conversion for the
Raspberry Pi deployment (T4.1).

### 2.3 Transformer-attention branch (`TRD.md §3.2.3`, T2.4)

| Layer | Configuration |
|---|---|
| Input projection | Dense **64** (= `EMBED_DIM`, `d_model`), linear |
| Positional encoding | fixed sinusoidal, added to the projection |
| Encoder × **2** | each: MultiHeadAttention(**4 heads**, key_dim **16**) → residual + LayerNorm → FFN(Dense **128** ReLU → Dense 64) → residual + LayerNorm, dropout **0.1** |
| GlobalAveragePooling1D | → 64-d embedding |
| Dense head | 2 units, softmax |

- **`EMBED_DIM` = 64** matches the shared embedding width.
- **4 heads × key_dim 16 = 64 = `EMBED_DIM`**, the standard head-splitting relation. 4 heads over a
  10-timestep window gives each head a genuine subspace without slicing 64 dimensions too thin.
- **2 encoder layers**: with only 10 timesteps, one layer already reaches every position; a second
  allows composition, and beyond that depth buys nothing on this window length.
- **FFN 128 = 2 × `EMBED_DIM`** (the usual expansion is 2–4×; 2× is chosen for Pi inference cost).
- **Dropout 0.1**, below the CNN/BiLSTM's 0.3, because attention layers here are shallow and
  residual-connected; 0.3 destabilises training at this depth.
- **Sinusoidal, not learned,** positional encoding: no extra parameters, and it generalises to
  windows shorter than 10 after padding.
- **GlobalAveragePooling** (not max, unlike the CNN) so this branch is deliberately biased toward
  window-wide evidence, complementing the CNN's spike sensitivity. Ensemble diversity is the point.

### 2.4 Training configuration (all three branches)

| Constant | Value | Note |
|---|---|---|
| Optimizer | Adam, learning rate **1e-3** | Matches the base paper's stated optimizer |
| Loss | categorical cross-entropy | 2-class softmax output |
| Batch size | **256** | |
| Max epochs | **50** | With early stopping, rarely reached |
| Early stopping | patience **5** on validation **F1** | Not val-accuracy: at 93.6% Attack, accuracy is a poor stopping signal on this imbalance |
| Seed | **42** | `src/config.py`, `Build-Instructions.md` §A.3 |
| Class handling | SMOTE on the training fold only | Never on validation/test (`TRD.md §2.2`) |

---

## 3. Confidence-weighted fusion (`TRD.md §3.3`, T2.5)

### 3.1 The formula — frozen

For branches `b ∈ {cnn, bilstm, transformer}` (plus `gnn` only if T2.7 returns "go"), with softmax
output `p_b ∈ R^2`:

```
confidence      c_b = max_k p_b[k]                    # the branch's own certainty
unnormalised    u_b = alpha_b * (c_b ^ gamma)
weight          w_b = u_b / sum_b'(u_b')              # weights sum to 1
fused           P   = sum_b (w_b * p_b)               # elementwise over the 2 classes
prediction          = argmax_k P[k]                   # 1 = Attack = POSITIVE (TRD.md §2.3)
ensemble conf.      = max_k P[k]
```

| Constant | Value | Justification |
|---|---|---|
| `CONFIDENCE_SHARPNESS` (`gamma`) | **1.0** | Weight proportional to raw confidence. `gamma > 1` sharpens toward a winner-take-all vote; 1.0 is the neutral starting point and the value to beat in T4.4's ablation. |
| `BRANCH_PRIORS` (`alpha_b`) | **1.0 for every branch** | No branch is privileged a priori. Kept as an explicit vector rather than hardcoded 1s because it is the **only learnable parameter of the fusion layer**, and T3.5's incremental learning ("fine-tune only the fusion layer") updates exactly these. |
| `EPSILON` | **1e-9** | Added to the denominator; guards the degenerate case of all confidences being 0. |

This is **not** a simple average and **not** a majority vote — `TRD.md §3.3` rules both out. With
`gamma = 1` and equal priors it reduces to a confidence-weighted mean, and a branch that is 95%
sure outvotes two branches that are 55% sure.

### 3.2 Required unit test (T2.5's VERIFY block)

Given synthetic branch outputs where one branch is highly confident and disagrees with two
low-confidence branches, the fused prediction must follow the **confident** branch. A simple
average would follow the majority; that difference is what the test asserts.

---

## 4. Traceability for Phase 2

| Task | Numbers it must use | Section |
|---|---|---|
| T2.1 `sequence_builder.py` | `SEQUENCE_LENGTH=10`, `TRAIN_STRIDE=5`, `INFERENCE_STRIDE=1`, `SESSION_GAP_SECONDS=1.0`, `MIN_SESSION_LENGTH=2`, `PAD_VALUE=0.0`, pre-padding, last-record labelling, session-level splitting, `assert SEQUENCE_LENGTH > 1` | §1.2–§1.5 |
| T2.2 `cnn_branch.py` | 3 Conv1D layers; 64/128/64 filters; kernel 3; ReLU; dropout 0.3; GlobalMaxPooling1D | §2.1 |
| T2.3 `bilstm_branch.py` | **2 layers, 64 units per direction**; dropout 0.3; `recurrent_dropout=0.0` | §2.2 |
| T2.4 `transformer_branch.py` | `EMBED_DIM=64`; **4 heads**, key_dim 16; **2 encoder layers**; FFN 128; dropout 0.1; sinusoidal positions | §2.3 |
| T2.5 `fusion.py` | `gamma=1.0`, `alpha_b=1.0`, `EPSILON=1e-9`, the §3.1 formula verbatim | §3.1 |

## 5. What is deliberately not decided here

- **GNN branch hyperparameters.** T2.7's sparsity go/no-go gate has not run. Fixing an architecture
  for a branch that may be cut would invite exactly the "quietly re-add the GNN" failure that
  `Build-Instructions.md` §A.1 rule 4 forbids. If and only if the check returns "go" does this
  document gain a §2.4.
- **Any latency or throughput target.** No number here is justified by speed on the Raspberry Pi,
  because no Pi measurement exists yet (T4.3). Where cost influenced a choice above it is stated as
  a *relative* preference ("keeps attention cost low"), never as a measured claim.
