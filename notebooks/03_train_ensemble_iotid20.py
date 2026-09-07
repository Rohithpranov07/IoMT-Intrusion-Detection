# %% [markdown]
# # T2.6 — Training the CNN + BiLSTM + Transformer ensemble on IoTID20
#
# **Build-Instructions task:** T2.6 · **This is the Review 2 deliverable** (`PRD.md §8`, 7 September)
# **Spec:** `TRD.md §3`, `docs/architecture_decision.md`
#
# ## What this notebook builds
#
# The committed core of Contribution 2: three fully-specified branches over genuine flow
# *sequences*, fused by confidence-weighted voting, trained on the leakage-free pipeline from T1.2
# and compared honestly against that pipeline's Random Forest baseline.
#
# | Branch | Exact specification (frozen in `docs/architecture_decision.md`) |
# |---|---|
# | CNN | 3 Conv1D layers, 64/128/64 filters, kernel 3, ReLU, dropout 0.3, GlobalMaxPooling1D |
# | BiLSTM | **2 layers, 64 units per direction** (128-d state), dropout 0.3 |
# | Transformer | 2 encoder layers, 4 heads, key_dim 16, embed dim 64, FFN 128, dropout 0.1 |
# | Fusion | `w_b = alpha_b · c_b^gamma / Σ`, gamma = 1.0 — not an average, not a majority vote |
#
# The BiLSTM row is the point: the base paper states **neither** its LSTM layer count **nor** its
# unit count (Objection #2, `PRD.md §2.1.2`). Ours are stated here, in the module docstring, in the
# decision document, and in the report this notebook writes.
#
# ## Positive-class convention
#
# **`Attack` (label 1) is the positive class** — `TRD.md §2.3`, the opposite of the base paper's
# choice. Every metric below is hand-verified from the raw confusion matrix under this convention.

# %%
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", force=True)

from src.config import RANDOM_STATE
from src.evaluation.leakage_check import synthetic_contamination
from src.evaluation.metrics import (
    POSITIVE_CLASS_STATEMENT,
    compute_metrics,
    hand_verify_metrics,
    results_table,
)
from src.models import bilstm_branch, cnn_branch, transformer_branch
from src.models.fusion import (
    confidence_weighted_fusion,
    describe_formula,
    majority_vote_fusion,
    simple_average_fusion,
)
from src.models.train_utils import (
    compute_class_weights,
    describe_training_configuration,
    scale_sequences,
    set_global_seeds,
    train_branch,
)
from src.preprocessing.clean import clean_iotid20, load_iotid20
from src.preprocessing.feature_select import IOTID20_N_FEATURES, RandomForestFeatureSelector
from src.preprocessing.sequence_builder import (
    SEQUENCE_LENGTH,
    TRAIN_STRIDE,
    assign_sessions,
    build_sequences,
    split_sessions,
)

set_global_seeds(RANDOM_STATE)
print(POSITIVE_CLASS_STATEMENT)
print("\n" + describe_training_configuration())
print("\n" + bilstm_branch.describe_architecture())
print("\n" + transformer_branch.describe_architecture())

# %% [markdown]
# ## Step 1 — Load, clean, and deduplicate
#
# Cleaning matches the **pipeline C** baseline from T1.2 Step 11: the honest split order *and*
# deduplication. T1.2 showed that fixing the SMOTE ordering alone leaves 58.2% duplicate rows in
# play, a larger leakage vector; both fixes are applied from here on.

# %%
raw = load_iotid20()
X, y, meta = clean_iotid20(raw, drop_duplicates=True)
print(f"Cleaned + deduplicated: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Attack (positive, 1): {int((y == 1).sum()):,}   Normal (0): {int((y == 0).sum()):,}")

# %% [markdown]
# ## Step 2 — Sessionise, then split by **session**
#
# Sessions are per-destination-device, broken by a 1-second inactivity gap
# (`docs/architecture_decision.md` §1.1). `Dst_IP` is the device being protected, and it is the only
# candidate key whose median group size exceeds 1 — grouping by `Src_IP` or `Flow_ID` would leave
# most windows padded from a single record, which is Objection #2 through the back door.
#
# **The split is by session, never by window** (§1.4). Adjacent windows within a session share up to
# 9 of their 10 records, so splitting windows directly would put near-identical windows on both
# sides — the very bug this project exists to expose, reintroduced by our own sequence builder.
#
# The assignment is also **size-aware**: IoTID20 session sizes are extremely skewed, and assigning
# sessions uniformly at random gives folds that are 98% / 39% / 96% Attack — not comparable to one
# another. Bin-packing sessions largest-first into window quotas fixes that.

# %%
sessions = assign_sessions(meta)
print(f"{sessions.nunique():,} sessions over {meta['Dst_IP'].nunique():,} destination devices")

train_sessions, test_sessions, val_sessions = split_sessions(sessions, y)
row_fold = {
    "train": sessions.isin(train_sessions).to_numpy(),
    "test": sessions.isin(test_sessions).to_numpy(),
    "val": sessions.isin(val_sessions).to_numpy(),
}
for name, mask in row_fold.items():
    print(f"  {name:<5s} {int(mask.sum()):>7,} rows  ({mask.mean():.1%})")

assert not (set(train_sessions) & set(test_sessions)), "session leaked between train and test"
assert not (set(train_sessions) & set(val_sessions)), "session leaked between train and validation"
print("\nNo session appears in more than one fold.")

# %% [markdown]
# ## Step 3 — Feature selection on the **training fold only** (83 → 62)
#
# Random Forest importance — *this project's own choice*, not the base paper's unreproducible PSO
# (`docs/feature_selection_decision.md`). Fitted only on rows belonging to training sessions.

# %%
selector = RandomForestFeatureSelector(n_features=IOTID20_N_FEATURES, random_state=RANDOM_STATE)
selector.fit(X[row_fold["train"]], y[row_fold["train"]])
X_selected = selector.transform(X)
selector.ranking_.to_csv(REPO_ROOT / "reports" / "iotid20_feature_ranking_sequences.csv", index=False)
print(f"Selected {len(selector.selected_features_)} features on the training fold only")

# %% [markdown]
# ## Step 4 — Build windows **within each fold**
#
# Windows are constructed per fold rather than globally, so no window can straddle the split.
# Window length **10**, training stride **5** (50% overlap). Each window is labelled by its **last**
# record — causal, and never requiring a label from the future (§1.3).

# %%
folds: dict[str, object] = {}
for name, mask in row_fold.items():
    folds[name] = build_sequences(
        X_selected[mask].reset_index(drop=True),
        y[mask].reset_index(drop=True),
        meta[mask].reset_index(drop=True),
        sequence_length=SEQUENCE_LENGTH,
        stride=TRAIN_STRIDE,
        session_ids=sessions[mask].reset_index(drop=True),
    )
    print(f"\n{name.upper()}\n{folds[name].summary()}")

# %% [markdown]
# ### The shape gate (`TRD.md §9`, "Sequences are real sequences")

# %%
train_ds, test_ds, val_ds = folds["train"], folds["test"], folds["val"]
print(f"Ensemble input shape: {train_ds.X.shape}  = (batch, sequence_length, features)")
assert train_ds.X.ndim == 3, "input must be 3-D, not (batch, features)"
assert train_ds.X.shape[1] == SEQUENCE_LENGTH > 1, "sequence_length must exceed 1"
print(f"sequence_length = {train_ds.X.shape[1]} > 1  ✓  (the base paper's input is a single record)")
print(f"features        = {train_ds.X.shape[2]}")

# %% [markdown]
# ## Step 5 — Scale, with statistics from the **training fold only**

# %%
X_train, X_test, X_val = scale_sequences(train_ds.X, test_ds.X, val_ds.X)
y_train, y_test, y_val = train_ds.y, test_ds.y, val_ds.y

for name, y_fold in (("train", y_train), ("test", y_test), ("validation", y_val)):
    print(f"  {name:<11s} {len(y_fold):>7,} windows  Attack rate {float((y_fold == 1).mean()):.3f}")

# %% [markdown]
# ### The test fold contains no synthetic data
#
# `train_utils` handles imbalance with **class weights, not SMOTE** — SMOTE applied to a flattened
# 620-dimensional flow sequence would fabricate sessions blending two devices' traffic over time,
# objects that cannot occur on a real network. See `src/models/train_utils.py`'s docstring. This
# difference from the row-level baseline is stated in the report.

# %%
print(synthetic_contamination(y_test, None, "SEQUENCE pipeline", fold_name="test").summary())
class_weights = compute_class_weights(y_train)
print(f"\nClass weights (inverse frequency): {({k: round(v, 4) for k, v in class_weights.items()})}")

# %% [markdown]
# ## Step 6 — Train the three branches

# %%
n_features = X_train.shape[2]
builders = {
    "cnn": cnn_branch.build_cnn_branch,
    "bilstm": bilstm_branch.build_bilstm_branch,
    "transformer": transformer_branch.build_transformer_branch,
}

models: dict[str, object] = {}
histories: dict[str, object] = {}
train_seconds: dict[str, float] = {}

for name, builder in builders.items():
    set_global_seeds(RANDOM_STATE)
    model = builder(SEQUENCE_LENGTH, n_features)
    print(f"\n=== {name} — {model.count_params():,} parameters ===")

    started = time.perf_counter()
    histories[name] = train_branch(
        model, X_train, y_train, X_val, y_val, class_weights=class_weights, verbose=0
    )
    train_seconds[name] = time.perf_counter() - started
    models[name] = model

    epochs_run = len(histories[name].history["loss"])
    best_f1 = max(histories[name].history["val_positive_f1"])
    print(f"    {epochs_run} epochs in {train_seconds[name]:.1f}s "
          f"(best validation positive-class F1 {best_f1:.4f})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for name, history in histories.items():
    axes[0].plot(history.history["loss"], label=f"{name} train")
    axes[0].plot(history.history["val_loss"], ls="--", label=f"{name} val")
    axes[1].plot(history.history["val_positive_f1"], label=name)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation positive-class F1")
axes[1].set_title("Validation F1 (Attack = positive)"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Step 7 — Per-branch test-set performance
#
# **Positive class = Attack (label 1).** Every metric is hand-verified against the confusion matrix.

# %%
branch_probabilities = {}
branch_results = []

for name, model in models.items():
    probabilities = model.predict(X_test, verbose=0)
    branch_probabilities[name] = probabilities
    result = hand_verify_metrics(
        compute_metrics(y_test, np.argmax(probabilities, axis=1), f"{name} branch alone"),
        verbose=False,
    )
    branch_results.append(result)
    print(result.report())
    print()

# %% [markdown]
# ## Step 8 — Confidence-weighted fusion

# %%
print(describe_formula())

# %%
branch_names = list(models.keys())
fusion = confidence_weighted_fusion(
    [branch_probabilities[name] for name in branch_names], branch_names
)
ensemble_result = hand_verify_metrics(
    compute_metrics(y_test, fusion.predictions, "ENSEMBLE (confidence-weighted fusion)"),
    verbose=True,
)
print()
print(ensemble_result.report())

# %% [markdown]
# ### Which branch actually drove each decision?
#
# The fusion layer records the weight each branch received per sample — the ensemble's own account
# of where a decision came from. This feeds directly into the SHAP/LIME output in Phase 3 (T3.1).

# %%
dominant = pd.Series(fusion.dominant_branch()).map(dict(enumerate(branch_names)))
print("Highest-weighted branch, share of test windows:")
print((dominant.value_counts(normalize=True) * 100).round(2).to_string())
print(f"\nMean fused confidence: {fusion.confidence.mean():.4f}")

# %% [markdown]
# ## Step 9 — Was confidence weighting worth it?
#
# `TRD.md §3.3` rules out the simple average and the majority vote. This measures what that choice
# actually buys on real data, rather than asserting it. A small or negative difference here is a
# finding to report, not to hide.

# %%
fusion_variants = [ensemble_result]
for label, predictions in (
    ("FUSION baseline: simple average",
     np.argmax(simple_average_fusion([branch_probabilities[n] for n in branch_names]), axis=1)),
    ("FUSION baseline: majority vote",
     majority_vote_fusion([branch_probabilities[n] for n in branch_names])),
):
    fusion_variants.append(
        hand_verify_metrics(compute_metrics(y_test, predictions, label), verbose=False)
    )

results_table(fusion_variants)[["model", "positive_class", "accuracy", "precision", "recall", "f1"]]

# %% [markdown]
# ### Step 9b — Diagnosing the fusion result
#
# The three fusion rules land within ~0.001 F1 of each other, and the best single branch beats all
# of them. That is not what the design predicted, so it needs an explanation rather than a shrug.
#
# The hypothesis: **confidence weighting can only do work when branch confidences actually differ.**
# Softmax outputs from over-parameterised networks are notoriously badly calibrated — they saturate
# near 1.0 whether or not the prediction is right. If every `c_b ≈ 1`, then `w_b ≈ 1/3` and the
# formula collapses to the simple average it was chosen to beat. The cells below test that directly.

# %%
confidence_stats = pd.DataFrame({
    name: {
        "mean confidence": probabilities.max(axis=1).mean(),
        "median confidence": np.median(probabilities.max(axis=1)),
        "std of confidence": probabilities.max(axis=1).std(),
        "share above 0.99": float((probabilities.max(axis=1) > 0.99).mean()),
        "share above 0.90": float((probabilities.max(axis=1) > 0.90).mean()),
    }
    for name, probabilities in branch_probabilities.items()
}).T.round(4)
print("Per-branch confidence distribution on the test fold:")
print(confidence_stats.to_string())

spread = np.stack([branch_probabilities[n].max(axis=1) for n in branch_names])
print(f"\nMean per-sample spread between the most and least confident branch: "
      f"{(spread.max(axis=0) - spread.min(axis=0)).mean():.4f}")
print(f"Mean fusion weight per branch: "
      f"{dict(zip(branch_names, fusion.branch_weights.mean(axis=0).round(4)))}")
print("\nIf those weights are all near 1/3 = 0.3333, confidence weighting is doing nothing.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for name, probabilities in branch_probabilities.items():
    axes[0].hist(probabilities.max(axis=1), bins=50, alpha=0.55, label=name)
axes[0].set_xlabel("branch confidence  max_k p_b[k]")
axes[0].set_ylabel("test windows")
axes[0].set_title("Branch confidence is saturated near 1.0")
axes[0].legend(fontsize=8)

# Does sharpening the weighting recover the best branch's performance?
gammas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
gamma_f1 = []
for gamma in gammas:
    swept = confidence_weighted_fusion(
        [branch_probabilities[n] for n in branch_names], branch_names, gamma=gamma
    )
    gamma_f1.append(compute_metrics(y_test, swept.predictions, f"gamma={gamma}").f1)

best_branch_f1 = max(r.f1 for r in branch_results)
axes[1].plot(gammas, gamma_f1, marker="o", label="confidence-weighted fusion")
axes[1].axhline(best_branch_f1, color="crimson", ls="--", label="best single branch")
axes[1].set_xscale("symlog"); axes[1].set_xlabel("gamma (CONFIDENCE_SHARPNESS)")
axes[1].set_ylabel("test F1 (Attack = positive)")
axes[1].set_title("Sharpening the weighting does not close the gap")
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"Best single branch F1        : {best_branch_f1:.4f}")
print(f"Fusion F1 at gamma = 1.0     : {gamma_f1[gammas.index(1.0)]:.4f}")
print(f"Fusion F1 at best gamma      : {max(gamma_f1):.4f} (gamma = {gammas[int(np.argmax(gamma_f1))]})")

# %% [markdown]
# **Reading the diagnosis.** If the weights sit near 1/3 and the gamma sweep is flat, then the
# fusion formula is not the problem — *branch calibration* is. Confidence-weighted voting is a sound
# rule that is being fed an input it cannot use, because every branch claims near-certainty.
#
# This is a concrete, actionable finding rather than a dead end, and it gives Phase 3 and Phase 4
# their direction:
#
# - **T3.5 (incremental learning)** already updates `BRANCH_PRIORS` (`alpha_b`), the fusion layer's
#   only adjustable parameter. Fitting `alpha_b` on the validation fold — rather than leaving all
#   three at 1.0 — is the natural way to let a genuinely better branch carry more weight, and it
#   needs no architectural change.
# - **T4.4 (ablation)** must report the single-branch results alongside the fused ones. On this
#   dataset the honest headline is that the best single branch beats the ensemble, and
#   `reports/ablation_study.md` has to say so.
# - A calibration step (temperature scaling on each branch's logits, fitted on the validation fold)
#   is the standard fix and would make the confidences informative. It is **not** implemented here,
#   because it is not in the frozen `docs/architecture_decision.md` §3.1 and inventing it mid-phase
#   would be exactly the undocumented drift this project criticises. It is recorded as a Phase 3
#   proposal.

# %% [markdown]
# ## Step 10 — The comparison that matters: ensemble vs. the leakage-free baseline
#
# `reports/t1_2_leakage_free_baseline.csv` is pipeline **C** from T1.2 — the honest, deduplicated
# Random Forest. That is the only number this project compares against.

# %%
baseline_path = REPO_ROOT / "reports" / "t1_2_leakage_free_baseline.csv"
baseline = pd.read_csv(baseline_path).iloc[0] if baseline_path.exists() else None

comparison = results_table(branch_results + fusion_variants)
if baseline is not None:
    comparison = pd.concat(
        [pd.DataFrame([{
            "model": "T1.2 baseline: Random Forest (rows, not sequences)",
            "positive_class": baseline["positive_class"],
            "accuracy": baseline["accuracy"], "precision": baseline["precision"],
            "recall": baseline["recall"], "f1": baseline["f1"],
            "TP": baseline["TP"], "FP": baseline["FP"],
            "TN": baseline["TN"], "FN": baseline["FN"],
            "hand_verified": baseline["hand_verified"],
        }]), comparison],
        ignore_index=True,
    )

comparison.to_csv(REPO_ROOT / "reports" / "t2_6_ensemble_results.csv", index=False)
print(POSITIVE_CLASS_STATEMENT)
print()
comparison[["model", "accuracy", "precision", "recall", "f1", "FP", "FN"]]

# %%
metadata = {
    "sequence_shape": list(X_train.shape),
    "sequence_length": int(SEQUENCE_LENGTH),
    "n_features": int(n_features),
    "train_windows": int(len(y_train)),
    "test_windows": int(len(y_test)),
    "val_windows": int(len(y_val)),
    "train_attack_rate": float((y_train == 1).mean()),
    "test_attack_rate": float((y_test == 1).mean()),
    "sessions": {"train": len(train_sessions), "test": len(test_sessions), "val": len(val_sessions)},
    "parameters": {name: int(model.count_params()) for name, model in models.items()},
    "epochs_run": {name: len(h.history["loss"]) for name, h in histories.items()},
    "train_seconds": {name: round(seconds, 1) for name, seconds in train_seconds.items()},
    "dominant_branch_share": (dominant.value_counts(normalize=True)).round(4).to_dict(),
    "mean_fused_confidence": float(fusion.confidence.mean()),
}
(REPO_ROOT / "reports" / "t2_6_run_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
print(json.dumps(metadata, indent=2))

# %% [markdown]
# ## Step 11 — Persist artifacts for Phase 3
#
# The XAI modules (T3.1 SHAP, T3.2 LIME) and the adaptive/incremental modules (T3.4, T3.5) all need
# the *trained* ensemble, not a retrained one — an explanation of a different model than the one
# evaluated above would be worthless. Everything Phase 3 needs is written to `artifacts/`
# (gitignored: model weights are build outputs, not source).
#
# The background sample for SHAP is drawn from the **training** fold. Drawing it from test data
# would leak the evaluation set into the explanations.

# %%
ARTIFACTS = REPO_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
(ARTIFACTS / "models").mkdir(exist_ok=True)

for name, model in models.items():
    model.save(ARTIFACTS / "models" / f"{name}.keras")

rng = np.random.default_rng(RANDOM_STATE)
background_index = rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)

np.savez_compressed(
    ARTIFACTS / "ensemble_artifacts.npz",
    X_test=X_test,
    y_test=y_test,
    X_background=X_train[background_index],
    fused_predictions=fusion.predictions,
    fused_confidence=fusion.confidence,
    branch_weights=fusion.branch_weights,
    **{f"probabilities_{name}": branch_probabilities[name] for name in branch_names},
)
(ARTIFACTS / "feature_names.json").write_text(
    json.dumps({"feature_names": list(selector.selected_features_),
                "branch_names": branch_names,
                "sequence_length": int(SEQUENCE_LENGTH)}, indent=2),
    encoding="utf-8",
)

print(f"Saved to {ARTIFACTS}:")
for path in sorted(ARTIFACTS.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(ARTIFACTS)}  ({path.stat().st_size / 1024:.0f} KB)")

# %% [markdown]
# ## Verdict (T2.6 VERIFY block)
#
# The VERIFY condition is: *"Report exists, states positive class explicitly, includes a
# hand-verified metric, and makes an honest (not inflated) comparison against the T1.2 baseline."*
#
# - **Positive class:** stated in the header, printed by every metrics block, carried as a column in
#   every saved table.
# - **Hand-verified:** every metric above passes `hand_verify_metrics`, which recomputes it from the
#   raw TP/FP/TN/FN cells and asserts agreement with scikit-learn.
# - **Honest comparison:** `reports/phase2_results.md` is written from these numbers, including the
#   caveats below.
#
# ### Caveats that must travel with these numbers
#
# 1. **The ensemble and the baseline handle imbalance differently.** The baseline uses SMOTE on the
#    training fold; the ensemble uses class weights, because SMOTE on flow *sequences* would
#    fabricate physically impossible sessions (`train_utils` docstring). The comparison is still
#    meaningful — both are leakage-free and evaluated on real, unseen traffic — but it is not a
#    single-variable comparison.
# 2. **They are evaluated on different units.** The baseline classifies individual flow *records*;
#    the ensemble classifies 10-record *windows*. A window-level F1 and a record-level F1 are not
#    the same quantity.
# 3. **IoTID20's sessions are 100% label-pure**, because each attack scenario was captured
#    separately. That makes the sequence task easier than a live deployment, where an attack begins
#    mid-session. This inflates all sequence-model results here relative to reality, and Edge-IIoTset
#    (T3.6) is the check on how much.
# 4. **Nothing here is compared to the base paper's headline numbers.** Where `final_comparison.md`
#    (T4.4) does cite them, the metric-inversion caveat (`PRD.md §10`) must appear in the same
#    paragraph.
