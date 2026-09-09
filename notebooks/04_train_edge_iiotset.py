# %% [markdown]
# # T3.6 — The ensemble on Edge-IIoTset
#
# **Build-Instructions task:** T3.6 (the second half) · **Spec:** `TRD.md §6.1`, `TRD.md §3`
#
# ## What changed, and why this notebook exists
#
# T3.6 asks for two things: repeat T1.2's leakage-free pipeline on Edge-IIoTset, *and* repeat
# T2.6's ensemble training on it. `reports/t3_6_edge_iiotset.md` delivered the first and declared
# the second impossible, on the grounds that
#
# > `frame.time` is corrupted [...] Records cannot be ordered in time. **Row order is not capture
# > order.**
#
# **The first clause is right and the second is wrong**, and the difference is the whole task.
# `frame.time` is damaged — an unquoted comma inside `Dec 26, 2021 22:14:30.939803000 IST` cost
# the file its calendar date — but the *clock* survives on 90.04% of rows. Recovering it shows the
# opposite of what the earlier report assumed:
#
# | Check | Result |
# |---|---|
# | Rows with a recoverable clock time | **142,088 of 157,800 (90.04%)** |
# | Blocks where row order agrees with the clock | **13 of 13**, 0.9992–1.0000 of adjacent pairs |
# | Backward steps after midnight-wrap repair | **13 of 142,088**, worst −0.024 s |
#
# Row order in this file **is** capture order, and the recovered clock proves it independently
# rather than assuming it. So sequences can be built, and the ensemble can be trained. The earlier
# claim was an inference from a corrupted column, not a measurement of it — the same species of
# error `docs/calibration_decision.md` records against this project once already.
#
# ## What this costs, stated before any result
#
# - **Two attack classes are lost.** `DDoS_UDP` (14,498 rows) and `MITM` (1,214) carry no
#   parseable clock at all. Imputing one would fabricate the ordering this notebook depends on, so
#   they are dropped. The ensemble half covers **13 of 15** attack types; the record-level half in
#   `reports/t3_6_edge_iiotset.md` still covers all 15.
# - **Two corrections are this project's own construction**, not recovered data: midnight-wrap
#   repair and block separation (`src/preprocessing/clean.py`, T3.6 section).
# - **The 98.91% duplicate rate does not go away.** It cannot be deduplicated here: deleting
#   records from a time series is not a sequence any more. §6 measures what it does to the result.
#
# ## Positive-class convention
#
# **`Attack` (label 1) is the positive class** — `TRD.md §2.3`, the opposite of the base paper's
# choice. Every metric below is hand-verified from its raw confusion matrix under this convention.

# %%
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.config import POSITIVE_CLASS_NAME, RANDOM_STATE, REPORTS_DIR
from src.evaluation.metrics import compute_metrics, hand_verify_metrics, results_table
from src.models.bilstm_branch import build_bilstm_branch
from src.models.cnn_branch import build_cnn_branch
from src.models.fusion import confidence_weighted_fusion
from src.models.train_utils import (
    compute_class_weights,
    scale_sequences,
    set_global_seeds,
    train_branch,
)
from src.models.transformer_branch import build_transformer_branch
from src.preprocessing.clean import (
    build_edge_iiotset_timeline,
    clean_edge_iiotset,
    load_edge_iiotset,
)
from src.preprocessing.sequence_builder import (
    SEQUENCE_LENGTH,
    TRAIN_STRIDE,
    assign_sessions,
    build_sequences,
    split_sessions,
    subset_by_sessions,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
set_global_seeds(RANDOM_STATE)
print(f"seed={RANDOM_STATE}  positive class={POSITIVE_CLASS_NAME}")

# %% [markdown]
# ## Step 1 — Clean, exactly as the record-level half did
#
# `clean_edge_iiotset` is reused unchanged, so the sequence half inherits every leakage removal the
# record-level half established: placeholder-spelling normalisation, the content columns that hold
# the attack strings themselves, and the ephemeral ports. **`drop_duplicates=False` here**, unlike
# the record-level run — see the header. 30 features survive, not the base paper's 46.

# %%
raw = load_edge_iiotset()
X, y, meta = clean_edge_iiotset(raw, drop_duplicates=False)
print(f"Cleaned: {X.shape[0]:,} rows x {X.shape[1]} features, Attack rate {y.mean():.4f}")
duplicate_rate = float(X.duplicated().mean())
print(f"Exact duplicate feature rows: {duplicate_rate:.4%}  (kept — see header)")

# %% [markdown]
# ## Step 2 — Recover the timeline
#
# The measurement that unblocks this notebook. Rows without a clock are dropped, never imputed.

# %%
timeline, keep_mask = build_edge_iiotset_timeline(meta)
X = X.loc[keep_mask].reset_index(drop=True)
y = y.loc[keep_mask].reset_index(drop=True)

dropped = pd.Series(meta.loc[~keep_mask, "Attack_type"]).value_counts()
print(f"Rows with a recoverable clock: {len(timeline):,} of {len(meta):,} ({keep_mask.mean():.2%})")
print("\nDropped for having no clock at all:")
print(dropped.to_string())

ordering = timeline["Timestamp"].astype("int64").to_numpy()
backsteps = int((np.diff(ordering) < 0).sum())
print(f"\nBackward steps in the reconstructed timeline: {backsteps:,}")

# %% [markdown]
# ## Step 3 — Sessionise, then split by **session**
#
# Same policy as T2.6 (`docs/architecture_decision.md` §1.1, §1.4): sessions are per destination
# device with a 1 s inactivity gap, and whole *sessions* — never windows — are assigned to folds,
# because adjacent windows share up to 9 of their 10 records.
#
# **One difference from notebook 03, forced by this dataset.** `split_sessions` balances its folds
# by counting how many entries each session contributes, and its signature documents those entries
# as *windows*. Notebook 03 passes it the *record*-level series, which is a fair proxy on IoTID20
# but not here: Edge-IIoTset's attack sessions run to 7,050 records (1,409 windows each) while its
# Normal sessions have a median length of 2 (1 window each). Counting records therefore
# mis-estimates window counts by three orders of magnitude between classes, and a first run of this
# notebook produced folds of 93.7% / 3.2% / 3.2% whose **test fold contained no Normal windows at
# all** — every model scored precision 1.0000 against zero negatives, which is not a result.
#
# So windows are built first and the split is taken over window-level session ids, as
# `split_sessions` documents. No window can straddle a fold either way: windows never cross a
# session boundary, and whole sessions are assigned.

# %%
sessions = assign_sessions(timeline)
sizes = sessions.value_counts()
print(f"{sessions.nunique():,} sessions over {timeline['Dst_IP'].nunique():,} destination hosts")
print(f"  median size {sizes.median():.0f}, mean {sizes.mean():.1f}, max {sizes.max():,}")
print(f"  sessions with >= 2 records: {int((sizes >= 2).sum()):,}")

purity = pd.DataFrame({"s": sessions, "y": y}).groupby("s")["y"].mean()
print(f"  label-pure sessions: {((purity == 0) | (purity == 1)).mean():.4%}")

# %% [markdown]
# ## Step 4 — Window, then split those windows by session

# %%
windowed = build_sequences(
    X, y, timeline,
    sequence_length=SEQUENCE_LENGTH,
    stride=TRAIN_STRIDE,
    session_ids=sessions,
)
print(windowed.summary())

train_sessions, test_sessions, val_sessions = split_sessions(windowed.session_ids, windowed.y)
assert not (set(train_sessions) & set(test_sessions)), "session leaked between train and test"
assert not (set(train_sessions) & set(val_sessions)), "session leaked between train and validation"

folds = {
    "train": subset_by_sessions(windowed, train_sessions),
    "test": subset_by_sessions(windowed, test_sessions),
    "val": subset_by_sessions(windowed, val_sessions),
}
for name, fold in folds.items():
    share = len(fold.X) / len(windowed.X)
    print(f"--- {name} ({share:.2%} of windows, Attack rate {fold.y.mean():.4f}) ---")
    print(fold.summary())

assert folds["train"].X.shape[1] == SEQUENCE_LENGTH > 1, "TRD.md §9: sequences must be real"
assert (folds["test"].y == 0).sum() > 0, "a test fold with no negatives makes precision meaningless"

# %%
X_train, X_test, X_val = scale_sequences(folds["train"].X, folds["test"].X, folds["val"].X)
y_train, y_test, y_val = folds["train"].y, folds["test"].y, folds["val"].y
n_features = X_train.shape[2]
print(f"Input shape (batch, sequence_length, features) = (n, {SEQUENCE_LENGTH}, {n_features})")

# %% [markdown]
# ## Step 5 — Train the three branches
#
# Same builders, same frozen hyperparameters as T2.6. Nothing is retuned for this dataset: the
# question T3.6 exists to answer is whether *the ensemble* transfers, and retuning the branches
# per dataset would confound it.

# %%
builders = {
    "cnn": build_cnn_branch,
    "bilstm": build_bilstm_branch,
    "transformer": build_transformer_branch,
}
class_weights = compute_class_weights(y_train)
print(f"class weights: {class_weights}")

models, histories, train_seconds = {}, {}, {}
for name, builder in builders.items():
    started = time.perf_counter()
    model = builder(sequence_length=SEQUENCE_LENGTH, n_features=n_features)
    histories[name] = train_branch(
        model, X_train, y_train, X_val, y_val, class_weights=class_weights, verbose=0
    )
    train_seconds[name] = time.perf_counter() - started
    models[name] = model
    epochs = len(histories[name].history["loss"])
    print(f"{name:12s} {epochs:2d} epochs  {train_seconds[name]:6.1f}s")

# %% [markdown]
# ## Step 6 — Evaluate every branch, then the fusion
#
# Metrics are hand-verified against the raw confusion matrix, as in T1.2 and T2.6.

# %%
branch_probabilities = [models[n].predict(X_test, verbose=0) for n in builders]
results = []
for name, probabilities in zip(builders, branch_probabilities):
    result = compute_metrics(y_test, probabilities.argmax(axis=1), model_name=f"{name} alone")
    results.append(hand_verify_metrics(result, verbose=False))

fusion = confidence_weighted_fusion(branch_probabilities, branch_names=list(builders))
fused_result = hand_verify_metrics(
    compute_metrics(y_test, fusion.predictions, model_name="full three-branch ensemble"), verbose=False
)
results.append(fused_result)

table = results_table(results).sort_values("f1", ascending=False)
print(f"POSITIVE CLASS = {POSITIVE_CLASS_NAME} (label 1)\n")
print(table.to_string(index=False))
print("\nhand-verified against confusion matrix:", all(r.hand_verified for r in results))

# %%
best_branch = max(results[:-1], key=lambda r: r.f1)
delta = fused_result.f1 - best_branch.f1
print(f"best single branch : {best_branch.model_name}  F1 {best_branch.f1:.4f}")
print(f"full ensemble      : F1 {fused_result.f1:.4f}")
print(f"ensemble - best branch = {delta:+.4f}")

predictions = np.stack([p.argmax(axis=1) for p in branch_probabilities])
unanimous = float((predictions == predictions[0]).all(axis=0).mean())
print(f"\nbranches agree on {unanimous:.2%} of test windows")

# %% [markdown]
# ## Step 7 — What the duplicate rate does to all of this
#
# 98.91% of rows are exact duplicates, and that cannot be cleaned away without destroying the
# sequences. The consequence is measurable rather than rhetorical: count how many test windows are
# *constant* (every timestep identical) and how many appear verbatim in the training fold.

# %%
def window_keys(tensor: np.ndarray) -> np.ndarray:
    """Hashable key per window, for exact-match comparisons across folds."""
    return np.array([hash(w.tobytes()) for w in np.ascontiguousarray(tensor)])

constant_test = float(
    np.mean([len(np.unique(w, axis=0)) == 1 for w in folds["test"].X])
)
train_keys = set(window_keys(folds["train"].X).tolist())
test_keys = window_keys(folds["test"].X)
verbatim = float(np.mean([k in train_keys for k in test_keys]))

print(f"test windows that are a single record repeated 10x : {constant_test:.2%}")
print(f"test windows appearing verbatim in the train fold  : {verbatim:.2%}")
print(f"distinct feature vectors among timed rows          : {X.drop_duplicates().shape[0]:,}")

# %% [markdown]
# ## Step 8 — Write the report

# %%
report_path = Path(REPORTS_DIR) / "t3_6_ensemble.md"
lines = [
    "# T3.6 (second half) — the ensemble on Edge-IIoTset",
    "",
    "**Task:** T3.6 · **Generated by:** `notebooks/04_train_edge_iiotset.ipynb` · "
    "**Spec:** `TRD.md §3`, `TRD.md §6.1`",
    "",
    "> POSITIVE CLASS = Attack (label 1); negative class = Normal (label 0). This is TRD.md §2.3's "
    "convention and is the OPPOSITE of the base paper's, which uses Normal as positive "
    "(PRD.md §2.1.3).",
    "",
    "## 1. This supersedes a claim in `reports/t3_6_edge_iiotset.md`",
    "",
    "That report stated that Edge-IIoTset's records \"cannot be ordered in time\" and that \"row "
    "order is not capture order\", and cut the ensemble half of T3.6 on that basis. The first half "
    "of the claim is true of the calendar date and false of the clock; the second is false "
    "outright. Measured here:",
    "",
    f"- **{keep_mask.mean():.2%}** of rows carry a recoverable clock time in `frame.time`.",
    "- Row order agrees with that clock in **all 13** timed capture blocks (0.9992-1.0000 of "
    "adjacent pairs).",
    f"- The reconstructed timeline has **{backsteps:,}** backward steps.",
    "",
    "The ordering exists. The ensemble therefore runs, and the numbers below are what it does.",
    "",
    "## 2. What was given up to get here",
    "",
    "| Concession | Cost |",
    "|---|---|",
    f"| Rows without a clock are dropped, never imputed | {int((~keep_mask).sum()):,} rows |",
    "| `DDoS_UDP` and `MITM` carry no clock at all | 13 of 15 attack types remain |",
    "| Midnight-wrap repair and block separation | this project's own construction, not data |",
    f"| Duplicates cannot be removed from a time series | {duplicate_rate:.2%} duplicate rows kept |",
    "",
    "## 3. Result",
    "",
    f"Windows: {len(folds['train'].X):,} train / {len(folds['test'].X):,} test / "
    f"{len(folds['val'].X):,} validation, shape "
    f"(batch, {SEQUENCE_LENGTH}, {n_features}). Split by session, never by window; "
    f"Attack rate {folds['train'].y.mean():.4f} / {folds['test'].y.mean():.4f} / "
    f"{folds['val'].y.mean():.4f}.",
    "",
    "> **A first run of this notebook produced an unusable version of this table**, and the reason "
    "is worth recording. `split_sessions` balances folds by counting each session's entries, and "
    "documents those entries as *windows*; notebook 03 passes it *records*. On IoTID20 that is a "
    "fair proxy. Here it is not — attack sessions run to 7,050 records against a Normal median of "
    "2 — and it produced 93.7%/3.2%/3.2% folds whose **test fold held no Normal windows at all**. "
    "Every model then scored precision 1.0000 against zero negatives. Splitting window-level ids, "
    "as the function documents, gives the fold balance above. `notebooks/03` still makes the "
    "record-level call; its folds were checked against this and are unaffected "
    "(81.2%/9.6%/9.2% of windows, Attack rate 0.896 train against 0.871 test), because IoTID20's "
    "session sizes are nowhere near this skewed. The proxy is safe there and wrong here.",
    "",
    "| Model | Positive class | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    *[
        f"| {r.model_name} | {r.positive_class} | {r.accuracy:.4f} | {r.precision:.4f} | "
        f"{r.recall:.4f} | {r.f1:.4f} | {r.tp:,} | {r.fp:,} | {r.tn:,} | {r.fn:,} |"
        for r in sorted(results, key=lambda r: r.f1, reverse=True)
    ],
    "",
    "Every row is hand-verified against its own confusion matrix.",
    "",
    "## 4. The question this task existed to answer",
    "",
    "`docs/calibration_decision.md` §6 designated T3.6 the decision point for whether the ensemble "
    "beats its best single branch on a second dataset. On IoTID20 it does not "
    "(`reports/ablation_study.md`).",
    "",
    f"- best single branch: **{best_branch.model_name}**, F1 {best_branch.f1:.4f}",
    f"- full three-branch ensemble: F1 {fused_result.f1:.4f}",
    f"- **difference: {delta:+.4f}**",
    f"- branches agree on {unanimous:.2%} of test windows",
    "",
    "## 5. Why this result should not be read as a clean answer",
    "",
    "The duplicate rate makes this dataset a poor referee, and the numbers say so:",
    "",
    f"- **{constant_test:.2%}** of test windows are a single record repeated {SEQUENCE_LENGTH} "
    "times — a window with no temporal variation for a recurrent or attention branch to model.",
    f"- **{verbatim:.2%}** of test windows appear **verbatim** in the training fold. The session "
    "split is honest — no session is shared — but with "
    f"{X.drop_duplicates().shape[0]:,} distinct feature vectors behind {len(X):,} timed rows, "
    "distinct sessions can still be composed of identical records.",
    "",
    "That is a property of the published CSV, not of the split, and no ordering fix reaches it. "
    "The ensemble comparison above is therefore **evidence, not a verdict**: it is drawn on a "
    "dataset where the branches have little to disagree about. The question stays open pending a "
    "dataset with intact timestamps and genuine record diversity.",
    "",
    "## 6. Reproducing",
    "",
    "```bash",
    ".venv/bin/jupyter nbconvert --to notebook --execute --inplace "
    "notebooks/04_train_edge_iiotset.ipynb",
    "```",
    "",
    f"Single seed ({RANDOM_STATE}), single split, as everywhere else in this project.",
    "",
]
report_path.write_text("\n".join(lines))
print(f"wrote {report_path}")

metadata = {
    "task": "T3.6-ensemble",
    "positive_class": POSITIVE_CLASS_NAME,
    "seed": RANDOM_STATE,
    "rows_timed": int(keep_mask.sum()),
    "rows_dropped_no_clock": int((~keep_mask).sum()),
    "timeline_backsteps": backsteps,
    "duplicate_rate": duplicate_rate,
    "n_features": int(n_features),
    "sequence_length": int(SEQUENCE_LENGTH),
    "windows": {k: int(len(v.X)) for k, v in folds.items()},
    "constant_test_windows": constant_test,
    "test_windows_verbatim_in_train": verbatim,
    "branch_agreement": unanimous,
    "results": {r.model_name: {"accuracy": r.accuracy, "precision": r.precision,
                         "recall": r.recall, "f1": r.f1} for r in results},
    "ensemble_minus_best_branch_f1": float(delta),
    "train_seconds": train_seconds,
}
(Path(REPORTS_DIR) / "t3_6_ensemble_metadata.json").write_text(json.dumps(metadata, indent=2))
print("wrote t3_6_ensemble_metadata.json")
