# %% [markdown]
# # T1.2 — The leakage-free baseline
#
# **Build-Instructions task:** T1.2 · **Spec:** `TRD.md §2.2`, `PRD.md §3` Contribution 1
#
# ## What this notebook is
#
# `01_reproduce_leakage.ipynb` showed the base paper's pipeline order is broken. This notebook
# rebuilds the same pipeline with **one thing changed** — the split moved before the resampling —
# and reports the resulting honest numbers.
#
# > **The numbers in this notebook are the ones this project cites everywhere else.** The T1.1
# > figures exist only as evidence of a defect; they are never a result and never a target.
#
# ## The corrected pipeline (`TRD.md §2.2`)
#
# ```
# clean → label-encode → feature-select (TRAIN FOLD ONLY)
#       → SPLIT 80/10/10
#       → SMOTE (TRAIN FOLD ONLY)
#       → MinMax scale (scaler FITTED ON TRAIN FOLD ONLY)
#       → train → evaluate
# ```
#
# Three things move, not one. Beyond the SMOTE ordering, **feature selection** and the **scaler**
# are also fitted on the training fold only. Fitting either on the full dataset leaks test-fold
# information into the model — a milder version of the same bug, and one the base paper's described
# order commits as well.
#
# Everything else — the cleaning, the 62-feature budget, the classifier and its hyperparameters,
# the seed — is held identical to T1.1, so the before/after difference is attributable to the
# pipeline order alone.
#
# ## Positive-class convention
#
# **`Attack` (label 1) is the positive class** — `TRD.md §2.3`, the opposite of the base paper's
# choice. Every metric below is hand-verified from the raw confusion matrix under this convention.
# This is the check that would have caught the base paper's swapped precision/recall labels
# (Objection #3, `PRD.md §2.1.3` / `§10`) before publication, and `TRD.md §2.3` requires we apply it
# to our own results, not just theirs.

# %%
import logging
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", force=True)

from src.config import RANDOM_STATE
from src.evaluation.leakage_check import (
    check_leakage,
    compare_reports,
    synthetic_contamination,
)
from src.evaluation.metrics import (
    POSITIVE_CLASS_STATEMENT,
    compute_metrics,
    hand_verify_metrics,
    results_table,
)
from src.models.baseline import MAX_DEPTH, N_ESTIMATORS, fit_predict
from src.preprocessing.clean import clean_iotid20, load_iotid20
from src.preprocessing.feature_select import IOTID20_N_FEATURES, RandomForestFeatureSelector
from src.preprocessing.resample import (
    resample_training_fold,
    scale_split,
    split_dataset,
)

print(POSITIVE_CLASS_STATEMENT)
print(f"\nSeed: {RANDOM_STATE} | baseline RF: {N_ESTIMATORS} trees, max_depth={MAX_DEPTH} "
      "(identical to T1.1)")

# %% [markdown]
# ## Step 1 — Load and clean (identical to T1.1)

# %%
raw = load_iotid20()
X, y, meta = clean_iotid20(raw, drop_duplicates=False)
print(f"Cleaned: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Attack (positive, 1): {int((y == 1).sum()):,}   Normal (0): {int((y == 0).sum()):,}")

# %% [markdown]
# ## Step 2 — **Split first.** Nothing has been resampled or scaled yet.
#
# 80/10/10 train/test/validation, stratified on the label so every fold keeps IoTID20's real
# 93.6% / 6.4% class balance. The test and validation folds are now frozen: from here on, nothing
# fitted on them can touch the model, and nothing synthetic can enter them.

# %%
split = split_dataset(X, y, random_state=RANDOM_STATE)
print(split.summary())

# %% [markdown]
# ## Step 3 — Feature selection, fitted on the **training fold only** (83 → 62)
#
# Same method and same 62-feature budget as T1.1 (Random Forest importance — *this project's own
# choice*, not the base paper's unreproducible PSO; see `docs/feature_selection_decision.md`). The
# difference is what it is allowed to see.

# %%
selector = RandomForestFeatureSelector(n_features=IOTID20_N_FEATURES, random_state=RANDOM_STATE)
selector.fit(split.X_train, split.y_train)
selector.ranking_.to_csv(REPO_ROOT / "reports" / "iotid20_feature_ranking_honest.csv", index=False)

split.X_train = selector.transform(split.X_train)
split.X_test = selector.transform(split.X_test)
split.X_val = selector.transform(split.X_val)
print(f"Selected {len(selector.selected_features_)} features on the training fold only")
selector.ranking_.head(15)

# %% [markdown]
# ### How much did fitting the selector on the training fold change the choice?
#
# If the two rankings agree closely, selection leakage was a minor effect on this dataset — worth
# knowing, and worth reporting honestly either way.

# %%
leaky_ranking_path = REPO_ROOT / "reports" / "iotid20_feature_ranking_leaky.csv"
if leaky_ranking_path.exists():
    leaky_ranking = pd.read_csv(leaky_ranking_path)
    leaky_kept = set(leaky_ranking.loc[leaky_ranking["kept"], "feature"])
    honest_kept = set(selector.selected_features_)
    print(f"Features kept by both      : {len(leaky_kept & honest_kept)}")
    print(f"Only in the leaky ranking  : {sorted(leaky_kept - honest_kept)}")
    print(f"Only in the honest ranking : {sorted(honest_kept - leaky_kept)}")
else:
    print("Run 01_reproduce_leakage.ipynb first to enable this comparison.")

# %% [markdown]
# ## Step 4 — SMOTE, **training fold only**
#
# `resample_training_fold` refuses to run on a split that came from the leaky path, so the broken
# order cannot be reached by accident. Test and validation folds are returned untouched: they must
# keep the real 93.6/6.4 class distribution, because that is the distribution the deployed system
# would actually meet.

# %%
resampled = resample_training_fold(split, random_state=RANDOM_STATE)
print(resampled.summary())

# %% [markdown]
# ## Step 5 — Scale, with the scaler fitted on the **training fold only**

# %%
honest = scale_split(resampled)
print(f"Scaler fitted on {len(honest.X_train):,} training rows, applied to all three folds")

# %% [markdown]
# ## Step 6 — Confirm the test fold is clean
#
# Zero synthetic rows, by construction. This is the mirror image of T1.1's Step 4.

# %%
honest_contamination = synthetic_contamination(
    honest.y_test, honest.synthetic_test, "HONEST (split before SMOTE)", fold_name="test"
)
print(honest_contamination.summary())

# %% [markdown]
# ## Step 7 — Train and evaluate
#
# **Positive class = Attack (label 1)**, per `TRD.md §2.3`.

# %%
model, y_pred = fit_predict(
    honest.X_train, honest.y_train, honest.X_test, random_state=RANDOM_STATE
)
honest_metrics = compute_metrics(honest.y_test, y_pred, "HONEST baseline (split before SMOTE)")
print(honest_metrics.report())

# %% [markdown]
# ### Hand-verification against the confusion matrix
#
# `TRD.md §2.3` requires this before any table is published. Each metric is recomputed from the raw
# TP/FP/TN/FN cells using the literal formulas, then asserted equal to scikit-learn's value. An
# `AssertionError` here would mean the positive-class convention had drifted somewhere in our own
# code — Objection #3 happening to us.

# %%
honest_metrics = hand_verify_metrics(honest_metrics, verbose=True)

# %%
# Independent cross-check: recompute with the roles reversed to show the labels are NOT swapped.
# With Normal as positive (the BASE PAPER'S convention) the numbers change — which is precisely
# why a results table is meaningless without stating its convention.
tp, fp, tn, fn = honest_metrics.tp, honest_metrics.fp, honest_metrics.tn, honest_metrics.fn
print("Same confusion matrix, both conventions:\n")
print(f"{'':<28}{'Attack positive (OURS)':>24}{'Normal positive (paper)':>26}")
print(f"{'Precision':<28}{tp / (tp + fp):>24.6f}{tn / (tn + fn):>26.6f}")
print(f"{'Recall':<28}{tp / (tp + fn):>24.6f}{tn / (tn + fp):>26.6f}")
print("\nThe two columns differ. Any table omitting its convention is ambiguous — that ambiguity")
print("is what Objection #3 is about, and it is why every table in this repo states its own.")

# %% [markdown]
# ## Step 8 — Validation fold
#
# The validation fold was never used for fitting or selection. Close agreement with the test fold
# indicates the test number is not itself a lucky draw.

# %%
y_pred_val = model.predict(honest.X_val)
val_metrics = hand_verify_metrics(
    compute_metrics(honest.y_val, y_pred_val, "HONEST baseline (validation fold)"), verbose=False
)
print(val_metrics.report())

# %% [markdown]
# ## Step 9 — Nearest-neighbour check on the honest pipeline
#
# Compared side by side with T1.1's. Read this table with the caveat stated in notebook 01: IoTID20
# has 363,884 exact duplicate rows in the 69-feature space (58.2% of it), so the honest pipeline
# shows some near-zero distances too.
# The *direct* evidence is the synthetic-contamination contrast — T1.1's test fold is substantially
# fabricated, this one is 0.00%.

# %%
honest_nn = check_leakage(
    honest.X_train, honest.y_train, honest.X_test, honest.y_test,
    pipeline_name="HONEST (split before SMOTE)",
)
print(honest_nn.summary())

# %% [markdown]
# ## Step 10 — Before / after
#
# The headline of Contribution 1.

# %%
leaky_path = REPO_ROOT / "reports" / "t1_1_leaky_result.csv"
honest_summary = results_table([honest_metrics, val_metrics])
honest_summary["pipeline_order"] = "split BEFORE resample (TRD.md §2.2)"
honest_summary["test_fold_synthetic_fraction"] = [honest_contamination.fraction_synthetic, np.nan]
honest_summary["nn_near_zero_fraction"] = [honest_nn.near_zero_fraction, np.nan]

if leaky_path.exists():
    comparison = pd.concat([pd.read_csv(leaky_path), honest_summary], ignore_index=True)
else:
    comparison = honest_summary

comparison.to_csv(REPO_ROOT / "reports" / "t1_2_baseline_comparison.csv", index=False)
comparison.T

# %%
if leaky_path.exists():
    leaky_row = pd.read_csv(leaky_path).iloc[0]
    print(POSITIVE_CLASS_STATEMENT)
    print("\n" + "=" * 78)
    print(f"{'':<14}{'LEAKY (T1.1)':>16}{'HONEST (T1.2)':>18}{'change':>16}")
    print("=" * 78)
    for metric in ("accuracy", "precision", "recall", "f1"):
        lo, ho = float(leaky_row[metric]), getattr(honest_metrics, metric)
        print(f"{metric:<14}{lo:>16.6f}{ho:>18.6f}{ho - lo:>+16.6f}")
    print("-" * 78)
    print(f"{'test fold':<14}{leaky_row['test_fold_synthetic_fraction']:>15.2%} synthetic"
          f"{honest_contamination.fraction_synthetic:>17.2%} synthetic")
    print("=" * 78)

# %% [markdown]
# ## Step 11 — The result that did not go as expected, and what it means
#
# **T1.2's VERIFY block predicts the honest accuracy will be *measurably lower* than T1.1's. On
# IoTID20 it is not — it is marginally higher.** That is not a bug to tune away; it is a finding,
# and burying it would be precisely the kind of convenient omission this project criticises.
#
# Two things explain it, and the second is the important one:
#
# 1. **The two test folds are different populations.** T1.1's fold is ~50/50 Attack/Normal (a slice
#    of the balanced post-SMOTE dataset); this one keeps IoTID20's real 93.6/6.4 imbalance, where
#    accuracy is dominated by an easy majority class. The two accuracies are not commensurable.
# 2. **A second, larger leakage vector survives the fix.** Step 9's nearest-neighbour check on the
#    *honest* pipeline still shows ~42% of minority test rows sitting at near-zero distance from a
#    training row. Those are not synthetic — Step 6 proved the fold is 0.00% synthetic. They are
#    **exact duplicate records**: 363,884 of IoTID20's 625,415 cleaned rows (58.2%) are duplicates
#    in the 69-feature space. Random splitting therefore puts identical records on both sides
#    regardless of when SMOTE runs.
#
# So: fixing the SMOTE ordering removes *fabricated* test data (46.61% → 0.00%), which is real and
# necessary. But on this dataset it is not the binding constraint on how trustworthy the number is.
# **Deduplication is.** Neither the base paper nor the senior's prior work mentions it.
#
# The run below applies both fixes. This is the number this project cites as its baseline.

# %%
X_dedup, y_dedup, _ = clean_iotid20(raw, drop_duplicates=True)
print(f"Deduplicated: {X_dedup.shape[0]:,} rows (from {X.shape[0]:,}) × {X_dedup.shape[1]} features")
print(f"Attack (positive, 1): {int((y_dedup == 1).sum()):,}   Normal (0): {int((y_dedup == 0).sum()):,}")
print(f"Class imbalance: {(y_dedup == 1).mean():.1%} Attack")

# %%
split_d = split_dataset(X_dedup, y_dedup, random_state=RANDOM_STATE)

selector_d = RandomForestFeatureSelector(n_features=IOTID20_N_FEATURES, random_state=RANDOM_STATE)
selector_d.fit(split_d.X_train, split_d.y_train)
split_d.X_train = selector_d.transform(split_d.X_train)
split_d.X_test = selector_d.transform(split_d.X_test)
split_d.X_val = selector_d.transform(split_d.X_val)

dedup = scale_split(resample_training_fold(split_d, random_state=RANDOM_STATE))
print(dedup.summary())

# %%
model_d, y_pred_d = fit_predict(
    dedup.X_train, dedup.y_train, dedup.X_test, random_state=RANDOM_STATE
)
dedup_metrics = hand_verify_metrics(
    compute_metrics(
        dedup.y_test, y_pred_d, "LEAKAGE-FREE baseline (split before SMOTE + deduplicated)"
    ),
    verbose=True,
)
print()
print(dedup_metrics.report())

# %%
dedup_contamination = synthetic_contamination(
    dedup.y_test, dedup.synthetic_test, "DEDUPLICATED honest", fold_name="test"
)
dedup_nn = check_leakage(
    dedup.X_train, dedup.y_train, dedup.X_test, dedup.y_test,
    pipeline_name="DEDUPLICATED honest (split before SMOTE, duplicates removed)",
)
print(dedup_contamination.summary())
print()
print(dedup_nn.summary())

# %% [markdown]
# ### The three pipelines side by side
#
# **Positive class = Attack (label 1)** in every column (`TRD.md §2.3`).

# %%
leaky_row = pd.read_csv(leaky_path).iloc[0] if leaky_path.exists() else None

rows = [
    ("A. LEAKY (base paper's order)", leaky_row, leaky_nn_frac := float(leaky_row["nn_near_zero_fraction"]) if leaky_row is not None else np.nan),
    ("B. Honest order, duplicates kept", honest_metrics, honest_nn.near_zero_fraction),
    ("C. Honest order + deduplicated", dedup_metrics, dedup_nn.near_zero_fraction),
]

print(POSITIVE_CLASS_STATEMENT)
print("
" + "=" * 100)
print(f"{'pipeline':<36}{'accuracy':>11}{'precision':>11}{'recall':>10}{'F1':>10}"
      f"{'synthetic':>12}{'NN near-0':>11}")
print("=" * 100)
for name, m, nn_frac in rows:
    if m is None:
        continue
    get = (lambda k: float(m[k])) if isinstance(m, pd.Series) else (lambda k: getattr(m, k))
    synth = (
        float(m["test_fold_synthetic_fraction"]) if isinstance(m, pd.Series)
        else (dedup_contamination.fraction_synthetic if "dedup" in name.lower()
              else honest_contamination.fraction_synthetic)
    )
    print(f"{name:<36}{get('accuracy'):>11.6f}{get('precision'):>11.6f}"
          f"{get('recall'):>10.6f}{get('f1'):>10.6f}{synth:>11.2%}{nn_frac:>11.2%}")
print("=" * 100)
print("
'synthetic' = share of the test fold fabricated by SMOTE.")
print("'NN near-0' = share of minority test rows at <1e-6 distance from a training row.")
print("
C is this project's baseline. Every later result is compared against C, not A or B.")

# %%
final = results_table([dedup_metrics])
final["pipeline_order"] = "split BEFORE resample + deduplicated (TRD.md §2.2 + T1.2 Step 11)"
final["test_fold_synthetic_fraction"] = dedup_contamination.fraction_synthetic
final["nn_near_zero_fraction"] = dedup_nn.near_zero_fraction
final.to_csv(REPO_ROOT / "reports" / "t1_2_leakage_free_baseline.csv", index=False)
final.T

# %% [markdown]
# ## Verdict (T1.2 VERIFY block)
#
# The VERIFY condition for T1.2 is: *"Reported accuracy is measurably lower than T1.1's leaky
# number; the notebook contains an explicit positive-class statement and a hand-verified metric."*
#
# - **Positive-class statement:** stated in the header, printed by every metrics block, and carried
#   as a column in every saved table.
# - **Hand-verified metric:** Step 7 recomputes all four metrics from the raw confusion-matrix cells
#   and asserts agreement with scikit-learn; Step 7's cross-check shows the numbers under the base
#   paper's opposite convention, demonstrating why the label matters.
# - **Lower, honest number:** **not observed as predicted, and reported as such.** See Step 11:
#   removing the SMOTE ordering bug alone did *not* lower the accuracy on IoTID20, because a larger
#   leakage vector (58.2% duplicate rows) survives it. Applying both fixes gives pipeline C, which
#   is the number this project cites. The VERIFY block's expectation was reasonable but turned out
#   to be wrong on this dataset; `Build-Instructions.md` T1.2 should be annotated accordingly.
#

# ### What this baseline is, and is not
#
# This is a **Random Forest**, not the project's architecture. It exists to establish an honest
# reference point on a leakage-free pipeline. The committed architecture — the CNN + BiLSTM +
# Transformer ensemble over genuine flow *sequences* (`TRD.md §3`) — is built in Phase 2 and
# evaluated against this exact number in `reports/phase2_results.md`.
#
# ### On comparing any of this to the base paper
#
# HIDS-IoMT's published 99.92% / 99.91% / 99.99% / 99.95% must never be cited in this project
# without noting, **in the same paragraph**, that its precision and recall labels are very likely
# swapped (`PRD.md §2.1.3`, worked in full at `PRD.md §10`) and that its evaluation fold was
# contaminated in the manner reproduced in notebook 01. Those numbers are the object of a critique,
# not a target to approach.
