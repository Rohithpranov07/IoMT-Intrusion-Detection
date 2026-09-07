# %% [markdown]
# # T1.1 — Reproducing the base paper's data leakage, empirically
#
# **Build-Instructions task:** T1.1 · **Spec:** `TRD.md §2.1`, `PRD.md §2.1.1` (Objection #1)
#
# ## What this notebook claims, and how it proves it
#
# Objection #1 is that HIDS-IoMT (Berguiga et al.) applies SMOTE to the **full** dataset and only
# then splits 80/10/10. The textual evidence is the paper's own Table 5: its training/testing/
# validation totals (936,548 / 117,069 / 117,069) sum to 1,170,686 — within **2 rows** of its stated
# post-SMOTE instance count of 1,170,684. That is only possible if synthetic samples were dealt into
# the test and validation folds. Step 1 below strengthens this considerably: IoTID20's cleaned
# majority class is 585,342 rows, and 2 × 585,342 = 1,170,684 **exactly** reproduces the paper's
# post-SMOTE total — so that total is provably a 50/50 SMOTE balance of the *whole* dataset, and the
# folds it is split into are therefore folds of synthetic-contaminated data.
#
# A paper-reading argument is not enough. This notebook reproduces that pipeline order on IoTID20
# and produces three independent pieces of measured evidence:
#
# 1. **Direct synthetic contamination** — a *count*, not an inference, of how many test-fold rows
#    were fabricated by SMOTE rather than observed on the wire.
# 2. **Nearest-neighbour distances** — the distribution of distances from each minority-class test
#    row to its closest training row. A cluster at ~0 is the near-duplicate signature.
# 3. **An implausibly high test accuracy** on a fold that is not actually unseen.
#
# ## ⚠️ Nothing in this notebook is a result
#
# Every number here comes from a pipeline we are asserting is **broken**. The honest baseline is
# `02_leakage_free_baseline.ipynb` (T1.2), and that is the only number cited elsewhere in this
# project.
#
# ## Positive-class convention
#
# **This project treats `Attack` (label 1) as the positive class** — `TRD.md §2.3`. This is
# deliberately the *opposite* of the base paper, which uses `Normal` as positive and, in doing so,
# swaps its own precision and recall labels (`PRD.md §2.1.3` / `§10` — Objection #3). Every metric
# below is computed and hand-verified under the Attack-positive convention.

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

from src.config import POSITIVE_CLASS_NAME, POSITIVE_LABEL, RANDOM_STATE
from src.evaluation.leakage_check import check_leakage, synthetic_contamination
from src.evaluation.metrics import (
    POSITIVE_CLASS_STATEMENT,
    compute_metrics,
    hand_verify_metrics,
    results_table,
)
from src.models.baseline import MAX_DEPTH, N_ESTIMATORS, fit_predict
from src.preprocessing.clean import clean_iotid20, load_iotid20
from src.preprocessing.feature_select import IOTID20_N_FEATURES, RandomForestFeatureSelector
from src.preprocessing.resample import leaky_resample_then_split, scale_split

print(POSITIVE_CLASS_STATEMENT)
print(f"\nSeed: {RANDOM_STATE} | baseline RF: {N_ESTIMATORS} trees, max_depth={MAX_DEPTH}")

# %% [markdown]
# ## Step 1 — Load and clean IoTID20
#
# The raw CSV is 625,783 rows × 86 columns = **83 feature columns** (matching the base paper's
# stated raw feature count, `TRD.md §6.1`) **+ 3 label columns** (`Label`, `Cat`, `Sub_Cat`).
#
# `clean_iotid20` follows the base paper's §II.G.1 cleaning step and states every drop explicitly:
# identifiers move to metadata, ±inf becomes NaN, NaN rows are dropped, and zero-variance columns
# go. Exact duplicate rows are **kept** (`drop_duplicates=False`) because the base paper never
# mentions removing them — we stay faithful to the pipeline under critique. Their count is
# reported below, since duplicates are a *separate* leakage vector we return to at the end.

# %%
raw = load_iotid20()
print(f"Raw: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
print(f"\nLabel distribution (raw):\n{raw['Label'].value_counts()}")
print(f"\nAttack sub-categories:\n{raw['Sub_Cat'].value_counts()}")

# %%
X, y, meta = clean_iotid20(raw, drop_duplicates=False)
print(f"\nCleaned feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Attack (positive, 1): {int((y == 1).sum()):,}   Normal (0): {int((y == 0).sum()):,}")
print(f"Class imbalance: {(y == 1).mean():.1%} Attack")
print(f"\nExact duplicate feature rows retained: {int(X.duplicated().sum()):,}")

# %% [markdown]
# ### Table 5 arithmetic, checked against the real dataset
#
# The base paper reports 1,170,684 post-SMOTE instances. Balancing IoTID20's cleaned majority
# (Attack) class 50/50 gives 2 × 585,342, which the cell below shows is **exactly** 1,170,684.
#
# Two consequences, both reported honestly:
#
# 1. The match is exact, so the paper's post-SMOTE total is confirmed to be a 50/50 SMOTE balance of
#    the entire dataset — stronger evidence than `PRD.md §2.1.1` claimed.
# 2. The paper's own Table 5 split totals sum to 1,170,686, which is **2 more** than that. `PRD.md
#    §2.1.1` describes these as matching "exactly"; they do not — they are off by 2. The
#    discrepancy is trivial in size (1.7 parts per million, almost certainly rounding in the
#    paper's split reporting) and does not weaken Objection #1, but this project does not get to
#    round in its own favour. `PRD.md §2.1.1`'s wording should be corrected to "within 2 rows of".

# %%
n_attack = int((y == 1).sum())
paper_post_smote_total = 1_170_684
paper_split_totals = (936_548, 117_069, 117_069)

print(f"IoTID20 majority (Attack) rows            : {n_attack:,}")
print(f"Balanced 2 × majority                     : {2 * n_attack:,}")
print(f"Base paper's stated post-SMOTE total      : {paper_post_smote_total:,}")
print(f"Difference                                : {abs(2 * n_attack - paper_post_smote_total):,} "
      f"({abs(2 * n_attack - paper_post_smote_total) / paper_post_smote_total:.2%})")
print(f"\nBase paper's train/test/val totals        : {paper_split_totals}")
print(f"Their sum                                 : {sum(paper_split_totals):,}")
print(f"Equal to its post-SMOTE total?            : {sum(paper_split_totals) == paper_post_smote_total}")

# %% [markdown]
# ## Step 2 — Feature selection (83 → 62)
#
# **This is not the base paper's PSO.** The paper's Algorithm 5 states no particle count, no
# iteration count, no inertia weight `w`, no acceleration constants `c1`/`c2`, and no fitness
# function (`TRD.md §6.1`), so it cannot be reproduced. Rather than invent those values and present
# them as the paper's — forbidden by `Build-Instructions.md` §A.1 — we use Random Forest impurity
# importance, a fully specified alternative the paper also mentions. See
# `docs/feature_selection_decision.md`.
#
# We keep the **same number** of features (62) so dimensionality stays comparable, but the selected
# subset is almost certainly different from theirs — and the paper never lists its subset, so no
# comparison is possible.
#
# Note the selector is fitted on the **full** dataset here. That is itself part of the broken
# pipeline being reproduced; the honest notebook fits it on the training fold only.

# %%
selector = RandomForestFeatureSelector(n_features=IOTID20_N_FEATURES, random_state=RANDOM_STATE)
X_selected = selector.fit_transform(X, y)
print(f"Selected {X_selected.shape[1]} of {X.shape[1]} features")
selector.ranking_.to_csv(REPO_ROOT / "reports" / "iotid20_feature_ranking_leaky.csv", index=False)
selector.ranking_.head(15)

# %% [markdown]
# ## Step 3 — The broken order: SMOTE the full dataset, *then* split
#
# `leaky_resample_then_split` is named so this mistake can never be made silently. It tags every
# row SMOTE fabricates, which is what makes Step 4's contamination count exact.

# %%
leaky = scale_split(leaky_resample_then_split(X_selected, y, random_state=RANDOM_STATE))
print(leaky.summary())

# %% [markdown]
# ## Step 4 — Evidence #1: direct synthetic contamination of the test fold
#
# This is a count, not a statistical inference. Every synthetic row is by construction an
# interpolation between two real minority rows — rows the model trains on. Their presence in the
# test fold is leakage **by definition**.

# %%
leaky_contamination = synthetic_contamination(
    leaky.y_test, leaky.synthetic_test, "LEAKY (SMOTE before split)", fold_name="test"
)
print(leaky_contamination.summary())

# %% [markdown]
# ## Step 5 — Evidence #2: an implausibly high test accuracy
#
# Same classifier used in T1.2, so the only thing that differs between the two notebooks is the
# pipeline order.
#
# **Positive class = Attack (label 1).** Metrics are hand-verified against the raw confusion matrix
# below — the exact check that would have caught the base paper's Objection #3 before publication.

# %%
_, y_pred_leaky = fit_predict(leaky.X_train, leaky.y_train, leaky.X_test, random_state=RANDOM_STATE)
leaky_metrics = compute_metrics(leaky.y_test, y_pred_leaky, "LEAKY pipeline (SMOTE before split)")
print(leaky_metrics.report())

# %%
# Hand-verify every metric straight from TP/FP/TN/FN, with Attack as the positive class.
leaky_metrics = hand_verify_metrics(leaky_metrics, verbose=True)

# %% [markdown]
# ## Step 6 — Evidence #3: nearest-neighbour distances
#
# For each minority-class (`Normal`) test row, the Euclidean distance to its nearest **training**
# row in scaled feature space. A cluster at ~0 means the test row is a near-duplicate of something
# the model already saw.
#
# One honest caveat, stated up front: after cleaning, 363,884 of IoTID20's 625,415 rows (58.2%) are
# exact duplicates in the 69-feature space, so a *correctly ordered* pipeline also shows some
# near-zero distances. That confound is why Step 4's direct count is the stronger evidence.
# Notebook 02 runs this same check on the honest pipeline for comparison.

# %%
leaky_nn = check_leakage(
    leaky.X_train, leaky.y_train, leaky.X_test, leaky.y_test,
    pipeline_name="LEAKY (SMOTE before split)",
)
print(leaky_nn.summary())

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
finite = leaky_nn.distances[leaky_nn.distances > 0]
ax.hist(np.log10(finite), bins=60, color="#c0392b", alpha=0.85)
ax.axvline(np.log10(1e-6), color="black", ls="--", lw=1.2, label="near-zero threshold (1e-6)")
ax.set_xlabel("log10( distance to nearest training row )")
ax.set_ylabel("minority-class test rows")
ax.set_title("LEAKY pipeline: test rows sit on top of training rows")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7 — Save the leaky result for the T1.2 comparison

# %%
summary = results_table([leaky_metrics])
summary["pipeline_order"] = "resample BEFORE split (base paper's order)"
summary["test_fold_synthetic_fraction"] = leaky_contamination.fraction_synthetic
summary["nn_near_zero_fraction"] = leaky_nn.near_zero_fraction
summary.to_csv(REPO_ROOT / "reports" / "t1_1_leaky_result.csv", index=False)
summary.T

# %% [markdown]
# ## Verdict (T1.1 VERIFY block)
#
# The VERIFY condition for T1.1 is: *"Notebook output shows both an implausibly high test accuracy
# and a nearest-neighbour distance distribution with a cluster of near-zero distances for
# oversampled classes."* Both are shown above, plus a third and stronger piece of evidence — the
# direct count of fabricated test rows.
#
# **Objection #1 is now measured, not asserted.** The number this project actually reports as its
# baseline is produced in `02_leakage_free_baseline.ipynb`.
#
# ### Secondary finding, recorded for later
#
# IoTID20's 363,884 exact duplicate rows (58.2% of the cleaned dataset) are an **independent**
# leakage vector that the SMOTE ordering fix does not address. The base paper does not mention deduplication; neither does the
# senior's prior work. `clean_iotid20(drop_duplicates=True)` exists so this can be quantified as a
# sensitivity check, and it is worth a paragraph in the Review 2 report — it is a defect in the
# *dataset's* usage across this literature, not just in one paper.
