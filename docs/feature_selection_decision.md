# Feature-Selection Method Decision (T1.3)

**Status:** Decided · **Applies to:** IoTID20 (Phase 1–2) and Edge-IIoTset (Phase 3)
**Spec refs:** `TRD.md §6.1`, `Build-Instructions.md` §A.1 rule 2, T1.3

---

## 1. The problem this decision exists to solve

The base paper (Berguiga et al., *HIDS-IoMT*) reduces IoTID20 from 83 → 62 features and
Edge-IIoTset from 61 → 46 features using PSO (its Algorithm 5). It **never states**:

- the particle count,
- the iteration count,
- the inertia weight `w`,
- the acceleration constants `c1` / `c2`,
- the fitness function being optimised.

That step is therefore **not reproducible as written**. `TRD.md §6.1` and `Build-Instructions.md`
§A.1 forbid guessing those values and presenting them as the paper's. We must instead pick one of:

- **(a)** run our own PSO with our own, clearly-labelled hyperparameters, or
- **(b)** use a different, fully-specified selector — the base paper also mentions Random Forest
  feature importance — and state plainly that this is a deliberate deviation.

## 2. Decision

**We choose (b): Random Forest impurity-based feature importance, with a fixed target feature
count.**

## 3. Why

| Criterion | PSO (our own params) | RF importance | Winner |
|---|---|---|---|
| Reproducibility across team members | Stochastic swarm; needs seed + 5 documented hyperparameters, and results still shift with population init | One seeded `RandomForestClassifier`; deterministic given `random_state` | **RF** |
| Risk of being *mistaken* for the paper's PSO | High — a reader sees "PSO, 62 features" and assumes we reproduced Algorithm 5 | Nil — obviously and visibly a different method | **RF** |
| Compute cost | Every fitness evaluation trains a classifier; ~particles × iterations model fits | One model fit | **RF** |
| Comparability of feature *count* to the paper | Can match 62 | Can also match 62 (we take the top-k) | tie |
| Honesty about the paper's gap | Muddies it — implies the step was reproducible | Makes the gap explicit and citable | **RF** |

The decisive point is the third row of `Build-Instructions.md` §C's failure-mode table: inventing
PSO hyperparameters "fabricates reproducibility that doesn't exist." Choosing a different, fully
specified method makes the base paper's omission part of our *evidence*, not something we paper
over. Contribution 1 of this project is honesty about methodology; the feature-selection step
should not be the one place we quietly guess.

Selecting the **same number** of features (62 for IoTID20, 46 for Edge-IIoTset) keeps the
dimensionality comparable to the base paper's even though the *selection criterion* differs — so a
performance gap can be attributed to the pipeline (leakage) rather than to a different feature
budget.

## 4. Exact parameters (all named constants in `src/preprocessing/feature_select.py`)

| Constant | Value | Justification |
|---|---|---|
| `N_ESTIMATORS` | `200` | Enough trees for importance ranks to stabilise on a 600k-row dataset; beyond ~200 the top-k ordering stops moving while fit time keeps growing. |
| `MAX_DEPTH` | `None` (grow fully) | Importances are read off the fitted forest, not used for prediction; depth-limiting biases importance toward features that happen to split early. |
| `MIN_SAMPLES_LEAF` | `5` | Prevents single-row leaves from inflating the importance of near-unique columns (ports, IDs). |
| `CLASS_WEIGHT` | `"balanced_subsample"` | IoTID20 is imbalanced; without this, importance is dominated by whatever separates the majority class. Note this is **inside** the selector only — it is not a substitute for the resampling in `resample.py`. |
| `N_JOBS` | `-1` | Wall-clock only; does not affect the result given a fixed `random_state`. |
| `RANDOM_STATE` | `42` (from `src/config.py`) | Project-wide seed (`Build-Instructions.md` §A.3). |
| `IOTID20_N_FEATURES` | `62` | Matches the base paper's post-selection count for comparability (`TRD.md §6.1`). |
| `EDGE_IIOTSET_N_FEATURES` | `46` | Same, for Edge-IIoTset. |

**Selection rule:** rank all features by `feature_importances_` descending, take the top *k* where
*k* is the dataset's target count above. Ties are broken by original column order, which is
deterministic. The full ranked list (feature name, importance, kept/dropped) is written out
alongside the selection so the choice is auditable.

## 5. Leakage caveat that applies to this step too

Random Forest feature importance is **fit on the training fold only** in the leakage-free pipeline
(T1.2). Fitting the selector on the full dataset would leak test-fold information into the feature
choice — a milder version of the very bug this project exists to expose. The T1.1 *leaky*
reproduction deliberately does the opposite (selects on the full dataset before splitting), because
that is the pipeline order being demonstrated as broken.

## 6. What we do **not** claim

- We do **not** claim to have reproduced the base paper's PSO.
- We do **not** claim our 62 features are the same 62 features it selected — they are almost
  certainly a different subset, and the paper does not list its subset, so no comparison is possible.
- Any report citing "62 features" must say "62 selected by Random Forest importance (this
  project's own method — see `docs/feature_selection_decision.md`)", never "62 as in the base paper."
