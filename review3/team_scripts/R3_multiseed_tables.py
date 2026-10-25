"""Phase 7.1 and 7.3 on the full pipeline: Tables 3, 7 and 8 over 20 partitions.

RUN FROM THE PROJECT REPO ROOT on the machine that has the datasets (the Mac, or Colab with the
repo uploaded), after `scripts/download_data.py` has been run for both datasets:

    python team_scripts/R3_multiseed_tables.py --dataset iotid20 --table all     # ~2 h on a GPU
    python team_scripts/R3_multiseed_tables.py --dataset edge    --table 7
    python team_scripts/R3_multiseed_tables.py --summarise

Every finished partition is appended to reports/r3_multiseed/<dataset>_<table>.csv immediately, so
an interrupted run resumes where it stopped (rerun the same command).

Table 3 (leakage regimes, record-level Random Forest): LEAKY (SMOTE then split), ORDER (split then
         SMOTE), HONEST (dedup, split, SMOTE on training only); 20 seeds each.
Table 7 (ensemble ablation): the three frozen branches and the confidence-weighted fusion,
         re-trained from scratch on 20 session-level partitions (seeds 1..20).
Table 8 (explanation stability): SHAP and LIME top-5 Jaccard under 1% noise, on 5 of the 20
         trained ensembles (seeds 1..5; SHAP/LIME cost is the limit), 8 windows each.
Statistics: mean, sd, 95% CI (t, df = n-1); paired Wilcoxon signed-rank with Bonferroni correction.
Attack is the positive class throughout.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
OUT = REPO / "reports" / "r3_multiseed"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(1, 21))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def done(path, key="seed"):
    return set(pd.read_csv(path)[key]) if path.exists() else set()


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def metrics(y, p):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn), accuracy=(tp + tn) / (tp + tn + fp + fn),
                precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def call_with_seed(fn, *args, seed):
    """Pass the seed through whichever keyword the project function accepts."""
    params = inspect.signature(fn).parameters
    for name in ("random_state", "seed", "rng_seed"):
        if name in params:
            return fn(*args, **{name: seed})
    raise SystemExit(f"{fn.__name__} takes no seed argument; add `random_state` to it before running 20 seeds")


# ---------------------------------------------------------------------------------- data
def load(dataset):
    from src.preprocessing import clean
    if dataset == "iotid20":
        raw = clean.load_iotid20()
        X, y, meta = clean.clean_iotid20(raw, drop_duplicates=False)
        return X, y, meta
    raw = clean.load_edge_iiotset()
    X, y, meta = clean.clean_edge_iiotset(raw, drop_duplicates=False)
    return X, y, meta


# ---------------------------------------------------------------------------------- Table 3
def table3(dataset):
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    X, y, _ = load(dataset)
    X, y = X.to_numpy(float), y.to_numpy()
    keep = ~pd.DataFrame(X).duplicated().to_numpy()
    path = OUT / f"{dataset}_table3.csv"
    have = set(map(tuple, pd.read_csv(path)[["regime", "seed"]].values)) if path.exists() else set()
    for regime in ("LEAKY", "ORDER", "HONEST"):
        for s in SEEDS:
            if (regime, s) in have:
                continue
            t0 = time.time()
            if regime == "LEAKY":
                Xr, yr = SMOTE(random_state=s).fit_resample(X, y)
                synth = np.r_[np.zeros(len(y), bool), np.ones(len(yr) - len(y), bool)]
                tr, te = train_test_split(np.arange(len(yr)), test_size=0.2, random_state=s, stratify=yr)
                m = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1).fit(Xr[tr], yr[tr])
                row = metrics(yr[te], m.predict(Xr[te]))
                row["test_synthetic_pct"] = 100 * synth[te].mean()
            else:
                XX, yy = (X[keep], y[keep]) if regime == "HONEST" else (X, y)
                tr, te = train_test_split(np.arange(len(yy)), test_size=0.2, random_state=s, stratify=yy)
                Xt, yt = SMOTE(random_state=s).fit_resample(XX[tr], yy[tr])
                m = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1).fit(Xt, yt)
                row = metrics(yy[te], m.predict(XX[te]))
                row["test_synthetic_pct"] = 0.0
            row.update(regime=regime, seed=s, seconds=round(time.time() - t0, 1))
            append(path, row)
            print(f"T3 {dataset} {regime} seed {s}: F1 {row['f1']:.4f}", flush=True)


# ---------------------------------------------------------------------------------- Table 7
def train_partition(dataset, seed):
    """Notebook 03 (IoTID20) or 04 (Edge-IIoTset), with the partition seed exposed."""
    from src.models import bilstm_branch, cnn_branch, transformer_branch
    from src.models.fusion import confidence_weighted_fusion
    from src.models.train_utils import compute_class_weights, scale_sequences, set_global_seeds, train_branch
    from src.preprocessing import sequence_builder as sb

    set_global_seeds(seed)
    if dataset == "iotid20":
        from src.preprocessing.feature_select import IOTID20_N_FEATURES, RandomForestFeatureSelector
        from src.preprocessing import clean as _clean
        X, y, meta = _clean.clean_iotid20(_clean.load_iotid20(), drop_duplicates=True)  # as notebook 03
        X, y, meta = X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)
        sessions = sb.assign_sessions(meta)
        tr_s, te_s, va_s = call_with_seed(sb.split_sessions, sessions, y, seed=seed)
        masks = {k: sessions.isin(v).to_numpy() for k, v in (("train", tr_s), ("test", te_s), ("val", va_s))}
        sel = RandomForestFeatureSelector(n_features=IOTID20_N_FEATURES, random_state=seed)
        sel.fit(X[masks["train"]], y[masks["train"]])
        Xs = sel.transform(X)
        folds = {k: sb.build_sequences(Xs[m].reset_index(drop=True), y[m].reset_index(drop=True), meta[m].reset_index(drop=True),
                                       sequence_length=sb.SEQUENCE_LENGTH, stride=sb.TRAIN_STRIDE,
                                       session_ids=sessions[m].reset_index(drop=True)) for k, m in masks.items()}
    else:
        X, y, meta = load(dataset)
        from src.preprocessing.clean import build_edge_iiotset_timeline
        timeline, keep = build_edge_iiotset_timeline(meta)
        X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        sessions = sb.assign_sessions(timeline)
        w = sb.build_sequences(X, y, timeline, sequence_length=sb.SEQUENCE_LENGTH, stride=sb.TRAIN_STRIDE, session_ids=sessions)
        tr_s, te_s, va_s = call_with_seed(sb.split_sessions, w.session_ids, w.y, seed=seed)
        folds = {k: sb.subset_by_sessions(w, v) for k, v in (("train", tr_s), ("test", te_s), ("val", va_s))}
    Xtr, Xte, Xva = scale_sequences(folds["train"].X, folds["test"].X, folds["val"].X)
    ytr, yte, yva = folds["train"].y, folds["test"].y, folds["val"].y
    cw = compute_class_weights(ytr)
    builders = {"cnn": cnn_branch.build_cnn_branch, "bilstm": bilstm_branch.build_bilstm_branch,
                "transformer": transformer_branch.build_transformer_branch}
    models, probs = {}, {}
    for name, b in builders.items():
        set_global_seeds(seed)
        m = b(Xtr.shape[1], Xtr.shape[2], seed=seed)
        train_branch(m, Xtr, ytr, Xva, yva, class_weights=cw, verbose=0)
        models[name], probs[name] = m, m.predict(Xte, verbose=0)
    fused = confidence_weighted_fusion([probs[n] for n in builders], list(builders))
    return models, probs, fused, (Xtr, Xte, yte)


def table7(dataset):
    path = OUT / f"{dataset}_table7.csv"
    for s in SEEDS:
        if s in done(path):
            continue
        t0 = time.time()
        _, probs, fused, (_, _, yte) = train_partition(dataset, s)
        row = {"seed": s, "test_windows": int(len(yte)), "test_attack_rate": float(yte.mean())}
        for n, p in probs.items():
            for k, v in metrics(yte, p.argmax(1)).items():
                row[f"{n}_{k}"] = v
        for k, v in metrics(yte, fused.predictions).items():
            row[f"ensemble_{k}"] = v
        agree = np.mean((probs["cnn"].argmax(1) == probs["bilstm"].argmax(1)) & (probs["bilstm"].argmax(1) == probs["transformer"].argmax(1)))
        oracle = np.mean(np.any(np.stack([p.argmax(1) == yte for p in probs.values()]), axis=0))
        best = max(row[f"{n}_accuracy"] for n in probs)
        row.update(branch_agreement=float(agree), oracle_accuracy=float(oracle), oracle_ceiling_over_best=float(oracle - best),
                   seconds=round(time.time() - t0, 1))
        append(path, row)
        print(f"T7 {dataset} seed {s}: ensemble F1 {row['ensemble_f1']:.4f} best-branch acc {best:.4f}", flush=True)


# ---------------------------------------------------------------------------------- Table 8
def table8(dataset):
    from src.models.fusion import confidence_weighted_fusion
    from src.xai.lime_explainer import EnsembleLimeExplainer
    from src.xai.metrics import measure_stability
    from src.xai.shap_explainer import EnsembleShapExplainer
    path = OUT / f"{dataset}_table8.csv"
    for s in SEEDS[:5]:
        if s in done(path):
            continue
        models, _, fused, (Xtr, Xte, _) = train_partition(dataset, s)
        names = list(models)
        feature_names = [f"f{i}" for i in range(Xtr.shape[2])]
        bg = Xtr[np.random.default_rng(s).choice(len(Xtr), min(500, len(Xtr)), replace=False)]

        def predict_fn(w):
            return confidence_weighted_fusion([models[n].predict(w, verbose=0) for n in names], names).probabilities

        flagged = np.flatnonzero(fused.predictions == 1)[:8]
        row = {"seed": s}
        for method, Ex in (("shap", EnsembleShapExplainer), ("lime", EnsembleLimeExplainer)):
            kw = {"background_size": 100} if "background_size" in inspect.signature(Ex).parameters else {}
            ex = Ex(models, feature_names, bg, **kw)
            r = measure_stability(ex, predict_fn, Xte[flagged], method, random_state=s)
            row.update({f"{method}_jaccard": r.jaccard, f"{method}_spearman": r.spearman, f"{method}_flip_rate": r.prediction_flip_rate})
        append(path, row)
        print(f"T8 {dataset} seed {s}: SHAP {row['shap_jaccard']:.3f} LIME {row['lime_jaccard']:.3f}", flush=True)


# ---------------------------------------------------------------------------------- summary
def summarise():
    from scipy.stats import t as tdist
    from scipy.stats import wilcoxon
    lines = ["# Review 3 multi-partition summary (Attack = positive)\n"]
    tests = []
    for f in sorted(OUT.glob("*_table*.csv")):
        df = pd.read_csv(f)
        lines.append(f"## {f.stem}  (n = {df.seed.nunique()} partitions)\n")
        groups = df.groupby("regime") if "regime" in df else [("all", df)]
        for g, d in groups:
            for c in [c for c in d.columns if c.endswith(("f1", "precision", "recall", "accuracy", "jaccard", "oracle_ceiling_over_best", "test_synthetic_pct"))]:
                v = d[c].dropna().to_numpy()
                if len(v) > 1:
                    ci = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                    lines.append(f"- {g} {c}: {v.mean():.4f} ± {v.std(ddof=1):.4f} (95% CI ± {ci:.4f})")
        if "regime" in df:
            for a, b in (("LEAKY", "ORDER"), ("ORDER", "HONEST"), ("LEAKY", "HONEST")):
                x = df[df.regime == a].sort_values("seed").f1.to_numpy()
                y = df[df.regime == b].sort_values("seed").f1.to_numpy()
                if len(x) == len(y) > 5 and not np.allclose(x, y):
                    tests.append((f"{f.stem} F1 {a} vs {b}", wilcoxon(x, y).pvalue))
        if "ensemble_f1" in df:
            best = df[["cnn_f1", "bilstm_f1", "transformer_f1"]].max(axis=1)
            tests.append((f"{f.stem} ensemble F1 vs best branch F1", wilcoxon(df.ensemble_f1, best).pvalue))
        if "shap_jaccard" in df and len(df) > 4:
            tests.append((f"{f.stem} SHAP vs LIME Jaccard", wilcoxon(df.shap_jaccard, df.lime_jaccard).pvalue))
        lines.append("")
    lines.append(f"## Paired Wilcoxon tests, Bonferroni over {len(tests)}\n")
    for name, p in tests:
        lines.append(f"- {name}: p = {p:.3g}, Bonferroni p = {min(1, p * len(tests)):.3g}")
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iotid20", "edge"])
    ap.add_argument("--table", choices=["3", "7", "8", "all"], default="all")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        summarise()
    else:
        if a.table in ("3", "all"):
            table3(a.dataset)
        if a.table in ("7", "all"):
            table7(a.dataset)
        if a.table in ("8", "all"):
            table8(a.dataset)
        summarise()
