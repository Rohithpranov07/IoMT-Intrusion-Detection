"""Phase 7 (record-level part, executed here): multi-partition detection, leakage regimes,
statistical tests, and held-out-attack generalisation.

Datasets reachable from this machine: Edge-IIoTset (ML file, project's 30-feature modelling space,
reproduced from clean.py) and WUSTL-EHMS-2020 (network + biometric features, identifiers removed).
IoTID20 is not reachable here; its multi-partition run is in notebooks/R3_multiseed_colab.py.

Regimes (Attack = positive class throughout):
  LEAKY   SMOTE on the whole dataset, then split (the order audited in HIDS-IoMT and Hosain 2025)
  ORDER   split, then SMOTE on the training fold only; duplicates kept
  HONEST  exact duplicates removed in the modelling space, then split, then SMOTE on training only
Each regime is run on 20 stratified 80/20 partitions (seeds 1..20) with a Random Forest
(100 trees, seeded). Paired Wilcoxon signed-rank tests compare regimes, Bonferroni-corrected.

Generalisation (Phase 7.2 substitute, see paper): leave-one-attack-type-out on the HONEST data,
reporting recall on an attack type never seen in training.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_dataset_leakage import edge_model_space  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "phase7")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(1, 21))


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn), accuracy=(tp + tn) / (tp + tn + fp + fn),
                precision=prec, recall=rec, f1=f1, fpr=fp / (fp + tn) if fp + tn else 0.0)


def rf(seed):
    return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)


def run_regime(X, y, regime, seed):
    X = X.to_numpy(dtype=float)
    y = y.to_numpy()
    if regime == "LEAKY":
        Xr, yr = SMOTE(random_state=seed).fit_resample(X, y)
        synth = np.zeros(len(yr), bool)
        synth[len(y):] = True  # imblearn appends synthetic rows after the originals
        idx = np.arange(len(yr))
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=yr)
        m = rf(seed).fit(Xr[tr], yr[tr])
        out = metrics(yr[te], m.predict(Xr[te]))
        out["test_synthetic_pct"] = 100.0 * synth[te].mean()
        return out
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    Xtr, ytr = SMOTE(random_state=seed).fit_resample(X[tr], y[tr])
    m = rf(seed).fit(Xtr, ytr)
    out = metrics(y[te], m.predict(X[te]))
    out["test_synthetic_pct"] = 0.0
    return out


def load_sets():
    edge = pd.read_csv(os.path.join(ROOT, "data", "edge_iiotset", "ML-EdgeIIoT-dataset.csv"), low_memory=False)
    Xe = edge_model_space(edge)
    ye = edge.loc[Xe.index, "Attack_label"].astype(int)
    te = edge.loc[Xe.index, "Attack_type"]
    w = pd.read_csv(os.path.join(ROOT, "data", "wustl_ehms_2020", "wustl-ehms-2020.csv"), low_memory=False)
    drop = ["Label", "Attack Category", "SrcAddr", "DstAddr", "Sport", "Dport", "SrcMac", "DstMac", "Dir", "Flgs"]
    Xw = w.drop(columns=[c for c in drop if c in w.columns]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    yw = w["Label"].astype(int)
    tw = w["Attack Category"]
    return {"Edge-IIoTset": (Xe.reset_index(drop=True), ye.reset_index(drop=True), te.reset_index(drop=True)),
            "WUSTL-EHMS-2020": (Xw, yw, tw)}


def dedup(X, y, t):
    key = pd.util.hash_pandas_object(X, index=False)
    keep = ~key.duplicated()
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), t[keep].reset_index(drop=True)


def main():
    sets = load_sets()
    rows = []
    for name, (X, y, t) in sets.items():
        Xd, yd, td = dedup(X, y, t)
        for regime, seed in itertools.product(("LEAKY", "ORDER", "HONEST"), SEEDS):
            XX, yy = (Xd, yd) if regime == "HONEST" else (X, y)
            r = run_regime(XX, yy, regime, seed)
            r.update(dataset=name, regime=regime, seed=seed, rows=len(yy))
            rows.append(r)
            print(json.dumps({k: r[k] for k in ("dataset", "regime", "seed", "f1", "precision", "recall", "test_synthetic_pct")}), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "multiseed_runs.csv"), index=False)

    # summary and paired tests
    summ, tests = [], []
    for (name, regime), g in df.groupby(["dataset", "regime"]):
        s = {"dataset": name, "regime": regime, "n_partitions": len(g), "rows": int(g.rows.iloc[0])}
        for m in ("accuracy", "precision", "recall", "f1", "fpr", "test_synthetic_pct"):
            v = g[m].to_numpy()
            half = 2.093 * v.std(ddof=1) / np.sqrt(len(v))  # t(0.975, 19)
            s[f"{m}_mean"], s[f"{m}_std"], s[f"{m}_ci95"] = v.mean(), v.std(ddof=1), half
        summ.append(s)
    pairs = [("LEAKY", "ORDER"), ("ORDER", "HONEST"), ("LEAKY", "HONEST")]
    n_tests = len(pairs) * len(sets) * 2
    for name in sets:
        for a, b in pairs:
            for m in ("f1", "precision"):
                va = df[(df.dataset == name) & (df.regime == a)].sort_values("seed")[m].to_numpy()
                vb = df[(df.dataset == name) & (df.regime == b)].sort_values("seed")[m].to_numpy()
                if np.allclose(va, vb):
                    p = 1.0
                else:
                    p = wilcoxon(va, vb).pvalue
                tests.append({"dataset": name, "comparison": f"{a} vs {b}", "metric": m,
                              "median_diff": float(np.median(va - vb)), "p_raw": p,
                              "p_bonferroni": min(1.0, p * n_tests), "n_tests_corrected": n_tests})
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "multiseed_summary.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(OUT, "multiseed_wilcoxon.csv"), index=False)

    # leave-one-attack-type-out on HONEST data
    lo = []
    for name, (X, y, t) in sets.items():
        Xd, yd, td = dedup(X, y, t)
        types = [x for x in td.unique() if str(x).lower() != "normal"]
        normal_idx = np.flatnonzero(yd.to_numpy() == 0)
        for held in types:
            for seed in (1, 2, 3, 4, 5):
                rng = np.random.default_rng(seed)
                n_te_norm = max(1, int(0.2 * len(normal_idx)))
                te_norm = rng.choice(normal_idx, n_te_norm, replace=False)
                held_idx = np.flatnonzero(td.to_numpy() == held)
                train = np.setdiff1d(np.arange(len(yd)), np.concatenate([te_norm, held_idx]))
                if yd.to_numpy()[train].min() == yd.to_numpy()[train].max():
                    continue
                Xtr, ytr = Xd.to_numpy(float)[train], yd.to_numpy()[train]
                k = min(5, int((ytr == 0).sum()) - 1, int((ytr == 1).sum()) - 1)
                if k >= 1:
                    Xtr, ytr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xtr, ytr)
                m = rf(seed).fit(Xtr, ytr)
                p_held = m.predict(Xd.to_numpy(float)[held_idx])
                p_norm = m.predict(Xd.to_numpy(float)[te_norm])
                lo.append({"dataset": name, "held_out_attack": held, "seed": seed, "held_rows": len(held_idx),
                           "recall_on_unseen_attack": float(p_held.mean()), "fpr_on_normal": float(p_norm.mean())})
    lod = pd.DataFrame(lo)
    lod.to_csv(os.path.join(OUT, "leave_one_attack_out_runs.csv"), index=False)
    lod.groupby(["dataset", "held_out_attack"]).agg(held_rows=("held_rows", "first"),
                                                    recall_mean=("recall_on_unseen_attack", "mean"),
                                                    recall_std=("recall_on_unseen_attack", "std"),
                                                    fpr_mean=("fpr_on_normal", "mean")).reset_index().to_csv(
        os.path.join(OUT, "leave_one_attack_out_summary.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
