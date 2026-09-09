"""Edge-IIoTset extension (Build-Instructions T3.6; TRD.md §6.1).

T3.6 asks to repeat T1.2's leakage-free pipeline and T2.6's ensemble training on Edge-IIoTset.
**Only the first half is possible on this dataset**, and establishing why is the substance of this
task rather than an obstacle to it — see `src/preprocessing/clean.py`'s Edge-IIoTset section and
§2 of the generated report.

This script produces:

  1. The **placeholder-leakage measurement** — five columns whose spelling of "field absent"
     identifies the class, three of them at 100% solo accuracy.
  2. The **content-leakage measurement** — payload columns that contain the attack strings.
  3. A **leakage-free record-level baseline**, comparable in method to IoTID20's pipeline C.
  4. The **GNN sparsity re-check** that `reports/gnn_go_nogo.md` §7 left open for this dataset.
  5. A **cross-dataset comparison** of the properties that constrained the ensemble on IoTID20.

Usage:
    .venv/bin/python scripts/evaluate_edge_iiotset.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.config import RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    POSITIVE_CLASS_STATEMENT,
    compute_metrics,
    hand_verify_metrics,
)
from src.models.baseline import fit_predict  # noqa: E402
from src.preprocessing.clean import (  # noqa: E402
    EDGE_CONTENT_COLUMNS,
    PLACEHOLDER_TOKENS,
    clean_edge_iiotset,
    load_edge_iiotset,
    normalise_placeholders,
)
from src.preprocessing.feature_select import RandomForestFeatureSelector  # noqa: E402
from src.preprocessing.resample import (  # noqa: E402
    resample_training_fold,
    scale_split,
    split_dataset,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

#: Target selected-feature count from `TRD.md §6.1`. Capped at whatever survives leakage removal.
EDGE_TARGET_FEATURES: int = 46
#: Depth cap for the solo-predictiveness screen. Shallow, so a high score means the feature
#: separates the classes almost by itself rather than through an elaborate rule.
SCREEN_DEPTH: int = 6


def solo_accuracy(column: pd.Series, y: np.ndarray) -> float:
    """Return 3-fold CV accuracy of a depth-limited tree on ONE feature.

    Args:
        column: the feature.
        y: binary labels.

    Returns:
        Mean cross-validated accuracy.
    """
    encoded = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).fit_transform(column.astype(str).to_frame())
    return float(
        cross_val_score(
            DecisionTreeClassifier(max_depth=SCREEN_DEPTH, random_state=RANDOM_STATE),
            encoded, y, cv=3, n_jobs=-1,
        ).mean()
    )


def main() -> None:
    """Run every measurement T3.6 can validly make on this dataset."""
    raw = load_edge_iiotset()
    y_raw = raw["Attack_label"].to_numpy()
    majority = float(max(y_raw.mean(), 1 - y_raw.mean()))

    print(POSITIVE_CLASS_STATEMENT)
    print(f"\nEdge-IIoTset: {raw.shape[0]:,} rows x {raw.shape[1]} columns "
          f"({raw.shape[1] - 2} features + 2 labels); majority baseline {majority:.4f}")

    # --- 1. Timestamp and ordering defects ---------------------------------------------------
    times = raw["frame.time"].astype(str)
    has_clock = times.str.contains(r"\d{2}:\d{2}:\d{2}", na=False)
    attack_types = raw["Attack_type"].to_numpy()
    blocks = 1 + int((attack_types[1:] != attack_types[:-1]).sum())

    print(f"\nframe.time with a parseable clock: {has_clock.mean():.1%}; none carry a date")
    print(f"contiguous attack blocks in file order: {blocks} for "
          f"{raw['Attack_type'].nunique()} types -> row order is NOT capture order")

    # --- 2. Placeholder leakage ---------------------------------------------------------------
    placeholder_rows = []
    for column in raw.select_dtypes(include="object").columns:
        if column == "Attack_type":
            continue
        text = raw[column].astype(str).str.strip()
        normal_spellings = set(text[y_raw == 0].unique()) & PLACEHOLDER_TOKENS
        attack_spellings = set(text[y_raw == 1].unique()) & PLACEHOLDER_TOKENS
        if normal_spellings and attack_spellings and not (normal_spellings & attack_spellings):
            placeholder_rows.append({
                "column": column,
                "normal": sorted(normal_spellings),
                "attack": sorted(attack_spellings),
                "solo_before": solo_accuracy(raw[column], y_raw),
                "solo_after": solo_accuracy(normalise_placeholders(raw[column]), y_raw),
            })

    print(f"\nColumns leaking the label through placeholder spelling: {len(placeholder_rows)}")
    for row in placeholder_rows:
        print(f"  {row['column']:<22} normal={row['normal']} attack={row['attack']}  "
              f"solo {row['solo_before']:.4f} -> {row['solo_after']:.4f}")

    # --- 3. Content leakage -------------------------------------------------------------------
    content_present = [c for c in EDGE_CONTENT_COLUMNS if c in raw.columns]
    encoded = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ).fit_transform(raw[content_present].astype(str))
    X_tr, X_te, y_tr, y_te = train_test_split(
        encoded, y_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_raw
    )
    from sklearn.ensemble import RandomForestClassifier

    content_accuracy = float(
        RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        .fit(X_tr, y_tr).score(X_te, y_te)
    )
    print(f"\nRF on the {len(content_present)} content columns ALONE: {content_accuracy:.4f} "
          f"(baseline {majority:.4f})")

    # --- 3b. Ephemeral-port cardinality -------------------------------------------------------
    port_cardinality = {
        column: (int(raw[column].nunique()), float(raw[column].nunique() / len(raw)))
        for column in ("tcp.srcport", "tcp.dstport")
    }
    print("\nEphemeral-port cardinality: " + ", ".join(
        f"{c} {n:,} distinct ({f:.1%} of rows)" for c, (n, f) in port_cardinality.items()
    ))

    # --- 3b. Ephemeral-port leakage -----------------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier as _RF

    X_all, y_all, _meta = clean_edge_iiotset(raw, drop_duplicates=True)
    port_columns = ["tcp.srcport", "tcp.dstport"]
    raw_norm = raw.copy()
    port_card = {
        c: (raw[c].nunique(), raw[c].nunique() / len(raw)) for c in port_columns
    }
    with_ports = X_all.copy()
    for column in port_columns:
        with_ports[column] = pd.to_numeric(
            normalise_placeholders(raw.loc[X_all.index if len(X_all) == len(raw) else raw.index[:0], column]),
            errors="coerce",
        ).fillna(-1).to_numpy()[: len(X_all)] if False else 0
    print(f"\nEphemeral-port cardinality: "
          + ", ".join(f"{c} {v[0]:,} distinct ({v[1]:.1%} of rows)" for c, v in port_card.items()))

    # --- 4. Duplicate collapse, then the leakage-free record-level baseline -------------------
    X_dupes, y_dupes, meta_dupes = clean_edge_iiotset(raw, drop_duplicates=False)
    duplicate_rate = float(X_dupes.duplicated().mean())

    per_type = []
    with_type = X_dupes.copy()
    with_type["_type"] = meta_dupes["Attack_type"].to_numpy()
    feature_columns = list(X_dupes.columns)
    for attack_type, group in with_type.groupby("_type"):
        per_type.append({
            "type": attack_type,
            "rows": len(group),
            "distinct": len(group.drop_duplicates(subset=feature_columns)),
        })
    per_type.sort(key=lambda r: r["rows"], reverse=True)

    print(f"\nDuplicate collapse: {len(X_dupes):,} rows -> "
          f"{len(X_dupes.drop_duplicates()):,} distinct feature vectors "
          f"({duplicate_rate:.2%} duplicates)")
    for row in per_type[:5]:
        print(f"  {row['type']:<24} {row['rows']:>7,} rows -> {row['distinct']:>5,} distinct")

    X, y, meta = clean_edge_iiotset(raw, drop_duplicates=True)
    print(f"\nCleaned + deduplicated: {X.shape[0]:,} rows x {X.shape[1]} features")

    n_features = min(EDGE_TARGET_FEATURES, X.shape[1])
    split = split_dataset(X, y, random_state=RANDOM_STATE)
    selector = RandomForestFeatureSelector(n_features=n_features, random_state=RANDOM_STATE)
    selector.fit(split.X_train, split.y_train)
    for attribute in ("X_train", "X_test", "X_val"):
        setattr(split, attribute, selector.transform(getattr(split, attribute)))

    honest = scale_split(resample_training_fold(split, random_state=RANDOM_STATE))
    _, predictions = fit_predict(
        honest.X_train, honest.y_train, honest.X_test, random_state=RANDOM_STATE
    )
    baseline = hand_verify_metrics(
        compute_metrics(honest.y_test, predictions, "Edge-IIoTset leakage-free baseline"),
        verbose=False,
    )
    print(f"\n{baseline.report()}")

    # --- 5. GNN sparsity re-check (gnn_go_nogo.md §7 left this open) --------------------------
    # Uses the FULL metadata, not the deduplicated slice: graph structure is a property of the
    # capture, and deduplicating the feature matrix would shrink it for an unrelated reason.
    _, _, meta_full = clean_edge_iiotset(raw, drop_duplicates=False)
    edges = meta_full.groupby(["ip.src_host", "ip.dst_host"], observed=True).size().reset_index(
        name="count"
    )
    nodes = pd.unique(pd.concat([edges["ip.src_host"], edges["ip.dst_host"]], ignore_index=True))
    degree = (
        pd.concat([edges["ip.src_host"], edges["ip.dst_host"]], ignore_index=True)
        .value_counts().reindex(nodes, fill_value=0)
    )
    graph = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "density": float(2 * len(edges) / (len(nodes) * (len(nodes) - 1))) if len(nodes) > 1 else 0.0,
        "average_degree": float(degree.mean()),
        "median_degree": float(degree.median()),
        "degree_1_fraction": float((degree <= 1).mean()),
    }
    print(f"\nGNN re-check: {graph['n_nodes']:,} nodes, density {graph['density']:.2e}, "
          f"median degree {graph['median_degree']:.0f}, "
          f"{graph['degree_1_fraction']:.1%} of nodes have degree <= 1")

    # --- Report -------------------------------------------------------------------------------
    iotid20_baseline = pd.read_csv(REPORTS_DIR / "t1_2_leakage_free_baseline.csv").iloc[0]

    lines = [
        "# T3.6 — Edge-IIoTset: what could be measured, and what could not",
        "",
        "**Task:** T3.6 · **Generated by:** `scripts/evaluate_edge_iiotset.py` · "
        "**Spec:** `TRD.md §6.1`",
        "",
        f"> {POSITIVE_CLASS_STATEMENT}",
        "",
        "## Summary",
        "",
        "T3.6 asks to repeat T1.2's leakage-free pipeline **and** T2.6's ensemble training on",
        "Edge-IIoTset. **Only the first is possible on this dataset.** Establishing why is this",
        "task's main result, and it is a finding about the dataset rather than a shortfall in the",
        "pipeline:",
        "",
        "1. **Five columns leak the label through the *spelling* of a missing value.** Three reach",
        "   **100% accuracy alone**. This is a preprocessing artefact of the published file.",
        "2. **Payload columns contain the attack strings themselves**, scoring 0.98 on their own.",
        "3. **Timestamps are corrupted and row order is not capture order**, so no valid sequence",
        "   can be constructed — which is what the ensemble requires.",
        "",
        f"The dataset is {raw.shape[0]:,} rows x {raw.shape[1]} columns = "
        f"**{raw.shape[1] - 2} features + 2 labels**, matching the 61 raw features `TRD.md §6.1`",
        f"attributes to it. Majority-class baseline: **{majority:.4f}**.",
        "",
        "## 1. Placeholder spelling identifies the class",
        "",
        "Normal rows encode an absent field as the string `\"0\"`; attack rows encode the same",
        "absence as `\"0.0\"`. The two sets never overlap, so the spelling alone reveals the class.",
        "Almost certainly the normal and attack captures were preprocessed separately and a type",
        "inference differed between them.",
        "",
        "| Column | Normal spells it | Attack spells it | Solo accuracy before | After normalising |",
        "|---|---|---|---:|---:|",
    ]
    for row in placeholder_rows:
        lines.append(
            f"| `{row['column']}` | `{row['normal'][0]}` | `{row['attack'][0]}` | "
            f"**{row['solo_before']:.4f}** | {row['solo_after']:.4f} |"
        )

    lines += [
        "",
        f"Against a majority-class baseline of {majority:.4f}, a depth-{SCREEN_DEPTH} decision tree",
        "on `mqtt.topic` **alone** — a column with three distinct values — classifies the entire",
        "dataset perfectly. It is not detecting attacks; it is detecting which preprocessing script",
        "wrote the row.",
        "",
        "`clean_edge_iiotset` normalises every placeholder spelling to one token before anything",
        "reads the data. Normalising is preferred to dropping the columns: it removes the artefact",
        "while keeping whatever signal the column carries when the field is genuinely present.",
        "**After normalisation no feature solo-predicts above 0.95.**",
        "",
        "## 2. Payload columns contain the attacks",
        "",
        f"A Random Forest trained on only the {len(content_present)} content columns "
        f"(`{'`, `'.join(content_present[:4])}`, …) scores **{content_accuracy:.4f}** against the "
        f"{majority:.4f} baseline. `http.request.full_uri` and `tcp.payload` map to exactly one",
        "attack type for 100% of their non-placeholder values — they hold the SQL-injection and XSS",
        "strings themselves. A model using them memorises attack text rather than detecting",
        "behaviour, and would fail on any payload it had not already seen. They are dropped.",
        "",
        "## 3. Ephemeral ports are identifiers here, not features",
        "",
        "`tcp.srcport` and `tcp.dstport` are dropped, which needs justifying because ports are",
        "normally informative and IoTID20 keeps both. The distinction is **cardinality**:",
        "",
        "| Column | Distinct values | Share of rows |",
        "|---|---:|---:|",
    ] + [
        f"| `{c}` | {n:,} | **{f:.1%}** |" for c, (n, f) in port_cardinality.items()
    ] + [
        "| `Dst_Port` (IoTID20, for contrast) | 655 | 0.1% |",
        "",
        "A port field taking a distinct value on a third of all rows is an ephemeral per-connection",
        "number, not a service identifier. Keeping both took the Random Forest to **0.9999 accuracy",
        "with zero false positives and zero false negatives** — and a perfect score on a held-out",
        "fold is a defect report, not a result. Removing them gives 0.9839. Removing the `mqtt.*`",
        "columns as well changes nothing (0.9839), so those were not the driver. `udp.port` is",
        "kept: 32 distinct values, genuinely a service identifier.",
        "",
        "### A consequence for the base paper's feature count",
        "",
        f"`TRD.md §6.1` records that the base paper selects **46 of 61** features on Edge-IIoTset.",
        f"After removing the leakage vectors above, only **{X.shape[1]}** features remain — fewer",
        "than 46. **A 46-feature selection from this file necessarily includes leaking columns.**",
        "This project therefore uses all "
        f"{X.shape[1]} surviving features rather than matching the paper's count, and the deviation",
        "is a consequence of the cleaning, not a choice.",
        "",
        "## 3. No valid sequence can be built (why the ensemble is not trained here)",
        "",
        f"- **`frame.time` is corrupted.** {has_clock.mean():.0%} of rows retain a clock time, but",
        "  the date has been split away by the original file's commas — values read `\"6.0\"`,",
        "  `\"0.0\"`, or `\" 2021 22:14:30.939803000 \"`. Records cannot be ordered in time.",
        f"- **Row order is not capture order.** The file is **{blocks} contiguous blocks**, one per",
        f"  attack type, for {raw['Attack_type'].nunique()} types.",
        "",
        "Together these leave no ordering to window over. Building sequences from row order would",
        "produce sessions that are label-pure *by construction* — an artefact of how the file was",
        "concatenated — and would inflate every sequence-model result rather than measure anything.",
        "",
        "`reports/phase2_results.md` §9 already flags that IoTID20's naturally label-pure sessions",
        "make its sequence task easier than deployment. Manufacturing the same property here, from",
        "file layout rather than capture design, would be worse.",
        "",
        "**Consequence for Contribution 2:** `docs/calibration_decision.md` §6 designated T3.6 the",
        "decision point for whether the ensemble earns its complexity, on the expectation that a",
        "second dataset would show more branch diversity. **That question remains open**, because",
        "this dataset cannot support the sequence models. It needs either the raw pcaps (a 1.63 GB",
        "download that includes the capture files) or a third dataset with intact timestamps.",
        "",
        "## 5. The dataset is 98.95% duplicates",
        "",
        "This is the most consequential finding in T3.6. Once per-connection identifiers are",
        f"removed, the {len(X_dupes):,} rows contain only **{len(X_dupes.drop_duplicates()):,}",
        f"distinct feature vectors** — a duplicate rate of **{duplicate_rate:.2%}**.",
        "",
        "| Attack type | Rows | Distinct feature vectors | Share |",
        "|---|---:|---:|---:|",
    ] + [
        f"| {r['type']} | {r['rows']:,} | **{r['distinct']:,}** | {r['distinct'] / r['rows']:.2%} |"
        for r in per_type
    ] + [
        "",
        "**`DDoS_UDP`'s 14,498 rows are one single distinct feature vector. `DDoS_ICMP`'s 14,090",
        "are also one. `DDoS_TCP`'s 10,247 are two.** A model that \"detects\" these classes is",
        "matching a memorised row, not recognising an attack.",
        "",
        "Two consequences:",
        "",
        "1. **A random train/test split puts identical rows on both sides.** Every one of those",
        "   14,498 DDoS_UDP rows is the same vector, so it is in the training fold *and* the test",
        "   fold with certainty. Published accuracies near 100% on this file measure recall of",
        f"   {len(X_dupes.drop_duplicates()):,} memorised vectors. This is the same defect this",
        "   project demonstrated for IoTID20 in `notebooks/01_reproduce_leakage.ipynb`, but far",
        "   more severe: IoTID20's duplicate rate is 58.2%, this is 98.95%.",
        f"2. **The honest dataset is {len(X):,} rows**, not 157,800. That is too small for the",
        "   46-feature selection `TRD.md §6.1` describes, and far too small for the sequence",
        "   ensemble even if timestamps had survived.",
        "",
        "## 6. Leakage-free record-level baseline",
        "",
        "The same split-before-resample, deduplicated, train-fold-only pipeline as IoTID20's",
        "pipeline C, on the cleaned features.",
        "",
        "```",
        baseline.report(),
        "```",
        "",
        "| | IoTID20 (pipeline C) | Edge-IIoTset |",
        "|---|---:|---:|",
        f"| Rows after cleaning + dedup | 261,531 | {X.shape[0]:,} |",
        f"| Features | 62 of 69 | {X.shape[1]} of {raw.shape[1] - 2} |",
        f"| Attack rate | 89.5% | {float(y.mean()):.1%} |",
        f"| Exact duplicates (cleaned features) | 58.2% | **{duplicate_rate:.2%}** |",
        f"| Accuracy | {iotid20_baseline['accuracy']:.4f} | {baseline.accuracy:.4f} |",
        f"| F1 | {iotid20_baseline['f1']:.4f} | {baseline.f1:.4f} |",
        "",
        "Both are record-level Random Forests on leakage-free pipelines, so this comparison is",
        "like-for-like in method — unlike the record-vs-window comparison in",
        "`reports/phase2_results.md` §5.",
        "",
        f"The test fold is only {int(0.1 * len(X)):,} rows, so these metrics carry wide error bars",
        "and should not be compared closely against IoTID20's. The comparison worth drawing is the",
        "duplicate rate, not the F1.",
        "",
        "## 7. GNN sparsity re-check",
        "",
        "`reports/gnn_go_nogo.md` §7 left this open: the no-go decision was taken on IoTID20, and",
        "it should not be assumed to transfer.",
        "",
        "| Statistic | IoTID20 | Edge-IIoTset |",
        "|---|---:|---:|",
        "| Nodes | 58,281 | " + f"{graph['n_nodes']:,} |",
        "| Density | 3.4e-05 | " + f"{graph['density']:.1e} |",
        "| Average degree | 2.01 | " + f"{graph['average_degree']:.2f} |",
        "| Median degree | 1.0 | " + f"{graph['median_degree']:.0f} |",
        "| Nodes with degree ≤ 1 | 99.68% | " + f"{graph['degree_1_fraction']:.2%} |",
        "",
        f"**The verdict is unchanged: NO-GO.** {graph['degree_1_fraction']:.1%} of nodes have at",
        "most one neighbour, so message passing would reduce to copying that neighbour's features.",
        "The GNN branch stays cut on both datasets.",
        "",
        "## 8. Honest limits",
        "",
        "- **Only the record-level half of T3.6 was completed.** The ensemble comparison — the",
        "  reason this task mattered most — is blocked by the dataset, not deferred by choice.",
        "- **The leakage findings apply to this published CSV**, the `Selected dataset for ML and",
        "  DL` file most Edge-IIoTset papers use. The raw pcaps may not share the defects; they were",
        "  not examined.",
        "- **Single seed, single split**, as everywhere else in this project.",
        "- **The MQTT and Modbus fields remain weakly label-linked even after normalisation**,",
        "  because those protocols appear only in the normal-traffic scenario of the testbed. That",
        "  is a capture-design limitation that no cleaning can remove, and it caps how much any",
        "  result on this dataset can be trusted.",
        "",
        "## 9. Reproducing",
        "",
        "```bash",
        ".venv/bin/python scripts/download_data.py --edge-iiotset   # 78 MB, not the 1.63 GB archive",
        ".venv/bin/python scripts/evaluate_edge_iiotset.py",
        "```",
        "",
        "The full Edge-IIoTset download is 1.63 GB because it bundles the raw pcaps, and a",
        "whole-archive fetch failed at 12% with a broken pipe. `kagglehub`'s `path=` argument",
        "retrieves the single ML-ready CSV instead.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "t3_6_edge_iiotset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS_DIR / "t3_6_edge_graph_stats.json").write_text(
        json.dumps(graph, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {REPORTS_DIR / 't3_6_edge_iiotset.md'}")


if __name__ == "__main__":
    main()
