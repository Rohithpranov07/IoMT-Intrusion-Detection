"""Branch and fusion-rule ablation (Build-Instructions T4.4; TRD.md §2.3).

T4.4 asks for each ensemble branch ablated against the leakage-free baseline. This computes every
combination from the saved branch probabilities — no retraining, so the ablated models are exactly
the ones `reports/phase2_results.md` evaluated rather than fresh runs that would differ by seed.

The GNN branch is absent because T2.7's sparsity check returned **no-go**
(`reports/gnn_go_nogo.md`): 99.68% of nodes in IoTID20's IP-pair graph have degree <= 1. Three
branches are ablated, not four, and `Build-Instructions.md` §A.1 rule 4 forbids it reappearing.

Every metric states **Attack as the positive class** (`TRD.md §2.3`) and is hand-verified against
its confusion matrix.

Usage:
    .venv/bin/python scripts/ablation_study.py
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ARTIFACTS_DIR, REPORTS_DIR  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    POSITIVE_CLASS_STATEMENT,
    compute_metrics,
    hand_verify_metrics,
)
from src.models.fusion import (  # noqa: E402
    CONFIDENCE_SHARPNESS,
    confidence_weighted_fusion,
    majority_vote_fusion,
    simple_average_fusion,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

#: Gamma values swept, to show whether sharpening the confidence weighting recovers anything.
GAMMA_SWEEP: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def main() -> None:
    """Run every ablation and write the report."""
    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    meta = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    branch_names = meta["branch_names"]

    y_test = data["y_test"]
    probabilities = {n: data[f"probabilities_{n}"] for n in branch_names}

    rows: list[dict[str, object]] = []

    def record(label: str, predictions: np.ndarray, group: str) -> None:
        """Compute, hand-verify, and record one configuration."""
        result = hand_verify_metrics(
            compute_metrics(y_test, predictions, label), verbose=False
        )
        rows.append({
            "group": group, "model": label,
            "accuracy": result.accuracy, "precision": result.precision,
            "recall": result.recall, "f1": result.f1,
            "TP": result.tp, "FP": result.fp, "TN": result.tn, "FN": result.fn,
        })

    # 1. Each branch alone.
    for name in branch_names:
        record(f"{name} alone", probabilities[name].argmax(axis=1), "single branch")

    # 2. Every pair, and the full three, under the deployed fusion rule.
    for size in (2, 3):
        for subset in itertools.combinations(branch_names, size):
            fused = confidence_weighted_fusion([probabilities[n] for n in subset], list(subset))
            label = " + ".join(subset) + (" (FULL ENSEMBLE)" if size == 3 else "")
            record(label, fused.predictions, f"{size}-branch fusion")

    # 3. Fusion rules TRD.md §3.3 excluded, on all three branches.
    all_probabilities = [probabilities[n] for n in branch_names]
    record(
        "simple average (excluded by TRD §3.3)",
        np.argmax(simple_average_fusion(all_probabilities), axis=1),
        "fusion rule",
    )
    record(
        "majority vote (excluded by TRD §3.3)",
        majority_vote_fusion(all_probabilities),
        "fusion rule",
    )

    # 4. Gamma sweep on the deployed rule.
    gamma_rows = []
    for gamma in GAMMA_SWEEP:
        fused = confidence_weighted_fusion(all_probabilities, branch_names, gamma=gamma)
        result = hand_verify_metrics(
            compute_metrics(y_test, fused.predictions, f"gamma={gamma}"), verbose=False
        )
        gamma_rows.append({"gamma": gamma, "f1": result.f1, "accuracy": result.accuracy})

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "t4_4_ablation.csv", index=False)

    best_single = table[table.group == "single branch"].sort_values("f1").iloc[-1]
    full = table[table.model.str.contains("FULL ENSEMBLE")].iloc[0]
    best_overall = table.sort_values("f1").iloc[-1]

    # 5. Why fusion has so little room: branch agreement and the ceiling on any fusion rule.
    predictions = np.stack([probabilities[n].argmax(axis=1) for n in branch_names])
    unanimous = (predictions == predictions[0]).all(axis=0)
    contested = ~unanimous
    oracle = np.where(unanimous, predictions[0], y_test)
    single_best_accuracy = max(float((predictions[i] == y_test).mean()) for i in range(len(branch_names)))
    fused_full = confidence_weighted_fusion(all_probabilities, branch_names)

    print(POSITIVE_CLASS_STATEMENT)
    print(f"\n{table.to_string(index=False)}")
    print(f"\nbest single branch : {best_single.model} F1={best_single.f1:.4f}")
    print(f"full ensemble      : F1={full.f1:.4f}")

    baseline_path = REPORTS_DIR / "t1_2_leakage_free_baseline.csv"
    baseline = pd.read_csv(baseline_path).iloc[0] if baseline_path.exists() else None

    lines = [
        "# T4.4 — Ablation study",
        "",
        "**Task:** T4.4 · **Generated by:** `scripts/ablation_study.py` · "
        "**Model:** the trained ensemble from T2.6",
        "",
        f"> {POSITIVE_CLASS_STATEMENT}",
        "",
        "Every configuration below is computed from the **saved branch probabilities**, so each",
        "ablated model is exactly the one `reports/phase2_results.md` evaluated rather than a fresh",
        "run that would differ by seed. Every metric is hand-verified against its confusion matrix.",
        "",
        "**Three branches are ablated, not four.** T2.7's sparsity check returned **no-go**",
        "(`reports/gnn_go_nogo.md`): 99.68% of nodes in IoTID20's IP-pair graph have degree ≤ 1, so",
        "the GNN branch was cut and never built.",
        "",
        "## Headline",
        "",
        f"**The best single branch beats the full ensemble.** `{best_single.model}` scores "
        f"**F1 {best_single.f1:.4f}** against the three-branch fusion's **{full.f1:.4f}**.",
        "",
        "That is the honest result on IoTID20 and it is stated first rather than buried. §3 explains",
        "why, and `docs/calibration_decision.md` records the two candidate fixes that were measured",
        "and both failed.",
        "",
        "## 1. Every configuration",
        "",
        "| Group | Configuration | Accuracy | Precision | Recall | F1 | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if baseline is not None:
        lines.append(
            f"| baseline | T1.2 leakage-free Random Forest *(records, not windows)* | "
            f"{baseline['accuracy']:.4f} | {baseline['precision']:.4f} | "
            f"{baseline['recall']:.4f} | {baseline['f1']:.4f} | {int(baseline['FP'])} | "
            f"{int(baseline['FN'])} |"
        )
    for row in rows:
        emphasis = "**" if "FULL ENSEMBLE" in row["model"] else ""
        lines.append(
            f"| {row['group']} | {emphasis}{row['model']}{emphasis} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {emphasis}{row['f1']:.4f}{emphasis} | "
            f"{row['FP']} | {row['FN']} |"
        )

    lines += [
        "",
        "The baseline row is **not directly comparable** and must never be quoted as though it",
        "were: it classifies individual flow *records*, while every other row classifies 10-record",
        "*windows*, on differently-drawn folds and with different imbalance handling",
        "(`reports/phase2_results.md` §5). It is included because T4.4 asks for the comparison, not",
        "because the numbers sit on the same scale.",
        "",
        "## 2. Fusion rules",
        "",
        "`TRD.md §3.3` requires confidence weighting specifically over a simple average or a",
        "majority vote. On this data all three land within a thousandth of each other:",
        "",
        "| Rule | F1 |",
        "|---|---:|",
    ]
    for row in rows:
        if row["group"] in ("fusion rule",) or "FULL ENSEMBLE" in row["model"]:
            lines.append(f"| {row['model']} | {row['f1']:.4f} |")

    lines += [
        "",
        "### Gamma sweep",
        "",
        "| gamma | F1 | Accuracy |",
        "|---:|---:|---:|",
    ]
    for row in gamma_rows:
        lines.append(f"| {row['gamma']:g} | {row['f1']:.4f} | {row['accuracy']:.4f} |")

    lines += [
        "",
        f"Sharpening the weighting does not close the gap to the best single branch "
        f"({best_single.f1:.4f}): the best gamma reaches "
        f"{max(r['f1'] for r in gamma_rows):.4f}.",
        "",
        "## 3. Why fusion has so little to work with",
        "",
        "| Measurement | Value |",
        "|---|---:|",
        f"| Test windows | {len(y_test):,} |",
        f"| All three branches agree | **{unanimous.mean():.1%}** |",
        f"| Contested windows | {int(contested.sum()):,} ({contested.mean():.1%}) |",
        f"| Best branch's accuracy on contested windows | "
        f"{float((predictions[int(np.argmax([float((predictions[i] == y_test).mean()) for i in range(len(branch_names))]))][contested] == y_test[contested]).mean()):.1%} |",
        f"| Fusion's accuracy on contested windows | "
        f"{float((fused_full.predictions[contested] == y_test[contested]).mean()):.1%} |",
        f"| Oracle accuracy (perfect on every disagreement) | {float((oracle == y_test).mean()):.4f} |",
        f"| Best single branch accuracy | {single_best_accuracy:.4f} |",
        f"| **Ceiling on ANY fusion rule** | "
        f"**{float((oracle == y_test).mean()) - single_best_accuracy:+.4f}** |",
        "",
        "Two things follow, and they are the whole explanation:",
        "",
        f"1. **The branches agree on {unanimous.mean():.1%} of windows**, so fusion can only change",
        f"   the outcome on {contested.mean():.1%} of them. Even an oracle resolving every",
        f"   disagreement perfectly would gain just "
        f"{float((oracle == y_test).mean()) - single_best_accuracy:.4f} accuracy over the best branch.",
        "2. **On the contested windows, fusion is worse than its best branch** — the two weaker",
        "   branches jointly outvote the stronger one, and confidence weighting cannot correct it",
        "   because they are *confidently wrong* there rather than miscalibrated.",
        "",
        "Both candidate fixes were measured and both failed: per-branch temperature calibration",
        "(the branches were already well calibrated, ECE 0.012–0.023) and fitting the fusion's",
        "`BRANCH_PRIORS` (+0.0017 validation F1, −0.0003 on test). See",
        "`docs/calibration_decision.md`.",
        "",
        "## 4. What each branch contributes",
        "",
        "| Removed branch | Remaining pair's F1 | Change vs full ensemble |",
        "|---|---:|---:|",
    ]
    full_f1 = float(full.f1)
    for row in rows:
        if row["group"] == "2-branch fusion":
            present = {n for n in branch_names if n in row["model"]}
            removed = (set(branch_names) - present).pop()
            lines.append(
                f"| {removed} | {row['f1']:.4f} | {float(row['f1']) - full_f1:+.4f} |"
            )

    lines += [
        "",
        "A **positive** change means the ensemble improved when that branch was removed — the",
        "branch was making the ensemble worse, not contributing to it.",
        "",
        "## 5. Honest limits",
        "",
        "- **One dataset, one seed, one split.** Every number is IoTID20 at seed 42. No repeated",
        "  runs and no significance testing across seeds, so small differences here should not be",
        "  read as real. `PRD.md §2.2` criticises the senior's prior work for exactly that omission,",
        "  and this study inherits the same limitation.",
        "- **IoTID20's sessions are 100% label-pure** (`reports/phase2_results.md` §9), which makes",
        "  the task easier than deployment and leaves the branches little to disagree about. That is",
        "  the most likely reason fusion has no room here, and it is a property of the dataset rather",
        "  than of the method.",
        "- **Edge-IIoTset (T3.6) has not been run.** It is the check on whether this conclusion",
        "  generalises, and until it does, 'the ensemble does not earn its complexity' is a claim",
        "  about IoTID20 only.",
        "- **No hyperparameter search was performed.** The branches use the first justified choices",
        "  from `docs/architecture_decision.md`. A tuned CNN or BiLSTM might contribute more —",
        "  though tuning until fusion wins would be fitting the conclusion.",
        "",
        "## 6. Recommendation",
        "",
        "On the evidence available, **the Transformer branch alone is the stronger deployment**",
        "**candidate for IoTID20**: higher F1, a third of the ensemble's inference cost, and one",
        "model to explain rather than three recombined. The ensemble should not be presented as an",
        "improvement on this dataset.",
        "",
        "This does **not** mean Contribution 2 failed. `PRD.md §3` defines its success as a fully",
        "specified architecture trained on genuine sequences with confidence-weighted fusion,",
        "evaluated honestly against the leakage-free baseline — all of which was delivered, and the",
        "specification is the direct fix for Objection #2. Whether the ensemble *earns its",
        "complexity* is a separate question that this study answers, for this dataset, as no.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "ablation_study.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
