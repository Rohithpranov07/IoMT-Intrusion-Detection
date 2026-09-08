"""Measure what the adaptive threshold actually buys (Build-Instructions T3.4; PRD.md §3).

T3.4's VERIFY block only requires that two criticality levels produce two different thresholds.
That is necessary but nowhere near sufficient: a scheme could pass it and still make detection
worse. `PRD.md §3` sets the real bar for Contribution 4 — **false positives reduced by >30%** — and
this script measures it on the held-out test fold.

>>> THE CRITICALITY ASSIGNMENT IS SYNTHETIC. READ THIS BEFORE QUOTING ANY NUMBER BELOW. <<<
--------------------------------------------------------------------------------------------
IoTID20 is a general IoT capture. It contains **no medical devices and no criticality labels**, so
the criticality of each window has to be assigned by this script. That makes every number here a
function of an assumption this project invented.

It also makes the headline number trivially gameable: assign most devices to NON_CLINICAL and the
false-positive count falls as far as you like. Quoting a single figure from one favourable mix
would be exactly the kind of convenient measurement this project criticises the base paper for.

So the script sweeps **several criticality mixes** and reports the whole range, and it reports the
recall cost alongside every false-positive gain. Adaptive thresholding creates no information; it
**reallocates errors** between device classes. Any claim that it is a free win is false.

Usage:
    .venv/bin/python scripts/evaluate_adaptive_threshold.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.adaptive.threshold import (  # noqa: E402
    DETECTION_BASE,
    Criticality,
    apply_threshold,
    threshold_table,
)
from src.config import ARTIFACTS_DIR, POSITIVE_LABEL, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    POSITIVE_CLASS_STATEMENT,
    compute_metrics,
    hand_verify_metrics,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

#: Criticality mixes swept, as (name, description, proportions per class in Criticality order).
#: None is presented as "the" mix -- IoTID20 cannot tell us which is realistic.
MIXES: list[tuple[str, str, tuple[float, float, float, float]]] = [
    ("all-standard", "every device treated alike (the base paper's implicit assumption)",
     (0.00, 0.00, 1.00, 0.00)),
    ("critical-heavy", "an ICU segment: mostly life-sustaining and monitoring devices",
     (0.40, 0.35, 0.20, 0.05)),
    ("balanced", "a mixed ward: clinical and non-clinical devices in similar numbers",
     (0.15, 0.20, 0.35, 0.30)),
    ("periphery-heavy", "a hospital's general network: mostly non-clinical endpoints",
     (0.05, 0.10, 0.25, 0.60)),
]

#: Fog-node load assumed. Neutral, so the reported effect is criticality's alone.
LOAD: float = 0.5


def main() -> None:
    """Sweep criticality mixes and write the T3.4 evaluation report."""
    if not (ARTIFACTS_DIR / "ensemble_artifacts.npz").exists():
        raise SystemExit("artifacts/ not found. Run notebooks/03_train_ensemble_iotid20.ipynb.")

    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    meta = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    branch_names = meta["branch_names"]

    y_test = data["y_test"]
    # Fused P(Attack) is column 1 of the fused probabilities; recompute from the saved branch
    # probabilities rather than re-running the models.
    from src.models.fusion import confidence_weighted_fusion

    fusion = confidence_weighted_fusion(
        [data[f"probabilities_{name}"] for name in branch_names], branch_names
    )
    attack_probability = fusion.probabilities[:, POSITIVE_LABEL]

    # Baseline: the flat 0.5 boundary applied to every device, as the base paper does.
    flat_predictions = (attack_probability >= DETECTION_BASE).astype(np.int8)
    flat_metrics = hand_verify_metrics(
        compute_metrics(y_test, flat_predictions, "FLAT threshold (0.50 for every device)"),
        verbose=False,
    )
    print(POSITIVE_CLASS_STATEMENT)
    print("\n" + flat_metrics.report())

    rng = np.random.default_rng(RANDOM_STATE)
    classes = list(Criticality)
    results = []

    for name, description, proportions in MIXES:
        assignment = rng.choice(len(classes), size=len(y_test), p=proportions)
        criticalities = [classes[i] for i in assignment]

        adaptive_predictions = apply_threshold(
            attack_probability, criticalities, loads=LOAD
        )
        adaptive_metrics = hand_verify_metrics(
            compute_metrics(y_test, adaptive_predictions, f"ADAPTIVE ({name})"), verbose=False
        )

        fp_change = (adaptive_metrics.fp - flat_metrics.fp) / max(flat_metrics.fp, 1)
        fn_change = (adaptive_metrics.fn - flat_metrics.fn) / max(flat_metrics.fn, 1)

        # The reallocation: recall on life-critical devices specifically.
        critical_mask = np.array([c is Criticality.LIFE_CRITICAL for c in criticalities])
        attacks_on_critical = critical_mask & (y_test == POSITIVE_LABEL)
        critical_recall_flat = (
            float(flat_predictions[attacks_on_critical].mean())
            if attacks_on_critical.any() else float("nan")
        )
        critical_recall_adaptive = (
            float(adaptive_predictions[attacks_on_critical].mean())
            if attacks_on_critical.any() else float("nan")
        )

        results.append({
            "name": name, "description": description, "metrics": adaptive_metrics,
            "fp_change": fp_change, "fn_change": fn_change,
            "critical_recall_flat": critical_recall_flat,
            "critical_recall_adaptive": critical_recall_adaptive,
            "n_critical_attacks": int(attacks_on_critical.sum()),
        })
        print(f"\n{name:<16} FP {flat_metrics.fp:>4} -> {adaptive_metrics.fp:<4} "
              f"({fp_change:+.1%})   FN {flat_metrics.fn:>4} -> {adaptive_metrics.fn:<4} "
              f"({fn_change:+.1%})   recall on life-critical "
              f"{critical_recall_flat:.3f} -> {critical_recall_adaptive:.3f}")

    best = min(results, key=lambda r: r["fp_change"])
    target_met = [r for r in results if r["fp_change"] <= -0.30]

    lines = [
        "# T3.4 — Does the adaptive threshold actually help?",
        "",
        "**Task:** T3.4 · **Generated by:** `scripts/evaluate_adaptive_threshold.py` · "
        "**Model:** the trained ensemble from T2.6",
        "",
        f"> {POSITIVE_CLASS_STATEMENT}",
        "",
        "## Read this first: the criticality assignment is synthetic",
        "",
        "IoTID20 is a general IoT capture. It contains **no medical devices and no criticality**",
        "**labels**, so every window's criticality had to be assigned by this script. Every number",
        "below is therefore a function of an assumption this project invented, not a measurement of",
        "a real hospital.",
        "",
        "That also makes the headline figure trivially gameable — assign most devices to",
        "NON_CLINICAL and false positives fall as far as you like. Rather than pick one favourable",
        "mix, the sweep below reports **four**, and pairs every false-positive gain with its recall",
        "cost. Adaptive thresholding creates no information; it **reallocates errors** between",
        "device classes.",
        "",
        "## Thresholds applied",
        "",
        "```",
        threshold_table(LOAD),
        "```",
        "",
        "## Baseline: one flat threshold for every device",
        "",
        "```",
        flat_metrics.report(),
        "```",
        "",
        "## Sweep over criticality mixes",
        "",
        "| Mix | What it represents | FP | Δ FP | FN | Δ FN | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| *flat baseline* | every device alike | {flat_metrics.fp} | — | {flat_metrics.fn} | "
        f"— | {flat_metrics.precision:.4f} | {flat_metrics.recall:.4f} |",
    ]
    for result in results:
        m = result["metrics"]
        lines.append(
            f"| **{result['name']}** | {result['description']} | {m.fp} | "
            f"**{result['fp_change']:+.1%}** | {m.fn} | {result['fn_change']:+.1%} | "
            f"{m.precision:.4f} | {m.recall:.4f} |"
        )

    lines += [
        "",
        "## The reallocation, made explicit",
        "",
        "The table above is not the whole story, because a fall in total false positives can hide a",
        "fall in detection where it matters most. This is the number that shows whether the scheme",
        "does what it is *for*: detection on life-sustaining devices.",
        "",
        "| Mix | Attacks on life-critical devices | Recall (flat) | Recall (adaptive) |",
        "|---|---:|---:|---:|",
    ]
    for result in results:
        if result["n_critical_attacks"]:
            lines.append(
                f"| {result['name']} | {result['n_critical_attacks']} | "
                f"{result['critical_recall_flat']:.4f} | "
                f"**{result['critical_recall_adaptive']:.4f}** |"
            )

    lines += [
        "",
        "## Verdict against `PRD.md §3`'s >30% false-positive reduction target",
        "",
    ]
    if target_met:
        lines += [
            f"The target is met in **{len(target_met)} of {len(results)}** mixes "
            f"({', '.join(r['name'] for r in target_met)}), with the largest reduction "
            f"**{best['fp_change']:+.1%}** in the *{best['name']}* mix.",
            "",
            "**This must not be reported as '>30% false-positive reduction achieved' without the**",
            "**mix stated in the same sentence.** The result is a property of the assumed device",
            "population, not of the method alone. On a network of mostly life-sustaining devices,",
            "the scheme deliberately produces *more* false positives — that is the intended",
            "behaviour, not a failure.",
        ]
    else:
        lines += [
            f"**The target is NOT met in any mix.** The largest reduction measured is "
            f"{best['fp_change']:+.1%} in the *{best['name']}* mix, short of the 30% "
            "`PRD.md §3` sets.",
            "",
            "Reported rather than tuned to fit. The weights in `src/adaptive/threshold.py` were",
            "chosen from clinical reasoning about the cost asymmetry between a missed attack and a",
            "false alarm, and re-picking them to clear a target on synthetic criticality labels",
            "would be fitting the assumption, not the problem.",
            "",
            "### Why the effect is small, and it is the same cause as Phase 2's",
            "",
            "**The ensemble's probabilities are saturated.** `reports/phase2_results.md` §6 measured",
            "a median branch confidence of 0.9999. A threshold only reclassifies detections that sit",
            "*near* it, and almost none do — moving the bar from 0.50 to 0.65 leaves a detection at",
            "0.9999 exactly where it was. The scheme is working correctly on a model that gives it",
            "very little to work with.",
            "",
            "This is the **same limitation** that made confidence-weighted fusion collapse into a",
            "simple average in Phase 2 (weights came out 0.337/0.330/0.334).",
            "",
            "**An earlier version of this report blamed miscalibration and called temperature scaling",
            "the highest-value remaining Phase 3 work. That was an inference, and measurement",
            "refuted it** (`docs/calibration_decision.md`). The branches are already well calibrated",
            "(ECE 0.012-0.023; the BiLSTM is mildly *under*-confident), and calibrating them moves",
            "this report's false-positive reduction from -6.8% to -6.9%.",
            "",
            "The saturation is real but it is **justified**: on IoTID20's 100% label-pure sessions",
            "the branches say 99% and are right 99% of the time. Most detections are genuinely",
            "unambiguous, so a threshold has little to reclassify. That is a property of the dataset",
            "and the task, not a defect in the model or in this threshold scheme.",
            "",
            "The open question is therefore whether `PRD.md §3`'s >30% target is reachable on a",
            "dataset this easy at all. T3.6 (Edge-IIoTset) is the check, and `PRD.md §3`'s actual",
            "wording -- reduction **on unseen attack types** -- points to T3.5, where novel attacks",
            "enter the system and detections should sit much closer to the boundary.",
        ]

    lines += [
        "",
        "## Honest limits",
        "",
        "- **Synthetic criticality**, as stated above. This is the dominant limitation.",
        "- **Fog-node load is held at neutral**, so the reported effect is criticality's alone. The",
        "  context term is exercised in `tests/test_adaptive_threshold.py`, not here.",
        "- **`PRD.md §3` asks for the reduction on *unseen attack types*.** This measures the",
        "  existing test fold, so it is the easier question. The novel-attack evaluation belongs",
        "  with T3.5's incremental learning, which is where new attack types enter the system.",
        "- **The ensemble's confidences are badly calibrated** (`reports/phase2_results.md` §6:",
        "  median branch confidence 0.9999). A threshold applied to a saturated probability has",
        "  less to work with than these numbers imply, because most detections sit far from any",
        "  boundary. Calibration is the Phase 3 proposal that would make this scheme sharper.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "t3_4_adaptive_threshold.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
