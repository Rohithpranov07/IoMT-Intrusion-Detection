"""Does per-branch calibration unblock fusion and the adaptive threshold? (calibration decision)

`reports/phase2_results.md` §6 and `reports/t3_4_adaptive_threshold.md` both trace a limitation to
the same cause: the branches' probabilities are saturated (median confidence 0.9999), so the
confidence-weighted fusion degenerates into a simple average and the adaptive threshold has almost
no detections near it to reclassify.

`docs/calibration_decision.md` records the decision. This script produces the evidence it rests on,
because a decision record asserting that calibration "should help" would be worth nothing.

Everything is fitted on the VALIDATION fold and evaluated on TEST. Fitting a calibration parameter
on the evaluation fold is the class of leakage Contribution 1 exists to expose.

Usage:
    .venv/bin/python scripts/evaluate_calibration.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.adaptive.threshold import DETECTION_BASE, Criticality, apply_threshold  # noqa: E402
from src.config import ARTIFACTS_DIR, POSITIVE_LABEL, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    POSITIVE_CLASS_STATEMENT,
    compute_metrics,
    hand_verify_metrics,
)
from src.models.calibration import (  # noqa: E402
    apply_temperature,
    expected_calibration_error,
    fit_all_branches,
)
from src.models.fusion import confidence_weighted_fusion  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

#: Criticality mix used for the threshold comparison. The mix that came closest to PRD.md §3's
#: target uncalibrated, so the comparison is against calibration's best previous showing.
PERIPHERY_HEAVY: tuple[float, float, float, float] = (0.05, 0.10, 0.25, 0.60)
LOAD: float = 0.5


def main() -> None:
    """Measure calibration's effect on fusion and on the adaptive threshold."""
    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    meta = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    branch_names = meta["branch_names"]

    y_val, y_test = data["y_val"], data["y_test"]
    val_probabilities = {n: data[f"val_probabilities_{n}"] for n in branch_names}
    test_probabilities = {n: data[f"probabilities_{n}"] for n in branch_names}

    print(POSITIVE_CLASS_STATEMENT)
    print(f"\nFitting temperatures on {len(y_val):,} VALIDATION windows; "
          f"evaluating on {len(y_test):,} TEST windows.\n")

    results = fit_all_branches(val_probabilities, y_val)
    for result in results.values():
        print(result.summary(), "\n")

    calibrated_test = {
        name: apply_temperature(test_probabilities[name], results[name].temperature)
        for name in branch_names
    }

    # --- Effect on fusion -------------------------------------------------------------------
    fusion_before = confidence_weighted_fusion(
        [test_probabilities[n] for n in branch_names], branch_names
    )
    fusion_after = confidence_weighted_fusion(
        [calibrated_test[n] for n in branch_names], branch_names
    )

    metrics_before = hand_verify_metrics(
        compute_metrics(y_test, fusion_before.predictions, "Fusion (uncalibrated)"), verbose=False
    )
    metrics_after = hand_verify_metrics(
        compute_metrics(y_test, fusion_after.predictions, "Fusion (calibrated)"), verbose=False
    )
    branch_results = {
        name: hand_verify_metrics(
            compute_metrics(y_test, test_probabilities[name].argmax(axis=1), name), verbose=False
        )
        for name in branch_names
    }
    best_branch = max(branch_results.values(), key=lambda m: m.f1)

    weights_before = fusion_before.branch_weights.mean(axis=0)
    weights_after = fusion_after.branch_weights.mean(axis=0)
    spread_before = float(
        (fusion_before.branch_weights.max(axis=1) - fusion_before.branch_weights.min(axis=1)).mean()
    )
    spread_after = float(
        (fusion_after.branch_weights.max(axis=1) - fusion_after.branch_weights.min(axis=1)).mean()
    )

    print("=" * 70)
    print("EFFECT ON FUSION")
    print("=" * 70)
    print(f"  mean weights before : {dict(zip(branch_names, weights_before.round(4)))}")
    print(f"  mean weights after  : {dict(zip(branch_names, weights_after.round(4)))}")
    print(f"  mean weight spread  : {spread_before:.4f} -> {spread_after:.4f}")
    print(f"  fusion F1           : {metrics_before.f1:.4f} -> {metrics_after.f1:.4f}")
    print(f"  best single branch  : {best_branch.f1:.4f} ({best_branch.model_name})")

    # --- Effect on the adaptive threshold ---------------------------------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    classes = list(Criticality)
    assignment = rng.choice(len(classes), size=len(y_test), p=PERIPHERY_HEAVY)
    criticalities = [classes[i] for i in assignment]

    threshold_rows = []
    for label, fusion in (("uncalibrated", fusion_before), ("calibrated", fusion_after)):
        attack_probability = fusion.probabilities[:, POSITIVE_LABEL]
        flat = hand_verify_metrics(
            compute_metrics(
                y_test, (attack_probability >= DETECTION_BASE).astype(np.int8), f"flat {label}"
            ),
            verbose=False,
        )
        adaptive = hand_verify_metrics(
            compute_metrics(
                y_test,
                apply_threshold(attack_probability, criticalities, loads=LOAD),
                f"adaptive {label}",
            ),
            verbose=False,
        )
        change = (adaptive.fp - flat.fp) / max(flat.fp, 1)
        near = float(((attack_probability > 0.05) & (attack_probability < 0.95)).mean())
        threshold_rows.append((label, flat, adaptive, change, near))
        print(f"\n  {label:<13} FP {flat.fp} -> {adaptive.fp} ({change:+.1%}); "
              f"{near:.1%} of detections lie in the actionable band 0.05-0.95")

    ece_before = float(np.mean([
        expected_calibration_error(test_probabilities[n], y_test) for n in branch_names
    ]))
    ece_after = float(np.mean([
        expected_calibration_error(calibrated_test[n], y_test) for n in branch_names
    ]))

    lines = [
        "# Calibration — measured evidence for the decision record",
        "",
        "**Generated by:** `scripts/evaluate_calibration.py` · "
        "**Decision:** `docs/calibration_decision.md` · **Model:** the trained ensemble from T2.6",
        "",
        f"> {POSITIVE_CLASS_STATEMENT}",
        "",
        f"Temperatures fitted on **{len(y_val):,} validation** windows, all results below measured "
        f"on **{len(y_test):,} held-out test** windows. Fitting on test would be the leakage "
        "Contribution 1 exists to expose.",
        "",
        "## 1. Were the branches actually miscalibrated?",
        "",
        "| Branch | Fitted T | NLL (val) | ECE (val) | Mean confidence | Accuracy |",
        "|---|---:|---|---|---|---|",
    ]
    for name in branch_names:
        r = results[name]
        lines.append(
            f"| {name} | **{r.temperature:.2f}** | {r.nll_before:.4f} → {r.nll_after:.4f} | "
            f"{r.ece_before:.4f} → {r.ece_after:.4f} | "
            f"{r.mean_confidence_before:.4f} → {r.mean_confidence_after:.4f} | "
            f"{r.accuracy_before:.4f} → {r.accuracy_after:.4f} |"
        )
    lines += [
        "",
        f"Mean ECE on the **test** fold: **{ece_before:.4f} → {ece_after:.4f}**.",
        "",
        "Accuracy is identical before and after in every branch, as it must be: dividing all logits",
        "by one positive scalar is monotonic, so `argmax` cannot move. `fit_temperature` raises if",
        "it ever does, because that would mean the implementation is wrong rather than that a",
        "trade-off was made.",
        "",
        "## 2. Effect on the fusion (Contribution 2)",
        "",
        "| | Uncalibrated | Calibrated |",
        "|---|---|---|",
        f"| Mean fusion weights | {dict(zip(branch_names, weights_before.round(3)))} | "
        f"{dict(zip(branch_names, weights_after.round(3)))} |",
        f"| Mean per-sample weight spread | {spread_before:.4f} | **{spread_after:.4f}** |",
        f"| Fusion F1 | {metrics_before.f1:.4f} | **{metrics_after.f1:.4f}** |",
        f"| Best single branch F1 | {best_branch.f1:.4f} | {best_branch.f1:.4f} |",
        "",
        "Weight *spread* is the diagnostic that matters. Uniform weights mean the confidence term",
        "conveys nothing and the formula is a simple average; a spread that widens means confidence",
        "has become informative and the fusion is finally doing what `TRD.md §3.3` specified.",
        "",
        "## 3. Effect on the adaptive threshold (Contribution 4)",
        "",
        "Periphery-heavy criticality mix, neutral load — the mix that came closest to "
        "`PRD.md §3`'s >30% target uncalibrated.",
        "",
        "| | False positives | Δ vs flat | Detections in the actionable band 0.05–0.95 |",
        "|---|---:|---:|---:|",
    ]
    for label, flat, adaptive, change, near in threshold_rows:
        lines.append(
            f"| {label} | {flat.fp} → {adaptive.fp} | **{change:+.1%}** | {near:.1%} |"
        )

    lines += [
        "",
        "The last column is the mechanism: a threshold can only reclassify detections that sit near",
        "it. If calibration moves detections off the 0/1 rails into the actionable band, the",
        "threshold gains something to act on; if it does not, no threshold scheme can help.",
        "",
        "## 4. Honest limits",
        "",
        "- Temperature scaling fixes the *magnitude* of confidences, not their *ordering*. If a",
        "  branch is confidently wrong, calibration makes it less confidently wrong — it does not",
        "  make it right.",
        "- One scalar per branch is the simplest possible calibrator. It cannot correct a branch",
        "  whose miscalibration varies across the input space.",
        "- ECE is bin-count sensitive (15 equal-width bins here) and is a summary, not a guarantee.",
        "- These are test-fold numbers on IoTID20, whose sessions are 100% label-pure",
        "  (`reports/phase2_results.md` §9). Edge-IIoTset (T3.6) is the check on generality.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "calibration_evidence.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
