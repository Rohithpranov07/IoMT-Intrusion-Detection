"""SHAP vs LIME head-to-head on the trained ensemble (Build-Instructions T3.2 VERIFY).

T3.2's VERIFY condition is: *"Runs measurably faster than the SHAP path from T3.1 on the same sample
input."* This script measures that claim rather than assuming it, on the actual trained model, and
writes `reports/t3_2_lime_vs_shap.md` with whatever the answer turns out to be.

It also compares the two methods' *agreement*: if SHAP and LIME rank entirely different features as
most important, that is a finding the Review 3 report needs, because it means at least one of them
is not describing the model faithfully.

Warm-up is excluded from all timings. The first call to either method pays one-off graph tracing and
JIT costs that have nothing to do with per-explanation latency, and including it would overstate
both methods -- and overstate SHAP more, since it traces three branch graphs.

Prerequisite: `notebooks/03_train_ensemble_iotid20.ipynb` (writes `artifacts/`).

Usage:
    .venv/bin/python scripts/benchmark_xai.py
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import ARTIFACTS_DIR, POSITIVE_LABEL, REPORTS_DIR  # noqa: E402
from src.evaluation.metrics import POSITIVE_CLASS_STATEMENT  # noqa: E402
from src.models.train_utils import load_branch  # noqa: E402
from src.xai.lime_explainer import NUM_SAMPLES, EnsembleLimeExplainer  # noqa: E402
from src.xai.shap_explainer import DEFAULT_BACKGROUND_SIZE, EnsembleShapExplainer  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

#: Flagged windows timed per method. Enough for a median and a spread, cheap enough to re-run.
N_TIMED: int = 10
#: Features compared when measuring rank agreement between the two methods.
TOP_N: int = 5


def rank_agreement(shap_features: list[str], lime_features: list[str]) -> float:
    """Return the overlap between two ranked feature lists, as a fraction.

    Args:
        shap_features: SHAP's top-N feature names, most important first.
        lime_features: LIME's top-N feature names.

    Returns:
        `|intersection| / N`. 1.0 means the same features (in any order), 0.0 means no overlap.
    """
    return len(set(shap_features) & set(lime_features)) / max(len(shap_features), 1)


def main() -> None:
    """Time both explainers on identical inputs and write the comparison report."""
    if not (ARTIFACTS_DIR / "ensemble_artifacts.npz").exists():
        raise SystemExit("artifacts/ not found. Run notebooks/03_train_ensemble_iotid20.ipynb.")

    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    meta = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    feature_names = meta["feature_names"]
    branch_names = meta["branch_names"]

    models = {
        name: load_branch(ARTIFACTS_DIR / "models" / f"{name}.keras") for name in branch_names
    }
    background = data["X_background"]
    X_test = data["X_test"]
    predictions = data["fused_predictions"]

    flagged = np.flatnonzero(predictions == POSITIVE_LABEL)[:N_TIMED]
    print(f"Timing {len(flagged)} flagged windows with both methods\n")

    shap_explainer = EnsembleShapExplainer(
        models, feature_names, background, background_size=DEFAULT_BACKGROUND_SIZE
    )
    lime_explainer = EnsembleLimeExplainer(models, feature_names, background)

    # Warm-up, excluded from the timings (see module docstring).
    shap_explainer.explain(X_test[flagged[0]])
    lime_explainer.explain(X_test[flagged[0]])

    shap_times: list[float] = []
    lime_times: list[float] = []
    agreements: list[float] = []
    rows: list[dict[str, object]] = []

    for index in flagged:
        shap_explanation = shap_explainer.explain(X_test[index], top_n=TOP_N)
        lime_explanation = lime_explainer.explain(X_test[index], top_n=TOP_N)

        shap_times.append(shap_explanation.elapsed_seconds)
        lime_times.append(lime_explanation.elapsed_seconds)

        shap_top = [f.feature_name for f in shap_explanation.top_features]
        lime_top = [f.feature_name for f in lime_explanation.top_features]
        agreements.append(rank_agreement(shap_top, lime_top))

        rows.append({
            "window": int(index),
            "shap_seconds": shap_explanation.elapsed_seconds,
            "lime_seconds": lime_explanation.elapsed_seconds,
            "shap_top": shap_top,
            "lime_top": lime_top,
            "agreement": agreements[-1],
        })
        print(f"  window {index:>5d}  SHAP {shap_explanation.elapsed_seconds:5.2f}s   "
              f"LIME {lime_explanation.elapsed_seconds:5.2f}s   "
              f"top-{TOP_N} overlap {agreements[-1]:.0%}")

    shap_median = statistics.median(shap_times)
    lime_median = statistics.median(lime_times)
    faster, slower = ("LIME", "SHAP") if lime_median < shap_median else ("SHAP", "LIME")
    ratio = max(shap_median, lime_median) / max(min(shap_median, lime_median), 1e-9)
    verify_passed = lime_median < shap_median

    print(f"\nSHAP median {shap_median:.3f}s | LIME median {lime_median:.3f}s")
    print(f"{faster} is {ratio:.1f}x faster than {slower}")
    print(f"Mean top-{TOP_N} agreement: {statistics.mean(agreements):.0%}")
    print(f"\nT3.2 VERIFY ('LIME faster than SHAP'): {'PASS' if verify_passed else 'FAIL'}")

    lines = [
        "# T3.2 — LIME vs SHAP, measured",
        "",
        "**Task:** T3.2 · **Generated by:** `scripts/benchmark_xai.py` · "
        "**Model:** the trained ensemble from T2.6",
        "",
        f"> {POSITIVE_CLASS_STATEMENT}",
        "",
        "## Result",
        "",
        f"| | SHAP (`GradientExplainer`) | LIME (`num_samples={NUM_SAMPLES}`) |",
        "|---|---:|---:|",
        f"| Median seconds per explanation | **{shap_median:.3f}** | **{lime_median:.3f}** |",
        f"| Mean | {statistics.mean(shap_times):.3f} | {statistics.mean(lime_times):.3f} |",
        f"| Fastest | {min(shap_times):.3f} | {min(lime_times):.3f} |",
        f"| Slowest | {max(shap_times):.3f} | {max(lime_times):.3f} |",
        "",
        f"Windows timed: {len(flagged)} flagged detections, identical inputs to both methods, "
        "warm-up excluded.",
        "",
        f"### T3.2's VERIFY block: **{'PASS' if verify_passed else 'FAIL'}**",
        "",
    ]

    if verify_passed:
        lines += [
            f"LIME is {ratio:.1f}x faster than SHAP on this configuration, as "
            "`docs/xai_survey.md` §2.2 predicted.",
        ]
    else:
        lines += [
            f"**LIME is {ratio:.1f}x SLOWER than SHAP here, so T3.2's VERIFY condition does not "
            "hold.** This is reported rather than tuned away, and the reason is specific and worth "
            "recording:",
            "",
            "`docs/xai_survey.md` §2.2 judged LIME the faster method by comparing it against SHAP",
            "*generically* — and the slow SHAP variant is `KernelExplainer`, which needs thousands",
            "of forward passes per explanation. **T3.1 never adopted `KernelExplainer`.** It uses",
            "`GradientExplainer`, which computes attributions from gradients in a handful of passes",
            "and is therefore cheap. LIME's speed advantage was an advantage over a SHAP variant",
            "this project does not use.",
            "",
            "LIME's cost here is structural, not a tuning mistake: it must run "
            f"`num_samples={NUM_SAMPLES}` perturbed windows through **all three branches plus the",
            "fusion** for every single explanation. Lowering `num_samples` would buy speed directly",
            "at the cost of surrogate stability — already LIME's documented weakness, and the exact",
            "property T3.3 has to measure. Trading away stability to win a benchmark would be the",
            "wrong call.",
            "",
            "**What this changes.** `docs/xai_survey.md` §4's recommendation should be annotated:",
            "LIME remains worth keeping, but as the **more faithful** method rather than the faster",
            "one. It explains the fused ensemble as a single black box, exactly as deployed, whereas",
            "SHAP cannot — the fusion is not differentiable end to end, so T3.1 runs per branch and",
            "recombines the maps with the fusion weights, a construction this project invented and",
            "has to defend. LIME needs no such step.",
            "",
            "**What this does not change.** SHAP stays the primary method. It is faster *and* it has",
            "the additive-consistency property that made it the primary choice in the first place",
            "(`docs/xai_survey.md` §2.1). No fallback-on-latency logic is needed, because the",
            "primary method is already the cheap one.",
        ]

    lines += [
        "",
        "## Do the two methods agree?",
        "",
        f"Mean top-{TOP_N} feature overlap: **{statistics.mean(agreements):.0%}** "
        f"(min {min(agreements):.0%}, max {max(agreements):.0%}).",
        "",
        "| Window | SHAP (s) | LIME (s) | SHAP top features | LIME top features | Overlap |",
        "|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['window']} | {row['shap_seconds']:.3f} | {row['lime_seconds']:.3f} | "
            f"{', '.join(f'`{f}`' for f in row['shap_top'][:3])} | "
            f"{', '.join(f'`{f}`' for f in row['lime_top'][:3])} | {row['agreement']:.0%} |"
        )

    lines += [
        "",
        "Agreement is a **consistency** check, not a correctness one: two methods can agree and both",
        "be wrong. Whether either explanation actually reflects the model is what T3.3's *fidelity*",
        "metric measures, by ablating the top-attributed features and checking the prediction moves",
        "in the predicted direction.",
        "",
        "## Caveats",
        "",
        "- These are **desktop** timings (Apple-silicon CPU). They are not, and must not be quoted",
        "  as, deployment latency. Any real-time claim needs the Raspberry Pi 4B measurement from",
        "  T4.3 (`Build-Instructions.md` §A.1 rule 3).",
        "- LIME's timing scales linearly with `num_samples`; the value used here is fixed in",
        "  `src/xai/lime_explainer.py` and stated in the table header.",
        "- Seeding makes a single LIME explanation reproducible. It does **not** make LIME stable",
        "  under input perturbation, which is a different property and T3.3's job to measure.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "t3_2_lime_vs_shap.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
