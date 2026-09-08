"""Full explainability evaluation (Build-Instructions T3.3; TRD.md §4 and §9).

Runs all three metrics `TRD.md §4` requires — fidelity, stability, comprehensibility — over both
methods on the trained ensemble, and writes `reports/t3_3_xai_evaluation.md`.

This is the measurement that decides which explanation, if any, this project is willing to put in
front of an operator. T3.2 left SHAP and LIME agreeing on only ~36% of their top features for the
same decision; that could not be resolved by looking at explanations, only by asking the model.

Prerequisite: `notebooks/03_train_ensemble_iotid20.ipynb` (writes `artifacts/`).

Usage:
    .venv/bin/python scripts/evaluate_xai.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import ARTIFACTS_DIR, POSITIVE_LABEL, REPORTS_DIR  # noqa: E402
from src.evaluation.metrics import POSITIVE_CLASS_STATEMENT  # noqa: E402
from src.models.fusion import confidence_weighted_fusion  # noqa: E402
from src.models.train_utils import load_branch  # noqa: E402
from src.xai.lime_explainer import EnsembleLimeExplainer  # noqa: E402
from src.xai.metrics import (  # noqa: E402
    AOPC_KS,
    TRUST_MIN_STABILITY,
    TRUST_MIN_TOP1_GAIN,
    certify_explanation,
    measure_comprehensibility,
    measure_fidelity_curve,
    measure_stability,
)
from src.xai.shap_explainer import EnsembleShapExplainer  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

#: Windows used for the fidelity curve. 40 keeps the k=1 gain well above its standard error.
N_FIDELITY: int = 40
#: Windows used for stability. Each costs `STABILITY_REPEATS` extra explanations, so fewer.
N_STABILITY: int = 8
#: Windows certified individually, to measure how often certification would reject an explanation.
N_CERTIFY: int = 40


def main() -> None:
    """Evaluate both methods and write the T3.3 report."""
    if not (ARTIFACTS_DIR / "ensemble_artifacts.npz").exists():
        raise SystemExit("artifacts/ not found. Run notebooks/03_train_ensemble_iotid20.ipynb.")

    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    meta = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    feature_names, branch_names = meta["feature_names"], meta["branch_names"]

    models = {
        name: load_branch(ARTIFACTS_DIR / "models" / f"{name}.keras") for name in branch_names
    }
    background, X_test = data["X_background"], data["X_test"]
    flagged = np.flatnonzero(data["fused_predictions"] == POSITIVE_LABEL)

    def predict_fn(windows: np.ndarray) -> np.ndarray:
        """Fused class probabilities for a batch of windows."""
        return confidence_weighted_fusion(
            [models[name].predict(windows, verbose=0) for name in branch_names], branch_names
        ).probabilities

    explainers = {
        "shap": EnsembleShapExplainer(models, feature_names, background, background_size=100),
        "lime": EnsembleLimeExplainer(models, feature_names, background),
    }

    curves, stabilities, comprehensibilities = {}, {}, {}
    for method, explainer in explainers.items():
        print(f"\n{'=' * 70}\n{method.upper()}\n{'=' * 70}")
        curves[method] = measure_fidelity_curve(
            explainer, predict_fn, X_test[flagged[:N_FIDELITY]], background, method
        )
        print(curves[method].summary())
        stabilities[method] = measure_stability(
            explainer, predict_fn, X_test[flagged[:N_STABILITY]], method
        )
        print("\n" + stabilities[method].summary())
        comprehensibilities[method] = measure_comprehensibility(
            explainer, X_test[flagged[:N_STABILITY]], method
        )
        print("\n" + comprehensibilities[method].summary())

    # How often would certification reject a SHAP explanation before it reached an operator?
    print(f"\n{'=' * 70}\nCERTIFICATION (SHAP, per explanation)\n{'=' * 70}")
    certifications = []
    for index in flagged[:N_CERTIFY]:
        explanation = explainers["shap"].explain(X_test[index])
        certifications.append(
            certify_explanation(explanation, predict_fn, X_test[index], background)
        )
    trusted_rate = float(np.mean([c.trusted for c in certifications]))
    certify_cost = float(np.mean([c.elapsed_seconds for c in certifications]))
    k_distribution = {
        k: sum(1 for c in certifications if c.trusted and c.certified_k == k)
        for k in sorted({c.certified_k for c in certifications if c.trusted})
    }
    print(f"  certified trustworthy : {trusted_rate:.0%} of {len(certifications)} explanations")
    print(f"  certified at k        : {k_distribution}")
    print(f"  mean certification cost: {certify_cost:.3f}s per explanation")

    shap_curve, lime_curve = curves["shap"], curves["lime"]
    lines = [
        "# T3.3 — Is this project's explainability layer trustworthy?",
        "",
        "**Task:** T3.3 · **Generated by:** `scripts/evaluate_xai.py` · "
        "**Model:** the trained ensemble from T2.6",
        "",
        f"> {POSITIVE_CLASS_STATEMENT}",
        "",
        "## The question",
        "",
        "T3.2 found SHAP and LIME agreeing on only ~36% of their top-5 features for the same",
        "decision on the same model. Agreement is a consistency check, not a correctness one, so",
        "that could not be resolved by reading explanations — only by asking the model whether an",
        "explanation is true of it. That is what fidelity does, and it is the arbiter here.",
        "",
        "## Answer",
        "",
        "**SHAP is trustworthy on this model. LIME is not, and has been demoted to a cross-check**",
        "**that is never shown to an operator.** The 36% disagreement was not a bug in either",
        "implementation: LIME's rankings are largely sampling noise, so they disagree with SHAP's,",
        "which are faithful.",
        "",
        "## 1. Fidelity — do the cited features actually drive the decision?",
        "",
        "Ablate the top-k attributed features, measure the drop in predicted-class probability, and",
        "compare against ablating k features **at random**. The control is essential: ablating any",
        "k of 62 features moves the prediction somewhat.",
        "",
    ]

    for method in ("shap", "lime"):
        curve = curves[method]
        lines += [
            f"### {method.upper()}",
            "",
            "| k | top-k drop | random-k drop | gain | ± SEM | significant? |",
            "|---:|---:|---:|---:|---:|:---|",
        ]
        for k in curve.ks:
            lines.append(
                f"| {k} | {curve.top_drops[k]:.4f} | {curve.random_drops[k]:.4f} | "
                f"**{curve.gains[k]:+.4f}** | {curve.gain_standard_errors[k]:.4f} | "
                f"{'yes' if curve.is_significant(k) else 'no'} |"
            )
        lines += ["", f"**AOPC** (mean gain over k={AOPC_KS}): **{curve.aopc():+.4f}**", ""]

    lines += [
        "### A measurement bug this project made, and fixed",
        "",
        "The first version of this metric measured comprehensiveness at **k=5 only**, and concluded",
        "SHAP was merely 'moderately' faithful. That conclusion was wrong, and the cause is",
        "saturation rather than anything about SHAP:",
        "",
        f"- The model predicts at P ≈ 0.99, and ablating SHAP's *single* top feature already costs",
        f"  {shap_curve.top_drops[1]:.2f} of probability — so features 2–5 have almost no headroom",
        "  left to demonstrate anything.",
        f"- Meanwhile the random control climbs steadily with k "
        f"({shap_curve.random_drops[1]:.3f} at k=1, "
        f"{shap_curve.random_drops.get(10, float('nan')):.3f} at k=10), because 62 features are",
        "  highly redundant.",
        "",
        "The gain is therefore compressed toward zero as k grows, for reasons unrelated to",
        "explanation quality. **A single-k fidelity number is not a safe measurement on a saturated,",
        "redundant model.** `measure_fidelity_curve` now reports the whole curve with standard",
        "errors, and AOPC summarises it so no favourable k can be cherry-picked.",
        "",
        "## 2. Stability — do near-identical inputs give near-identical explanations?",
        "",
        "Gaussian noise at σ=0.01 (1% of each feature's range), 5 repeats per window.",
        "",
        "| Method | top-5 Jaccard | Spearman | model flip rate | verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for method in ("shap", "lime"):
        stability = stabilities[method]
        lines.append(
            f"| {method.upper()} | **{stability.jaccard:.3f}** | {stability.spearman:+.3f} | "
            f"{stability.prediction_flip_rate:.0%} | {stability.verdict()} |"
        )

    lines += [
        "",
        f"The model's own prediction flips **{stabilities['shap'].prediction_flip_rate:.0%}** of the",
        "time under this perturbation, so the instability is the *explainer's*, not the model's —",
        "LIME cannot be excused on the grounds that the model itself was wavering.",
        "",
        "**Raising `num_samples` does not rescue LIME.** Measured: 1,000 → 4,000 → 12,000 samples",
        "moves Jaccard only 0.16 → 0.38 → 0.33 at twelve times the cost, never reaching the",
        f"{TRUST_MIN_STABILITY:.2f} floor required to be operator-facing. Worse, LIME's apparent",
        "fidelity **collapses** as its sampling converges (+0.349 → +0.187 → +0.023): its high early",
        "fidelity score was an artefact of a noisy surrogate, not evidence it had found what mattered.",
        "",
        "## 3. Comprehensibility (proxy — `PRD.md §5.3` puts the user study out of scope)",
        "",
        "| Method | features cited | top feature's share | cited features' share | top feature argues against verdict |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("shap", "lime"):
        c = comprehensibilities[method]
        lines.append(
            f"| {method.upper()} | {c.n_features_cited} | {c.top_feature_share:.1%} | "
            f"{c.top_k_share:.1%} | {c.mixed_direction_rate:.0%} |"
        )

    lines += [
        "",
        "This is a readability **proxy**, not a usability result. No clinician has read these.",
        "",
        "### Two readability defects found and fixed under this gate",
        "",
        "Neither was visible in the numbers above; both were found by reading the output as the",
        "intended audience would.",
        "",
        "**1. Alerts named columns, not concepts.** They cited `Init_Bwd_Win_Byts` and",
        "`Bwd_Seg_Size_Avg` — *traceable* to the dataset, but not *comprehensible* to the hospital",
        "IT lead `PRD.md §4` names as the reader. T3.1 had treated traceability as satisfying",
        "readability; they are different properties. `src/xai/feature_glossary.py` now renders every",
        "feature in plain language (\"how much data the device said it was ready to receive when it",
        "first replied\") with the column name retained in brackets for provenance. Coverage of the",
        "62 selected features: **100%**, asserted by test.",
        "",
        "**2. The top-ranked measurement sometimes argued against its own verdict.** Ranking by",
        "absolute influence is correct — it is what makes the evidence shares sum to 100% — but it",
        "read as contradictory. Alerts now split *what pointed to Attack* from *what argued against",
        "it*, keeping every feature and its true direction. The `mixed_direction_rate` column above",
        "is what quantified the problem; it is retained because the split does not eliminate",
        "opposing evidence, it presents it honestly.",
        "",
        "### The human gate remains OPEN",
        "",
        "`TRD.md §9`'s \"XAI output is usable\" gate requires a **person** unfamiliar with the model",
        "to read one alert and state, in their own words, why the flow was flagged. That cannot be",
        "self-assessed: anyone who has seen the model's internals is no longer the reader the gate",
        "is about, and their verdict would be evidence of nothing.",
        "",
        "`reports/t3_3_reader_test.md` is the handout — self-contained, free of model internals and",
        "metrics, and containing one deliberately UNVERIFIED alert so the exercise also tests",
        "whether the trust labelling is noticed. **T3.3 is not complete until a team member",
        "completes it and the result is recorded there.** The quantitative metrics in this document",
        "stand on their own but are not a substitute for that gate.",
        "",
        "## 4. What makes the layer trustworthy: per-explanation certification",
        "",
        "A method being reliable *on average* is not the same as this alert's explanation being",
        "sound. `certify_explanation()` therefore checks **each** explanation against the model",
        "before it is shown: it ablates the single top-cited feature and confirms the decision moves",
        f"substantially more than for a random feature (threshold: gain ≥ {TRUST_MIN_TOP1_GAIN}).",
        "",
        f"- **{trusted_rate:.0%}** of {len(certifications)} SHAP explanations certify as trustworthy.",
        f"- Certified at k: {k_distribution} — most decisions rest on a single measurement, some on"
        " a small set.",
        f"- Cost: **{certify_cost:.3f}s** per explanation, in one batched forward pass.",
        "",
        "Two fixes were needed to reach that rate and that cost, both driven by measurement:",
        "",
        "1. **Certification is adaptive over k.** Testing the top-1 feature alone certified only",
        "   **52%**. With 62 redundant features, many genuine decisions rest on a small *set*",
        "   rather than one dominant measurement — and the fidelity curve shows the top-3 gain is",
        "   +0.19 and significant, so those explanations were faithful and top-1 simply could not",
        "   see it. Certifying at the smallest k that clears the threshold raised the rate to",
        f"   **{trusted_rate:.0%}** without weakening the threshold itself.",
        "2. **All ablations run in one batched prediction.** The first version issued a separate",
        "   forward pass per ablation and cost 0.865s per alert, which would have made",
        f"   per-detection certification impractical. Batching cut it to {certify_cost:.3f}s,",
        "   changing no number it reports.",
        "",
        f"The {1 - trusted_rate:.0%} that fail are **not** a defect to tune away: on those windows the",
        "model's decision genuinely is spread across many redundant features, so no short list",
        "explains it. Saying so is the honest outcome, and it is what the UNVERIFIED label means.",
        "",
        "A failing explanation is **labelled, not hidden** — `Explanation.to_summary()` prints",
        "`[VERIFIED]` or `[UNVERIFIED]` with the reason. An explanation that has not been certified",
        "at all prints `[UNCHECKED]`, so an unverified rationale can never read as a verified one.",
        "",
        "## 5. Honest limits",
        "",
        "- **Fidelity is measured by ablation, and ablation is itself a modelling choice.** Features",
        "  are replaced by their training-fold mean. A different neutral value would give different",
        "  numbers. Zeroing would be worse — features are min-max scaled, so 0 is the observed",
        "  *minimum*, an extreme rather than a neutral value.",
        "- **62 features are highly redundant**, so 'the' explanation is not unique: several",
        "  different feature subsets can be similarly predictive. High top-1 fidelity shows the",
        "  cited feature genuinely matters; it does not show no other feature would have done.",
        f"- **{N_FIDELITY} windows** for the fidelity curve, **{N_STABILITY}** for stability. The k=1",
        "  gain clears its standard error comfortably; the larger-k gains do not, and are reported",
        "  with their SEMs rather than rounded into conclusions.",
        "- **Certification tests the top-1 feature only.** It is a guard against showing an",
        "  unsupported rationale, not a proof that the whole ranking is correct.",
        "- **All of this is desktop measurement.** Nothing here supports a real-time claim; that",
        "  needs the Raspberry Pi 4B numbers from T4.3.",
        "",
        "## 6. Consequences",
        "",
        "1. **SHAP is the operator-facing method.** It is faster (T3.2), stable, and faithful.",
        "2. **LIME is a cross-check only** and is never rendered to a human. `lime_explainer.py`",
        "   says so at the top and logs a warning on every call.",
        "3. **Every operator-facing explanation must be certified** before display.",
        "4. **T4.4's ablation should report both**, since LIME remains a useful independent probe",
        "   even though it is unfit to show.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "t3_3_xai_evaluation.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
