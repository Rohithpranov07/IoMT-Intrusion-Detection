"""Generate the reader-test handout for T3.3's human VERIFY step.

T3.3's VERIFY condition is the one thing in this project no test can perform:

    "Have a team member unfamiliar with the model internals read one sample explanation output and
     correctly state, in their own words, why that flow was flagged."

`TRD.md §9` phrases the same gate as "XAI output is usable". This script produces
`reports/t3_3_reader_test.md`: a self-contained handout a teammate can complete without opening the
codebase, plus the scoring guidance needed for their answer to mean something.

Two design choices worth stating, because they decide whether the exercise measures anything:

  1. The handout contains **no model internals, no metrics, and no method names**. A reader told
     that SHAP scored an AOPC of +0.28 is no longer an uninformed reader, and their answer stops
     being evidence about readability.
  2. It includes an **UNVERIFIED** explanation alongside verified ones. If a reader cannot tell
     which explanations the system itself does not stand behind, the labelling has failed — and
     that is more important to know than whether the prose reads nicely.

Prerequisite: `notebooks/03_train_ensemble_iotid20.ipynb` (writes `artifacts/`).

Usage:
    .venv/bin/python scripts/make_reader_test.py
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
from src.models.fusion import confidence_weighted_fusion  # noqa: E402
from src.models.train_utils import load_branch  # noqa: E402
from src.xai.metrics import certify_explanation  # noqa: E402
from src.xai.shap_explainer import EnsembleShapExplainer  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")

#: Verified explanations to include, plus one unverified one if the sample contains any.
N_VERIFIED: int = 2
#: How many flagged windows to search for a verified/unverified pair.
SEARCH_LIMIT: int = 30


def main() -> None:
    """Build the handout and write it to reports/."""
    if not (ARTIFACTS_DIR / "ensemble_artifacts.npz").exists():
        raise SystemExit("artifacts/ not found. Run notebooks/03_train_ensemble_iotid20.ipynb.")

    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    meta = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    feature_names, branch_names = meta["feature_names"], meta["branch_names"]

    models = {
        name: load_branch(ARTIFACTS_DIR / "models" / f"{name}.keras") for name in branch_names
    }
    background, X_test, y_test = data["X_background"], data["X_test"], data["y_test"]

    def predict_fn(windows: np.ndarray) -> np.ndarray:
        """Fused class probabilities."""
        return confidence_weighted_fusion(
            [models[name].predict(windows, verbose=0) for name in branch_names], branch_names
        ).probabilities

    explainer = EnsembleShapExplainer(models, feature_names, background, background_size=100)
    flagged = np.flatnonzero(data["fused_predictions"] == POSITIVE_LABEL)[:SEARCH_LIMIT]

    verified: list[tuple[int, object]] = []
    unverified: list[tuple[int, object]] = []
    for index in flagged:
        explanation = explainer.explain(X_test[index])
        explanation.certification = certify_explanation(
            explanation, predict_fn, X_test[index], background
        )
        bucket = verified if explanation.certification.trusted else unverified
        if len(bucket) < (N_VERIFIED if bucket is verified else 1):
            bucket.append((int(index), explanation))
        if len(verified) >= N_VERIFIED and len(unverified) >= 1:
            break

    selected = verified + unverified
    print(f"Selected {len(verified)} verified and {len(unverified)} unverified explanation(s)")

    lines = [
        "# Reader test — can a non-specialist use these alerts? (T3.3 VERIFY)",
        "",
        "**For the reader:** you do not need any machine-learning background, and you should not",
        "look at the code before answering. That is the point of the exercise — if these alerts",
        "only make sense to someone who built the model, they have failed at their job.",
        "",
        "## Background you need (and no more)",
        "",
        "A monitoring system watches network traffic going to medical devices on a hospital network.",
        "It reads the traffic in short windows of 10 consecutive records. For each window it decides",
        "**Attack** or **Normal**, and when it says Attack it produces the note below explaining",
        "itself. Three separate detectors vote on each decision, and the note says how much weight",
        "each one carried.",
        "",
        "Each note begins with a status line in square brackets. Part of this exercise is finding",
        "out whether that line means anything to you without being told what it means.",
        "",
        "Each measurement is described in words, with the underlying technical field name in",
        "brackets after it. You should not need the bracketed name; it is there so a network",
        "engineer could trace the alert back to the raw traffic.",
        "",
        "## Your task",
        "",
        "For **each** alert below, write down — in your own words, in one or two sentences:",
        "",
        "1. **What did the system decide, and how sure was it?**",
        "2. **Why did it decide that?** Name the specific measurements it based the decision on.",
        "3. **Would you act on this alert?** Why or why not?",
        "",
        "Then answer once, overall:",
        "",
        "4. **Did any alert below look less reliable than the others?** If so, which, and what told",
        "   you?",
        "5. **What did you not understand?** Every term that stopped you is a defect worth fixing.",
        "6. **Were the measurement descriptions clear**, or did you find yourself relying on the",
        "   bracketed technical names? If any description left you guessing, quote it.",
        "",
        "---",
        "",
    ]

    for position, (index, explanation) in enumerate(selected, start=1):
        lines += [
            f"## Alert {position}",
            "",
            "```",
            explanation.to_summary(),
            "```",
            "",
            "**Your answers**",
            "",
            "1. Decision and confidence: ",
            "2. Reason: ",
            "3. Would you act on it? ",
            "",
            "---",
            "",
        ]

    lines += [
        "## Overall",
        "",
        "4. Did any alert look less reliable? Which, and what told you? ",
        "",
        "5. What did you not understand? ",
        "",
        "---",
        "",
        "# Scoring guidance — for the team, AFTER the reader has answered",
        "",
        "> Do not show this section to the reader beforehand.",
        "",
        "### What counts as a pass",
        "",
        "The gate (`TRD.md §9`, T3.3's VERIFY) is that the reader **correctly states why the flow",
        "was flagged**. Concretely, a pass means their answer to question 2 names at least one of",
        "the measurements the alert actually cited, and does not invent a reason the alert did not",
        "give. They do **not** need to know what the measurement means in networking terms —",
        "`Pkt_Size_Avg` naming a packet-size average is enough. Judgement about whether that",
        "*should* be suspicious is a networking question, not an explainability one.",
        "",
        "### The answer that matters most",
        "",
        "**Question 4.** One alert below is one the system itself does not stand behind: its",
        "`[UNVERIFIED]` line says removing the measurements it cites barely changes the decision, so",
        "the stated reason is not supported by the model. If the reader did not notice, **the",
        "labelling has failed**, and that is a more serious finding than any wording problem —",
        "it means an operator could act on a rationale the system knows to be unsupported.",
        "",
        "The fix in that case is presentational (make the banner louder, or withhold unverified",
        "explanations from the default view), not a change to the model. Record the outcome either",
        "way in `reports/t3_3_xai_evaluation.md`.",
        "",
        "### Two changes already made in response to this gate",
        "",
        "Both were defects a reader test would have surfaced, found before the test was run, and",
        "fixed rather than left for the reader to trip over:",
        "",
        "1. **Measurements are now described in words.** Alerts previously cited raw column names",
        "   such as `Init_Bwd_Win_Byts` and `Bwd_Seg_Size_Avg`. Those are *traceable* to the data",
        "   but not *comprehensible* to the intended reader, and the two had been conflated.",
        "   `src/xai/feature_glossary.py` now covers 100% of the 62 features the model can cite.",
        "2. **Evidence is split by direction.** Features are ranked by absolute influence, which is",
        "   correct and is what makes the shares sum to 100% — but it meant the top-ranked",
        "   measurement sometimes argued *against* its own verdict (12% of SHAP explanations, 88%",
        "   of LIME's). Alerts now separate 'what pointed to Attack' from 'what argued against it',",
        "   keeping every feature and its true direction while removing the contradiction.",
        "",
        "Question 6 exists to check whether the first change actually worked. If the reader still",
        "leans on the bracketed technical names, the descriptions are not doing their job.",
        "",
        "### Recording the result",
        "",
        "| | |",
        "|---|---|",
        "| Reader (name, role) | |",
        "| Date | |",
        "| Q2 correct for each alert? | |",
        "| Q4 — spotted the unverified alert? | |",
        "| Terms that blocked them (Q5) | |",
        "| **VERIFY passed?** | |",
        "",
        "Paste the completed table into `reports/t3_3_xai_evaluation.md` under a new",
        "'Reader test' heading. Until then, T3.3's VERIFY block is **incomplete** — the quantitative",
        "metrics in that report stand on their own, but they are not a substitute for this gate.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "t3_3_reader_test.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
