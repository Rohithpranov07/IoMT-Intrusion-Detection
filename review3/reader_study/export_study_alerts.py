"""Phase 6: export the 20 real alerts for the human reader study.

Run ON THE MACHINE THAT HOLDS artifacts/ (the trained ensemble from notebook 03), from the repo
root, the same way scripts/make_reader_test.py is run:

    .venv/bin/python reader_study/export_study_alerts.py --out reader_study/alerts_20.json

It scans flagged test windows, explains and certifies each one exactly as the operator view does,
and keeps a stratified set:
    CERTIFIED       certification.trusted is True and the top-ranked feature supports the verdict
    UNCERTIFIED     certification.trusted is False
    CONTRADICTORY   the top-ranked (largest |attribution|) feature argues AGAINST the verdict
Default quotas 8 / 6 / 6. Every alert is a real model output; nothing is written by hand. The
JSON is loaded into reader_study/study.html with its "Load alert set" button.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from src.config import ARTIFACTS_DIR, POSITIVE_LABEL  # noqa: E402
from src.models.fusion import confidence_weighted_fusion  # noqa: E402
from src.models.train_utils import load_branch  # noqa: E402
from src.xai.feature_glossary import describe_feature  # noqa: E402
from src.xai.metrics import certify_explanation  # noqa: E402
from src.xai.shap_explainer import EnsembleShapExplainer  # noqa: E402


def category(explanation) -> str:
    top = explanation.top_features[0]
    if top.attribution <= 0:  # top-ranked feature pushes toward Normal, but the verdict is Attack
        return "CONTRADICTORY"
    return "CERTIFIED" if explanation.certification.trusted else "UNCERTIFIED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reader_study/alerts_20.json")
    ap.add_argument("--quota", default="8,6,6", help="CERTIFIED,UNCERTIFIED,CONTRADICTORY")
    ap.add_argument("--search-limit", type=int, default=400)
    a = ap.parse_args()
    quota = dict(zip(("CERTIFIED", "UNCERTIFIED", "CONTRADICTORY"), map(int, a.quota.split(","))))

    data = np.load(ARTIFACTS_DIR / "ensemble_artifacts.npz")
    meta = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text(encoding="utf-8"))
    names, branches = meta["feature_names"], meta["branch_names"]
    models = {b: load_branch(ARTIFACTS_DIR / "models" / f"{b}.keras") for b in branches}
    X_test, y_test, background = data["X_test"], data["y_test"], data["X_background"]

    def predict_fn(w):
        return confidence_weighted_fusion([models[b].predict(w, verbose=0) for b in branches], branches).probabilities

    explainer = EnsembleShapExplainer(models, names, background, background_size=100)
    flagged = np.flatnonzero(data["fused_predictions"] == POSITIVE_LABEL)[: a.search_limit]
    kept = {k: [] for k in quota}
    for idx in flagged:
        e = explainer.explain(X_test[idx])
        e.certification = certify_explanation(e, predict_fn, X_test[idx], background)
        c = category(e)
        if len(kept[c]) >= quota[c]:
            continue
        kept[c].append({
            "id": f"{c[:3]}-{int(idx)}",
            "category": c,
            "test_index": int(idx),
            "true_label": "Attack" if int(y_test[idx]) == POSITIVE_LABEL else "Normal",
            "confidence": float(e.confidence),
            "certified": bool(e.certification.trusted),
            "summary": e.to_summary(),
            "features": [{"rank": f.rank, "column": f.feature_name, "description": describe_feature(f.feature_name),
                          "share": float(f.share_of_total), "direction": "attack" if f.attribution > 0 else "normal",
                          "peak_record": int(f.peak_timestep) + 1} for f in e.top_features],
        })
        if all(len(kept[k]) >= quota[k] for k in quota):
            break
    alerts = [x for k in quota for x in kept[k]]
    short = {k: quota[k] - len(kept[k]) for k in quota if len(kept[k]) < quota[k]}
    out = {"source": "real model outputs from artifacts/ (notebook 03 ensemble); SHAP + per-alert certification",
           "counts": {k: len(v) for k, v in kept.items()}, "shortfall": short, "alerts": alerts}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(alerts)} alerts to {a.out}; counts {out['counts']}; shortfall {short or 'none'}")


if __name__ == "__main__":
    main()
