"""Phase 6.2 analysis: reader_study_responses.csv (exported by study.html) -> statistics + paper text.

Usage:  python3 reader_study/analyse_reader_study.py reader_study_responses.csv

Measures (per alert category: CERTIFIED / UNCERTIFIED / CONTRADICTORY):
  accuracy   q2_correct: reader names >= 1 measurement the alert cited FOR its verdict and picks no
             counter-evidence and no distractor (the pass rule of make_reader_test.py)
  decision   q1_correct (verdict read correctly), q1_confidence_correct (system confidence band)
  action     share Act / Escalate / Ignore
  confidence q4 self-confidence (1-7), trust q5 (1-7), time-to-decision (s)
Inference is within-participant: each participant contributes one mean per category, compared with
paired Wilcoxon signed-rank tests, Bonferroni-corrected across all tests run. Action counts by
category use a chi-square test of independence.
"""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wilcoxon

df = pd.read_csv(sys.argv[1], keep_default_na=False, na_values=[""])
df["time_s"] = df["time_ms"] / 1000.0
cats = [c for c in ("CERTIFIED", "UNCERTIFIED", "CONTRADICTORY") if c in set(df.category)]
n_part = df.participant.nunique()

print(f"# Human evaluation of explanation stability\n\nParticipants: {n_part}. Responses: {len(df)}.\n")
demo = df.drop_duplicates("participant")
print("Roles: " + ", ".join(f"{k} {v}" for k, v in demo.role.value_counts().items()))
print("Network experience: " + ", ".join(f"{k} {v}" for k, v in demo.net_experience.value_counts().items()))
print(f"ML familiarity median {demo.ml_familiarity.median():.0f}/7, security familiarity median {demo.security_familiarity.median():.0f}/7\n")

per = df.groupby(["participant", "category"]).agg(acc=("q2_correct", "mean"), verdict=("q1_correct", "mean"),
                                                   conf_read=("q1_confidence_correct", "mean"), selfconf=("q4_self_confidence", "mean"),
                                                   trust=("q5_trust", "mean"), time_s=("time_s", "median")).reset_index()
rows = []
for c in cats:
    g = per[per.category == c]
    acts = df[df.category == c].q3_action.value_counts(normalize=True)
    rows.append({"category": c, "alerts_per_reader": int(df[df.category == c].groupby("participant").size().median()),
                 "reason_correct_%": 100 * g.acc.mean(), "sd": 100 * g.acc.std(ddof=1),
                 "verdict_read_%": 100 * g.verdict.mean(), "confidence_band_read_%": 100 * g.conf_read.mean(),
                 "self_confidence_1to7": g.selfconf.mean(), "trust_1to7": g.trust.mean(),
                 "median_time_s": g.time_s.median(),
                 "act_%": 100 * acts.get("act", 0), "escalate_%": 100 * acts.get("escalate", 0), "ignore_%": 100 * acts.get("ignore", 0)})
tab = pd.DataFrame(rows)
print(tab.round(2).to_markdown(index=False))

tests = []
for a, b in itertools.combinations(cats, 2):
    wide = per.pivot(index="participant", columns="category")
    for m in ("acc", "trust", "selfconf", "time_s"):
        x, y = wide[m][a].dropna(), wide[m][b].dropna()
        idx = x.index.intersection(y.index)
        x, y = x.loc[idx], y.loc[idx]
        p = 1.0 if np.allclose(x, y) else wilcoxon(x, y).pvalue
        tests.append({"comparison": f"{a} vs {b}", "measure": m, "n": len(idx), "median_diff": float(np.median(x - y)), "p_raw": p})
t = pd.DataFrame(tests)
t["p_bonferroni"] = (t.p_raw * len(t)).clip(upper=1.0)
print("\n## Paired Wilcoxon tests (participant means), Bonferroni over", len(t), "tests\n")
print(t.round(4).to_markdown(index=False))

ct = pd.crosstab(df.category, df.q3_action)
chi = chi2_contingency(ct)
print(f"\n## Action by category\n\n{ct.to_markdown()}\n\nchi-square = {chi.statistic:.2f}, dof = {chi.dof}, p = {chi.pvalue:.4g}")
post = demo[["post_t1", "post_t2", "post_t3"]].astype(float)
print(f"\nPost-study agreement (1-7, mean): notes helped {post.post_t1.mean():.2f}; could tell when unsure {post.post_t2.mean():.2f}; would rely on them {post.post_t3.mean():.2f}")
print("\nFree-text answers to 'which alerts looked less reliable' must be coded by two team members independently "
      "(spotted the UNVERIFIED banner: yes / no / partly) and agreement reported as Cohen's kappa before quoting a rate.")
