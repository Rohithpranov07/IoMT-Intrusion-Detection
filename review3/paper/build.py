"""Assemble the manuscript: fill generated tables and result snippets, then compile with xelatex.

Placeholders %%NAME%% in sec*.tex are replaced by paper/snippets/NAME.tex when it exists, or by a
red PENDING marker when it does not, so an unfinished experiment is always visible in the PDF.
E1TABLE and E7TABLE are generated here from results/ns3/*.csv.
"""
from decimal import Decimal, ROUND_HALF_UP
import os
import re
import shutil
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "ns3")
SNIP = os.path.join(HERE, "snippets")
BUILD = os.path.join(HERE, "build")
os.makedirs(SNIP, exist_ok=True)
NAMES = {"A": "A flat", "B": "B EWMA", "C": "C criticality", "B'hys": "B$'$ hysteresis", "B'dual": "B$'$ dual", "B'cap": "B$'$ capacity"}
ORDER = ["A", "B", "C", "B'hys", "B'dual", "B'cap"]



def h(x, n):
    """Round half up, as printed values are read by people."""
    q = Decimal(1).scaleb(-n)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))

def e1table():
    s = pd.read_csv(os.path.join(RES, "E1_summary.csv"))
    rows = []
    for lat in (500, 2000):
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{lat:,} µs per flow}}}}\\\\")
        for a in ORDER:
            r = s[(s.idsLatencyUs == lat) & (s.arch == a)].iloc[0]
            rows.append(f"{NAMES[a]} & {h(r.attackersBlocked_mean,1)} & {h(r.wearablesBlocked_mean,2)} $\\pm$ {h(r.wearablesBlocked_sd,2)} & "
                        f"{h(r.fpAlerts_mean,1)} $\\pm$ {h(r.fpAlerts_sd,1)} & {h(r.legitPdr_mean,3)} & {h(r.appDelayMeanMs_mean,1)} & "
                        f"{h(r.energyJ_mean,0)} & {int(round(r.latched_mean * r.n))}/{int(r.n)} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    return ("\\begin{table*}[t]\n\\caption{Baseline scenario (20 wearables, 60 bots, 4 fog nodes, star), 20 seeds per row, mean (and sd where it is informative). "
            "Bots: of 60 blocked. Quar.: wearables quarantined of 20. PDR: wearable delivery ratio. Delay: wearable mean, including inspection queueing. "
            "Energy: modelled fog inspection energy over 30\\,s. Latched: runs with $<$50\\,\\% of bots blocked and queue overflow.}\n\\label{tab:e1}\n\\centering\\footnotesize\n"
            "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n\\textbf{Rule} & \\textbf{Bots} & \\textbf{Quar.} & \\textbf{FP alerts} & \\textbf{PDR} & \\textbf{Delay (ms)} & \\textbf{Energy (J)} & \\textbf{Latched} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def e7table():
    s = pd.read_csv(os.path.join(RES, "E7_two_stage_summary.csv"))
    lab = {"none (C alone)": "C alone", "bilstm": "C + BiLSTM", "transformer": "C + transformer", "ensemble": "C + ensemble"}
    rows = []
    for br in ("none (C alone)", "bilstm", "transformer", "ensemble"):
        for mode in ("-", "independent", "correlated"):
            r = s[(s.branch == br) & (s.stage2Mode == mode)]
            if not len(r):
                continue
            r = r.iloc[0]
            m = "" if mode == "-" else (" (indep.)" if mode == "independent" else " (per device)")
            rows.append(f"{lab[br]}{m} & {h(r.attackersBlocked_mean,2)} & {h(r.wearablesBlocked_mean,2)} & {h(r.fpAlerts_mean,1)} & "
                        f"{("+" if r.fpAlerts_reduction_vs_C_pct>=0 else "")+h(r.fpAlerts_reduction_vs_C_pct,1)} & {("+" if r.wearables_reduction_vs_C_pct>=0 else "")+h(r.wearables_reduction_vs_C_pct,1)} & {h(r.fpAlerts_reduction_vs_A_pct,1)} \\\\")
    return ("\\begin{table}[t]\n\\caption{Two-stage rule in the baseline scenario, 500\\,µs, 20 seeds. Reductions in \\%; positive is fewer.}\n\\label{tab:twostage}\n"
            "\\centering\\footnotesize\n\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n& \\textbf{Bots} & \\textbf{Quar.} & \\textbf{FP} & \\textbf{$\\Delta$FP} & \\textbf{$\\Delta$Quar.} & \\textbf{$\\Delta$FP} \\\\\n"
            "& /60 & /20 & alerts & vs C & vs C & vs A \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


GEN = {"E1TABLE": e1table, "E7TABLE": e7table}


def fill(text):
    def rep(m):
        k = m.group(1)
        if k in GEN:
            try:
                return GEN[k]()
            except Exception as ex:  # noqa: BLE001
                return f"\\pending{{{k} could not be generated: {type(ex).__name__}}}"
        p = os.path.join(SNIP, f"{k}.tex")
        return open(p).read() if os.path.exists(p) else f"\\pending{{{k}: experiment still running}}"
    return re.sub(r"%%([A-Z0-9]+)%%", rep, text)


def main():
    if os.path.exists(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(BUILD)
    for f in os.listdir(HERE):
        if f.endswith(".tex"):
            open(os.path.join(BUILD, f), "w").write(fill(open(os.path.join(HERE, f)).read()))
    for _ in range(2):
        p = subprocess.run(["xelatex", "-interaction=nonstopmode", "paper_r3.tex"], cwd=BUILD, capture_output=True, text=True)
    log = open(os.path.join(BUILD, "paper_r3.log"), errors="ignore").read()
    errs = [l for l in log.splitlines() if l.startswith("!")]
    pend = sum(open(os.path.join(BUILD, f)).read().count("\\pending{") for f in os.listdir(BUILD) if f.endswith(".tex"))
    print("errors:", errs[:10] or "none", "| pending markers:", pend, "| pdf:", os.path.exists(os.path.join(BUILD, "paper_r3.pdf")))


if __name__ == "__main__":
    main()
