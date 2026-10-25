"""Aggregate every ns-3 experiment into tables, statistics and 300-DPI figures.

Reads runs/E*.txt (one RESULT line per run), writes results/ns3/*.csv, *.md and figs/*.png.
Re-runnable at any time; it uses whatever runs exist and reports n per cell, so partial data is
never mistaken for complete data.
Statistics: mean, sd, 95% CI (t, df = n-1) per configuration; paired Wilcoxon signed-rank tests
(pairs share rngRun, hence identical traffic realisations); Bonferroni across each family.
"""
import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import ARCH_NAME, ARCH_ORDER, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ns3")
FIG = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# Reference categorical palette, fixed order: A, B, C, B'hys, B'dual, B'cap
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "B'hys": "#eda100", "B'dual": "#e87ba4", "B'cap": "#008300"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight"})

METRICS = ["attackersBlocked", "wearablesBlocked", "fpAlerts", "fnAttackers", "legitPdr", "appDelayMeanMs",
           "appDelayP99Ms", "netPdr", "ctrlOverheadPct", "energyJ", "meanTimeToBlockS", "latched", "queueDrops"]


def exists(exp):
    return os.path.exists(os.path.join(ROOT, "runs", f"{exp}.txt"))


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan
    return tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def summarise(df, keys, metrics=METRICS):
    rows = []
    for k, g in df.groupby(keys):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["n"] = len(g)
        for m in metrics:
            if m in g:
                v = g[m].astype(float)
                r[f"{m}_mean"], r[f"{m}_sd"], r[f"{m}_ci95"] = v.mean(), v.std(ddof=1), ci(v)
        rows.append(r)
    return pd.DataFrame(rows)


def paired_tests(df, group_col, groups, metrics, by=None, family=""):
    out = []
    for key, sub in (df.groupby(by) if by else [((), df)]):
        for a, b in itertools.combinations(groups, 2):
            for m in metrics:
                x = sub[sub[group_col] == a].set_index("rngRun")[m]
                y = sub[sub[group_col] == b].set_index("rngRun")[m]
                idx = x.index.intersection(y.index)
                if len(idx) < 6:
                    continue
                d = (x.loc[idx] - y.loc[idx]).to_numpy(float)
                p = 1.0 if np.allclose(d, 0) else wilcoxon(x.loc[idx], y.loc[idx]).pvalue
                out.append({"family": family, "by": key, "a": a, "b": b, "metric": m, "n_pairs": len(idx),
                            "mean_a": x.loc[idx].mean(), "mean_b": y.loc[idx].mean(), "median_diff": float(np.median(d)), "p_raw": p})
    t = pd.DataFrame(out)
    if len(t):
        t["p_bonferroni"] = (t.p_raw * len(t)).clip(upper=1.0)
    return t


def fmt(m, s, digits=2):
    return f"{m:.{digits}f} ± {s:.{digits}f}"


# ---------------------------------------------------------------- E1 baseline
def e1():
    df = load("E1")
    df.to_csv(os.path.join(OUT, "E1_raw_runs.csv"), index=False)  # deliverable: 20 rows per configuration
    s = summarise(df, ["idsLatencyUs", "arch"])
    s.to_csv(os.path.join(OUT, "E1_summary.csv"), index=False)
    tests = paired_tests(df, "arch", ["A", "B", "C"], ["wearablesBlocked", "fpAlerts", "attackersBlocked", "legitPdr", "appDelayMeanMs", "energyJ"],
                         by="idsLatencyUs", family="E1 A/B/C")
    tests2 = paired_tests(df, "arch", ["B", "B'hys", "B'dual", "B'cap"], ["attackersBlocked", "wearablesBlocked", "legitPdr"],
                          by="idsLatencyUs", family="E1 B vs mitigations")
    tests2 = tests2[tests2.a == "B"]
    pd.concat([tests, tests2]).to_csv(os.path.join(OUT, "E1_wilcoxon.csv"), index=False)
    md = ["| Inspection cost | Arch | Attackers blocked /60 | Wearables quarantined /20 | FP alerts | Wearable PDR | App delay mean (ms) | App delay p99 (ms) | IDS energy (J) | Latched runs |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in s.sort_values(["idsLatencyUs", "arch"], key=lambda c: c.map({a: i for i, a in enumerate(ARCH_ORDER)}) if c.name == "arch" else c).iterrows():
        md.append(f"| {r.idsLatencyUs:.0f} µs | {ARCH_NAME[r.arch]} | {fmt(r.attackersBlocked_mean, r.attackersBlocked_sd, 1)} | {fmt(r.wearablesBlocked_mean, r.wearablesBlocked_sd, 1)} | "
                  f"{fmt(r.fpAlerts_mean, r.fpAlerts_sd, 1)} | {fmt(r.legitPdr_mean, r.legitPdr_sd, 3)} | {fmt(r.appDelayMeanMs_mean, r.appDelayMeanMs_sd, 1)} | "
                  f"{fmt(r.appDelayP99Ms_mean, r.appDelayP99Ms_sd, 0)} | {r.energyJ_mean:.0f} | {int(round(r.latched_mean * r.n))}/{r.n} |")
    open(os.path.join(OUT, "E1_table.md"), "w").write("\n".join(md) + "\n")

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharey="col")
    panels = [("attackersBlocked", "Attackers blocked (of 60)"), ("wearablesBlocked", "Wearables quarantined (of 20)"),
              ("legitPdr", "Wearable delivery ratio"), ("appDelayP99Ms", "Wearable p99 delay (ms)")]
    for row, lat in enumerate((500, 2000)):
        sub = s[s.idsLatencyUs == lat].set_index("arch").reindex(ARCH_ORDER)
        for col, (m, title) in enumerate(panels):
            ax = axes[row, col]
            y = sub[f"{m}_mean"].to_numpy()
            e = sub[f"{m}_ci95"].fillna(0).to_numpy()
            ax.bar(range(6), y, color=[COL[a] for a in ARCH_ORDER], width=0.72, edgecolor="white", linewidth=1)
            ax.errorbar(range(6), y, yerr=e, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            ax.set_xticks(range(6), ["A", "B", "C", "B'h", "B'd", "B'c"])
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(title, fontsize=8.5, color=INK)
            if col == 0:
                ax.set_ylabel(f"{lat} µs per flow", color=INK)
    fig.suptitle("Baseline scenario, 20 seeds per bar (mean, 95% CI). B latches off at 2000 µs; every mitigation restores detection.",
                 fontsize=8.5, color=INK2, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e1_baseline.png"))
    plt.close(fig)
    return s, tests


# ---------------------------------------------------------------- E2 scenario matrix
def e2():
    frames = []
    base = load("E1")
    for exp in ("E2a", "E2b"):
        if exists(exp):
            frames.append(load(exp))
    if not frames:
        return None
    df = pd.concat(frames)
    b = base.copy()
    b["scenario"] = "baseline"
    df = pd.concat([df, b])
    df["scenario"] = df["scenario"].fillna("baseline")
    s = summarise(df, ["idsLatencyUs", "scenario", "arch"], ["attackersBlocked", "wearablesBlocked", "fpAlerts", "legitPdr", "appDelayMeanMs", "latched", "nAttackers", "nWearables"])
    s.to_csv(os.path.join(OUT, "E2_scenario_summary.csv"), index=False)
    order = ["baseline", "wear=10", "wear=40", "wear=80", "bots=20", "bots=40", "bots=80", "bots=120", "fog=1", "fog=2", "fog=8",
             "topo=clustered", "topo=grid", "topo=random", "mix=icu", "mix=periphery"]
    for lat in sorted(s.idsLatencyUs.unique()):
        md = [f"Inspection cost {lat:.0f} µs per flow. Cells: mean ± sd (n). Wearables quarantined as % of wearables present; attackers blocked as % of attackers present.",
              "", "| Scenario | " + " | ".join(ARCH_NAME[a] for a in ARCH_ORDER) + " |", "|---|" + "---:|" * 6]
        sub = s[s.idsLatencyUs == lat]
        for sc in order:
            cells = []
            for a in ARCH_ORDER:
                r = sub[(sub.scenario == sc) & (sub.arch == a)]
                if not len(r):
                    cells.append("")
                    continue
                r = r.iloc[0]
                w = 100 * r.wearablesBlocked_mean / r.nWearables_mean
                at = 100 * r.attackersBlocked_mean / r.nAttackers_mean
                cells.append(f"PDR {r.legitPdr_mean:.2f} · quar {w:.0f}% · det {at:.0f}% · FP {r.fpAlerts_mean:.0f} ({int(r.n)})")
            md.append(f"| {sc} | " + " | ".join(cells) + " |")
        open(os.path.join(OUT, f"E2_table_{int(lat)}us.md"), "w").write("\n".join(md) + "\n")

    for lat in sorted(s.idsLatencyUs.unique()):
        sub = s[s.idsLatencyUs == lat]
        groups = [("wear", "Legitimate wearables", [10, 20, 40, 80]), ("bots", "Attacker-bots", [20, 40, 60, 80, 120]),
                  ("fog", "Fog nodes", [1, 2, 4, 8]), ("topo", "Topology", ["star", "clustered", "grid", "random"]),
                  ("mix", "Device mix", ["balanced", "icu", "periphery"])]
        base_label = {"wear": "20", "bots": "60", "fog": "4", "topo": "star", "mix": "balanced"}
        fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.6), sharey="row")
        for col, (g, title, levels) in enumerate(groups):
            for a in ARCH_ORDER:
                qa, da = [], []
                for lv in levels:
                    sc = "baseline" if str(lv) == base_label[g] else f"{g}={lv}"
                    r = sub[(sub.scenario == sc) & (sub.arch == a)]
                    qa.append(100 * r.wearablesBlocked_mean.iloc[0] / r.nWearables_mean.iloc[0] if len(r) else np.nan)
                    da.append(100 * r.attackersBlocked_mean.iloc[0] / r.nAttackers_mean.iloc[0] if len(r) else np.nan)
                x = range(len(levels))
                axes[0, col].plot(x, da, marker="o", ms=3.5, lw=1.4, color=COL[a], label=a)
                axes[1, col].plot(x, qa, marker="o", ms=3.5, lw=1.4, color=COL[a], label=a)
            for r_ in (0, 1):
                axes[r_, col].set_xticks(range(len(levels)), [str(v) for v in levels], fontsize=7, rotation=30 if g == "topo" else 0)
            axes[0, col].set_title(title, fontsize=8.5, color=INK)
        axes[0, 0].set_ylabel("Attackers blocked (%)")
        axes[1, 0].set_ylabel("Wearables quarantined (%)")
        axes[0, 0].set_ylim(-5, 105)
        axes[1, 0].set_ylim(-5, 105)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, [ARCH_NAME[x] for x in l], loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"One-factor-at-a-time scenario matrix at {lat:.0f} µs per flow (means over seeds)", fontsize=8.5, color=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig_e2_matrix_{int(lat)}us.png"))
        plt.close(fig)
    return s


# ---------------------------------------------------------------- E3 sensitivity (tornado)
def e3():
    if not exists("E3"):
        return None
    df = load("E3")
    base = load("E1")
    s = summarise(df, ["factor", "arch", "idsLatencyUs", "baseMs", "ewmaAlpha", "weightScheme", "blockAfter"],
                  ["attackersBlocked", "wearablesBlocked", "fpAlerts", "legitPdr", "latched"])
    s.to_csv(os.path.join(OUT, "E3_sensitivity_summary.csv"), index=False)
    # tornado: for each rule, range (min..max over levels incl. baseline) of an outcome per factor
    rows = []
    for arch, outcome in (("A", "wearablesBlocked"), ("B", "attackersBlocked"), ("B", "wearablesBlocked"), ("C", "wearablesBlocked"), ("C", "fpAlerts")):
        b500 = base[(base.arch == arch) & (base.idsLatencyUs == 500)][outcome].mean()
        b2000 = base[(base.arch == arch) & (base.idsLatencyUs == 2000)][outcome].mean()
        for f in ("tau0", "alpha", "weights", "cost", "patience"):
            sub = s[(s.factor == f) & (s.arch == arch)]
            if not len(sub):
                continue
            ref = b2000 if f == "patience" else b500
            vals = list(sub[f"{outcome}_mean"]) + [ref]
            if f == "alpha":
                sub5 = sub[sub.idsLatencyUs == 500]
                vals = list(sub5[f"{outcome}_mean"]) + [b500]
            rows.append({"arch": arch, "outcome": outcome, "factor": f, "low": min(vals), "high": max(vals), "baseline": ref, "range": max(vals) - min(vals)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "E3_tornado.csv"), index=False)
    panels = [(a, o) for a, o in (("A", "wearablesBlocked"), ("B", "wearablesBlocked"), ("B", "attackersBlocked"), ("C", "wearablesBlocked")) if len(t[(t.arch == a) & (t.outcome == o)])]
    oname = {"wearablesBlocked": "wearables quarantined (of 20)", "attackersBlocked": "attackers blocked (of 60)", "fpAlerts": "false-positive alerts"}
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(1.85 * len(panels), 2.3))
        axes = np.atleast_1d(axes)
        names = {"tau0": "base threshold τ0", "alpha": "EWMA α", "weights": "criticality weights", "cost": "inspection cost", "patience": "patience (blockAfter)"}
        for ax, (a, o) in zip(axes, panels):
            sub = t[(t.arch == a) & (t.outcome == o)].sort_values("range")
            y = range(len(sub))
            ax.barh(y, sub.high - sub.low, left=sub.low, color=COL[a], height=0.6, edgecolor="white")
            ax.axvline(sub.baseline.iloc[0], color=INK, lw=0.8, ls="--")
            ax.set_yticks(list(y), [names[f] for f in sub.factor], fontsize=7.5)
            ax.set_title(f"{ARCH_NAME[a]}\n{oname[o]}", fontsize=7.5, color=INK)
            ax.grid(axis="y", visible=False)
        fig.suptitle("Sensitivity: range of each outcome as one factor varies (dashed = baseline)", fontsize=8.5, color=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_e3_tornado.png"))
        plt.close(fig)
    return t


# ---------------------------------------------------------------- E4 latch-off phase diagram
def e4():
    if not exists("E4"):
        return None
    df = load("E4")
    # per-fog offered utilisation: (bots/fog x 200 pps + wearables/fog x 1/0.6 s) x inspection cost
    df["rho"] = ((df.nAttackers / df.nFog) * (1000 / 5.0) + (df.nWearables / df.nFog) * (1000 / 600.0)) * df.idsLatencyUs * 1e-6
    df.to_csv(os.path.join(OUT, "E4_raw_runs.csv"), index=False)
    g = df.groupby(["blockAfter", "ewmaAlpha", "idsLatencyUs", "nAttackers"]).agg(latched=("latched", "mean"), rho=("rho", "first"),
                                                                                    blocked=("attackersBlocked", "mean"), n=("latched", "size")).reset_index()
    g.to_csv(os.path.join(OUT, "E4_phase_cells.csv"), index=False)
    # boundary table: smallest rho that latches, per (patience, alpha)
    bt = []
    for (p, a), sub in g.groupby(["blockAfter", "ewmaAlpha"]):
        lat = sub[sub.latched >= 0.5]
        ok = sub[sub.latched < 0.5]
        bt.append({"blockAfter": p, "ewmaAlpha": a, "max_rho_functional": ok.rho.max() if len(ok) else np.nan,
                   "min_rho_latched": lat.rho.min() if len(lat) else np.nan, "cells_latched": int((sub.latched >= 0.5).sum()), "cells": len(sub)})
    pd.DataFrame(bt).to_csv(os.path.join(OUT, "E4_boundary.csv"), index=False)
    costs = sorted(g.idsLatencyUs.unique())
    bots = sorted(g.nAttackers.unique())
    pats = sorted(g.blockAfter.unique())
    alphas = sorted(g.ewmaAlpha.unique())
    fig, axes = plt.subplots(len(pats), len(alphas), figsize=(1.55 * len(alphas) + 0.8, 1.3 * len(pats) + 0.6), squeeze=False)
    for i, p in enumerate(pats):
        for j, a in enumerate(alphas):
            sub = g[(g.blockAfter == p) & (g.ewmaAlpha == a)].pivot(index="idsLatencyUs", columns="nAttackers", values="latched").reindex(index=costs, columns=bots)
            ax = axes[i, j]
            ax.imshow(sub.to_numpy(float), origin="lower", cmap="Oranges", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(bots)), [f"{int(b)}" for b in bots], fontsize=5.5)
            ax.set_yticks(range(len(costs)), [f"{c:g}" for c in costs], fontsize=5.5)
            ax.grid(False)
            if i == 0:
                ax.set_title(f"α = {a:g}", fontsize=7.5, color=INK)
            if j == 0:
                ax.set_ylabel(f"patience {int(p)}\ncost µs", fontsize=7)
            if i == len(pats) - 1:
                ax.set_xlabel("attacker-bots", fontsize=6.5)
    fig.suptitle("Latch-off region of the EWMA rule (dark = detector latched off: <50% of bots blocked, IDS queue overflowing)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e4_phase_diagram.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for k, p in enumerate(pats):
        sub = g[g.blockAfter == p].sort_values("rho")
        ax.scatter(sub.rho, sub.latched + (k - 2) * 0.015, s=9, color=list(COL.values())[k % 6], label=f"patience {int(p)}")
    ax.set_xscale("log")
    ax.axvline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("offered utilisation per fog node ρ (log)")
    ax.set_ylabel("fraction of runs latched")
    ax.legend(fontsize=6.5, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e4_latch_vs_rho.png"))
    plt.close(fig)
    return g


# ---------------------------------------------------------------- E5/E6/E7 trade-off and two-stage
def e5():
    if not exists("E5"):
        return None
    df = load("E5")
    df["fnRate"] = df.fnAttackers / df.nAttackers
    df["fpRate"] = df.wearablesBlocked / df.nWearables
    s = df.groupby(["arch", "baseMs"]).agg(fp=("fpRate", "mean"), fn=("fnRate", "mean"), fpA=("fpAlerts", "mean"), n=("fpRate", "size")).reset_index()
    two = None
    if exists("E6"):
        d6 = load("E6")
        d6["fnRate"] = d6.fnAttackers / d6.nAttackers
        d6["fpRate"] = d6.wearablesBlocked / d6.nWearables
        two = d6.groupby(["branch", "stage2Mode", "baseMs"]).agg(fp=("fpRate", "mean"), fn=("fnRate", "mean"), n=("fpRate", "size")).reset_index()
        two.to_csv(os.path.join(OUT, "E6_two_stage_curve.csv"), index=False)
    s.to_csv(os.path.join(OUT, "E5_tradeoff_curve.csv"), index=False)
    # Pareto frontier over every threshold-rule operating point
    pts = s[["fp", "fn"]].to_numpy()
    front = [i for i, (x, y) in enumerate(pts) if not any((px <= x and py <= y) and (px < x or py < y) for px, py in pts)]
    s.iloc[front].sort_values("fp").to_csv(os.path.join(OUT, "E5_pareto.csv"), index=False)
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    fr = s.iloc[front].sort_values("fp")
    for a in ("A", "B", "C"):
        sub = s[s.arch == a].sort_values("baseMs")
        ax.plot(100 * sub.fp, 100 * sub.fn, marker="o", ms=3, lw=1.3, color=COL[a], label=ARCH_NAME[a] + " (τ0 50 to 1500 ms)")
        op = sub[sub.baseMs == 500]
        ax.scatter(100 * op.fp, 100 * op.fn, s=60, facecolor="white", edgecolor=COL[a], linewidth=1.8, zorder=5)
    if two is not None:
        for br, ls in (("bilstm", "-"), ("transformer", "--")):
            sub = two[(two.branch == br) & (two.stage2Mode == "independent")].sort_values("baseMs")
            if len(sub):
                ax.plot(100 * sub.fp, 100 * sub.fn, ls=ls, lw=1.3, color="#6250d6", marker="s", ms=2.5, label=f"two-stage: C gated by {'BiLSTM' if br == 'bilstm' else 'transformer'}")
    ax.set_xlabel("Wearables quarantined (% of legitimate)")
    ax.set_ylabel("Attackers missed (% of attackers)")
    ax.legend(fontsize=6.3, loc="upper right")
    ax.set_title("Varying τ0 slides each rule along its curve; a second-stage\nclassifier moves the curve. Open circles: τ0 = 500 ms", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e5_tradeoff.png"))
    plt.close(fig)
    return s


def e7():
    if not exists("E7"):
        return None
    df = load("E7")
    base = load("E1")
    c = base[(base.arch == "C") & (base.idsLatencyUs == 500)].copy()
    c["branch"], c["stage2Mode"] = "none (C alone)", "-"
    allr = pd.concat([c, df])
    s = summarise(allr, ["branch", "stage2Mode"], ["attackersBlocked", "wearablesBlocked", "fpAlerts", "legitPdr", "meanTimeToBlockS"])
    cfp, cw = c.fpAlerts.mean(), c.wearablesBlocked.mean()
    s["fpAlerts_reduction_vs_C_pct"] = 100 * (1 - s.fpAlerts_mean / cfp)
    s["wearables_reduction_vs_C_pct"] = 100 * (1 - s.wearablesBlocked_mean / cw)
    a = base[(base.arch == "A") & (base.idsLatencyUs == 500)]
    s["fpAlerts_reduction_vs_A_pct"] = 100 * (1 - s.fpAlerts_mean / a.fpAlerts.mean())
    s["wearables_reduction_vs_A_pct"] = 100 * (1 - s.wearablesBlocked_mean / a.wearablesBlocked.mean())
    s.to_csv(os.path.join(OUT, "E7_two_stage_summary.csv"), index=False)
    return s


def e8():
    if not exists("E8"):
        return None
    df = load("E8")
    s = summarise(df, ["factor", "idsLatencyUs", "arch", "nAttackers"], ["attackersBlocked", "wearablesBlocked", "fpAlerts", "latched", "legitPdr", "appDelayP99Ms"])
    s.to_csv(os.path.join(OUT, "E8_measured_cost_summary.csv"), index=False)
    md = ["| Measured cost (source) | Rule | 20 bots | 40 bots | 60 bots | 80 bots |", "|---|---|---:|---:|---:|---:|"]
    for (f, lat, a), sub in s.groupby(["factor", "idsLatencyUs", "arch"]):
        cells = []
        for n in (20, 40, 60, 80):
            r = sub[sub.nAttackers == n]
            cells.append(f"{100 * r.attackersBlocked_mean.iloc[0] / n:.0f}% ({int(r.n.iloc[0])})" if len(r) else "")
        md.append(f"| {lat:.1f} µs ({f}) | {ARCH_NAME[a]} | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "E8_table5_measured.md"), "w").write("Attackers blocked, mean over seeds (n).\n\n" + "\n".join(md) + "\n")
    return s


if __name__ == "__main__":
    for f in (e1, e2, e3, e4, e5, e7, e8):
        try:
            r = f()
            print(f.__name__, "ok" if r is not None else "no data yet")
        except Exception as ex:  # noqa: BLE001
            print(f.__name__, "FAILED:", repr(ex))
