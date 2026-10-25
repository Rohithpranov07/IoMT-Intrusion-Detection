"""Slide-scale charts (large type, off-white ground) for the Review 3 deck. Same data as the paper."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figs", "deck")
os.makedirs(FIG, exist_ok=True)
BG, INK, INK2, GRID = "#FAFAF7", "#17252F", "#4C5A63", "#E3E5E0"
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "B'hys": "#eda100", "B'dual": "#e87ba4", "B'cap": "#008300", "gate": "#6250d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "savefig.dpi": 150})
RES = os.path.join(ROOT, "results")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)


def e1():
    s = pd.read_csv(os.path.join(RES, "ns3", "E1_summary.csv"))
    sub = s[s.idsLatencyUs == 500].set_index("arch").reindex(["A", "B", "C"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (m, title, fmt) in zip(axes, [("attackersBlocked", "Attackers blocked (of 60)", "{:.0f}"),
                                          ("wearablesBlocked", "Wearables quarantined (of 20)", "{:.1f}"),
                                          ("fpAlerts", "False-positive alerts", "{:.1f}")]):
        y = sub[f"{m}_mean"].to_numpy()
        ax.bar(range(3), y, color=[COL[a] for a in "ABC"], width=0.6, edgecolor=BG, linewidth=2)
        ax.errorbar(range(3), y, yerr=sub[f"{m}_ci95"].fillna(0), fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
        ci = sub[f"{m}_ci95"].fillna(0).to_numpy()
        for k, v in enumerate(y):
            ax.text(k, v + ci[k] + max(y) * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK)
        ax.set_xticks(range(3), ["A flat", "B EWMA", "C criticality"])
        ax.set_ylim(0, max(y) * 1.2)
        ax.set_title(title, fontsize=16, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "deck_e1.png")


def latch2000():
    s = pd.read_csv(os.path.join(RES, "ns3", "E1_summary.csv"))
    arch = ["B", "B'hys", "B'dual", "B'cap"]
    lab = ["B EWMA", "B′ hysteresis", "B′ dual", "B′ capacity"]
    sub = s[s.idsLatencyUs == 2000].set_index("arch").reindex(arch)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, (m, title, fmt) in zip(axes, [("attackersBlocked", "Attackers blocked (of 60), 20 seeds", "{:.1f}"),
                                          ("appDelayMeanMs", "Wearable mean delay (ms)", "{:.0f}")]):
        y = sub[f"{m}_mean"].to_numpy()
        ax.bar(range(4), y, color=[COL[a] for a in arch], width=0.6, edgecolor=BG, linewidth=2)
        for k, v in enumerate(y):
            ax.text(k, v + max(y) * 0.02, f"{v:.2f}" if 0 < v < 1 else fmt.format(v), ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK)
        ax.set_xticks(range(4), lab)
        ax.set_ylim(0, max(y) * 1.2)
        ax.set_title(title, fontsize=16, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "deck_latch2000.png")


def funnel():
    d = pd.read_csv(os.path.join(RES, "phase2", "dataset_leakage.csv"))
    e = d[d.dataset == "Edge-IIoTset"]
    labels = ["all 61 raw\ncolumns", "minus\nidentifiers", "minus nonces\nand ports", "minus\npayloads", "modelling\nspace (30)"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(e))
    ax.bar(x, e.duplicate_pct, color="#2a78d6", width=0.62, edgecolor=BG, linewidth=2)
    for i, v in enumerate(e.duplicate_pct):
        ax.text(i, v + 2, f"{v:.2f}%", ha="center", fontsize=15, color=INK, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=13.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("exact duplicate rows (%)")
    ax.grid(axis="x", visible=False)
    save(fig, "deck_funnel.png")


def order():
    s = pd.read_csv(os.path.join(RES, "phase7", "multiseed_summary.csv"))
    w = s[s.dataset == "WUSTL-EHMS-2020"].set_index("regime")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    names = ["Our pipeline,\nsplit first", "Tarish 2026,\nsplit first", "Our pipeline,\nSMOTE first", "Hosain 2025,\nSMOTE first"]
    vals = [w.loc["HONEST", "f1_mean"], 0.6234, w.loc["LEAKY", "f1_mean"], 0.9912]
    errs = [w.loc["HONEST", "f1_ci95"], 0, w.loc["LEAKY", "f1_ci95"], 0]
    cols = ["#1baf7a", "#1baf7a", "#eb6834", "#eb6834"]
    ax.barh(range(4), vals, xerr=errs, color=cols, alpha=1, height=0.6, edgecolor=BG, linewidth=2, error_kw={"ecolor": INK})
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=15, color=INK, fontweight="bold")
    ax.set_yticks(range(4), names, fontsize=13.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 on WUSTL-EHMS-2020 (Attack positive)")
    ax.grid(axis="y", visible=False)
    save(fig, "deck_order.png")


def tradeoff():
    s = pd.read_csv(os.path.join(RES, "ns3", "E5_tradeoff_curve.csv"))
    two = pd.read_csv(os.path.join(RES, "ns3", "E6_two_stage_curve.csv"))
    fig, ax = plt.subplots(figsize=(10, 6))
    names = {"A": "A flat (base paper)", "B": "B EWMA", "C": "C criticality"}
    for a in ("A", "B", "C"):
        sub = s[s.arch == a].sort_values("baseMs")
        ax.plot(100 * sub.fp, 100 * sub.fn, marker="o", ms=6, lw=2.6, color=COL[a], label=names[a])
        op = sub[sub.baseMs == 500]
        ax.scatter(100 * op.fp, 100 * op.fn, s=170, facecolor=BG, edgecolor=COL[a], linewidth=3, zorder=5)
    sub = two[(two.branch == "transformer") & (two.stage2Mode == "independent")].sort_values("baseMs")
    ax.plot(100 * sub.fp, 100 * sub.fn, ls="--", lw=2.6, color=COL["gate"], marker="s", ms=6, label="C gated by transformer")
    ax.set_xlabel("wearables quarantined (%)")
    ax.set_ylabel("attackers missed (%)")
    ax.legend(fontsize=13.5, loc="upper right")
    save(fig, "deck_tradeoff.png")


def matrix():
    s = pd.read_csv(os.path.join(RES, "ns3", "E2_scenario_summary.csv"))
    sub = s[s.idsLatencyUs == 500]
    scen = ["baseline", "fog=1", "topo=random", "mix=icu", "mix=periphery"]
    lab = ["baseline", "1 fog node", "random\ntopology", "ICU-heavy\nward", "periphery-\nheavy ward"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    w = 0.26
    for k, a in enumerate(("A", "B", "C")):
        det, qua = [], []
        for sc in scen:
            r = sub[(sub.scenario == sc) & (sub.arch == a)].iloc[0]
            det.append(100 * r.attackersBlocked_mean / r.nAttackers_mean)
            qua.append(100 * r.wearablesBlocked_mean / r.nWearables_mean)
        x = np.arange(len(scen)) + (k - 1) * w
        axes[0].bar(x, det, width=w, color=COL[a], label={"A": "A flat", "B": "B EWMA", "C": "C criticality"}[a], edgecolor=BG, linewidth=1.5)
        axes[1].bar(x, qua, width=w, color=COL[a], edgecolor=BG, linewidth=1.5)
        for ax_, vals in ((axes[0], det), (axes[1], qua)):
            for xi, v in zip(x, vals):
                if v < 1:
                    ax_.text(xi, 2, "0", ha="center", va="bottom", fontsize=14, fontweight="bold", color=COL[a])
    for ax, t in zip(axes, ("Attackers blocked (%)", "Wearables quarantined (%)")):
        ax.set_xticks(range(len(scen)), lab, fontsize=12.5)
        ax.set_title(t, loc="left", fontsize=16, color=INK)
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, fontsize=14, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, "deck_matrix.png")


def phase():
    p = os.path.join(RES, "ns3", "E4_phase_cells.csv")
    if not os.path.exists(p):
        return
    g = pd.read_csv(p)
    sub = g[g.ewmaAlpha == 0.3]
    costs = sorted(sub.idsLatencyUs.unique())
    bots = sorted(sub.nAttackers.unique())
    pats = sorted(sub.blockAfter.unique())
    fig, axes = plt.subplots(1, len(pats), figsize=(16, 4.2), sharey=True)
    for ax, pt in zip(axes, pats):
        m = sub[sub.blockAfter == pt].pivot(index="idsLatencyUs", columns="nAttackers", values="latched").reindex(index=costs, columns=bots)
        from matplotlib.colors import ListedColormap
        ax.imshow(m.to_numpy(float), origin="lower", cmap=ListedColormap(["#DCE8E3", "#C4511F"]), vmin=0, vmax=1, aspect="auto")
        for c in (500, 2000):  # baseline operating points: 60 bots, Review 2 patience 3
            if pt == 3 and c in costs and 60 in bots:
                ax.add_patch(plt.Rectangle((bots.index(60) - 0.5, costs.index(c) - 0.5), 1, 1, fill=False, ec="#17252F", lw=2.5))
        ax.set_xticks(range(len(bots)), [f"{int(b)}" for b in bots], fontsize=12)
        ax.set_yticks(range(len(costs)), [f"{c:,.0f}" for c in costs], fontsize=12)
        ax.set_title(f"patience {int(pt)}", fontsize=15, color=INK)
        ax.set_xlabel("attacker-bots", fontsize=13)
        ax.grid(False)
    axes[0].set_ylabel("inspection cost (µs)", fontsize=13)
    save(fig, "deck_phase.png")


if __name__ == "__main__":
    for f in (e1, latch2000, funnel, order, tradeoff, matrix, phase):
        try:
            f()
            print(f.__name__, "ok")
        except Exception as ex:  # noqa: BLE001
            print(f.__name__, "FAILED", repr(ex))
