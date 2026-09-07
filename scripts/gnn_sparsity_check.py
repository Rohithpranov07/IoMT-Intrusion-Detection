"""GNN go/no-go sparsity analysis (Build-Instructions T2.7; TRD.md §3.4).

Constructs the communication graph IoTID20's flow records induce -- nodes = unique IPs, edges =
observed source/destination pairs weighted by flow count -- and reports the density and
connectivity statistics that decide whether a GNN branch is worth Phase 2/3 engineering time.

`PRD.md §9` and `TRD.md §3.4` pre-agree that a graph too sparse to carry neighbourhood structure
means the branch is CUT, and that this is not a project failure. This script produces the evidence;
`reports/gnn_go_nogo.md` records the decision.

Usage:
    .venv/bin/python scripts/gnn_sparsity_check.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import REPORTS_DIR  # noqa: E402
from src.preprocessing.clean import clean_iotid20, load_iotid20  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

#: A graph whose average degree falls below this carries little neighbourhood structure for a
#: graph convolution to aggregate over.
MIN_USEFUL_AVERAGE_DEGREE: float = 2.0
#: Above this share of degree-1 nodes, most nodes have exactly one neighbour, so message passing
#: reduces to copying that neighbour's features.
MAX_USEFUL_DEGREE_ONE_FRACTION: float = 0.75


def analyse_graph(meta: pd.DataFrame) -> dict[str, object]:
    """Compute density and connectivity statistics for the induced communication graph.

    Args:
        meta: metadata frame containing `Src_IP` and `Dst_IP`.

    Returns:
        Dictionary of graph statistics.
    """
    edges = (
        meta.groupby(["Src_IP", "Dst_IP"], observed=True).size().reset_index(name="flow_count")
    )
    nodes = pd.unique(pd.concat([edges["Src_IP"], edges["Dst_IP"]], ignore_index=True))

    n_nodes = len(nodes)
    n_edges = len(edges)
    # Undirected simple-graph density: 2E / (N(N-1)).
    density = (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    degree = (
        pd.concat([edges["Src_IP"], edges["Dst_IP"]], ignore_index=True)
        .value_counts()
        .reindex(nodes, fill_value=0)
    )

    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "density": float(density),
        "average_degree": float(degree.mean()),
        "median_degree": float(degree.median()),
        "max_degree": int(degree.max()),
        "degree_1_fraction": float((degree <= 1).mean()),
        "degree_2_or_less_fraction": float((degree <= 2).mean()),
        "isolated_fraction": float((degree == 0).mean()),
        "top_10_degree_share": float(degree.nlargest(10).sum() / degree.sum()),
        "n_source_ips": int(meta["Src_IP"].nunique()),
        "n_destination_ips": int(meta["Dst_IP"].nunique()),
        "degree_percentiles": {
            f"p{p}": float(np.percentile(degree, p)) for p in (50, 75, 90, 95, 99)
        },
    }


def decide(stats: dict[str, object]) -> tuple[str, list[str]]:
    """Apply the go/no-go criteria to the computed statistics.

    Args:
        stats: output of `analyse_graph`.

    Returns:
        Tuple of the decision ("go" or "no-go") and the reasons behind it.
    """
    reasons: list[str] = []

    if stats["average_degree"] < MIN_USEFUL_AVERAGE_DEGREE:
        reasons.append(
            f"average degree {stats['average_degree']:.2f} is below the "
            f"{MIN_USEFUL_AVERAGE_DEGREE} threshold for useful neighbourhood aggregation"
        )
    if stats["degree_1_fraction"] > MAX_USEFUL_DEGREE_ONE_FRACTION:
        reasons.append(
            f"{stats['degree_1_fraction']:.1%} of nodes have degree <= 1, above the "
            f"{MAX_USEFUL_DEGREE_ONE_FRACTION:.0%} threshold; message passing would mostly copy "
            "a single neighbour"
        )

    return ("no-go" if reasons else "go"), reasons


def main() -> None:
    """Run the analysis and write the statistics to reports/."""
    raw = load_iotid20()
    _, _, meta = clean_iotid20(raw, drop_duplicates=True)

    stats = analyse_graph(meta)
    decision, reasons = decide(stats)
    stats["decision"] = decision
    stats["reasons"] = reasons

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "gnn_graph_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"\nDECISION: {decision.upper()}")
    for reason in reasons:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
