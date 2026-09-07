# GNN Branch — Go/No-Go Decision (T2.7)

**Decision: NO-GO. The GNN branch is CUT.**

**Task:** T2.7 · **Spec:** `TRD.md §3.4`, `PRD.md §5.2`, `PRD.md §9` (risk register)
**Evidence:** `scripts/gnn_sparsity_check.py`, raw output in `reports/gnn_graph_stats.json`
**Dataset:** IoTID20, cleaned and deduplicated (261,531 flow records)

---

## 1. Why this gate exists

`PRD.md §5.2` makes the GNN branch an **explicit stretch goal**, and `PRD.md §9`'s risk register
pre-agrees the exit condition:

> *"GNN needs topology data the flow-record datasets don't natively provide → Construct the
> communication graph from source/destination IP pairs in the flow records; if the induced graph is
> too sparse to be informative, cut the GNN branch — pre-agreed, not a failure."*

`TRD.md §3.4` requires this check to run **before** further GNN engineering time is spent. This
document is that check. The decision is recorded whichever way it goes, so that a later reader can
see it was made on evidence rather than on how the semester was going.

## 2. Graph construction

Exactly as `TRD.md §3.4` specifies:

- **Nodes** — unique IP addresses appearing as a source or destination in the flow records.
- **Edges** — observed (source, destination) pairs.
- **Edge weight** — number of flows between that pair.

## 3. Measured statistics

| Statistic | Value |
|---|---:|
| Nodes (unique IPs) | 58,281 |
| Edges (unique src→dst pairs) | 58,504 |
| **Graph density** | **0.0000344** (3.4 × 10⁻⁵) |
| **Average degree** | **2.01** |
| **Median degree** | **1.0** |
| Degree p75 / p90 / p95 / p99 | 1.0 / 1.0 / 1.0 / 1.0 |
| **Nodes with degree ≤ 1** | **99.68%** |
| Nodes with degree ≤ 2 | 99.92% |
| Isolated nodes (degree 0) | 0% |
| Maximum degree | 43,441 |
| Share of all degree held by the top 10 nodes | 50.0% |
| Distinct source IPs | 57,957 |
| Distinct destination IPs | 471 |

## 4. What the numbers mean

**The induced graph is a star, not a network.** 57,957 source IPs communicate with just 471
destinations, and 99.68% of all nodes have exactly one neighbour — the 99th percentile of the degree
distribution is 1. A single hub node reaches 43,441 others, and ten nodes account for half of all
edge endpoints.

This is not an incidental property of the sample; it is what the dataset *is*. IoTID20's attack
scenarios are scans and floods, which by their nature emit traffic from a large number of ephemeral,
frequently spoofed source addresses toward a handful of victim devices. The "topology" the flow
records induce is therefore a record of *who was attacked*, not of a communication structure with
neighbourhoods to learn from.

**Why that defeats a GNN specifically.** Graph convolution and graph attention both work by
aggregating a node's features over its neighbours. For 99.68% of these nodes there is exactly one
neighbour, so aggregation reduces to copying that neighbour's feature vector — the layer learns a
per-node transformation dressed up as message passing, and contributes nothing the CNN, BiLSTM, and
Transformer branches do not already extract from the flow features directly. The remaining 0.32% of
nodes are hubs whose neighbourhoods number in the tens of thousands, where aggregation averages away
any individual signal.

Both go/no-go criteria were evaluated:

| Criterion | Threshold | Measured | Passed? |
|---|---|---|---|
| Average degree | ≥ 2.0 | 2.01 | marginally yes |
| Nodes with degree ≤ 1 | ≤ 75% | 99.68% | **no** |

The average degree only clears its threshold **because of the hubs** — the mean is 2.01 while the
median is 1.0, which is the signature of a degree distribution so skewed that its mean describes no
actual node. Reporting "average degree 2.01, threshold met" without the median beside it would be
precisely the kind of selective statistic this project criticises the base paper for. The degree-1
criterion fails by a margin that leaves nothing to argue about.

## 5. Consequences (binding)

Per `Build-Instructions.md` §A.1 rule 4 and §C's failure-mode table, this decision is **binding**,
not advisory:

- `src/models/gnn_branch.py` **has not been created and must not be**.
- `src/models/fusion.py` has **no** GNN branch wired in. `tests/test_fusion.py::
  test_no_gnn_branch_is_wired_in` asserts both of these automatically, so the branch cannot quietly
  reappear in a later commit.
- `reports/ablation_study.md` (T4.4) ablates **three** branches, not four.
- Phase 3 proceeds directly; no time is spent on graph learning.
- PyTorch Geometric (`TRD.md §8`) is **not** installed and is not a project dependency.

## 6. This is a result, not a shortfall

`PRD.md §5.2` and `§9` committed in advance to cutting this branch if the graph proved sparse, and
the graph is sparse by two orders of magnitude. Recording that finding *is* the deliverable for
T2.7.

It is also worth stating positively in the Review 3 report: the flow-record datasets used across
this literature — IoTID20 and Edge-IIoTset both — do not carry the topology that graph-based IDS
proposals assume. Any published work claiming a GNN over IP-pair graphs from data of this shape
should be expected to show its degree distribution. That is a small contribution in its own right,
and it costs nothing now that the measurement exists.

## 7. Reproducing this

```bash
.venv/bin/python scripts/gnn_sparsity_check.py
```

Deterministic — the analysis involves no randomness. Raw output: `reports/gnn_graph_stats.json`.

**Open item for T3.6.** The same check should be re-run on Edge-IIoTset when that dataset is loaded.
If its graph turned out to be substantially denser, this decision would be worth revisiting — but
that would require a new go/no-go entry in this document and an update to
`docs/architecture_decision.md` §5, not a quiet re-addition of the branch.
