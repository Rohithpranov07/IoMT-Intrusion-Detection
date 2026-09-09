# NS-3 architecture inventory (T-N1)

**Task:** T-N1 · **Subject:** `scratch/hids-iomt-adaptive.cc` · **Spec:** `NS3-Simulation.md §D`

## 0. The premise this task was written on does not hold

T-N1 says: *"Read `scratch/hids-iomt-adaptive.cc` in full"*, and `NS3-Simulation.md §B.1` lists it,
plus two completed runs (`C_batch1`, `C_batched`), as already existing.

**No such file existed.** Searched before any code was written:

| Where | Result |
|---|---|
| Repository working tree | no `.cc` file, no `scratch/` directory |
| Repository git history (all branches, all commits) | no `.cc` ever added |
| `~/ns-3-dev/scratch/` | stock ns-3 files only (`scratch-simulator.cc`, `subdir/`, `nested-subdir/`) |
| `reports/ns3_runs/` | did not exist |

So there was nothing to inventory and nothing to re-run. The `C_batch1` / `C_batched` outputs quoted
in §B.1 could not be reproduced, confirmed, or contradicted, and **nothing in this repository is
presented as a reproduction of them**. This document therefore records what the simulation *is*, as
built, rather than what a prior one was.

Two other §B.1 details are also corrected by contact with the machine: NS-3 here is **`ns-3-dev`**,
not `ns-3.47`, and its Python bindings are **disabled** in this build — which happens to satisfy
`§A.3`'s C++-only requirement by construction rather than by discipline.

## 1. Node roles (T-N1 point 1) — the ambiguity is removed by construction

`§A.1`'s last rule flags `wearables / compromised / fog : 20 / 60 / 4` as ambiguous, because 60
"compromised" cannot be a subset of 20 "wearables". The simulation resolves this the only way the
arithmetic allows: **three disjoint node sets.**

| Label in code | Count | Meaning | Where |
|---|---:|---|---|
| `wearables` | 20 | legitimate IoMT patient devices; periodic telemetry | `main()`, `NodeContainer wearables` |
| `attackers` | 60 | attacker-bot nodes, **never a subset of the wearables** | `main()`, `NodeContainer attackers` |
| `fog` | 4 | fog nodes running the IDS application | `main()`, `NodeContainer fog` |
| `edge` | 1 | aggregation switch; carries no IDS logic | `main()`, `NodeContainer edge` |

Total leaf senders = `nWearables + nAttackers` = 80, printed in every run's summary under
`total leaf senders`. The word "compromised" is **not used anywhere** in the code or output: a
compromised device and an attacker-bot are different claims, and only the second is simulated.

Each wearable is assigned a clinical criticality class round-robin, so all four classes are
represented and the assignment is reproducible: `LIFE_CRITICAL` 0.60, `HIGH` 0.80, `STANDARD` 1.00,
`NON_CLINICAL` 1.30 — the weights frozen in `src/adaptive/threshold.py` (T3.4).

Attacker-bots are registered as `STANDARD` (1.00). This is deliberate and slightly against this
project's own interest: handing attackers the most sensitive class would inflate the criticality
rule's apparent detection rate for free.

## 2. Fog IDS service capacity (T-N1 point 2) — per-node *and* aggregate, both printed

`§B.2` records that an earlier run printed a capacity exactly equal to `1/latency` without saying
whether that was one node or four. This build prints both and labels them, so the question cannot
recur:

```
capacity PER FOG NODE  : 2000.0000 flows/s
capacity AGGREGATE     : 8000.0000 flows/s (4 nodes)
```

The model: each fog node is an independent **single-server queue** (`FogIds::m_queue`), serving one
flow every `idsLatencyUs` (`FogIds::ServiceComplete`). Per-node capacity is `1e6 / idsLatencyUs`;
aggregate is `nFog x` that. Leaves are pinned to fog node `index % nFog`, so load is spread evenly
by construction rather than by routing.

## 3. The threshold rules (T-N1 point 3) — what each one actually keys on

`FogIds::EffectiveThresholdMs` is the whole of it. All three rules judge the same observable — the
**inter-message gap** for a source — which is the quantity `PRD.md §2.1.4` objects to the base paper
fixing flat at τ = 500 ms.

| Rule | Threshold | Keys off | Content-aware? |
|---|---|---|---|
| `fixed` | `500 ms` | nothing | no |
| `ewma` | `500 ms · (1 − load)` or `500 ms · load` | **queue state** + per-source EWMA of gaps (α = 0.30) | no |
| `criticality` | `clamp(500 · w_crit · (1 + 0.30·(load − 0.5)), 50, 2000)` | device criticality (dominant) + queue state (±15%) | no |

**None of the three inspects packet contents.** That answers T-N1's point 3 directly and is the
premise of T-N2's finding: a rule that reads only arrival timing and queue state is detecting
*traffic shape*, and its relationship to whether traffic is hostile is an assumption, not a
measurement.

The `ewma` rule is **this project's own reconstruction** of a congestion-adaptive detector matching
the shape `§B.2` describes. It is not a transcription of anyone's published code, and no finding
about it is a finding about anyone else's implementation.

## 4. Provenance of `14542.7 µs` and `57.9 µs` (T-N1 point 4)

**Unresolved, and unresolvable from this repository.** T-N1 asks to check "run scripts, comments, or
git history for context"; there are none of any of the three — see §0. The two values survive in
this work **only as sweep points**, labelled as such, and every run stamps a
`latency provenance` string into its own output so a saved run can never be read back without it.

`docs/latency_provenance.md` carries the full rule. The short version: no run in this repository may
be labelled with a Raspberry Pi number (none exists) or the base paper's number (it publishes none).
