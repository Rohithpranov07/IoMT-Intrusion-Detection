# NS3-Simulation.md — Network-Level Validation Layer (Phase 4 Deployment Demo)

> Companion to `PRD.md`, `TRD.md`, and `Build-Instructions.md`. This document governs the NS-3 simulation work that was decided to fold into **Phase 4 as the deployment demo** (replacing/extending the plain Pi-only latency benchmark, `TRD.md §7`). It picks up from work already started (`scratch/hids-iomt-adaptive.cc`, architectures `C_batch1` / `C_batched`) and turns it into a complete, reproducible experiment suite for Review 3.
>
> **Note on tooling:** the team's earlier decision was Mininet-folded-into-Phase-4. In practice, the simulation work already underway uses **NS-3** (`ns-3.47`, C++ scratch scripts). This document formalizes NS-3 as the actual tool in use — update `TRD.md §8`'s software stack and `Build-Instructions.md §B.3` to reflect this the next time those documents are touched, so they don't silently disagree with what's actually built.
>
> If this document and `TRD.md` disagree on a technical detail, **the TRD wins** — update this document to match.

---

## §A — Operating Contract (paste into `CLAUDE.md` alongside `Build-Instructions.md §A`)

### A.1 Anti-hallucination rules (specific to simulation work)
- **Never invent a CNN-LSTM/ensemble latency number for the base paper's side of a comparison.** The base paper never reports per-flow inference latency anywhere (`PRD.md §2.2`) — there is no "the paper's number" to plug into `--idsLatencyUs`. Any run representing "the base paper's architecture" must either (a) use this project's own measured latency for an equivalently-shaped model as an honest proxy, clearly labeled as a proxy, or (b) be explicitly framed as "an assumed/swept value, since the paper reports none" — never presented as if it were extracted from the paper.
- **This project's own architecture's latency must come from `T4.1`–`T4.3`'s real measurements, not a placeholder.** Once the trained CNN-BiLSTM-Transformer ensemble has a real measured Raspberry Pi inference latency (from `Build-Instructions.md T4.3`), that number — not an arbitrary sweep value — is what represents "our architecture" in the simulation. Sweep values remain useful for sensitivity analysis (§D, T-N6) but the headline comparison run must use the real number.
- **Never report a zero-detection or zero-alert result as a validated finding without root-causing the mechanism first.** The existing `C_batched` run (57.9µs/flow, 0 alerts, 0/60 blocked) is exactly this case — see T-N2, which is a blocking task. Do not proceed to the full experiment matrix (T-N6) until T-N2's root cause is understood and documented.
- **Never conflate a simulated network-level number with a physically measured Pi number.** NS-3's `mean end-to-end delay`, `jitter`, `throughput`, etc. are simulation outputs — they model what *would* happen at network scale, they are not a replacement for `T4.3`'s actual hardware benchmark. Every report/table must state whether a number came from the NS-3 model or from the physical Pi, and never blend them without saying so.
- **Never silently change the topology's meaning.** `wearables / compromised / fog : 20 / 60 / 4` is ambiguous as written (60 "compromised" can't be a subset of 20 "wearables"). Confirm and document explicitly whether this is `wearables (legitimate) / attacker-bot nodes / fog nodes` before running further experiments — see T-N1.

### A.2 Anti-drift rules
Each task in §D lists an explicit **Files** line. Do not edit files outside that list for that task.

### A.3 Quality gates
- Every scratch `.cc` file must have a header comment stating: the topology (node counts and roles), the threshold rule in use, and where each input latency value came from (measured, swept, or a documented proxy).
- All simulations must be pure C++ NS-3 scratch scripts, **not Python bindings** — `cppyy`-based Python bindings are broken on Apple Silicon Macs (confirmed via NS-3's own installation docs), and at least one team member is on Apple Silicon, so a Python-bound script would not be portable across the team.
- Every run's raw output must be saved (e.g., `tee`'d to `reports/ns3_runs/<label>.txt`, matching what's already being done) before any aggregation/summary is written — the summary must always be re-derivable from a saved raw run.
- Deterministic: fix the NS-3 RNG seed/run-number for any result that goes into a report, so a teammate can reproduce it exactly.

### A.4 Working method
1. Read `PRD.md`, `TRD.md`, and `Build-Instructions.md` in full, then this document.
2. Do T-N1 and T-N2 first, in order, before anything else in this document — they are blocking (inventory the existing script, then resolve the zero-detection anomaly). Do not run the full experiment matrix (T-N6) on top of an unexplained bug.
3. After each task, run the resulting scratch script and confirm its output matches the VERIFY block before moving on.
4. Commit per task with a conventional-commit message.

### A.5 Definition of done
Every gate in §F is checked, the zero-detection anomaly from the existing `C_batched` run is either fixed or explained and documented as a genuine finding (not left as an open bug), and every report clearly distinguishes simulated numbers from physically measured ones.

---

## §B — Canonical Spec Index & Confirmed Facts

### B.1 What already exists (as of the runs shown to Claude)
| Artifact | Status |
|---|---|
| `scratch/hids-iomt-adaptive.cc` | Exists and runs; produces the `C_batch1` and `C_batched` output format shown below |
| `C_batch1` run (`idsLatencyUs=14542.7`, `nCompromised=60`) | 1,928 alerts raised, 8/60 nodes blocked, 94.62% packet loss |
| `C_batched` run (`idsLatencyUs=57.9`, `nCompromised=60`) | **0 alerts raised, 0/60 blocked** — unexplained, flagged as a blocking issue (T-N2) |
| Consistency check: service capacity | `capacity = 1/latency` holds exactly in both runs (68.763/s ↔ 14542.7µs; 17271.2/s ↔ 57.9µs) |
| Consistency check: loss + delivery | Sums to 100% in both runs (5.3788+94.6212; 13.6732+86.3268) |

### B.2 Open items to resolve (do not guess these — see T-N1, T-N2, T-N3)
| Item | Why it's open |
|---|---|
| Whether `compromised` in the topology line means attacker-bot nodes distinct from the 20 wearables, or something else | The numbers as printed (20 wearables, 60 compromised) don't compose as a subset — needs confirming against the `.cc` source, not assumed |
| Whether "fog IDS service capacity" is per-node or aggregate across the 4 fog nodes | It currently matches `1/latency` exactly, which reads as single-node capacity |
| Source of the `14542.7µs` and `57.9µs` latency values | Unknown whether these are this project's own measured numbers, swept/synthetic values, or something else — must be confirmed and, going forward, replaced with real `T4.3` measurements for the headline comparison (A.1) |
| Mechanism behind `C_batched`'s zero detections | Leading hypothesis: the `C1 adaptive EWMA` rule may key off arrival-timing/queue-state rather than message content, so a fog node fast enough to avoid queue buildup never produces the timing signature the rule reacts to — **must be confirmed in the `.cc` source before treating this as a real finding** |

## §C — Failure-Mode Table

| Failure mode | Why it matters | Guarded by |
|---|---|---|
| Full experiment matrix run before the zero-detection bug is understood | Every subsequent result could be silently affected by the same bug | A.4, T-N2 is blocking |
| Base paper's latency represented by an invented number | Fabricates a "paper's number" that doesn't exist (`PRD.md §2.2` already established the paper reports none) | A.1, T-N3 |
| NS-3 simulated numbers reported as if they were the real Pi hardware measurement | Misrepresents what was actually measured vs. modeled | A.1, every report task |
| Python-bound NS-3 script written by one teammate, unusable by the Apple-Silicon teammate | Breaks reproducibility across the team | A.3 |
| Topology semantics (wearables vs. compromised vs. fog) left ambiguous in reports | A reader can't tell what was actually simulated | T-N1 |

---

## §D — Phased Tasks

### T-N1 — Inventory the existing scratch script (blocking, do first)
**Files:** none modified; produces `docs/ns3_architecture_inventory.md`
**Prompt:**
> Read `scratch/hids-iomt-adaptive.cc` in full. Document, in `docs/ns3_architecture_inventory.md`: (1) the exact meaning of the `wearables`/`compromised`/`fog` node counts — confirm whether "compromised" nodes are attacker-bots distinct from the wearables or something else, and rename the printed labels in the `.cc` file if they're misleading (e.g., to `wearables / attacker-bots / fog`); (2) whether "fog IDS service capacity" is computed per-node or aggregated across the 4 fog nodes, and fix the label if it's ambiguous; (3) the exact formula behind the `C1 adaptive EWMA` threshold rule, in particular whether it evaluates arrival-timing/queue state, packet content, or both; (4) where the `--idsLatencyUs` values used in existing runs (14542.7, 57.9) originated — check run scripts, comments, or git history for context. Do not change simulation *behavior* in this task, only labels/comments and the inventory doc.
**VERIFY:** `docs/ns3_architecture_inventory.md` answers all four points with direct references to line numbers/functions in `hids-iomt-adaptive.cc`, not guesses.

### T-N2 — Root-cause the zero-detection anomaly (blocking)
**Files:** `scratch/hids-iomt-adaptive.cc` (only if a real bug is found and fixed), `docs/zero_detection_investigation.md`
**Prompt:**
> Using T-N1's finding on the EWMA rule's mechanism, determine why `C_batched` (57.9µs/flow) produced 0 alerts / 0 blocked while `C_batch1` (14542.7µs/flow) produced 1,928 alerts / 8 blocked, with the same 60 attacker nodes present in both. Test the leading hypothesis first: if the EWMA/τ=500ms rule reacts to inter-arrival timing/queue buildup rather than message content, a faster fog server may never let a queue form long enough to trip the threshold. Confirm this (or find the actual mechanism) by instrumenting the `.cc` script to log the EWMA statistic's value over time in both runs and comparing. Write `docs/zero_detection_investigation.md` stating the confirmed mechanism. **If it's a genuine design property** (fast-enough infrastructure defeats a timing-based detector) — that is a legitimate, reportable finding (arguably a fifth objection: the base paper's adaptive rule is congestion-based, not attack-based) and should be written up as such, not silently normalized away. **If it's a bug** (e.g., an off-by-one in the ready-interval window, or the EWMA never being updated when latency is below some implicit threshold) — fix it in `hids-iomt-adaptive.cc` and re-run both `C_batch1` and `C_batched` to confirm the fix produces sane, non-zero detection in the fast case too.
**VERIFY:** `docs/zero_detection_investigation.md` states a confirmed mechanism (not a hypothesis) backed by the instrumented log data; if a code fix was made, both architectures re-run and produce plausible (non-zero, unless independently justified) detection numbers.

### T-N3 — Wire in real measured latency (depends on `Build-Instructions.md` T4.1–T4.3 being complete)
**Files:** `scratch/hids-iomt-adaptive.cc` or a new run-config wrapper, `docs/latency_provenance.md`
**Prompt:**
> Replace the ad hoc `--idsLatencyUs` sweep values used so far with two clearly labeled cases for the headline comparison: (1) **this project's own architecture** — the actual measured per-sequence inference latency from `Build-Instructions.md T4.3`'s real Raspberry Pi benchmark; (2) **the base paper's architecture, as a labeled proxy** — since the base paper reports no per-flow latency number (`PRD.md §2.2`), do not invent one; instead, either measure an equivalently-shaped CNN-LSTM model's latency yourselves and label it "proxy — base paper reports no comparable number," or omit this comparison arm entirely and say why. Write `docs/latency_provenance.md` stating exactly where every latency number used from this point forward comes from. Retain the ability to run arbitrary swept latency values for sensitivity analysis (T-N6), but the two headline cases must use real/honestly-labeled numbers only.
**VERIFY:** `docs/latency_provenance.md` traces every latency value used in any report-facing run to either a real T4.3 measurement or an explicitly labeled proxy/assumption — no unlabeled numbers.

### T-N4 — Implement this project's adaptive threshold as an alternate rule (fixes Contribution 4, `TRD.md §5.1`)
**Files:** `scratch/hids-iomt-adaptive.cc` (add a new threshold-rule branch alongside the existing `C1 adaptive EWMA`), `docs/threshold_rules.md`
**Prompt:**
> Add a second threshold rule implementing `TRD.md §5.1`'s criticality/context-weighted formula (`threshold = base_threshold * criticality_weight * context_factor`, using the exact weights decided in `Build-Instructions.md T3.4`), as a selectable alternative to the existing fixed-τ and `C1 adaptive EWMA` rules. Add a third architecture label (e.g., `C_criticality`) that runs the same topology/attack scenario with this new rule. Document all three rules side by side in `docs/threshold_rules.md`.
**VERIFY:** Running the same attack scenario under fixed-τ, `C1 adaptive EWMA`, and the new criticality-weighted rule produces three distinct alert/block outcomes, demonstrating the new rule actually changes behavior (not a no-op).

### T-N5 — Incremental learning scenario (fixes the other half of Contribution 4, `TRD.md §5.2`)
**Files:** `scratch/hids-iomt-adaptive.cc` (or a new scratch script `hids-iomt-incremental.cc`), `docs/incremental_scenario.md`
**Prompt:**
> Design a scenario that simulates a novel attack type appearing partway through the simulation (`simTime` split into a "before" and "after" phase), where the detection-probability/latency parameters change at the midpoint to represent an incrementally-updated model (per `Build-Instructions.md T3.5`'s before/after evaluation pattern, translated into network-simulation terms). Report alert/block statistics separately for the before and after phases. Document exactly what parameter change represents "the incremental update" and why, in `docs/incremental_scenario.md` — NS-3 cannot literally retrain a model, so this is a stand-in that must be clearly labeled as such.
**VERIFY:** `docs/incremental_scenario.md` clearly labels the incremental-update stand-in mechanism; before/after phase statistics are reported separately and show a plausible improvement in the "after" phase.

### T-N6 — Full experiment matrix (only after T-N1–T-N5 are complete)
**Files:** new run scripts under `scripts/ns3_experiments/`, raw outputs under `reports/ns3_runs/`
**Prompt:**
> Design and run a complete experiment matrix sweeping: (1) threshold rule (fixed-τ / EWMA / criticality-weighted, from T-N4); (2) latency (this project's real measured value, the labeled base-paper proxy, and at least 2–3 additional swept values for sensitivity analysis, from T-N3); (3) attack scale (vary `nCompromised`, e.g., 20/40/60/80 attacker-bots) at a fixed topology size. Save every raw run output under `reports/ns3_runs/<descriptive_label>.txt` (matching the existing `tee` convention). Do not aggregate results in this task — that's T-N7.
**VERIFY:** Every combination in the matrix has a corresponding saved raw output file; file naming makes the swept parameters identifiable without opening the file.

### T-N7 — Aggregate and report
**Files:** `reports/ns3_simulation_results.md`
**Prompt:**
> Aggregate all raw runs from T-N6 (plus the T-N2 fix/finding, T-N4's threshold comparison, and T-N5's incremental scenario) into `reports/ns3_simulation_results.md`. Structure it to feed directly into `Build-Instructions.md T4.4`'s final comparison: state clearly, for every number, whether it's a real Pi measurement (`T4.3`), an NS-3 simulated number, or a labeled proxy/assumption (per A.1). Include the resolved zero-detection finding from T-N2 as a named result, not a footnote. Explicitly connect this report's adaptive-threshold comparison (T-N4) to the PRD's >30% false-positive-reduction target (`PRD.md §3.1`) — state whether that target was met by the simulated numbers, and if not, say so plainly.
**VERIFY:** Report contains no number without a stated provenance (measured/simulated/proxy); explicitly addresses the PRD's adaptability success criterion with a yes/no/partial answer, not just raw numbers left for the reader to interpret.

---

## §E — Build Order

| Order | Tasks | Notes |
|---|---|---|
| 1 | T-N1 | Read-only inventory — do this before touching any simulation code |
| 2 | T-N2 | Blocking — do not proceed until the zero-detection mechanism is confirmed |
| 3 | T-N3 | Depends on `Build-Instructions.md T4.1–T4.3` being done — if those aren't finished yet, this task blocks here, not at the start |
| 4 | T-N4, T-N5 | Can run in parallel — independent additions to the scratch script |
| 5 | T-N6 | The big sweep — only after everything above is solid |
| 6 | T-N7 | Final aggregation, feeds `Build-Instructions.md T4.4` |

## §F — Final Acceptance Checklist

- [ ] Topology labels (`wearables`/`attacker-bots`/`fog`) are unambiguous in code and output (T-N1)
- [ ] Zero-detection anomaly in `C_batched` is either fixed or documented as a genuine, confirmed finding — not left unexplained (T-N2)
- [ ] No latency number used in a headline report is fabricated; every one traces to a real measurement or a labeled proxy (T-N3)
- [ ] This project's own adaptive threshold rule (criticality/context-weighted) is implemented and demonstrably distinct from fixed-τ and EWMA (T-N4)
- [ ] Incremental-learning stand-in scenario is implemented and clearly labeled as a simulation stand-in, not literal retraining (T-N5)
- [ ] Full experiment matrix has every raw run saved and reproducible (T-N6)
- [ ] `reports/ns3_simulation_results.md` exists, states provenance for every number, and directly answers the PRD's >30% false-positive-reduction target (T-N7)
- [ ] `TRD.md §8` and `Build-Instructions.md §B.3` updated to reflect NS-3 (not Mininet) as the actual tool in use
- [ ] No task touched files outside its declared **Files** line
