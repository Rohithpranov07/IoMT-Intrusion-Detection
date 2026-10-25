# Independent verification of the Review 3 manuscript and slide deck

Checked on 2026-10-21 against the raw sources in /home/claude/r3. All ns-3 statistics were recomputed from runs/*.txt (parsed with scripts/load.py, statistics recomputed with pandas and scipy 1.18.1), not taken from results/ns3/*.csv. Detection-pipeline statistics were recomputed from results/phase7/*_runs.csv. Audit metrics were recomputed from the confusion-matrix counts in audit/audit_additional_systems.md. Scratch scripts are in /tmp/claude-0/-home-claude/ed5a2d69-c791-538a-87e2-304cc2d3ac44/scratchpad/verify/.

Scope: paper/build/*.tex (text inside \pending{} and the E4/E2b placeholder subsections skipped) and the 19 slides in the deck (visible text and aside notes). E2b.txt and E4.txt were not used.

Severity key: **wrong** = the number or claim does not match the data; **imprecise** = rounding, wording, scope or labelling does not match the data, or a claim is stated more strongly than the data supports; **unverifiable** = no raw source available to me.

## (a) Discrepancies

### Manuscript

| # | File, location | Claim (quote) | Computed value | Source and method | Severity |
|---|---|---|---|---|---|
| M1 | sec5c_network.tex, Head-to-head paragraph 1 | C has "a sixth of B's spread" (also slide headtohead) | sd C 0.523 vs sd B 4.501: ratio 0.116, about one ninth. Range 2 vs 16: one eighth. IQR 0 vs 6.25 | E1.txt, idsLatencyUs=500, fpAlerts per arch, 20 seeds | wrong |
| M2 | sec5c_network.tex, Scenario matrix, last paragraph | "B, hysteresis and dual threshold quarantine 42 to 49 % regardless of mix" | Across all 16 scenarios (excluding B at fog=1, where it quarantines 0 % because it is latched): B 38.5 % (bots=120) to 47.0 % (fog=2); B'hys and B'dual 39.0 % (bots=120) to 49.0 % (bots=80). In the three mix scenarios only (baseline, ICU, periphery) all three are 45.0 to 46.5 %. No reading gives 42 % as the minimum | E2a.txt plus E1.txt (500 us), mean wearablesBlocked/nWearables per scenario and arch, all seeds present | wrong |
| M3 | sec6_discussion.tex, "Design guidance" | "in our runs it was the only rule that never disconnected a life-critical device" | False across our runs. Rule C quarantined life-critical devices in E3 tau0=1000 (5 of 5 in all 10 seeds), E3 uniform weights (48 over 10 seeds), E5 mimicry at tau0 >= 800 ms (25 of 25 per 5 seeds), E6 two-stage at tau0 1000 and 1500 (bilstm correlated 15, bilstm independent 25, transformer correlated 11, transformer independent 17 and 24). See section (d) for the full device-class audit | E3, E5, E6 .txt, sum of blockedLife for arch C | wrong |
| M4 | sec5e_threshold.tex, Revised conclusion | "in every scenario it never quarantined a life-critical or high-criticality device" | Holds in every E2a scenario and in E1, E7, E8 at default tau0 and weights. Fails in the mimicry scenario (E5) of the same section: high-criticality devices quarantined from tau0=600 ms (22 of 25), life-critical from 800 ms; also E6 at tau0 >= 1000, E3 tau0=1000, uniform and linear weights (linear: 24 high, 0 life). Needs "at the default threshold and weights" | as M3, plus blockedHigh | imprecise (overstated) |
| M5 | sec4_setup.tex, Network scenarios | "The full factorial would need 720 scenarios per architecture" | 4 wearables x 5 bots x 4 fog x 4 topologies x 3 mixes = 960; x 2 inspection costs = 1,920. 720 matches neither | arithmetic on the levels listed in the same paragraph | wrong |
| M6 | abstract.tex | "A congestion-coupled threshold latches off under load in 20 of 20 seeds, also when traffic is concentrated by topology or a single fog node" | 20/20 holds for the baseline at 2,000 us (E1) and 10/10 for fog=1 (E2a). On the random topology B latched in only 2 of 10 seeds (21/60 blocked in seeds 1 and 6; 60/60 in the other 8), mean detection 87 %. The sentence reads as if the topology case also latches in every seed. The measured-cost case (E8) is 10 of 10, not 20 of 20 | E2a.txt scenario=topo=random arch=B, fields latched, attackersBlocked; E8.txt | imprecise (overstated) |
| M7 | abstract.tex; sec1_intro.tex contribution 3; sec8_conclusion.tex | "three threshold rules block every attacker yet quarantine between 45 % and 100 %"; "collateral quarantine ranges from 45 % to 100 %"; "rules with identical detection rates quarantined between 45 % and 100 %" | True only in the baseline at 500 us (B 45.0 %, C 49.25 %, A 100 %). In the scenario matrix, rules that block every attacker quarantine as little as 20 % (C, ICU mix) or 0 to 0.25 % (B'cap, 20 bots). The range should be tied to the baseline | E1.txt, E2a.txt | imprecise |
| M8 | sec5b_leakage.tex, Table tab:order and text | Edge ORDER precision "0.999"; text "precision falls from 0.999 to 0.961" | ORDER precision mean 0.998453, which rounds to 0.998 | results/phase7/multiseed_runs.csv, mean of precision over 20 seeds | imprecise (rounding) |
| M9 | sec5c_network.tex, Scenario matrix, last paragraph | C quarantine "78 % in a periphery-heavy ward" (also slide matrix notes) | 15.70/20 = 78.5 %, which rounds half up to 79 % | E2a.txt scenario=mix=periphery arch=C, 10 seeds | imprecise (rounding) |
| M10 | sec5c_network.tex, Table tab:e1, rows B 500, B'hys 500, B'dual 500; text "36.5 +- 4.5" | FP alerts 36.5 | 36.55 exactly (731/20), which rounds half up to 36.6. Printed 36.5 is a float-representation artefact | E1.txt fpAlerts | imprecise (rounding) |
| M11 | sec5e_threshold.tex, Table tab:twostage | transformer per-device "89.2" FP vs A; ensemble per-device FP "10.4" | 1 - 6.45/60 = 89.25 % (slide twostage prints 89.3); ensemble per-device FP mean 10.45. Both are exact halves; the paper rounds down, the slide rounds up, so paper and slide disagree | E7.txt branch, stage2Mode, fpAlerts; E1.txt A fpAlerts = 60 | imprecise (rounding, paper vs slide inconsistent) |
| M12 | sec5f_hardware.tex, Table tab:measured | idle p50, B and C at 80 bots: "100 (3.6)" | B 3.5467 ms, C 3.5455 ms: both round to 3.5 | E8.txt factor=x86-idle-p50, appDelayP99Ms mean | imprecise (rounding) |
| M13 | sec5f_hardware.tex, Simulation with measured costs | "wearable p99 delay rises from about 3.5 ms to between 690 and 855 ms ... under A, C and the mitigated rule" | For A the idle-cost p99 is 3.5 to 8.4 ms (8.43 ms at 80 bots), so "from about 3.5 ms" holds for B and C, not A. The 690 to 855 range is correct (691.3 to 854.7) | E8.txt appDelayP99Ms | imprecise |
| M14 | sec5c_network.tex, Latch-off Mitigations | "Bonferroni p = 4x10^-4 for each against B" | Raw p = 1.19x10^-5 (all 20 pairs differ, exact minimum for n=20 with one tie structure). 4.3x10^-4 comes from multiplying by 36, because analyse_ns3.py corrects over all 6 pairs among B, B'hys, B'dual, B'cap x 3 metrics x 2 costs and then discards the pairs not involving B. The family actually reported has 18 tests (2.1x10^-4) and the paper never states the family. Conservative, not wrong, but the family is not what is reported | E1.txt, wilcoxon(B, B'x) on attackersBlocked at 2000 us, paired by rngRun; scripts/analyse_ns3.py lines 67 to 102 | imprecise |
| M15 | sec5c_network.tex, Head-to-head last paragraph | "Quarantine signalling is negligible for every rule (0.06 to 0.07 % of bytes)" | Non-latched runs 0.057 to 0.071 %; B at 2,000 us is 0.000 to 0.001 % (it quarantines nothing) | E1.txt ctrlOverheadPct min and max per arch and cost | imprecise |
| M16 | sec3_framework.tex, Step 5 | "Every configuration is run with independent ns-3 run numbers 1 to 20" | Only E1 and E7 use 20 for every configuration. E2a 10 to 20, E3 10, E5 and E6 5, E8 10 to 20. Contradicted by sec7_limits.tex | rngRun per configuration in each run file | imprecise |
| M17 | sec5c_network.tex, Scenario matrix intro and Fig. e2 | "10 seeds per cell, 20 for the baseline" | E2a raw data has 20 seeds in 34 cells (all six rules for wear=10/40/80 and bots=20; A, B, C at bots=40; A, B at bots=80) and 12 seeds for B'hys at bots=40. scripts/experiments.py defines E2a with SEEDS10 only, so the raw file holds runs the current definition does not generate (provenance of seeds 11 to 20 should be stated). Reported means use all seeds | E2a.txt groupby(scenario, arch) rngRun count | imprecise |
| M18 | sec5g_pipeline.tex, Attacks withheld | "On Edge-IIoTset recall on a withheld type is bimodal: near 100 % ... and near zero" | Two of eleven held-out types are intermediate: MITM 76 % (per seed 40.5 to 100 %), Fingerprinting 80 %. Bimodal is stated more strongly than the data | results/phase7/leave_one_attack_out_runs.csv, mean recall per type | imprecise |
| M19 | sec5h_xai.tex | "SHAP's top-5 features have a Jaccard stability of 0.83" | paper_v1.tex (Review 2, the designated source) reports SHAP 0.87 in its abstract, Table tab:xai, text and conclusion. The project doc t3_3_xai_evaluation.md reports 0.830 (Spearman +0.916). The two prior sources disagree; the new value is 0.04 lower than the published Review 2 figure and the change is not explained | paper_v1.tex lines 58, 352, 361, 411; project doc t3_3_xai_evaluation.md | wrong against paper_v1 (supported by project doc; needs a note) |
| M20 | sec4_setup.tex, Simulator validity | "on the same ns-3 build both produce identical detection counts in the normal and the latched regimes" | Normal regime confirmed: E1 seed 1 at 500 us gives A 321/60/20, B 300/39/10, C 291/30/10, identical to my runs of orig/hids-orig. Latched regime not checked (outside the three permitted runs) | E1.txt rngRun=1 vs scratchpad/verify/orig_*.txt | unverifiable (half) |
| M21 | sec5b_leakage.tex, How common is the leaky order | badiyani2026 "applies its balancing step before the train/test split" | No raw source available to me | none | unverifiable |
| M22 | refs.tex | internal consistency | All 34 cited keys exist. Two bibitems are never cited: nsnam, thulasiraman2019 | grep \cite vs \bibitem | imprecise (minor) |

### Slides

| # | Slide (file), location | Claim (quote) | Computed value | Source and method | Severity |
|---|---|---|---|---|---|
| S1 | headtohead, visible text | "with a sixth of the spread" | about one ninth (sd) or one eighth (range); see M1 | E1.txt | wrong |
| S2 | headtohead, visible text and footer | "(29.8 vs 36.5, p = 0.007) ... not significant (p = 0.08)", footer "paired Wilcoxon, Bonferroni-corrected" | 0.007 is the corrected p; 0.08 is the raw p (corrected 1.0). The slide mixes the two under one "Bonferroni-corrected" footer. Also 36.55 rounds to 36.6 (M10) | E1.txt; recomputed raw 0.0825 and 1.93x10^-4, x36 | imprecise |
| S3 | order, visible text and footer | "Paired Wilcoxon p = 2 x 10^-6" with footer "Bonferroni over 12 tests" | 2x10^-6 is raw; corrected is 2x10^-5 | multiseed_runs.csv, WUSTL LEAKY vs ORDER F1, 20 pairs | imprecise |
| S4 | measured, visible text | "3.5 -> 855 ms Wearable p99 delay under A at 80 bots" | Under A at 80 bots the idle-cost p99 is 8.43 ms, not 3.5 ms (3.5 is B and C) | E8.txt arch=A nAttackers=80 | wrong |
| S5 | twostage, visible text | "Transformer gate, errors stick to a device ... -89.3%" | 89.25 %; paper prints 89.2 (M11) | E7.txt | imprecise (inconsistent with paper) |
| S6 | mitigations, title | "Each mitigation turns 0 of 60 into 60 of 60" | B blocks 0.05 of 60 on average (1 of 60 in seed 14). Mitigation side correct (60/60 in 20/20) | E1.txt arch=B idsLatencyUs=2000 | imprecise (minor) |
| S7 | asked, row 8 | "22 verified references" | refs.tex has 36 bibitems, 34 of them cited | paper/build/refs.tex | wrong |
| S8 | asked, rows 1 and 3, status "Done" | "16-scenario matrix at two costs ... Done"; "Phase diagram, three mitigations, validated across the matrix ... Done" | The 2,000 us matrix (E2b) and the phase diagram (E4) are still running, and the manuscript has \pending placeholders for both (sec5d_latch.tex) | runs/E2b.txt, runs/E4.txt incomplete | imprecise (overstated) |
| S9 | matrix, notes | "That is sixteen scenarios, run for all six rules at two inspection costs" | Only 500 us is complete (E2a); 2,000 us (E2b) is incomplete | as S8 | imprecise (overstated) |
| S10 | matrix, footer | "full 16-scenario table in the paper" | The paper shows a figure and says "full table in the repository" (sec5c_network.tex) | sec5c_network.tex | imprecise |
| S11 | matrix, notes | "20 percent quarantined in an ICU-heavy ward, 78 percent in a periphery-heavy one" | 78.5 % (M9) | E2a.txt | imprecise (rounding) |
| S12 | reader, visible table and notes | "Top-5 Jaccard 0.83" | paper_v1.tex gives 0.87 (M19) | paper_v1.tex | wrong against paper_v1 |
| S13 | seeds, notes | "reproduces the Review 2 program's detection counts exactly, in both the normal and the latched regime" | Normal regime verified; latched regime not checked (M20) | as M20 | unverifiable (half) |
| S14 | close, visible text | "100% of wearables quarantined by the published rule in all 16 scenarios, at the same detection rate as the others" | 100 % confirmed at 500 us. "Same detection rate as the others" fails for B at fog=1 (0 %) and random topology (87 %) | E2a.txt | imprecise (minor) |

## (b) Claims that could not be verified, and why

1. Latched-regime equivalence of the R3 simulator and the Review 2 program (M20, S13): would need a further ns-3 run beyond the three permitted.
2. badiyani2026 oversampling order (M21): no source file.
3. "Ports with a distinct value on a third of all rows" (sec5b_leakage.tex, funnel notes): no per-column distinct counts in results/phase2.
4. LIME 1,000 to 12,000 samples, Jaccard 0.16 to 0.33, fidelity +0.349 to +0.023, 82 % of 40 certified: not in the designated sources except +0.349/+0.023 and 82 % (paper_v1.tex). The project doc t3_3_xai_evaluation.md supports all of them (it also shows a non-monotone 0.38 at 4,000 samples, which the paper omits).
5. Ensemble branch agreement 90.9 %, oracle gain 0.026, Edge-IIoTset branch accuracies, incremental learning values (8,646 parameters, 0.9746 to 0.9880, 0.9821 to 0.9863, 0.5231 to 0.6280): match paper_v1.tex, which is the only permitted source; not recomputable.
6. Berguiga split sums and Table tab:rederive: match paper_v1.tex; the original confusion matrices were not in the permitted sources, so not recomputed.
7. B'cap mechanism ("its utilisation estimate counts all arriving traffic, including floods from bots it has already quarantined"): consistent with sim code comments, not checked by a run.
8. Reader-study design figures (8/6/6 alerts, 10 to 15 participants, 25 minutes): design, not data.
9. Any claim depending on E2b or E4.
10. Window construction ("ten-record windows", session-level folds) and the TFLite file size "1.1 MB": file sizes sum to 1,146.5 KiB, consistent; the windowing is not recomputable here.

## (c) Count of claims checked

About 245 distinct numeric or factual claims were checked (manuscript about 185, slides and notes about 60, with repeated numbers counted once per location). 36 discrepancies are listed above (22 manuscript, 14 slides): 7 wrong, 25 imprecise, 4 unverifiable or partly unverifiable. Everything else matched the raw data.

## (d) Detail on the device-class claim ("no life-critical or high-criticality device was ever quarantined")

Sums of blockedLife and blockedHigh over every rule-C run:

| Experiment | Settings | Life-critical quarantined | High quarantined | Holds? |
|---|---|---|---|---|
| E1 | 500 and 2,000 us, 20 seeds each | 0 | 0 | yes |
| E2a | all 15 scenarios at 500 us (incl. ICU 0 of 10 and 0 of 6, periphery 0 of 1 and 0 of 3, balanced 0 of 5 and 0 of 5) | 0 | 0 | yes |
| E3 | cost 57.9, 1,000, 14,542.7 us; tau0 100, 250; exponential weights | 0 | 0 | yes |
| E3 | tau0 = 1,000 ms | 50 (5 of 5 in every seed) | 50 | **no** |
| E3 | uniform weights | 48 | 48 | **no** |
| E3 | linear weights | 0 | 24 (max 3 of 5 in a seed) | **no** (high only) |
| E7 | all six two-stage variants, 500 us, 20 seeds | 0 | 0 | yes |
| E8 | both measured costs, all attack scales | 0 | 0 | yes |
| E5 (mimicry, not in the requested list but in the same section) | tau0 600 / 700 ms | 0 | 22 / 25 | **no** (high) |
| E5 | tau0 >= 800 ms | 25 per level | 25 per level | **no** |
| E6 | tau0 1,000 and 1,500 ms | 11 to 25 per cell | 12 to 25 per cell | **no** |

So the claim holds for every rule-C run at the default tau0 = 500 ms with the frozen (or exponential) weights, across every scenario, cost and two-stage variant. It does not hold when tau0 is raised to 1,000 ms, with uniform or linear weights, or in the mimicry scenario at tau0 >= 600 ms. The slide "matrix" and sec5c_network.tex ("in every mix") are correct as scoped; sec5e_threshold.tex ("in every scenario") is overstated and sec6_discussion.tex ("in our runs ... never") is wrong.

## (e) Checks that passed (selection)

- Paired Wilcoxon recomputed from raw runs, paired by rngRun: B vs C at 500 us wearablesBlocked raw 0.0825 (Bonferroni x36: 1.0), fpAlerts raw 1.93x10^-4 (0.0070), legitPdr raw 3.22x10^-4 (0.0116, quoted 0.012); B vs each B' attackersBlocked at 2,000 us raw 1.19x10^-5; B vs B'cap wearablesBlocked at 500 us raw 0.0082 (x36 0.29, not significant). WUSTL LEAKY vs ORDER F1 raw 1.9x10^-6, x12 2.3x10^-5; Edge ORDER vs HONEST precision 2.3x10^-5 corrected.
- Table tab:e1: every mean, sd, PDR, delay and energy value (apart from M10); "2.2 times the modelled energy" (738.6/333.3 = 2.22, /336.9 = 2.19); queue overflow 2.8x10^5; 110 ms to quarantine; B'hys and B'dual identical to B in every seed and field at 500 us.
- Sensitivity: tau0 100/250/1000, alpha 5.4 to 19.9, latched 10 of 10 for every alpha at 2,000 us, uniform 19.3 and 59 false alerts, linear 12.4, frozen and exponential 9.8, 360 ms minimum gap.
- Trade-off and two-stage: 0 %/18 % at 350 ms, 82 %/7.7 % at 400 ms, 31 %/10.3 %, 49 %/3.0 %, 72 % at zero missed, 37 % (24.5 % fewer), 26 % (47 % fewer); all of Table tab:twostage apart from M11; stage-2 rates 0.9916/0.5103, 0.9735/0.2194, 0.9831/0.3529; 0.5 to 1.7 bots escape.
- Hardware: every cell of Table tab:latency; 72 % (0.330/0.457); eightfold (8.05x); 18 % median rise; 1,918 windows/s; 14 MB; 305,798 parameters; E8 latch 10 of 10 from 40 bots; 690 to 855 ms range.
- Leakage: every cell of Table tab:dup; 32 merged label-conflicting vectors; Table tab:order apart from M8; 42.7 % and 40.9 % synthetic; FPR 0.8 % to 13.3 %; F1 gap 0.286 (0.29).
- Audit: Hosain accuracy 99.2215, precision 98.3478, recall 99.9117, F1 99.1236 from TP 2,262, TN 2,836, FP 38, FN 2; 25,688 x 0.2 = 5,137.6; 2,264 > 2,046 (218); Tarish WUSTL recall 46.9055, F1 62.3377, ECU 99.9834/91.4919/95.5494.
- Held-out attacks: WUSTL 1.1 % and 0.09 %, backdoor 1.1 %, ransomware 0.7 %, false alarms 3.3 to 22.7 %; feature overlap 0/0/0/20.
- ns-3 version table: my runs of orig/hids-orig (rule fixed, ewma, criticality; 500 us; 60 bots; 30 s; rngRun 1) give 321/60/20, 300/39/10, 291/30/10, exactly the ns-3.44 row. The Review 2 row matches paper_v1.tex (327/60/20, 303/36/8, 296/29/9).
- Ensemble ablation (0.9230, 0.9271, 0.9487, 0.9399) matches t4_4_ablation.csv and t2_6_ensemble_results.csv; Review 2 criticality figures (10.8 %, 23.4 %, 48.6 %, FP+FN 294 to 311) match paper_v1.tex.
