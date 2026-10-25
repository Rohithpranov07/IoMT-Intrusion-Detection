# Network-level evaluation of IoMT intrusion detection (Review 3)

Code, raw simulation logs and analysis for the manuscript *Detection Rate Is Not Enough: A Simulation-in-the-Loop Framework for Network-Level Evaluation of IoMT Intrusion Detection* (Balaji, Prabhu, Pranov, Muthuswamy, V. P.; Vellore Institute of Technology).

Every number in the paper is regenerated from files in this repository by the scripts below. Nothing is typed in by hand.

## Layout

| Path | Contents |
|---|---|
| `sim/hids-iomt-r3.cc` | ns-3 simulator: fog IDS as a single-server queue in the data path; rules A (flat), B (congestion-coupled EWMA), C (criticality-weighted), B' mitigations (hysteresis, dual threshold, capacity coupling), two-stage gate, scenario and topology options. One `RESULT,` line per run. |
| `orig/hids-iomt-adaptive.cc` | the unmodified Review 2 simulator, kept for the regression check |
| `scripts/experiments.py` | every experiment (E1 to E9; E9 uses the costs measured on the Raspberry Pi 4) as a list of configurations; resumable runner |
| `runs/E*.txt` | raw output, one line per run, written before any aggregation |
| `scripts/analyse_ns3.py` | tables, paired Wilcoxon tests with Bonferroni correction, 300-DPI figures |
| `scripts/phase2_dataset_leakage.py` | duplicate and train/test-twin rates along each dataset's feature funnel |
| `scripts/phase7_detection_multiseed.py` | 20-partition leakage regimes, held-out-attack generalisation |
| `audit/` | audit of additional systems: extracted numbers with locations, re-derivation script |
| `hardware/` | latency/memory/CPU/energy harness (`measure_latency.py`), Raspberry Pi runbook; the Pi run itself is in `../reports/pi_run_2026-10-24/` |
| `reader_study/` | study app, export script for the 20 alerts, protocol and consent form, analysis script |
| `team_scripts/R3_multiseed_tables.py` | Tables 3, 7, 8 over 20 partitions on the full project pipeline (needs the project repo and IoTID20) |
| `paper/` | LaTeX sources; `build.py` fills generated tables and compiles |
| `results/`, `figs/` | generated outputs |
| `data/DATA_SOURCES.md` | where each dataset copy came from, with sha256 and row counts (datasets themselves are not redistributed) |

## Reproduce

```bash
pip install ns3==3.44.post0 numpy pandas scipy matplotlib scikit-learn imbalanced-learn tabulate
B=$(python3 -c "import ns,os;print(os.path.dirname(os.path.dirname(ns.__file__)))")
g++ -std=c++20 -O2 -I$B/include sim/hids-iomt-r3.cc -o sim/hids-r3 -L$B/lib64 -Wl,-rpath,$B/lib64 \
    -lns3.44-core -lns3.44-network -lns3.44-internet -lns3.44-point-to-point -lns3.44-applications \
    -lns3.44-flow-monitor -lns3.44-stats -lns3.44-mobility -lns3.44-traffic-control -lns3.44-bridge
python3 scripts/experiments.py E1 E7 E2a E3 E5 E6 E8 E2b E4 E9  # resumable; ~20 CPU-hours
python3 scripts/analyse_ns3.py
python3 scripts/phase2_dataset_leakage.py                      # needs data/ (see DATA_SOURCES.md)
python3 scripts/phase7_detection_multiseed.py
python3 paper/build.py
```

Regression check: with default options `sim/hids-r3` and `orig/hids-iomt-adaptive.cc` produce identical detection counts on the same ns-3 build (checked for all three rules at run 1).

## Conventions

- Attack is the positive class in every metric.
- Every inspection cost is labelled with its provenance (`latencySource`): swept values say so; measured values name the hardware.
- Every network result is a distribution over ns-3 run numbers (seed 42); comparisons are paired by run number.

## Licence and citation

This folder sits inside the project repository as `review3/`; paths in the table are relative to it.


Code: MIT (see `LICENSE`). Cite via `CITATION.cff`. After pushing to GitHub, enable the Zenodo GitHub integration and create a release to mint the DOI; `.zenodo.json` supplies the metadata.
