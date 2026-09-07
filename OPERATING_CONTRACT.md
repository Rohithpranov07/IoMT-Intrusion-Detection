# Operating Contract

> Source: `Build-Instructions.md` §A. Companions: `PRD.md` (what/why), `TRD.md` (exact technical contracts).
> **If this file and `TRD.md` disagree on a technical detail, the TRD wins.**

## Project in one line
A leakage-free IoMT intrusion-detection baseline + CNN-BiLSTM+Transformer ensemble + SHAP/LIME
explainability + adaptive threshold & incremental learning, built as an evidence-backed critique of
Berguiga et al., *HIDS-IoMT* (IEEE Access, 2025).

## Anti-hallucination rules (A.1)
1. **State the positive class explicitly, every time.** This project's convention is
   **Attack = positive** (`TRD.md §2.3`). Every metric computation, docstring, report, and slide
   must say so. Silently repeating the base paper's Objection #3 is the worst possible outcome.
2. **Never fabricate PSO hyperparameters and present them as the base paper's.** The base paper
   states no particle count, `w`, `c1`/`c2`, iteration count, or fitness function (`TRD.md §6.1`).
   Either choose and label our own values, or use a fully-specified alternative and say so.
3. **Never claim "real-time" or "deployed" without a measured number** from an actual Raspberry Pi
   4B run (`TRD.md §7`). No estimates, no desktop benchmarks dressed up as Pi numbers.
4. **Respect the GNN go/no-go gate.** If `reports/gnn_go_nogo.md` says no-go, no GNN code may
   re-enter the repo (`TRD.md §3.4`, `PRD.md §9`).
5. **Never let sequence length collapse to 1.** Ensemble input must be
   `(batch, sequence_length, features)` with `sequence_length > 1`, asserted in code.
6. **Never cite the base paper's 99.92 / 99.91 / 99.99 / 99.95 without the metric-inversion
   caveat in the same paragraph** (`PRD.md §2.1.3`, `§10`).

## Anti-drift rule (A.2)
Each task in `Build-Instructions.md` §D lists an explicit **Files** line. Do not edit files outside
that list while working that task.

## Quality gates (A.3)
- Module-level docstring on every model/pipeline file stating its exact hyperparameters as plain
  numbers (layer counts, unit counts, window sizes, thresholds).
- Deterministic: a fixed seed wherever randomness appears (splits, resampling, weight init).
  Project-wide seed constant: `RANDOM_STATE = 42` (`src/config.py`).
- No bare `except:` — catch and log specific exceptions.
- Type hints on all new function signatures.

## Working method (A.4)
Phase by phase, task by task, in `Build-Instructions.md` §D order. Run what each task produces and
confirm its VERIFY block before moving on. One conventional commit per task.

## Environment
- Python 3.11 in `.venv/` (`TRD.md §8`). Activate: `source .venv/bin/activate`.
- Dataset: IoTID20 via `kagglehub.dataset_download("rohulaminlabid/iotid20-dataset")`,
  fetched by `scripts/download_data.py`. `data/` is gitignored.
