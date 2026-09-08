# Raspberry Pi deployment — run-through for when the hardware arrives

**Covers:** T4.2 (`pi_inference.py`) and T4.3 (`benchmark.py`) · **Spec:** `TRD.md §7`, `TRD.md §9`

Everything on the software side is built, tested, and bundled. **The only thing missing is the
Raspberry Pi 4B.** This document is the sequence to follow once it is connected; nothing in it
requires further development work.

---

## What already works, and how it was checked

| | Status | Evidence |
|---|---|---|
| Ensemble exported to TFLite | done | `reports/t4_1_export.md` — 100% prediction agreement, all branches |
| Standalone runner, no TensorFlow | done | `test_pi_inference_imports_nothing_from_the_repo` parses the AST and fails on any `src.*` import |
| Fusion reproduced without the repo | done | `test_pi_fusion_matches_the_training_implementation` — identical to `src/models/fusion.py` to 1e-12 |
| Deployed = evaluated model | done | `test_runner_reproduces_the_trained_ensemble` — 100% agreement on real test windows |
| Bundle assembled | done | `artifacts/pi_bundle/`, ~2 MB, verified self-contained |
| **Latency / throughput on the Pi** | **BLOCKED — needs hardware** | — |

**No latency or throughput number for the Pi exists anywhere in this repository**, and the code
refuses to produce one off the hardware. `PRD.md §2.2` identifies the senior's prior work's central
failure as a real-time claim with no numbers behind it; `Build-Instructions.md` §A.1 rule 3 forbids
this project repeating it, and `write_deployment_report()` raises `NotOnTargetHardwareError` rather
than relying on anyone remembering that.

---

## Before the Pi arrives — rebuild the bundle if the model changed

```bash
.venv/bin/python scripts/export_for_pi.py     # writes artifacts/deployment/{float32,float16}/
.venv/bin/python scripts/make_pi_bundle.py    # writes artifacts/pi_bundle/
```

The bundle is gitignored (build output). Rebuild it after any retraining, or the Pi will run a
model that no report describes.

## Hardware and OS

- **Raspberry Pi 4B, 8 GB** (`TRD.md §7`).
- **64-bit Raspberry Pi OS.** The 32-bit image is a false economy here: `tflite-runtime` wheels for
  `armv7l` are older and harder to source than `aarch64` ones.
- Power supply and an SD card with ~1 GB free. The bundle is ~2 MB; the dependencies are the bulk.

## Step 1 — copy the bundle

```bash
scp -r artifacts/pi_bundle/ pi@raspberrypi.local:~/iomt-ids
```

## Step 2 — install the two dependencies

```bash
ssh pi@raspberrypi.local
cd ~/iomt-ids
python3 -m pip install -r requirements-pi.txt
```

Only `numpy` and `tflite-runtime`. **No TensorFlow** — that is the point of `TRD.md §7`.

If `tflite-runtime` has no wheel for the Pi's Python version, install `ai-edge-litert` instead. The
runner accepts either and prints which one it used.

## Step 3 — confirm it loads

```bash
python3 pi_inference.py --model-dir deployment --self-test
```

Expected:

```
  Raspberry Pi : True
  Self-test on 8 synthetic windows: PASS
  backend      : tflite_runtime
```

**If `Raspberry Pi` reads `False`,** the device tree was not readable and the benchmark will refuse
to write a report. **If `backend` reads `tensorflow`,** the standalone runtime is not installed, and
any timing taken would not be a `tflite-runtime` measurement — fix that before Step 5.

## Step 4 — prove the Pi reproduces the evaluated model

```bash
python3 pi_inference.py --model-dir deployment \
    --input sample_windows.npy --expected expected.npy
```

`agreement with supplied labels` must read **100.0000%**.

`expected.npy` holds this development machine's predictions for the same 256 real test windows.
Anything below 100% means the Pi is not running the model this project evaluated, and **no result
from the device should be believed until that is resolved.** Likely causes, in order: a truncated
`scp`, a mismatched `tflite-runtime` version, or a rebuilt bundle whose `expected.npy` is stale.

## Step 5 — benchmark (this is T4.3's deliverable)

```bash
python3 benchmark_pi.py --model-dir deployment --windows sample_windows.npy
```

Writes `reports/deployment_benchmark.md` plus a `.json`, containing:

- median / p90 / p95 / p99 per-window latency — percentiles, not a mean, because a fog node's worst
  case is what decides whether an alert is late;
- per-branch latency, so a slow branch is attributable (watch the BiLSTM: it is unrolled for TFLite,
  see `reports/t4_1_export.md`);
- sustained throughput in windows/second;
- cold-start time;
- the hardware, **read from the device tree at run time**, not configured.

Copy both files back into the repository's `reports/`.

## Step 6 — if it is too slow

`artifacts/deployment/float16/` is already exported and verified: **half the size, 100% prediction
agreement** (`reports/t4_1_export.md`). Copy that directory instead and repeat Steps 3–5.

It is not the default precision because §A.1 rule 3 forbids optimising against an unmeasured
latency claim. Once Step 5 has produced a real number, choosing float16 becomes an evidence-based
decision rather than a guess.

---

## What the benchmark will NOT establish

State these alongside any number the Pi produces:

- **Inference only.** Flow capture, feature extraction, and sessionisation are excluded. An
  end-to-end real-time claim needs the capture path measured too, which is outside this project's
  scope. The generated report says so.
- **A window is 10 flow records**, so per-window latency is the cost of one decision, not one packet.
- **Explanation cost is not included.** SHAP adds ~0.21 s per flagged detection on a desktop
  (`reports/t3_2_lime_vs_shap.md`) and certification ~0.07 s; neither has been measured on the Pi.
  Both apply only to flagged windows, but if explanations are shown to operators in real time, that
  cost belongs in the budget and should be measured in a follow-up run.
- **Synthetic load.** `--windows sample_windows.npy` replays real captured windows back to back.
  That is a throughput ceiling, not a live-traffic measurement.

## If something fails

| Symptom | Cause | Fix |
|---|---|---|
| `No TFLite interpreter found` | neither runtime installed | `pip install tflite-runtime` (or `ai-edge-litert`) |
| `Raspberry Pi : False` on a Pi | `/proc/device-tree/model` unreadable | check permissions; the benchmark refuses to write a report until this reads True |
| `NotOnTargetHardwareError` | not running on the Pi | intended. `--allow-non-pi` writes a `DRAFT_` file that cannot be mistaken for evidence |
| agreement below 100% | corrupted copy, or a stale bundle | re-run `make_pi_bundle.py`, re-copy, and re-check |
| `branch graph ... is missing` | partial copy | copy the whole `deployment/` directory, manifest included |
