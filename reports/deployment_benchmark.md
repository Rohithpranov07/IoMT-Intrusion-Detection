# T4.3: Deployment latency and throughput on the Raspberry Pi 4B

## Hardware (read from the device at run time, not configured)

```
Host: Raspberry Pi 4 Model B Rev 1.5
  architecture : aarch64
  cores        : 4
  RAM          : 3.7 GB
  OS           : Linux 6.18.50+rpt-rpi-v8
  Python       : 3.13.5
  Raspberry Pi : True
```

Interpreter backend: **ai_edge_litert**  ·  model precision: **float32**

## Results

200 windows timed after 20 discarded warm-up iterations.

| Measurement | Value |
|---|---:|
| Median latency (p50) | **2.89 ms** |
| p90 latency | 2.96 ms |
| p95 latency | **2.99 ms** |
| p99 latency | 3.03 ms |
| Throughput | **344.9 windows/s** |
| Cold start | 0.04 s |

Percentiles rather than a mean alone: a fog node's worst case is what determines whether an alert is late, and a mean hides it.

### Per branch

| Branch | Median | p95 |
|---|---:|---:|
| cnn | 0.28 ms | 0.33 ms |
| bilstm | 2.09 ms | 2.18 ms |
| transformer | 0.51 ms | 0.56 ms |

## What these numbers do and do not support

- A window is **10 flow records**, so the per-window latency is the cost of one decision,
  not one packet.
- Latency here excludes flow capture, feature extraction, and sessionisation. It is
  **model inference only**. An end-to-end real-time claim needs the capture path measured
  too, which is not in this project's scope.
- Explanation cost is **not** included. SHAP adds ~0.21 s per flagged detection on a
  desktop (`reports/t3_2_lime_vs_shap.md`) and has not been measured here; certification
  adds ~0.07 s. Both apply only to flagged windows.

## Reproducing

```bash
python3 scripts/benchmark_pi.py --model-dir deployment/float32
```

---

## Supplement: verification, hardware detail and behaviour under simulated load

Added after the run from files captured on the same Pi on 24 October 2026. Every value below was read from the device or measured on it; nothing is estimated. Raw files: `reports/pi_run_2026-10-24/`.

### Exact hardware and software

| Item | Value | Source |
|---|---|---|
| Board | Raspberry Pi 4 Model B Rev 1.5 | `/proc/device-tree/model` |
| RAM | **4 GB model** (3.7 GiB visible to Linux) | `free -h` |
| CPU | 4 cores, aarch64 (Cortex-A72) | harness |
| OS | Raspberry Pi OS 64-bit, Debian GNU/Linux 13.7 (trixie) base | `/etc/os-release` |
| Kernel | Linux 6.18.50+rpt-rpi-v8 | harness |
| Python | 3.13.5 | harness |
| Runtime | ai-edge-litert 2.3.0 (no TensorFlow installed) | harness |
| Power and thermals | `throttled=0x0` before and after; 38.9 °C to 44.3 °C | `vcgencmd` |

Build-Instructions T4.2 names a Pi 4B with 8 GB. The board measured is the 4 GB model; it is reported as measured. The ensemble's resident memory is about 53 MB, so RAM size does not limit it.

### T4.2: the Pi runs the evaluated model

| Check | Result |
|---|---|
| `pi_inference.py --self-test` | PASS (8 synthetic windows, mean confidence 0.9999) |
| Agreement with the training environment on 256 real test windows | **100.0000%** argmax agreement |
| Largest fused-probability difference (separate check, `verify_pi.json`) | 6.5 × 10⁻⁸ |

### T4.3: latency under simulated CPU load

Same trained float32 models; k background processes each saturate one core while the ensemble is timed (1,000 windows per level after warm-up, `measure_latency.py`).

| Background loads | Ensemble p50 (ms) | p95 (ms) | p99 (ms) |
|---:|---:|---:|---:|
| 0 | 3.00 | 3.03 | 3.11 |
| 1 | 2.99 | 3.02 | 3.08 |
| 2 | 3.01 | 3.03 | 3.08 |
| 3 | 3.02 | 3.16 | 3.41 |
| 4 (every core busy) | 3.01 | **6.18** | **7.04** |

The median is unaffected by load. The tail roughly doubles only when every core is busy, because the inference thread then competes for a core. Sustained single-threaded throughput over 30 s: **334 windows/s** at 100% of one core. Resident memory rises from 30.5 MB to 52.9 MB when the three models load; model files total 1.42 MB (1,420,424 bytes).

### Reference: the same models on a cloud x86 host

| Host | p50 idle | p95 with all cores busy | Sustained |
|---|---:|---:|---:|
| Raspberry Pi 4B (this report) | 3.00 ms | 6.18 ms | 334 windows/s |
| Intel Xeon 2.1 GHz, 2 vCPU (cloud VM) | 0.47 ms | 4.67 ms | 1,999 windows/s |

The Pi is about 6.4 times slower at the median. Per-window cost is dominated by the BiLSTM on both hosts (2.09 of 2.89 ms on the Pi).

### Not measured

Energy: the Pi 4 has no power sensor and no external USB meter was used, so no energy figure is reported.
