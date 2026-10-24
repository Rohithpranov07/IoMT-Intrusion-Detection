# T4.3 — Deployment latency and throughput on the Raspberry Pi 4B

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
