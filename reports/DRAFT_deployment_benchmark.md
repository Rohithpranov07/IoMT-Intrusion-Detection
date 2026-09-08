# DRAFT — benchmark harness output (NOT deployment evidence)

> ## ⚠️ These numbers are NOT deployment evidence and must not be cited as such.
>
> They were measured on **Darwin arm64**, which is not a Raspberry Pi. This
> file exists to show the harness works end to end and to fix the report's format
> before the hardware arrives. `PRD.md §2.2` identifies the senior's prior work's
> central failure as a real-time claim with no hardware behind it; presenting a
> desktop measurement as a Pi measurement would be the same failure.
>
> Re-run `scripts/benchmark_pi.py` **on the Pi** to produce the real report.

## Hardware (read from the device at run time, not configured)

```
Host: Darwin arm64
  architecture : arm64
  cores        : 8
  RAM          : 8.0 GB
  OS           : Darwin 25.2.0
  Python       : 3.11.15
  Raspberry Pi : False
```

Interpreter backend: **tensorflow**  ·  model precision: **float32**

> The **tensorflow** backend was used, not `tflite-runtime`. A measurement taken
> through full TensorFlow is not a `tflite-runtime` measurement and must not be quoted
> as one. On the Pi, install `tflite-runtime` so this line reads `tflite_runtime`.

## Results

60 windows timed after 20 discarded warm-up iterations.

| Measurement | Value |
|---|---:|
| Median latency (p50) | **0.19 ms** |
| p90 latency | 0.21 ms |
| p95 latency | **0.21 ms** |
| p99 latency | 0.22 ms |
| Throughput | **5078.8 windows/s** |
| Cold start | 2.30 s |

Percentiles rather than a mean alone: a fog node's worst case is what determines whether an alert is late, and a mean hides it.

### Per branch

| Branch | Median | p95 |
|---|---:|---:|
| cnn | 0.03 ms | 0.03 ms |
| bilstm | 0.13 ms | 0.14 ms |
| transformer | 0.04 ms | 0.04 ms |

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
