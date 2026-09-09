# Latency provenance (T-N3)

**Task:** T-N3 · **Spec:** `NS3-Simulation.md §A.1` rules 1–2, `§F`

## The rule

Every latency value that reaches any report traces to exactly one of three origins, and is labelled
with it at the point of use:

| Origin | May be described as | Currently available? |
|---|---|---|
| Physical Raspberry Pi 4B measurement (`Build-Instructions.md` T4.3) | "measured on the Pi" | **NO** |
| Desktop TFLite measurement (`reports/DRAFT_deployment_benchmark.json`) | "desktop proxy — not the Pi" | yes |
| Swept simulation input | "swept sensitivity value" | yes |

## Status: T-N3's headline comparison cannot be run

T-N3 asks for two labelled headline cases. **Neither can be produced right now:**

**(1) This project's architecture.** T-N3 requires "the actual measured per-sequence inference
latency from T4.3's real Raspberry Pi benchmark". That measurement **does not exist**. The hardware
has not been available, and `src/deployment/benchmark.py` raises `NotOnTargetHardwareError` rather
than emit a number off-device — the prohibition is in code, not in discipline. Until a Pi run
happens, no NS-3 run may be labelled as this project's architecture on real hardware.

**(2) The base paper's architecture.** The base paper reports **no per-flow inference latency
anywhere** (`PRD.md §2.2` — this is one of the four original objections). There is no number to
plug in. `§A.1` rule 1 offers two escapes: measure an equivalently-shaped CNN-LSTM as a labelled
proxy, or omit the arm and say why. **This work omits it**, because a proxy built from this
project's own model would be a comparison of this project against itself wearing a label, and the
label is the only thing making it a comparison.

So the headline comparison is **deferred, not approximated**. Everything in
`reports/ns3_simulation_results.md` is a sensitivity sweep.

## What the numbers actually used are

| Value | Origin | May be cited as |
|---|---|---|
| 57.9 µs | Named in `NS3-Simulation.md §B.1`; §B.2 records its own origin as never established, and no run script or history exists here to establish it | sweep point only |
| 14542.7 µs | as above | sweep point only |
| 500 µs, 2000 µs | chosen here to bracket the two above | sweep point only |

Every run stamps its own provenance string into its saved output:

```
  inspection cost/flow   : 500.0000 us
  latency provenance     : SWEPT-sensitivity-value-no-Pi-measurement-exists
```

so a raw run in `reports/ns3_runs/` cannot be read back, quoted, or pasted into a slide without its
provenance travelling with it.

## The desktop number, and why it is not used here

`reports/DRAFT_deployment_benchmark.json` holds a real measurement: 0.19 ms median per window,
float32 TFLite, **on Darwin arm64 — not a Raspberry Pi**, and through full TensorFlow rather than
`tflite-runtime`. It is deliberately **not** wired into any simulation run. Feeding it in would
produce a table that is arithmetically about a desktop and rhetorically about a fog node, and the
distance between those two readings is exactly where `§A.1` rule 4's failure lives.

## When the Pi arrives

1. Run `docs/raspberry_pi_setup.md` end to end; confirm 100.0000% prediction agreement.
2. `scripts/benchmark_pi.py` writes `reports/deployment_benchmark.md` with a real median latency.
3. Re-run the matrix with `--idsLatencyUs=<that number>` and
   `--latencySource=MEASURED-Pi4B-T4.3-<date>`.
4. That, and only that, is the run that may be labelled "this project's architecture".
