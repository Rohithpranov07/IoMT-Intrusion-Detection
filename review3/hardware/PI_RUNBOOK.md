# Phase 5 on the faculty's Raspberry Pi: about 45 minutes

Goal: replace the swept inspection costs with **measured** ones, then re-run the latency-load table in ns-3 with them. An emulator (Velxio, Wokwi) does not run a Linux TFLite workload and its timings are not hardware measurements, so use the borrowed board.

## What to bring
- The Pi (4B or 5) with Raspberry Pi OS 64-bit, its official power supply, a microSD card.
- A USB-C inline power meter (any cheap USB "doctor" that shows volts and amps). This is the only way to get energy on a Pi.
- A laptop on the same network.

## 1. Copy the bundle (on the Mac, in the repo)
```bash
python scripts/make_pi_bundle.py            # writes the float32 TFLite export, manifest and runner (already verified, 100% agreement with Keras)
scp -r artifacts/deployment/float32 hardware/measure_latency.py pi@<pi-ip>:~/r3/
```

## 2. Install the runtime (on the Pi)
```bash
python3 -m venv ~/r3/venv && source ~/r3/venv/bin/activate
pip install ai-edge-litert numpy psutil      # or: pip install tflite-runtime numpy psutil
```
Do not install TensorFlow on the Pi. The measurement must go through the standalone runtime.

## 3. Measure (on the Pi, nothing else running)
```bash
cd ~/r3
python3 measure_latency.py --model-dir float32 --out latency_pi.json
```
While the 10-second sustained phase at the end runs, read the meter. Note average watts at idle (before starting) and during the run. Then rerun with the reading so energy per window is recorded:
```bash
python3 measure_latency.py --model-dir float32 --out latency_pi.json --meter-watts <watts during run>
```
The file records the board model from the device tree, so it cannot be mislabelled. It reports p50, p90, p95, p99 per branch and for the full ensemble at 0 to 4 background CPU loads, sustained windows per second, CPU use, resident memory, model sizes, and energy per window.

Also run the existing project benchmark once, so both harnesses agree:
```bash
python3 scripts/benchmark_pi.py --model-dir float32
```

## 4. Send back
`latency_pi.json` and a photo of the meter reading. Claude converts it into the ns-3 runs (step 5) and the paper's hardware table.

## 5. ns-3 re-run with the measured costs (Mac, or Claude does it)
```bash
cp sim/hids-iomt-r3.cc ~/ns-3-dev/scratch/ && cd ~/ns-3-dev && ./ns3 build hids-iomt-r3
P50=$(python3 -c "import json;print(round(json.load(open('latency_pi.json'))['per_flow_cost_us_for_ns3']['idle_p50_us'],1))")
P95=$(python3 -c "import json;print(round(json.load(open('latency_pi.json'))['per_flow_cost_us_for_ns3']['full_load_p95_us'],1))")
for c in $P50 $P95; do for rule in fixed ewma criticality; do for atk in 20 40 60 80; do for s in $(seq 1 20); do
  ./ns3 run "hids-iomt-r3 --rule=$rule --idsLatencyUs=$c --nAttackers=$atk --rngRun=$s --simTime=30 --flowmon=0 \
     --latencySource=MEASURED-RaspberryPi-LiteRT-float32 --label=pi_${rule}_${c}_${atk}_${s}" | grep '^RESULT,' >> pi_table5.txt
done; done; done; done
```
`pi_table5.txt` has one RESULT line per run in the same format as the rest of the Review 3 experiments.

## What the x86 reference already shows (measured here, not a Pi)
On a 2-vCPU Intel Xeon 2.1 GHz cloud machine with LiteRT float32: ensemble p50 457 µs idle, p95 558 µs; with both cores loaded p50 540 µs but **p95 4,493 µs**. The BiLSTM is about 72% of the cost. A Pi 4B's Cortex-A72 is slower per core, so expect higher figures; the shape (median stable, tail exploding under contention) is what to check.
