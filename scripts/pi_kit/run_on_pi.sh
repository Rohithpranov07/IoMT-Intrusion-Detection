#!/usr/bin/env bash
# Phase 5 measurement on the Raspberry Pi. Run from inside the pi_kit folder.
# Usage: bash run_on_pi.sh            (first pass, no energy)
#        bash run_on_pi.sh 4.2        (second pass, 4.2 = average watts read on the USB meter during the run)
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip
  ./venv/bin/pip install ai-edge-litert numpy psutil || ./venv/bin/pip install tflite-runtime numpy psutil
fi
vcgencmd measure_temp 2>/dev/null || true
vcgencmd get_throttled 2>/dev/null || true
./venv/bin/python verify_models.py
if [ -n "$1" ]; then
  ./venv/bin/python measure_latency.py --model-dir models --out latency_pi.json --meter-watts "$1"
else
  ./venv/bin/python measure_latency.py --model-dir models --out latency_pi.json
fi
vcgencmd measure_temp 2>/dev/null || true
vcgencmd get_throttled 2>/dev/null || true
echo "Done. Send back latency_pi.json and verify_pi.json."
