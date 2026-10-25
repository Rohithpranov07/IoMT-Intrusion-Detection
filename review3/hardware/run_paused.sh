#!/usr/bin/env bash
# Pause the simulation queue, run the latency measurement on an otherwise idle machine, resume.
cd /home/claude/r3
pkill -STOP -f "experiments.py"; pkill -STOP -f "hids-r3"
trap 'pkill -CONT -f "experiments.py"; pkill -CONT -f "hids-r3"' EXIT
python3 hardware/measure_latency.py --out results/phase5/latency_x86.json --n 1000
