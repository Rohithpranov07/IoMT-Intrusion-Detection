HIDS-IoMT Review 3, Phase 5: Raspberry Pi measurement kit

Contents
  models/            YOUR TRAINED cnn, bilstm, transformer as TFLite float32, plus manifest.json
  check/             256 real test windows and the predictions they must reproduce
  verify_models.py   proves the Pi runs the evaluated model (must say PASS)
  measure_latency.py the measurement harness (records the board model itself, from the device tree)
  run_on_pi.sh       installs the runtime in a local venv and runs the measurement

On the Pi
  1. Copy this whole folder to the Pi (USB stick or scp) and open a terminal in it.
  2. Note the USB meter reading with the Pi idle for 1 minute (idle watts).
  3. bash run_on_pi.sh
     When the screen says SUSTAINED PHASE STARTED, read the meter for the 30 seconds and note the average watts.
  4. bash run_on_pi.sh <average watts>
  5. Send back latency_pi.json, verify_pi.json, the terminal output, and photos of the meter (idle and during the run).

Rules: nothing else running, official power supply, heatsink or fan fitted, Pi OS 64-bit.
Do not install TensorFlow on the Pi.

Building the kit from this repository
  models/  copy artifacts/deployment/float32/*.tflite and manifest.json (the trained export),
           plus a PROVENANCE.json describing it
  check/   copy sample_windows.npy, expected.npy and expected_probabilities.npy from
           artifacts/pi_bundle/
  The run of 24 October 2026 and its outputs are in reports/pi_run_2026-10-24/.
