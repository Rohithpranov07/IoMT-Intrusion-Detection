#!/usr/bin/env bash
# T-N6 — full experiment matrix for the NS-3 validation layer (NS3-Simulation.md §D).
#
# Every raw run is saved under reports/ns3_runs/<descriptive_label>.txt before any aggregation,
# per NS3-Simulation.md §A.3. The label encodes every swept parameter, so the matrix can be read
# off the filenames without opening a file.
#
# LATENCY PROVENANCE (§A.1 rules 1-2, T-N3): every value below is a SWEPT input. At the time of
# writing no Raspberry Pi 4B measurement exists (Build-Instructions.md T4.3 is hardware-blocked)
# and the base paper reports no per-flow latency at all, so no run here may be labelled as either.
# See docs/latency_provenance.md.
#
# Usage: scripts/ns3_experiments/run_matrix.sh [path-to-ns-3-dev]
set -euo pipefail

NS3_DIR="${1:-$HOME/ns-3-dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO_ROOT/reports/ns3_runs"
SIM_TIME=30
RNG_RUN=1

mkdir -p "$OUT"
cp "$REPO_ROOT/scratch/hids-iomt-adaptive.cc" "$NS3_DIR/scratch/"
( cd "$NS3_DIR" && ./ns3 build hids-iomt-adaptive >/dev/null 2>&1 )

run() {
  local label="$1"; shift
  echo "  -> $label"
  ( cd "$NS3_DIR" && ./ns3 run "hids-iomt-adaptive $* --simTime=$SIM_TIME --rngRun=$RNG_RUN --label=$label" ) \
    > "$OUT/$label.txt" 2>&1
}

PROV="SWEPT-sensitivity-value-no-Pi-measurement-exists"

echo "[1/4] threshold rule x IDS latency"
for rule in fixed ewma criticality; do
  for lat in 57.9 500 2000 14542.7; do
    run "rule-${rule}_lat-${lat}us_atk-60" --rule="$rule" --idsLatencyUs="$lat" --latencySource="$PROV"
  done
done

echo "[2/4] attack scale at fixed latency"
for rule in fixed ewma criticality; do
  for atk in 20 40 60 80; do
    run "rule-${rule}_lat-500us_atk-${atk}" --rule="$rule" --idsLatencyUs=500 \
        --nAttackers="$atk" --latencySource="$PROV"
  done
done

echo "[3/4] T-N2 congestion-coupling probe"
for coupling in inverse direct; do
  for lat in 57.9 14542.7; do
    run "tn2_ewma-${coupling}_lat-${lat}us" --rule=ewma --ewmaCoupling="$coupling" \
        --idsLatencyUs="$lat" --traceRule=1 --latencySource="$PROV"
  done
done

echo "[4/4] T-N5 incremental-learning stand-in"
for rule in fixed criticality; do
  run "tn5_${rule}_no-update" --rule="$rule" --idsLatencyUs=500 --novelAttackAt=10 \
      --incrementalAt=20 --incrementalBaseMs=500 --latencySource="$PROV"
  run "tn5_${rule}_with-update" --rule="$rule" --idsLatencyUs=500 --novelAttackAt=10 \
      --incrementalAt=20 --incrementalBaseMs=900 --latencySource="$PROV"
done

echo "done: $(ls "$OUT" | wc -l | tr -d ' ') raw runs in $OUT"
