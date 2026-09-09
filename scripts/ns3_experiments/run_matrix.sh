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

# T-N6's VERIFY requires EVERY combination in the matrix to have a saved raw output, so this is
# the full cross-product of rule x latency x attack scale (3 x 4 x 4 = 48), not a pair of 2-D
# slices through it. Slices would have hidden any interaction between the three axes -- and §5.1 of
# the aggregated report shows there is one: the ewma rule's collapse point moves with attack scale.
echo "[1/3] full cross-product: rule x IDS latency x attack scale (48 runs)"
for rule in fixed ewma criticality; do
  for lat in 57.9 500 2000 14542.7; do
    for atk in 20 40 60 80; do
      run "rule-${rule}_lat-${lat}us_atk-${atk}" --rule="$rule" --idsLatencyUs="$lat" \
          --nAttackers="$atk" --latencySource="$PROV"
    done
  done
done

echo "[2/3] T-N2 congestion-coupling probe"
for coupling in inverse direct; do
  for lat in 57.9 14542.7; do
    run "tn2_ewma-${coupling}_lat-${lat}us" --rule=ewma --ewmaCoupling="$coupling" \
        --idsLatencyUs="$lat" --traceRule=1 --latencySource="$PROV"
  done
done

echo "[3/4] tipping-point probe: does blocking outrun saturation?"
# The cross-product shows the ewma rule holding 100% at 40 attacker-bots and collapsing to 0% at
# 60, at the SAME latency. This isolates why: blockAfter is how many violations the rule waits for
# before acting, and it is the only variable here.
for ba in 1 2 3 4; do
  run "tn2_race_ewma_lat-2000us_atk-60_blockafter-${ba}" --rule=ewma --idsLatencyUs=2000 \
      --nAttackers=60 --blockAfter="$ba" --latencySource="$PROV"
done

echo "[4/4] T-N5 incremental-learning stand-in"
for rule in fixed criticality; do
  run "tn5_${rule}_no-update" --rule="$rule" --idsLatencyUs=500 --novelAttackAt=10 \
      --incrementalAt=20 --incrementalBaseMs=500 --latencySource="$PROV"
  run "tn5_${rule}_with-update" --rule="$rule" --idsLatencyUs=500 --novelAttackAt=10 \
      --incrementalAt=20 --incrementalBaseMs=900 --latencySource="$PROV"
done

echo "done: $(ls "$OUT" | wc -l | tr -d ' ') raw runs in $OUT"
