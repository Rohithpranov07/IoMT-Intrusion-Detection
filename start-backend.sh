#!/usr/bin/env bash
#
# Run the NS-3 network simulation: the backend half of this project.
#
#   ./start-backend.sh                run the complete suite (60 runs, about 3 minutes)
#   ./start-backend.sh --quick        one run per rule, to check the toolchain works
#   ./start-backend.sh --report-only  re-aggregate the saved runs without re-simulating
#   ./start-backend.sh --ns3 <path>   point at an ns-3 checkout other than ~/ns-3-dev
#
# There is no HTTP service here. This project's backend is a simulation and an analysis
# pipeline: it produces reports/ns3_simulation_results.md and the JSON the dashboard reads.
#
# EVERY NUMBER IT PRODUCES IS SIMULATION OUTPUT. No Raspberry Pi measurement exists in this
# repository, and nothing this script writes may be quoted as one (NS3-Simulation.md A.1).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS3_DIR="${NS3_DIR:-$HOME/ns-3-dev}"
MODE="full"
PROGRAM="hids-iomt-adaptive"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE="quick"; shift ;;
    --report-only) MODE="report"; shift ;;
    --ns3) NS3_DIR="${2:?--ns3 needs a path}"; shift 2 ;;
    -h|--help) sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

PYTHON="$REPO_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3 || true)"
if [[ -z "$PYTHON" ]]; then
  echo "error: no python found. Create the venv first:" >&2
  echo "  python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

aggregate() {
  echo
  echo "==> aggregating $(ls "$REPO_ROOT/reports/ns3_runs"/*.txt 2>/dev/null | wc -l | tr -d ' ') saved runs"
  "$PYTHON" "$REPO_ROOT/scripts/ns3_experiments/aggregate_results.py" --json
}

if [[ "$MODE" == "report" ]]; then
  if ! compgen -G "$REPO_ROOT/reports/ns3_runs/*.txt" >/dev/null; then
    echo "error: no saved runs in reports/ns3_runs/. Run without --report-only first." >&2
    exit 1
  fi
  aggregate
  exit 0
fi

# --- ns-3 toolchain -------------------------------------------------------------------------
if [[ ! -x "$NS3_DIR/ns3" ]]; then
  echo "error: no ns-3 build tool at $NS3_DIR/ns3" >&2
  echo >&2
  echo "  Point at your checkout:   ./start-backend.sh --ns3 /path/to/ns-3-dev" >&2
  echo "  Or set it once:           export NS3_DIR=/path/to/ns-3-dev" >&2
  echo >&2
  echo "  ns-3 is not vendored here. Build it once with:" >&2
  echo "    git clone https://gitlab.com/nsnam/ns-3-dev.git && cd ns-3-dev && ./ns3 configure && ./ns3 build" >&2
  exit 1
fi

echo "==> ns-3 at $NS3_DIR"
echo "==> syncing scratch/$PROGRAM.cc and building"
cp "$REPO_ROOT/scratch/$PROGRAM.cc" "$NS3_DIR/scratch/"
if ! ( cd "$NS3_DIR" && ./ns3 build "$PROGRAM" >/tmp/ns3-build.log 2>&1 ); then
  echo "error: the simulation did not compile. Last lines of /tmp/ns3-build.log:" >&2
  tail -20 /tmp/ns3-build.log >&2
  exit 1
fi
echo "    built"

mkdir -p "$REPO_ROOT/reports/ns3_runs"

if [[ "$MODE" == "quick" ]]; then
  echo
  echo "==> quick check: one run per threshold rule, 30 simulated seconds each"
  echo
  for rule in fixed ewma criticality; do
    printf '    %-12s ' "$rule"
    out=$( cd "$NS3_DIR" && ./ns3 run "$PROGRAM --rule=$rule --idsLatencyUs=500 --simTime=30 \
      --label=quick_$rule --latencySource=SWEPT-quick-check" 2>/dev/null )
    blocked=$(sed -n 's/.*attacker-bots blocked *: *\([0-9]*\) \/ \([0-9]*\).*/\1\/\2/p' <<<"$out")
    false_pos=$(sed -n 's/.*on wearables *: *\([0-9]*\).*/\1/p' <<<"$out")
    echo "attackers blocked $blocked, false positives $false_pos"
  done
  echo
  echo "Three different outcomes from the same scenario is the point: the rule changes behaviour."
  echo "Run without --quick for the complete 60-run suite."
  exit 0
fi

# --- the complete suite ---------------------------------------------------------------------
echo
echo "==> running the complete simulation suite"
echo "    48 cross-product runs, 4 congestion-coupling probes, 4 tipping-point probes,"
echo "    4 incremental-learning runs. About 3 minutes."
echo
"$REPO_ROOT/scripts/ns3_experiments/run_matrix.sh" "$NS3_DIR"

aggregate

# --- what it found --------------------------------------------------------------------------
RESULTS="$REPO_ROOT/reports/ns3_simulation_results.md"
echo
echo "================================================================"
echo " Complete. Every figure below is NS-3 SIMULATION OUTPUT."
echo " No Raspberry Pi measurement exists in this repository."
echo "================================================================"
echo
echo "  raw runs saved     reports/ns3_runs/  ($(ls "$REPO_ROOT/reports/ns3_runs"/*.txt | wc -l | tr -d ' ') files)"
echo "  report             reports/ns3_simulation_results.md"
echo "  dashboard data     dashboard/lib/ns3-results.json"
echo
sed -n '/^| Rule | Alerts/,/^$/p' "$RESULTS" | head -6
echo
echo "  The flat rule blocks every legitimate wearable in the simulation."
echo "  Full detail, including where the congestion-adaptive rule goes blind:"
echo "    $RESULTS"
echo
echo "  Show it in the browser:  ./start-frontend.sh"
