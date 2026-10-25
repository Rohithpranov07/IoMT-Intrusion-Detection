#!/usr/bin/env bash
# Phase 7: Tables 3, 7 and 8 over 20 partitions (IoTID20) and Tables 7 and 8 (Edge-IIoTset).
# The two datasets run IN PARALLEL, and Table 8 reuses the models trained for Table 7.
# Run from the repository root:  bash review3/team_scripts/run_phase7_mac.sh
# Resumable: if it stops, run the same command again and it continues where it left off.
cd "$(dirname "$0")/../.."
VENV="$HOME/Downloads/Projects/HIDS IoMT Intrusion Detection/tfenv"
PY="$VENV/bin/python3"
[ -x "$PY" ] || { echo "Python not found at $PY"; exit 1; }
export TF_CPP_MIN_LOG_LEVEL=2
"$PY" -c "from imblearn.over_sampling import SMOTE; import numpy as np; SMOTE().fit_resample(np.random.rand(20,2), [0]*10+[1]*10)" 2>/dev/null || "$PY" -m pip install -q -U imbalanced-learn
"$PY" -c "import sklearn, imblearn, tensorflow, shap, lime, scipy; print('environment ok')" || exit 1
S=review3/team_scripts/R3_multiseed_tables.py
mkdir -p reports/r3_multiseed
L=reports/r3_multiseed
caffeinate -dims env PY="$PY" bash -c "
( \"\$PY\" $S --dataset iotid20 --table 3 && \"\$PY\" $S --dataset iotid20 --table 7 && \"\$PY\" $S --dataset iotid20 --table 8 ) > $L/log_iotid20.txt 2>&1 &
( \"\$PY\" $S --dataset edge --table 7 && \"\$PY\" $S --dataset edge --table 8 ) > $L/log_edge.txt 2>&1 &
while [ \$(jobs -r | wc -l) -gt 0 ]; do
  sleep 60
  clear; date; echo '--- IoTID20 (latest) ---'; grep -E '^T[378] ' $L/log_iotid20.txt | tail -3
  echo '--- Edge-IIoTset (latest) ---'; grep -E '^T[78] ' $L/log_edge.txt | tail -3
  echo; echo 'Done so far:'; for f in $L/*_table*.csv; do [ -f \"\$f\" ] && echo \"  \$(basename \$f): \$(( \$(wc -l < \$f) - 1 )) rows\"; done
  echo; echo 'Targets: iotid20_table3 60, iotid20_table7 20, iotid20_table8 5, edge_table7 20, edge_table8 5'
done
wait
\"\$PY\" $S --summarise > $L/summary_log.txt 2>&1
"
echo "PHASE 7 FINISHED. Tell Claude: done."
