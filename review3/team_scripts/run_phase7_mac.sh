#!/usr/bin/env bash
# Phase 7: Tables 3, 7 and 8 over 20 partitions (IoTID20) and Tables 7 and 8 (Edge-IIoTset).
# Run from the repository root:  bash review3/team_scripts/run_phase7_mac.sh
# Resumable: if it stops, run the same command again and it continues where it left off.
cd "$(dirname "$0")/../.."
VENV="$HOME/Downloads/Projects/HIDS IoMT Intrusion Detection/tfenv"
source "$VENV/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=2
python -c "from imblearn.over_sampling import SMOTE; import numpy as np; SMOTE().fit_resample(np.random.rand(20,2), [0]*10+[1]*10)" 2>/dev/null || pip install -q -U imbalanced-learn
python -c "import sklearn, imblearn, tensorflow, shap, lime, scipy; print('environment ok')" || exit 1
S=review3/team_scripts/R3_multiseed_tables.py
mkdir -p reports/r3_multiseed
caffeinate -dims bash -c "
python $S --dataset iotid20 --table 3 &&
python $S --dataset iotid20 --table 7 &&
python $S --dataset iotid20 --table 8 &&
python $S --dataset edge --table 7 &&
python $S --dataset edge --table 8 ;
python $S --summarise
" 2>&1 | tee -a reports/r3_multiseed/run_log.txt
echo "PHASE 7 FINISHED. Tell Claude: done."
