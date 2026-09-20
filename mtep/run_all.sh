#!/usr/bin/env bash
# Full reproduction of every number, table and figure in the manuscript and supplement.
# Usage:  bash run_all.sh /path/to/CogBeacon-MultiModal_Dataset_for_Cognitive_Fatigue [--force]
#   --force  recompute results that already exist (otherwise completed steps are
#            skipped, so an interrupted run can simply be restarted).
set -euo pipefail
cd "$(dirname "$0")"
CORPUS="${1:?usage: bash run_all.sh /path/to/cogbeacon [--force]}"
FORCE="${2:-}"

echo "[1/9] feature table from the raw corpus (label/feature name check asserted)"
python code/build_dataset.py "$CORPUS" data/dataset.npz

echo "[2/9] Study 1 - conventional protocol (14 configurations)"
for j in $(seq 0 13); do python code/run_experiment.py "$j" $FORCE; done

echo "[3/9] Section 4.3 - position-decile matching attempts on the state label"
python code/run_matched.py $FORCE

echo "[4/9] Study 2 - MTEP matched case-control at lead L = 1 (9 configurations)"
echo "      job 0 is the position check: its pooled AUC CI must include 0.5"
for j in $(seq 0 8); do python code/run_casecontrol.py "$j" $FORCE; done

echo "[5/9] MTEP step 6b - reporting-act lead sweep (L = 0..3)"
python code/run_leadsweep.py $FORCE

echo "[6/9] paired significance tests"
python code/run_stats.py

echo "[7/9] sensitivity grid (33 cells, resumable)"
python code/run_sensitivity.py $FORCE

echo "[8/9] matched-set permutation null (200 permutations, resumable)"
python code/run_permutation.py 200 $FORCE

echo "[9/9] tables and figures"
python code/make_tables.py
python code/make_figures.py
echo "done. tables -> TABLES.md and results/tables.json, figures -> figures/"
