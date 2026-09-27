#!/usr/bin/env bash
# Full reproduction of every number, table and figure in the manuscript and supplement.
# Usage:  bash run_all.sh /path/to/CogBeacon-MultiModal_Dataset_for_Cognitive_Fatigue [--force]
#   --force  recompute results that already exist (otherwise completed steps are
#            skipped, so an interrupted run can simply be restarted).
set -euo pipefail
cd "$(dirname "$0")"
CORPUS="${1:?usage: bash run_all.sh /path/to/cogbeacon [--force]}"
FORCE="${2:-}"

echo "[1/11] feature table from the raw corpus (label/feature name check asserted)"
python code/build_dataset.py "$CORPUS" data/dataset.npz

echo "[2/11] Study 1 - conventional protocol (14 configurations)"
for j in $(seq 0 13); do python code/run_experiment.py "$j" $FORCE; done

echo "[3/11] Section 5.3 - position-decile matching attempts on the state label"
python code/run_matched.py $FORCE

echo "[4/11] Study 2 - MTEP matched case-control at lead L = 1 (9 configurations)"
echo "      job 0 is the position certificate: its pooled AUC CI must include 0.5"
for j in $(seq 0 8); do python code/run_casecontrol.py "$j" $FORCE; done

echo "[5/11] MTEP step 6b - lead sweep (L = 0..3)"
python code/run_leadsweep.py $FORCE

echo "[6/11] paired significance tests"
python code/run_stats.py

echo "[7/11] sensitivity grid (33 cells, resumable)"
python code/run_sensitivity.py $FORCE

echo "[8/11] matched-set permutation null (200 permutations, resumable)"
python code/run_permutation.py 200 $FORCE

echo "[9/11] certify report-independence on the corpus"
python code/run_leaddelta.py $FORCE

echo "[9b/11] is the certified lead robust to estimator and window length?"
python code/run_certify_robustness.py $FORCE

echo "[10/11] calibrate both certificates by simulation (resumable)"
echo "      20 replicates for the report-independence certificate, ~4 min each"
python code/run_simulation.py --reps 20 --workers 2
echo "      200 cells for the position certificate"
python code/run_simposition.py --reps 40 --workers 2

echo "[10b/11] head-to-head: do four designs reach the right conclusion? (resumable)"
python code/run_headtohead.py --reps 20 --workers 2

echo "[11/11] tables and figures"
python code/make_protocol_figure.py
python code/make_tables.py
python code/make_sim_tables.py
python code/make_figures.py
python code/make_merged_figures.py
python code/make_sim_figure.py
echo "done. tables -> TABLES.md and results/tables.json, figures -> figures/"
echo
echo "to check the design's invariants:      python code/test_mtep.py"
echo "to certify a corpus of your own:       python code/certify.py yourdata.npz"
