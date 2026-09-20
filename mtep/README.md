# MTEP — Matched Temporal Evaluation Protocol

Code and derived data accompanying *MTEP: A Matched Temporal Evaluation Protocol
for Multimodal Cognitive Fatigue Detection from EEG and Facial Behaviour*.

Every number, table and figure in the manuscript and the supplement is generated
by the scripts below from the files in `results/`. `make_tables.py` writes every
table (main Tables 1–7 and supplementary Tables S1–S7) to `TABLES.md` and
`results/tables.json`; `make_figures.py` writes all eight figures.

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh /path/to/CogBeacon-MultiModal_Dataset_for_Cognitive_Fatigue
```

Run from any directory; all paths are resolved relative to the repository.
Completed steps are skipped, so an interrupted run (e.g. a power cut) can simply
be restarted; the sensitivity grid and the permutation null checkpoint their
progress. Pass `--force` as the second argument to recompute everything.
CPU only; roughly 2–3 hours on one core, most of it the permutation null.

## Data

The CogBeacon corpus is not redistributed. Obtain it from its authors:
https://github.com/MikeMpapa/CogBeacon-MultiModal_Dataset_for_Cognitive_Fatigue
and extract the four archives (`eeg`, `face_keypoints`, `fatigue_self_report`,
`user_performance`) in place.

`data/dataset.npz` is the derived per-round feature matrix (6,972 rounds × 97
features, press events, participant/session indices). It contains exactly the arrays that
`build_dataset.py` produces from the corpus as distributed (verified), so the analysis can
be re-run without the raw data by skipping step 1 of `run_all.sh`.

## Layout

```
code/
  common.py            shared loading, windowing, models, metrics, grouped CV
  build_dataset.py     raw corpus -> per-round feature table (+ name check)
  run_experiment.py    Study 1: conventional sliding-window protocol, 14 configs
  run_matched.py       Section 4.3: position-decile matching on the state label
  run_casecontrol.py   Study 2: MTEP matched case-control at lead L = 1, 9 configs
  run_leadsweep.py     MTEP step 6b: reporting-act diagnostic, L = 0..3
  run_stats.py         paired Wilcoxon tests (paired by participant), Holm
  run_sensitivity.py   33-cell grid over the design's free parameters
  run_permutation.py   matched-set permutation null (same RF model as reported)
  make_tables.py       every table in the paper and supplement
  make_figures.py      all eight figures
data/dataset.npz       derived feature matrix
results/
  study1/job*.json     per configuration: pooled and per-participant metrics,
  study2/job*.json       and the out-of-fold prediction for every window
  study2/design.json   case/control construction, offsets and accounting
  leadsweep.json, matched_state.json, sensitivity.json, permutation.json,
  stats.json, tables.json
figures/               eight PNGs at 300 dpi
```

## Label/feature separation

`build_dataset.py` reads labels only from `fatigue_self_report/` and features only
from `eeg/`, `face_keypoints/` and `user_performance/`, in functions that share no
state. A closing assertion fails the build if a feature *name* references the
label stream. This guards against a self-report column becoming a feature. It
cannot detect behavioural traces of the reporting act in the other streams —
that is what the lead sweep (MTEP step 6b) is for.

## The two validity checks

1. **Position (step 6a).** `run_casecontrol.py` job 0 fits a model on session
   position alone over the matched sample. Its pooled ROC-AUC confidence
   interval must include 0.5; otherwise the matching has failed.
2. **Event unit (step 6b).** `run_leadsweep.py` repeats the design with the case
   window ending L = 0, 1, 2, 3 rounds before the press. On CogBeacon, including
   the press round (L = 0) inflates the result substantially, because pressing
   the button moves the face and lengthens the round. Interpret only L ≥ 1.

Passing both checks is necessary, not sufficient: each detects only the
confound it tests.

## Porting MTEP to another corpus

Replace (1) the feature extractors and session/participant parsing in
`build_dataset.py`, keeping the separation and the name check; (2) the event
definition in `run_experiment.make_labels`; (3) the position variable. Where a
questionnaire is scored once per recording and samples are drawn at a finer
granularity, the confound is *which recording* a sample came from: substitute a
unit-identifying variable for the position variable in the diagnostic.

## Determinism

Seeded throughout (`SEED = 1337` in `common.py`). Classical models reproduce
exactly; Transformer results can differ in the third decimal across PyTorch
versions or hardware.

## Licence

Code released under the MIT Licence. The CogBeacon corpus is distributed by its
authors under their own terms; this repository redistributes no raw data.
