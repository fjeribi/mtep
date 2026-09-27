# MTEP — Matched Temporal Evaluation Protocol

Code and derived data accompanying *MTEP: A Matched Temporal Evaluation Protocol
for Cognitive Fatigue Detection from In-Task Self-Reports*.

Every number, table and figure in the manuscript and the supplement is generated
by the scripts below from the files in `results/`. `make_tables.py` writes every
table to `TABLES.md` and
`results/tables.json`; `make_figures.py` and `make_merged_figures.py` write the figures.

## Where each table and figure comes from

`TABLES.md` uses internal labels. Mapping to the manuscript (v5) and supplement:

| Manuscript | Source |
|---|---|
| Table 1, Table 2 | text tables (design and requirement) |
| Tables 3, 4, 5 and 6 | `make_sim_tables.py` (calibration, head-to-head, certification) |
| Table 7 | `TABLES.md` T4 |
| Study 1 values in Section 5.3 | `TABLES.md` T2 |
| Tables S1–S7 | `TABLES.md` S1–S7 |
| Table S8 | `TABLES.md` T2, all rows |
| Table S9 | `make_sim_tables.py` (certified-lead robustness) |
| Fig. 1 | `fig1_protocol` |
| Fig. 2 | `fig2_calibration` |
| Fig. 3 | `fig3_confound_single` |
| Fig. 4 | `fig4_temporal` |
| Fig. S1 | `fig8_per_participant` |
| Fig. S2 | `fig5_design` |


## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh /path/to/CogBeacon-MultiModal_Dataset_for_Cognitive_Fatigue
```

Run from any directory; all paths are resolved relative to the repository.
Completed steps are skipped, so an interrupted run (e.g. a power cut) can simply
be restarted; the sensitivity grid and the permutation null checkpoint their
progress. Pass `--force` as the second argument to recompute everything.
CPU only; roughly 4-5 hours on two cores, most of it the permutation null and
the simulation, both of which checkpoint and can be interrupted.

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
  run_matched.py       Section 5.3: position-decile matching on the state label
  run_casecontrol.py   Study 2: MTEP matched case-control at lead L = 1, 9 configs
  mtep.py              THE DESIGN, as a function over arrays (see below)
  certify.py           command-line front end: certify your own corpus
  test_mtep.py         invariants of the design, asserted
  run_leadsweep.py     MTEP step 6b: the lead sweep, L = 0..3
  run_leaddelta.py     displacement statistic D(L) and the certified lead
  run_certify_robustness.py  the certified lead under other estimators and
                       window lengths
  simlib.py            simulation: synthetic events, injection, lean CV, rules
  run_simulation.py    Section 5.1: report-independence certificate, act vs state
  run_simposition.py   Section 5.1: position certificate against known imbalance
  run_headtohead.py    Section 5.2: four designs, four worlds, known truth
  run_stats.py         paired Wilcoxon tests (paired by participant), Holm
  run_sensitivity.py   33-cell grid over the design's free parameters
  run_permutation.py   matched-set permutation null (same RF model as reported)
  make_tables.py       every table in the paper and supplement
  make_sim_tables.py   the two calibration tables and the certification table
  make_figures.py      Fig. S1, per-participant ROC-AUC
  make_merged_figures.py  figures 3, 4 and 5 as published (merged panels)
  make_sim_figure.py   figure 2, calibration of both certificates
  make_protocol_figure.py  figure 1, the design as a flow diagram
data/dataset.npz       derived feature matrix
results/
  study1/job*.json     per configuration: pooled and per-participant metrics,
  study2/job*.json       and the out-of-fold prediction for every window
  study2/design.json   case/control construction, offsets and accounting
  sim_lead/rep*.json   one calibration replicate each
  headtohead/rep*.json one head-to-head replicate each
  leadsweep.json, lead_delta.json, certify_robustness.json,
  matched_state.json, sensitivity.json,
  permutation.json, sim_position.json, simulation.json, stats.json, tables.json
figures/               PNGs at 300 dpi (see mapping below)
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


## Applying the design to your own corpus

Everything above is the CogBeacon application. The design itself is
`code/mtep.py`, which takes arrays:

```python
import sys; sys.path.insert(0, "code")
from mtep import run_mtep

out = run_mtep(X, y_event, session, participant, unit_index)
print(out["verdict"])
```

| array | shape | meaning |
|---|---|---|
| `X` | (n_units, n_features) | features, one row per unit |
| `y_event` | (n_units,) | 1 at a unit where a report was made |
| `session` | (n_units,) | recording-unit id (a session, a shift) |
| `participant` | (n_units,) | person id; folds are formed on this |
| `unit_index` | (n_units,) | 1-based position within the session |
| `position` | (n_units,) | optional; defaults to unit_index / session length |

The result carries both certificates, the certified lead, and the estimate
there. From the command line, with the same arrays saved in an `.npz`:

```bash
python code/certify.py yourdata.npz --out report.json
```

which prints, for the bundled corpus:

```
  P3 position balance      granted   position-only AUC 0.493 [0.482, 0.512]
  P4 report-independence   granted   certified lead 1
       D(0 to 1) = +0.228 [+0.165, +0.276]
       D(1 to 2) = +0.014 [-0.050, +0.089]   <- certifies at lead 1
       D(2 to 3) = +0.055 [+0.000, +0.124]

  certified at lead 1: pooled AUC 0.659 [0.612, 0.733]
```

### Recalibrating the certificates for your corpus

The specificity and sensitivity in Section 5.1 were measured on CogBeacon (Section 5.1) and
on a design of roughly 66 cases and 198 controls from 20 participants. To
measure them for your own corpus instead of borrowing ours, point the pipeline
at your file and re-run the calibration:

```bash
export MTEP_DATA=/path/to/yourdata.npz
python code/run_simulation.py --reps 20 --workers 2
python code/run_simposition.py --reps 40 --workers 2
python code/make_sim_tables.py
```

`MTEP_DATA` is read by every script in the release.

## Tests

```bash
python code/test_mtep.py          # ~2 minutes
python code/test_mtep.py --fast   # construction checks only, ~4 seconds
```

Sixteen assertions over the design: that no window contains an event unit once
a lead is applied, that case windows end exactly `lead` units before their
event, that controls respect the gap and are used once, that matched sets are
disjoint and within a participant, that the released design counts are
unchanged, that injection at amplitude zero is inert and a spike touches only
event units, that the certified-lead rule matches the stored result, and that
the public API reproduces the manuscript.
