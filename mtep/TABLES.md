# Tables generated from results/


## T1. Characteristics of the analysed CogBeacon corpus.

| Property | Value |
|---|---|
| Session folders analysed | 77 |
| Participant identifiers | 20 |
| Rounds with complete EEG and task records | 6972 |
| Fatigue button presses (events) | 79 |
| Press rate per round | 1.13% |
| Sessions containing no press | 42 of 77 |
| Participants with at least one press | 15 |
| Features per round (EEG / facial / task and timing) | 97 (72 / 13 / 12) |
| Window length (rounds) | 10 |

## T2. Study 1, conventional sliding-window protocol, grouped leave-one-participant-out. Pooled ROC-AUC from concatenated out-of-fold predictions with participant-clustered bootstrap 95% CI; fold-wise values are mean ± SD over participants whose held-out windows contain both classes (Folds).

| Configuration | Prev. | Pooled ROC-AUC [95% CI] | Pooled PR-AUC | Fold ROC-AUC | Fold bal. acc. | Folds |
|---|---|---|---|---|---|---|
| State: all features (Transformer) | 0.346 | 0.678 [0.622, 0.744] | 0.590 | 0.850 ± 0.109 | 0.694 ± 0.107 | 15/20 |
| State: all features (LR) | 0.346 | 0.767 [0.701, 0.828] | 0.630 | 0.868 ± 0.087 | 0.724 ± 0.100 | 15/20 |
| State: all features (RF) | 0.346 | 0.732 [0.668, 0.802] | 0.531 | 0.893 ± 0.032 | 0.717 ± 0.120 | 15/20 |
| State: all features (SVM) | 0.346 | 0.750 [0.682, 0.815] | 0.600 | 0.878 ± 0.065 | 0.723 ± 0.087 | 15/20 |
| State: position only (normalised) (LR) | 0.346 | 0.789 [0.734, 0.852] | 0.601 | 0.933 ± 0.054 | 0.831 ± 0.047 | 15/20 |
| State: elapsed rounds only (LR) | 0.346 | 0.774 [0.724, 0.832] | 0.597 | 0.915 ± 0.067 | 0.819 ± 0.060 | 15/20 |
| State: EEG only (Transformer) | 0.346 | 0.563 [0.442, 0.692] | 0.452 | 0.603 ± 0.194 | 0.555 ± 0.109 | 15/20 |
| State: facial only (Transformer) | 0.346 | 0.545 [0.426, 0.655] | 0.441 | 0.518 ± 0.191 | 0.492 ± 0.101 | 15/20 |
| State: task/timing only (Transformer) | 0.346 | 0.721 [0.669, 0.777] | 0.532 | 0.819 ± 0.106 | 0.733 ± 0.084 | 15/20 |
| State: EEG + facial (Transformer) | 0.346 | 0.538 [0.419, 0.660] | 0.412 | 0.564 ± 0.181 | 0.526 ± 0.095 | 15/20 |
| Event, D = 0: all features (Transformer) | 0.012 | 0.708 [0.582, 0.830] | 0.051 | 0.872 ± 0.115 | 0.733 ± 0.165 | 15/20 |
| Event, D = 1: all features (Transformer) | 0.025 | 0.728 [0.650, 0.815] | 0.116 | 0.804 ± 0.139 | 0.709 ± 0.112 | 15/20 |
| Event, D = 3: all features (Transformer) | 0.049 | 0.607 [0.529, 0.715] | 0.098 | 0.748 ± 0.155 | 0.624 ± 0.111 | 15/20 |
| Event, D = 5: all features (Transformer) | 0.072 | 0.614 [0.521, 0.711] | 0.140 | 0.704 ± 0.146 | 0.582 ± 0.068 | 15/20 |

## T3. Reporting-act diagnostic: pooled ROC-AUC of the matched design as a function of the lead L between the end of the case window and the press. At L = 0 the case window contains the round in which the button was pressed.

| Lead L | Cases | Position (LR) | All (RF) | Task/timing (RF) | Facial (RF) | EEG+facial (RF) | EEG (RF) |
|---|---|---|---|---|---|---|---|
| 0 | 77 | 0.508 | 0.888 | 0.827 | 0.828 | 0.760 | 0.562 |
| 1 | 68 | 0.493 | 0.659 | 0.707 | 0.622 | 0.611 | 0.505 |
| 2 | 66 | 0.509 | 0.645 | 0.685 | 0.509 | 0.516 | 0.500 |
| 3 | 63 | 0.517 | 0.590 | 0.627 | 0.448 | 0.532 | 0.526 |

## T4. Study 2, MTEP within-session case-control design at lead L = 1 (68 cases, 204 controls, 34 sessions, 15 participants). Pooled values from concatenated out-of-fold predictions with participant-clustered bootstrap 95% CI; fold-wise values are mean ± SD over the held-out participants.

| Configuration | Pooled ROC-AUC [95% CI] | Pooled PR-AUC | Fold ROC-AUC | Fold bal. acc. |
|---|---|---|---|---|
| Position only (LR) (validity check) | 0.493 [0.482, 0.512] | 0.252 | 0.524 ± 0.098 | 0.512 ± 0.040 |
| All features (RF) | 0.659 [0.612, 0.733] | 0.412 | 0.759 ± 0.196 | 0.520 ± 0.084 |
| All features (LR) | 0.581 [0.532, 0.640] | 0.347 | 0.659 ± 0.176 | 0.543 ± 0.092 |
| Task/timing only (RF) | 0.707 [0.645, 0.789] | 0.467 | 0.776 ± 0.189 | 0.675 ± 0.194 |
| EEG + facial (RF) | 0.611 [0.551, 0.672] | 0.320 | 0.569 ± 0.219 | 0.489 ± 0.026 |
| EEG + facial (LR) | 0.528 [0.478, 0.585] | 0.294 | 0.524 ± 0.178 | 0.537 ± 0.094 |
| Facial only (RF) | 0.622 [0.537, 0.694] | 0.411 | 0.644 ± 0.223 | 0.554 ± 0.192 |
| EEG only (RF) | 0.516 [0.456, 0.562] | 0.280 | 0.455 ± 0.235 | 0.489 ± 0.043 |
| All features (Transformer) | 0.610 [0.541, 0.692] | 0.348 | 0.684 ± 0.243 | 0.585 ± 0.179 |

## T5. Label-permutation null for the controlled design: case status permuted within each matched set, full leave-one-participant-out procedure repeated, same random-forest model as reported. The null was run on a separate machine; its observed values differ from Table 4 in the third decimal, and both runs' observed values exceed every null value, so p is the minimum attainable, 1/(n + 1).

| Model | Observed pooled AUC | Null mean | Null 95th pct. | Permutations | p |
|---|---|---|---|---|---|
| All features (RF) | 0.661 | 0.497 | 0.556 | 200 | 0.005 |
| EEG + facial (RF) | 0.610 | 0.495 | 0.560 | 200 | 0.005 |

## T6. Paired Wilcoxon signed-rank comparisons of fold-wise ROC-AUC, paired by participant, Holm-Bonferroni corrected across the family. W/T/L: participants on which A is better / tied / worse. Effect: matched-pairs rank-biserial correlation over non-zero differences.

| A | B | Mean A | Mean B | W/T/L | p (Holm) | Effect | Sig. |
|---|---|---|---|---|---|---|---|
| All features (RF) | All features (Transformer) | 0.759 | 0.684 | 6/4/5 | 1.000 | +0.33 | No |
| All features (RF) | All features (LR) | 0.759 | 0.659 | 9/3/3 | .399 | +0.64 | No |
| All features (RF) | EEG + facial (RF) | 0.759 | 0.569 | 11/0/4 | .511 | +0.53 | No |
| EEG + facial (RF) | EEG only (RF) | 0.569 | 0.455 | 8/2/5 | .511 | +0.56 | No |
| EEG + facial (RF) | Facial only (RF) | 0.569 | 0.644 | 6/1/8 | 1.000 | -0.16 | No |
| All features (RF) | Task/timing only (RF) | 0.759 | 0.776 | 4/5/6 | 1.000 | -0.27 | No |
| All features (RF) | Position only (LR) | 0.759 | 0.524 | 13/0/2 | .026 | +0.88 | Yes |
| EEG + facial (RF) | Position only (LR) | 0.569 | 0.524 | 8/1/6 | 1.000 | +0.30 | No |
| Task/timing only (RF) | Position only (LR) | 0.776 | 0.524 | 14/0/1 | .018 | +0.92 | Yes |
| All features (Transformer) | All features (LR) | 0.684 | 0.659 | 7/2/6 | 1.000 | +0.05 | No |

## T7. Sensitivity of the controlled result (L = 1) across 33 design configurations varying window length, minimum gap, maximum offset, controls per case and sampling seed, and across the 17 configurations that pass the position check (position-only CI includes 0.5). Last column: number of configurations whose 95% CI lies entirely above / below 0.5.

| Configurations | Model | Median pooled AUC | Minimum | Maximum | CI > .5 / CI < .5 |
|---|---|---|---|---|---|
| All 33 | Position only (validity check) | 0.505 | 0.444 | 0.536 | 12 / 4 |
| All 33 | EEG + facial (RF) | 0.575 | 0.494 | 0.622 | 20 / 0 |
| All 33 | Task/timing only (RF) | 0.696 | 0.633 | 0.739 | 33 / 0 |
| All 33 | All features (RF) | 0.657 | 0.616 | 0.703 | 33 / 0 |
| Passing (17) | Position only (validity check) | 0.500 | 0.493 | 0.514 | 0 / 0 |
| Passing (17) | EEG + facial (RF) | 0.580 | 0.494 | 0.619 | 11 / 0 |
| Passing (17) | Task/timing only (RF) | 0.707 | 0.656 | 0.739 | 17 / 0 |
| Passing (17) | All features (RF) | 0.659 | 0.616 | 0.703 | 17 / 0 |

## S1. Per-participant ROC-AUC in the controlled design (L = 1), by CogBeacon participant identifier. Cases: number of case windows for that participant; folds with one or two cases can only take a few discrete AUC values, which is why pooled estimates are primary.

| Participant | Cases | Windows | Position | EEG | Facial | EEG+fac. | Task | All RF | All Tr. |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 9 | 36 | 0.519 | 0.556 | 0.621 | 0.551 | 0.646 | 0.638 | 0.564 |
| 1 | 6 | 24 | 0.546 | 0.676 | 0.565 | 0.537 | 0.741 | 0.602 | 0.426 |
| 2 | 6 | 24 | 0.519 | 0.444 | 0.421 | 0.481 | 0.741 | 0.574 | 0.630 |
| 3 | 11 | 44 | 0.463 | 0.507 | 0.758 | 0.730 | 0.534 | 0.612 | 0.590 |
| 4 | 5 | 20 | 0.613 | 0.613 | 0.560 | 0.773 | 0.693 | 0.827 | 0.560 |
| 6 | 5 | 20 | 0.480 | 0.827 | 0.840 | 0.667 | 0.680 | 0.653 | 0.333 |
| 9 | 4 | 16 | 0.562 | 0.271 | 0.646 | 0.833 | 0.688 | 0.646 | 0.729 |
| 10 | 1 | 4 | 0.667 | 0.000 | 0.333 | 0.000 | 1.000 | 1.000 | 1.000 |
| 11 | 4 | 16 | 0.521 | 0.625 | 0.438 | 0.625 | 0.396 | 0.458 | 0.479 |
| 12 | 2 | 8 | 0.667 | 0.250 | 1.000 | 0.500 | 0.750 | 0.583 | 0.667 |
| 13 | 2 | 8 | 0.333 | 0.500 | 0.917 | 0.917 | 1.000 | 1.000 | 1.000 |
| 14 | 3 | 12 | 0.556 | 0.556 | 0.407 | 0.519 | 1.000 | 1.000 | 0.370 |
| 16 | 2 | 8 | 0.583 | 0.417 | 0.417 | 0.500 | 1.000 | 1.000 | 1.000 |
| 17 | 1 | 4 | 0.333 | 0.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 |
| 19 | 7 | 28 | 0.503 | 0.585 | 0.735 | 0.571 | 0.776 | 0.789 | 0.918 |

## S2. Complete sensitivity grid (33 configurations) at lead L = 1: pooled ROC-AUC. W window length, g minimum gap to any press, m maximum offset, k controls per case.

| W | g | m | k | Seed | Cases | n | Position | Check | EEG+facial | Task | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 3 | 20 | 3 | 1337 | 74 | 296 | 0.508 | FAIL | 0.575 | 0.633 | 0.629 |
| 5 | 3 | 30 | 3 | 1337 | 74 | 296 | 0.508 | FAIL | 0.575 | 0.633 | 0.629 |
| 5 | 3 | 50 | 3 | 1337 | 74 | 296 | 0.508 | FAIL | 0.575 | 0.633 | 0.629 |
| 5 | 5 | 20 | 3 | 1337 | 74 | 296 | 0.526 | FAIL | 0.559 | 0.696 | 0.653 |
| 5 | 5 | 30 | 3 | 1337 | 74 | 296 | 0.526 | FAIL | 0.559 | 0.696 | 0.653 |
| 5 | 5 | 50 | 3 | 1337 | 74 | 296 | 0.526 | FAIL | 0.559 | 0.696 | 0.653 |
| 5 | 10 | 20 | 3 | 1337 | 74 | 296 | 0.536 | FAIL | 0.622 | 0.691 | 0.675 |
| 5 | 10 | 30 | 3 | 1337 | 74 | 296 | 0.536 | FAIL | 0.622 | 0.691 | 0.675 |
| 5 | 10 | 50 | 3 | 1337 | 74 | 296 | 0.536 | FAIL | 0.622 | 0.691 | 0.675 |
| 10 | 3 | 20 | 3 | 1337 | 68 | 272 | 0.505 | pass | 0.592 | 0.716 | 0.662 |
| 10 | 3 | 30 | 3 | 1337 | 68 | 272 | 0.505 | pass | 0.592 | 0.716 | 0.662 |
| 10 | 3 | 50 | 3 | 1337 | 68 | 272 | 0.505 | pass | 0.592 | 0.716 | 0.662 |
| 10 | 5 | 20 | 3 | 1337 | 68 | 272 | 0.493 | pass | 0.611 | 0.707 | 0.659 |
| 10 | 5 | 30 | 1 | 1337 | 68 | 136 | 0.472 | FAIL | 0.521 | 0.699 | 0.639 |
| 10 | 5 | 30 | 2 | 1337 | 68 | 204 | 0.514 | pass | 0.575 | 0.726 | 0.676 |
| 10 | 5 | 30 | 3 | 7 | 68 | 272 | 0.499 | pass | 0.580 | 0.739 | 0.678 |
| 10 | 5 | 30 | 3 | 42 | 68 | 272 | 0.502 | pass | 0.619 | 0.725 | 0.703 |
| 10 | 5 | 30 | 3 | 1337 | 68 | 272 | 0.493 | pass | 0.611 | 0.707 | 0.659 |
| 10 | 5 | 30 | 3 | 2024 | 68 | 272 | 0.500 | pass | 0.612 | 0.706 | 0.677 |
| 10 | 5 | 30 | 5 | 1337 | 68 | 408 | 0.493 | pass | 0.579 | 0.708 | 0.657 |
| 10 | 5 | 50 | 3 | 1337 | 68 | 272 | 0.493 | pass | 0.611 | 0.707 | 0.659 |
| 10 | 10 | 20 | 3 | 1337 | 68 | 272 | 0.531 | FAIL | 0.580 | 0.726 | 0.698 |
| 10 | 10 | 30 | 3 | 1337 | 68 | 272 | 0.531 | FAIL | 0.580 | 0.726 | 0.698 |
| 10 | 10 | 50 | 3 | 1337 | 68 | 272 | 0.531 | FAIL | 0.580 | 0.726 | 0.698 |
| 15 | 3 | 20 | 3 | 1337 | 60 | 240 | 0.507 | pass | 0.573 | 0.656 | 0.624 |
| 15 | 3 | 30 | 3 | 1337 | 60 | 240 | 0.507 | pass | 0.573 | 0.656 | 0.624 |
| 15 | 3 | 50 | 3 | 1337 | 60 | 240 | 0.507 | pass | 0.573 | 0.656 | 0.624 |
| 15 | 5 | 20 | 3 | 1337 | 60 | 240 | 0.500 | pass | 0.494 | 0.695 | 0.616 |
| 15 | 5 | 30 | 3 | 1337 | 60 | 240 | 0.500 | pass | 0.494 | 0.695 | 0.616 |
| 15 | 5 | 50 | 3 | 1337 | 60 | 240 | 0.500 | pass | 0.494 | 0.695 | 0.616 |
| 15 | 10 | 20 | 3 | 1337 | 58 | 231 | 0.444 | FAIL | 0.515 | 0.682 | 0.617 |
| 15 | 10 | 30 | 3 | 1337 | 58 | 231 | 0.444 | FAIL | 0.515 | 0.682 | 0.617 |
| 15 | 10 | 50 | 3 | 1337 | 58 | 231 | 0.444 | FAIL | 0.515 | 0.682 | 0.617 |

## S3. Complete paired tests. W: Wilcoxon statistic; test: exact null when no zero or tied differences, otherwise normal approximation; zero differences discarded (Wilcoxon zero method).

| A | B | n | W/T/L | Median diff. | W | Test | p | p (Holm) | Effect |
|---|---|---|---|---|---|---|---|---|---|
| All features (RF) | All features (Transformer) | 15 | 6/4/5 | +0.000 | 22.0 | normal approx. | .328 | 1.000 | +0.33 |
| All features (RF) | All features (LR) | 15 | 9/3/3 | +0.027 | 14.0 | normal approx. | .050 | .399 | +0.64 |
| All features (RF) | EEG + facial (RF) | 15 | 11/0/4 | +0.083 | 28.0 | exact | .073 | .511 | +0.53 |
| EEG + facial (RF) | EEG only (RF) | 15 | 8/2/5 | +0.037 | 20.0 | normal approx. | .075 | .511 | +0.56 |
| EEG + facial (RF) | Facial only (RF) | 15 | 6/1/8 | -0.028 | 44.0 | normal approx. | .594 | 1.000 | -0.16 |
| All features (RF) | Task/timing only (RF) | 15 | 4/5/6 | +0.000 | 20.0 | normal approx. | .445 | 1.000 | -0.27 |
| All features (RF) | Position only (LR) | 15 | 13/0/2 | +0.173 | 7.5 | normal approx. | .003 | .026 | +0.88 |
| EEG + facial (RF) | Position only (LR) | 15 | 8/1/6 | +0.033 | 37.0 | normal approx. | .331 | 1.000 | +0.30 |
| Task/timing only (RF) | Position only (LR) | 15 | 14/0/1 | +0.200 | 5.0 | normal approx. | .002 | .018 | +0.92 |
| All features (Transformer) | All features (LR) | 15 | 7/2/6 | +0.000 | 43.0 | normal approx. | .861 | 1.000 | +0.05 |

## S4. Feature inventory in the column order of data/dataset.npz, with missingness before imputation (training-fold median). Electrode labels follow the Muse SDK channel order (TP9, AF7, AF8, TP10); the corpus documentation does not list the order. Tags: A-prefixed = absolute band power, unprefixed = relative band power (a alpha, b beta, d delta, g gamma, t theta), as documented in the corpus README.

| # | Feature | Group | Description | Missing |
|---|---|---|---|---|
| 1 | eeg_Aa_mean_ch0 | EEG | electrode TP9 | 0.4% |
| 2 | eeg_Aa_mean_ch1 | EEG | electrode AF7 | 0.4% |
| 3 | eeg_Aa_mean_ch2 | EEG | electrode AF8 | 0.4% |
| 4 | eeg_Aa_mean_ch3 | EEG | electrode TP10 | 0.4% |
| 5 | eeg_Ab_mean_ch0 | EEG | electrode TP9 | 0.4% |
| 6 | eeg_Ab_mean_ch1 | EEG | electrode AF7 | 0.4% |
| 7 | eeg_Ab_mean_ch2 | EEG | electrode AF8 | 0.4% |
| 8 | eeg_Ab_mean_ch3 | EEG | electrode TP10 | 0.4% |
| 9 | eeg_Ad_mean_ch0 | EEG | electrode TP9 | 0.4% |
| 10 | eeg_Ad_mean_ch1 | EEG | electrode AF7 | 0.4% |
| 11 | eeg_Ad_mean_ch2 | EEG | electrode AF8 | 0.4% |
| 12 | eeg_Ad_mean_ch3 | EEG | electrode TP10 | 0.4% |
| 13 | eeg_Ag_mean_ch0 | EEG | electrode TP9 | 0.4% |
| 14 | eeg_Ag_mean_ch1 | EEG | electrode AF7 | 0.4% |
| 15 | eeg_Ag_mean_ch2 | EEG | electrode AF8 | 0.4% |
| 16 | eeg_Ag_mean_ch3 | EEG | electrode TP10 | 0.4% |
| 17 | eeg_At_mean_ch0 | EEG | electrode TP9 | 0.4% |
| 18 | eeg_At_mean_ch1 | EEG | electrode AF7 | 0.4% |
| 19 | eeg_At_mean_ch2 | EEG | electrode AF8 | 0.4% |
| 20 | eeg_At_mean_ch3 | EEG | electrode TP10 | 0.4% |
| 21 | eeg_a_mean_ch0 | EEG | electrode TP9 | 1.3% |
| 22 | eeg_a_mean_ch1 | EEG | electrode AF7 | 2.9% |
| 23 | eeg_a_mean_ch2 | EEG | electrode AF8 | 32.8% |
| 24 | eeg_a_mean_ch3 | EEG | electrode TP10 | 1.1% |
| 25 | eeg_b_mean_ch0 | EEG | electrode TP9 | 1.3% |
| 26 | eeg_b_mean_ch1 | EEG | electrode AF7 | 2.9% |
| 27 | eeg_b_mean_ch2 | EEG | electrode AF8 | 32.8% |
| 28 | eeg_b_mean_ch3 | EEG | electrode TP10 | 1.1% |
| 29 | eeg_d_mean_ch0 | EEG | electrode TP9 | 1.3% |
| 30 | eeg_d_mean_ch1 | EEG | electrode AF7 | 2.9% |
| 31 | eeg_d_mean_ch2 | EEG | electrode AF8 | 32.8% |
| 32 | eeg_d_mean_ch3 | EEG | electrode TP10 | 1.1% |
| 33 | eeg_g_mean_ch0 | EEG | electrode TP9 | 1.3% |
| 34 | eeg_g_mean_ch1 | EEG | electrode AF7 | 2.9% |
| 35 | eeg_g_mean_ch2 | EEG | electrode AF8 | 32.8% |
| 36 | eeg_g_mean_ch3 | EEG | electrode TP10 | 1.1% |
| 37 | eeg_t_mean_ch0 | EEG | electrode TP9 | 1.3% |
| 38 | eeg_t_mean_ch1 | EEG | electrode AF7 | 2.9% |
| 39 | eeg_t_mean_ch2 | EEG | electrode AF8 | 32.8% |
| 40 | eeg_t_mean_ch3 | EEG | electrode TP10 | 1.1% |
| 41 | eeg_a_sd_ch0 | EEG | electrode TP9 | 1.3% |
| 42 | eeg_a_sd_ch1 | EEG | electrode AF7 | 2.9% |
| 43 | eeg_a_sd_ch2 | EEG | electrode AF8 | 32.8% |
| 44 | eeg_a_sd_ch3 | EEG | electrode TP10 | 1.1% |
| 45 | eeg_b_sd_ch0 | EEG | electrode TP9 | 1.3% |
| 46 | eeg_b_sd_ch1 | EEG | electrode AF7 | 2.9% |
| 47 | eeg_b_sd_ch2 | EEG | electrode AF8 | 32.8% |
| 48 | eeg_b_sd_ch3 | EEG | electrode TP10 | 1.1% |
| 49 | eeg_d_sd_ch0 | EEG | electrode TP9 | 1.3% |
| 50 | eeg_d_sd_ch1 | EEG | electrode AF7 | 2.9% |
| 51 | eeg_d_sd_ch2 | EEG | electrode AF8 | 32.8% |
| 52 | eeg_d_sd_ch3 | EEG | electrode TP10 | 1.1% |
| 53 | eeg_g_sd_ch0 | EEG | electrode TP9 | 1.3% |
| 54 | eeg_g_sd_ch1 | EEG | electrode AF7 | 2.9% |
| 55 | eeg_g_sd_ch2 | EEG | electrode AF8 | 32.8% |
| 56 | eeg_g_sd_ch3 | EEG | electrode TP10 | 1.1% |
| 57 | eeg_t_sd_ch0 | EEG | electrode TP9 | 1.3% |
| 58 | eeg_t_sd_ch1 | EEG | electrode AF7 | 2.9% |
| 59 | eeg_t_sd_ch2 | EEG | electrode AF8 | 32.8% |
| 60 | eeg_t_sd_ch3 | EEG | electrode TP10 | 1.1% |
| 61 | eeg_theta_alpha_ch0 | EEG | electrode TP9 | 1.3% |
| 62 | eeg_theta_alpha_ch1 | EEG | electrode AF7 | 2.9% |
| 63 | eeg_theta_alpha_ch2 | EEG | electrode AF8 | 32.8% |
| 64 | eeg_theta_alpha_ch3 | EEG | electrode TP10 | 1.1% |
| 65 | eeg_engagement_ch0 | EEG | electrode TP9 | 1.3% |
| 66 | eeg_engagement_ch1 | EEG | electrode AF7 | 2.9% |
| 67 | eeg_engagement_ch2 | EEG | electrode AF8 | 32.8% |
| 68 | eeg_engagement_ch3 | EEG | electrode TP10 | 1.1% |
| 69 | eeg_quality_ch0 | EEG | electrode TP9 | 0.4% |
| 70 | eeg_quality_ch1 | EEG | electrode AF7 | 0.4% |
| 71 | eeg_quality_ch2 | EEG | electrode AF8 | 0.4% |
| 72 | eeg_quality_ch3 | EEG | electrode TP10 | 0.4% |
| 73 | face_eye_extent_mean | Facial | eye landmark extent ratio (h/w), mean | 11.2% |
| 74 | face_eye_extent_sd | Facial | eye extent ratio, SD | 11.2% |
| 75 | face_mouth_extent_mean | Facial | mouth extent ratio, mean | 11.2% |
| 76 | face_mouth_extent_sd | Facial | mouth extent ratio, SD | 11.2% |
| 77 | face_bbox_w_px | Facial | face box width, pixels | 0.0% |
| 78 | face_bbox_h_px | Facial | face box height, pixels | 0.0% |
| 79 | face_bbox_x_px | Facial | face box x, pixels | 0.0% |
| 80 | face_bbox_y_px | Facial | face box y, pixels | 0.0% |
| 81 | face_motion_mean | Facial | centroid motion / box width, mean | 0.0% |
| 82 | face_motion_sd | Facial | centroid motion, SD | 0.0% |
| 83 | face_motion_max | Facial | centroid motion, max | 0.0% |
| 84 | perf_round_frames | Task/timing | video frames in round (duration at 2 fps) | 0.0% |
| 85 | face_head_x_sd | Facial | centroid x / box width, SD | 0.0% |
| 86 | face_head_y_sd | Facial | centroid y / box width, SD | 0.0% |
| 87 | perf_level | Task/timing |  | 0.0% |
| 88 | perf_score | Task/timing |  | 0.0% |
| 89 | perf_response_time | Task/timing |  | 0.0% |
| 90 | perf_correct | Task/timing |  | 0.0% |
| 91 | perf_question | Task/timing |  | 0.0% |
| 92 | perf_persistence | Task/timing |  | 0.0% |
| 93 | perf_round_norm | Task/timing | round index / session length | 0.0% |
| 94 | perf_d_nonper_err | Task/timing |  | 0.0% |
| 95 | perf_d_per_err | Task/timing |  | 0.0% |
| 96 | perf_rt_dev | Task/timing | response time z-score vs last 10 rounds | 0.0% |
| 97 | perf_acc_run | Task/timing | accuracy over last 10 rounds | 0.0% |

## S5. Failed attempts to remove the position confound from the cumulative-state label by position-decile matching (Section 4.3). The position-only model remains far above chance under both.

| Matching | Model | Windows | Pooled ROC-AUC [95% CI] | Fold ROC-AUC |
|---|---|---|---|---|
| Corpus-level decile matching | Position only (LR) | 4328 | 0.703 [0.628, 0.798] | 0.923 ± 0.096 |
| Corpus-level decile matching | All features (LR) | 4328 | 0.704 [0.608, 0.791] | 0.790 ± 0.096 |
| Within-participant decile matching | Position only (LR) | 2278 | 0.894 [0.848, 0.938] | 0.919 ± 0.093 |
| Within-participant decile matching | All features (LR) | 2278 | 0.655 [0.592, 0.726] | 0.733 ± 0.139 |

## S6. Model settings, identical in both studies. In every outer fold all models are trained on the same participants; the validation participants are used only for Transformer early stopping.

| Model | Settings |
|---|---|
| Logistic regression | L2, C = 1.0, class_weight = balanced, max_iter = 3000; input: window mean, SD and final value |
| Random forest | 400 trees, min_samples_leaf = 3, class_weight = balanced_subsample; same input |
| SVM (Study 1) | RBF kernel, C = 1.0, gamma = scale, class_weight = balanced; score = logistic(decision function), so 0.5 is the SVM decision boundary; same input |
| Transformer | linear projection to 48, sinusoidal positions, 2 layers, 4 heads, GELU, dropout 0.2, AdamW (lr 1e-3, wd 1e-2), batch 256, ≤ 15 epochs, patience 4 on validation ROC-AUC, class-weighted BCE, gradient clipping 1.0 |
| Validation split | 1/8 of training participants (at least one), drawn once per outer fold with seed 1337 |
| Hyperparameter tuning | none for any model |
| Early-stopping fallback folds | state|all|transformer: 2; state|eeg_only|transformer: 2; state|face_only|transformer: 2; state|task_only|transformer: 2; state|eeg+face|transformer: 2; event_d0|all|transformer: 2; event_d1|all|transformer: 2; event_d3|all|transformer: 2; event_d5|all|transformer: 2; All features (Transformer): 0 |

## S7. Medians of selected per-round features in the 79 rounds containing a press and in all other rounds (Section 4.4).

| Feature | Press rounds | Other rounds | Ratio |
|---|---|---|---|
| Maximum centroid displacement (box widths) | 0.607 | 0.0304 | 19.94 |
| Video frames in round | 13 | 9 | 1.44 |
| Response time | 3 | 2 | 1.50 |