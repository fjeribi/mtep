"""
make_tables.py -- every table in the manuscript and the supplement, generated
from results/ only. Writes results/tables.json (consumed by the document
builders) and TABLES.md (human-readable).

Usage: python code/make_tables.py
"""
import glob, json, os
import numpy as np
import common as C

f3 = lambda x: f"{x:.3f}"
ci = lambda r: f"{r['pooled']['roc_auc']:.3f} [{r['pooled']['auc_ci95'][0]:.3f}, {r['pooled']['auc_ci95'][1]:.3f}]"
msd = lambda d: f"{d['mean']:.3f} ± {d['sd']:.3f}" if d else "—"
pfmt = lambda p: "< .001" if p < .001 else f"{p:.3f}".lstrip("0")


def load_dir(d):
    return [json.load(open(f)) for f in sorted(glob.glob(C.rpath(d, "job*.json")))]


def main():
    X, y_event, meta, names, groups = C.load()
    S1, S2 = load_dir("study1"), load_dir("study2")
    by2 = {r["name"]: r for r in S2}
    DES = C.load_json("study2", "design.json")
    LEAD = C.load_json("leadsweep.json")
    PERM = C.load_json("permutation.json") if os.path.exists(C.rpath("permutation.json")) else {}
    STATS = C.load_json("stats.json")
    SENS = C.load_json("sensitivity.json")
    MATCH = C.load_json("matched_state.json")
    T = {}

    sess_press = [int(y_event[m].sum()) for _, m in C.session_order(meta)]
    T["T1"] = dict(caption="Characteristics of the analysed CogBeacon corpus.",
        header=["Property", "Value"], rows=[
        ["Session folders analysed", str(len(sess_press))],
        ["Participant identifiers", str(len(np.unique(meta[:, 1])))],
        ["Rounds with complete EEG and task records", str(len(y_event))],
        ["Fatigue button presses (events)", str(int(y_event.sum()))],
        ["Press rate per round", f"{100 * y_event.mean():.2f}%"],
        ["Sessions containing no press", f"{sum(p == 0 for p in sess_press)} of {len(sess_press)}"],
        ["Participants with at least one press", str(len(np.unique(meta[y_event == 1, 1])))],
        ["Features per round (EEG / facial / task and timing)",
         f"{X.shape[1]} ({len(groups['eeg'])} / {len(groups['face'])} / {len(groups['perf'])})"],
        ["Window length (rounds)", "10"]])

    def pretty(n):
        task, fs, model = n.split("|")
        task = "State" if task == "state" else "Event, D = " + task.split("_d")[1]
        fs = {"all": "all features", "position_normalised_only": "position only (normalised)",
              "elapsed_rounds_only": "elapsed rounds only", "eeg_only": "EEG only", "face_only": "facial only",
              "task_only": "task/timing only", "eeg+face": "EEG + facial"}[fs]
        model = {"logreg": "LR", "rf": "RF", "svm": "SVM", "transformer": "Transformer"}[model]
        return f"{task}: {fs} ({model})"
    rows = []
    for r in S1:
        rows.append([pretty(r["name"]), f3(r["prevalence"]), ci(r),
                     f3(r["pooled"]["pr_auc"]), msd(r["fold_mean"]["roc_auc"]),
                     msd(r["fold_mean"]["balanced_accuracy"]),
                     f"{r['n_participants_scored']}/{r['n_participants_total']}"])
    T["T2"] = dict(caption=("Study 1, conventional sliding-window protocol, grouped leave-one-participant-out. "
        "Pooled ROC-AUC from concatenated out-of-fold predictions with participant-clustered bootstrap 95% CI; "
        "fold-wise values are mean ± SD over participants whose held-out windows contain both classes (Folds)."),
        header=["Configuration", "Prev.", "Pooled ROC-AUC [95% CI]", "Pooled PR-AUC",
                "Fold ROC-AUC", "Fold bal. acc.", "Folds"], rows=rows)

    lab = {"position": "Position (LR)", "all": "All (RF)", "task": "Task/timing (RF)",
           "face": "Facial (RF)", "eegface": "EEG+facial (RF)", "eeg": "EEG (RF)"}
    keys = ["position", "all", "task", "face", "eegface", "eeg"]
    T["T3"] = dict(caption=("Reporting-act diagnostic: pooled ROC-AUC of the matched design as a function of the lead L "
        "between the end of the case window and the press. At L = 0 the case window contains the round in which "
        "the button was pressed."),
        header=["Lead L", "Cases"] + [lab[k] for k in keys],
        rows=[[str(r["lead"]), str(r["design"]["n_cases"])] + [f3(r[k]["pooled_auc"]) for k in keys]
              for r in LEAD])

    order = ["Position only (LR) [validity check]", "All features (RF)", "All features (LR)",
             "Task/timing only (RF)", "EEG + facial (RF)", "EEG + facial (LR)", "Facial only (RF)",
             "EEG only (RF)", "All features (Transformer)"]
    T["T4"] = dict(caption=(f"Study 2, MTEP within-session case-control design at lead L = 1 "
        f"({DES['n_cases']} cases, {DES['n_controls']} controls, {DES['n_sessions']} sessions, "
        f"{DES['n_participants']} participants). Pooled values from concatenated out-of-fold predictions with "
        "participant-clustered bootstrap 95% CI; fold-wise values are mean ± SD over the held-out participants."),
        header=["Configuration", "Pooled ROC-AUC [95% CI]", "Pooled PR-AUC", "Fold ROC-AUC", "Fold bal. acc."],
        rows=[[k.replace(" [validity check]", " (validity check)"), ci(by2[k]),
               f3(by2[k]["pooled"]["pr_auc"]), msd(by2[k]["fold_mean"]["roc_auc"]),
               msd(by2[k]["fold_mean"]["balanced_accuracy"])] for k in order])

    T["T5"] = dict(caption=("Label-permutation null for the controlled design: case status permuted within each matched "
        "set, full leave-one-participant-out procedure repeated, same random-forest model as reported. The null was run on a "
        "separate machine; its observed values differ from Table 4 in the third decimal, and both runs' observed values "
        "exceed every null value, so p is the minimum attainable, 1/(n + 1)."),
        header=["Model", "Observed pooled AUC", "Null mean", "Null 95th pct.", "Permutations", "p"],
        rows=[[v["label"], f3(v["observed"]), f3(v["null_mean"]), f3(v["null_p95"]), str(v["n_perm"]),
               f"{v['p_value']:.3f}"] for v in PERM.values() if "p_value" in v]
             or [["All features (RF)", "pending", "pending", "pending", "pending", "pending"],
                 ["EEG + facial (RF)", "pending", "pending", "pending", "pending", "pending"]])

    cl = lambda s: s.replace(" [validity check]", "")
    T["T6"] = dict(caption=("Paired Wilcoxon signed-rank comparisons of fold-wise ROC-AUC, paired by participant, "
        "Holm-Bonferroni corrected across the family. W/T/L: participants on which A is better / tied / worse. "
        "Effect: matched-pairs rank-biserial correlation over non-zero differences."),
        header=["A", "B", "Mean A", "Mean B", "W/T/L", "p (Holm)", "Effect", "Sig."],
        rows=[[cl(r["a"]), cl(r["b"]), f3(r["mean_a"]), f3(r["mean_b"]),
               f"{r['wins']}/{r['ties']}/{r['losses']}", pfmt(r["p_holm"]),
               f"{r['rank_biserial']:+.2f}", "Yes" if r["significant"] else "No"] for r in STATS])

    passes = lambda r: r["position"]["ci95"][0] <= .5 <= r["position"]["ci95"][1]
    OK = [r for r in SENS if passes(r)]
    srow = []
    for subset, sname in [(SENS, f"All {len(SENS)}"), (OK, f"Passing ({len(OK)})")]:
        for k, nm in [("position", "Position only (validity check)"), ("eegface", "EEG + facial (RF)"),
                      ("task", "Task/timing only (RF)"), ("all", "All features (RF)")]:
            v = np.array([r[k]["pooled_auc"] for r in subset])
            lo = np.array([r[k]["ci95"][0] for r in subset]); hi = np.array([r[k]["ci95"][1] for r in subset])
            srow.append([sname, nm, f3(np.median(v)), f3(v.min()), f3(v.max()),
                         f"{int(np.sum(lo > .5))} / {int(np.sum(hi < .5))}"])
    T["T7"] = dict(caption=(f"Sensitivity of the controlled result (L = 1) across {len(SENS)} design configurations "
        "varying window length, minimum gap, maximum offset, controls per case and sampling seed, and across the "
        f"{len(OK)} configurations that pass the position check (position-only CI includes 0.5). Last column: number "
        "of configurations whose 95% CI lies entirely above / below 0.5."),
        header=["Configurations", "Model", "Median pooled AUC", "Minimum", "Maximum", "CI > .5 / CI < .5"], rows=srow)

    # ---------------- supplement ----------------
    pcols = [("Position only (LR) [validity check]", "Position"), ("EEG only (RF)", "EEG"),
             ("Facial only (RF)", "Facial"), ("EEG + facial (RF)", "EEG+fac."),
             ("Task/timing only (RF)", "Task"), ("All features (RF)", "All RF"),
             ("All features (Transformer)", "All Tr.")]
    parts = [f["participant"] for f in by2["All features (RF)"]["per_fold"]]
    rows = []
    for pid in parts:
        f0 = [f for f in by2["All features (RF)"]["per_fold"] if f["participant"] == pid][0]
        row = [str(pid), str(f0["n_pos"]), str(f0["n"])]
        for k, _ in pcols:
            v = [f["roc_auc"] for f in by2[k]["per_fold"] if f["participant"] == pid]
            row.append(f3(v[0]) if v else "—")
        rows.append(row)
    T["S1"] = dict(caption=("Per-participant ROC-AUC in the controlled design (L = 1), by CogBeacon participant "
        "identifier. Cases: number of case windows for that participant; folds with one or two cases can only take a "
        "few discrete AUC values, which is why pooled estimates are primary."),
        header=["Participant", "Cases", "Windows"] + [n for _, n in pcols], rows=rows)

    T["S2"] = dict(caption=(f"Complete sensitivity grid ({len(SENS)} configurations) at lead L = 1: pooled ROC-AUC. "
        "W window length, g minimum gap to any press, m maximum offset, k controls per case."),
        header=["W", "g", "m", "k", "Seed", "Cases", "n", "Position", "Check", "EEG+facial", "Task", "All"],
        rows=[[str(r["W"]), str(r["gap"]), str(r["maxoff"]), str(r["k"]), str(r["seed"]), str(r["n_cases"]),
               str(r["n_windows"]), f3(r["position"]["pooled_auc"]), "pass" if passes(r) else "FAIL"]
              + [f3(r[k]["pooled_auc"]) for k in ["eegface", "task", "all"]]
              for r in sorted(SENS, key=lambda r: (r["W"], r["gap"], r["maxoff"], r["k"], r["seed"]))])

    T["S3"] = dict(caption=("Complete paired tests. W: Wilcoxon statistic; test: exact null when no zero or tied "
        "differences, otherwise normal approximation; zero differences discarded (Wilcoxon zero method)."),
        header=["A", "B", "n", "W/T/L", "Median diff.", "W", "Test", "p", "p (Holm)", "Effect"],
        rows=[[cl(r["a"]), cl(r["b"]), str(r["n_folds"]), f"{r['wins']}/{r['ties']}/{r['losses']}",
               f"{r['median_diff']:+.3f}", f"{r['W']:.1f}", r["method"], pfmt(r["p_raw"]),
               pfmt(r["p_holm"]), f"{r['rank_biserial']:+.2f}"] for r in STATS])

    ELEC = ["TP9", "AF7", "AF8", "TP10"]
    miss = np.isnan(X).mean(0)
    def grp(n):
        return "EEG" if n.startswith("eeg_") else ("Facial" if n.startswith("face_") else "Task/timing")
    def desc(n):
        if n.startswith("eeg_"):
            ch = int(n[-1]); return f"electrode {ELEC[ch]}"
        return {"face_eye_extent_mean": "eye landmark extent ratio (h/w), mean",
                "face_eye_extent_sd": "eye extent ratio, SD", "face_mouth_extent_mean": "mouth extent ratio, mean",
                "face_mouth_extent_sd": "mouth extent ratio, SD", "face_bbox_w_px": "face box width, pixels",
                "face_bbox_h_px": "face box height, pixels", "face_bbox_x_px": "face box x, pixels",
                "face_bbox_y_px": "face box y, pixels", "face_motion_mean": "centroid motion / box width, mean",
                "face_motion_sd": "centroid motion, SD", "face_motion_max": "centroid motion, max",
                "face_head_x_sd": "centroid x / box width, SD", "face_head_y_sd": "centroid y / box width, SD",
                "perf_round_frames": "video frames in round (duration at 2 fps)",
                "perf_round_norm": "round index / session length",
                "perf_rt_dev": "response time z-score vs last 10 rounds",
                "perf_acc_run": "accuracy over last 10 rounds"}.get(n, "")
    T["S4"] = dict(caption=("Feature inventory in the column order of data/dataset.npz, with missingness before "
        "imputation (training-fold median). Electrode labels follow the Muse SDK channel order (TP9, AF7, AF8, "
        "TP10); the corpus documentation does not list the order. Tags: A-prefixed = absolute band power, "
        "unprefixed = relative band power (a alpha, b beta, d delta, g gamma, t theta), as documented in the "
        "corpus README."),
        header=["#", "Feature", "Group", "Description", "Missing"],
        rows=[[str(i + 1), n, grp(n), desc(n), f"{100 * miss[i]:.1f}%"] for i, n in enumerate(names)])

    rows = []
    for lvl, nm in [("corpus", "Corpus-level decile matching"), ("participant", "Within-participant decile matching")]:
        for t, tn in [("position", "Position only (LR)"), ("all", "All features (LR)")]:
            v = MATCH[lvl][t]
            rows.append([nm, tn, str(v["n_windows"]), f"{v['pooled_auc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]",
                         f"{v['fold_auc']:.3f} ± {v['fold_sd']:.3f}"])
    T["S5"] = dict(caption=("Failed attempts to remove the position confound from the cumulative-state label by "
        "position-decile matching (Section 4.3). The position-only model remains far above chance under both."),
        header=["Matching", "Model", "Windows", "Pooled ROC-AUC [95% CI]", "Fold ROC-AUC"], rows=rows)

    fb = {r["name"]: r["transformer_fallback_folds"] for r in S1 + S2 if r["model"] == "transformer"}
    T["S6"] = dict(caption=("Model settings, identical in both studies. In every outer fold all models are trained "
        "on the same participants; the validation participants are used only for Transformer early stopping."),
        header=["Model", "Settings"], rows=[
        ["Logistic regression", "L2, C = 1.0, class_weight = balanced, max_iter = 3000; input: window mean, SD and final value"],
        ["Random forest", "400 trees, min_samples_leaf = 3, class_weight = balanced_subsample; same input"],
        ["SVM (Study 1)", "RBF kernel, C = 1.0, gamma = scale, class_weight = balanced; score = logistic(decision function), so 0.5 is the SVM decision boundary; same input"],
        ["Transformer", "linear projection to 48, sinusoidal positions, 2 layers, 4 heads, GELU, dropout 0.2, "
         "AdamW (lr 1e-3, wd 1e-2), batch 256, ≤ 15 epochs, patience 4 on validation ROC-AUC, class-weighted BCE, "
         "gradient clipping 1.0"],
        ["Validation split", "1/8 of training participants (at least one), drawn once per outer fold with seed 1337"],
        ["Hyperparameter tuning", "none for any model"],
        ["Early-stopping fallback folds", "; ".join(f"{k}: {v}" for k, v in fb.items())]])

    fi, ri = names.index("face_motion_max"), names.index("perf_round_frames")
    rt = names.index("perf_response_time")
    prs = []
    for i, nm in [(fi, "Maximum centroid displacement (box widths)"), (ri, "Video frames in round"),
                  (rt, "Response time")]:
        a, b = np.nanmedian(X[y_event == 1, i]), np.nanmedian(X[y_event == 0, i])
        prs.append([nm, f"{a:.3g}", f"{b:.3g}", f"{a / b:.2f}"])
    T["S7"] = dict(caption=("Medians of selected per-round features in the 79 rounds containing a press and in all "
        "other rounds (Section 4.4)."), header=["Feature", "Press rounds", "Other rounds", "Ratio"], rows=prs)

    C.save_json(T, "tables.json")
    md = ["# Tables generated from results/\n"]
    for k, t in T.items():
        md.append(f"\n## {k}. {t['caption']}\n")
        md.append("| " + " | ".join(t["header"]) + " |")
        md.append("|" + "---|" * len(t["header"]))
        md += ["| " + " | ".join(r) + " |" for r in t["rows"]]
    open(os.path.join(C.ROOT, "TABLES.md"), "w").write("\n".join(md))
    print(f"wrote {len(T)} tables -> results/tables.json, TABLES.md")


if __name__ == "__main__":
    main()
