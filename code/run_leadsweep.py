"""
run_leadsweep.py -- MTEP step 6b: reporting-act diagnostic.

Re-runs the matched design with the case window ending L rounds before the
press, L = 0 (window includes the press round), 1, 2, 3. A large drop from
L = 0 to L = 1 means the press round itself carries a trace of the reporting
act (movement, gaze, round duration) that a classifier can exploit.

Usage: python code/run_leadsweep.py [--force]
Outputs -> results/leadsweep.json
"""
import sys, os
import numpy as np
import common as C
import run_casecontrol as CC

CONFIGS = [("position", "logreg"), ("all", "rf"), ("eegface", "rf"),
           ("eeg", "rf"), ("face", "rf"), ("task", "rf")]


def main():
    dest = C.rpath("leadsweep.json")
    if os.path.exists(dest) and "--force" not in sys.argv:
        print("[exists, use --force]"); return
    X, y_event, meta, names, groups = C.load()
    FS = CC.feature_sets(names, groups)
    out = []
    for lead in [0, 1, 2, 3]:
        Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
            X, y_event, meta, np.arange(X.shape[1]), lead=lead)
        row = dict(lead=lead, design=info)
        for fs, kind in CONFIGS:
            r = C.loso(Xw[:, :, FS[fs]], yw, uw, kind)
            row[fs] = dict(pooled_auc=r["pooled"]["roc_auc"], ci95=r["pooled"]["auc_ci95"],
                           fold_auc=r["fold_mean"]["roc_auc"]["mean"],
                           fold_sd=r["fold_mean"]["roc_auc"]["sd"])
        out.append(row)
        print(f"L={lead} cases={info['n_cases']} " + " ".join(
            f"{fs}={row[fs]['pooled_auc']:.3f}" for fs, _ in CONFIGS), flush=True)
    C.save_json(out, "leadsweep.json")


if __name__ == "__main__":
    main()
