"""
run_sensitivity.py -- dependence of the controlled result on MTEP's free
parameters, at the default lead L = 1.

Grid: window W in {5,10,15} x gap g in {3,5,10} x max offset m in {20,30,50}
(controls per case k = 3, seed 1337) = 27 cells, plus k in {1,2,5} and seeds
{7,42,2024} at the default cell = 33 cells in total. The grid does not vary
the lead (see run_leadsweep.py) or the feature sets.

Resumable: completed cells are kept in results/sensitivity.json and skipped.
Usage: python code/run_sensitivity.py [--force]
"""
import sys, itertools, os
import numpy as np
import common as C
import run_casecontrol as CC


def grid():
    g = [(W, gp, mo, 3, 1337) for W, gp, mo in
         itertools.product([5, 10, 15], [3, 5, 10], [20, 30, 50])]
    g += [(10, 5, 30, k, 1337) for k in [1, 2, 5]]
    g += [(10, 5, 30, 3, s) for s in [7, 42, 2024]]
    return g


def main():
    path = C.rpath("sensitivity.json")
    out = [] if ("--force" in sys.argv or not os.path.exists(path)) else C.load_json("sensitivity.json")
    done = {(r["W"], r["gap"], r["maxoff"], r["k"], r["seed"]) for r in out}
    X, y_event, meta, names, groups = C.load()
    FS = CC.feature_sets(names, groups)
    for cell in grid():
        if cell in done:
            continue
        W, gp, mo, k, seed = cell
        Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
            X, y_event, meta, np.arange(X.shape[1]), w=W, gap=gp, maxoff=mo, k=k, seed=seed)
        r = dict(W=W, gap=gp, maxoff=mo, k=k, seed=seed, n_cases=info["n_cases"],
                 n_windows=int(len(yw)))
        for tag, kind in [("position", "logreg"), ("all", "rf"), ("eegface", "rf"), ("task", "rf")]:
            res = C.loso(Xw[:, :, FS[tag]], yw, uw, kind, seed=seed)
            r[tag] = dict(pooled_auc=res["pooled"]["roc_auc"], ci95=res["pooled"]["auc_ci95"],
                          fold_auc=res["fold_mean"]["roc_auc"]["mean"])
        out.append(r)
        C.save_json(out, "sensitivity.json")
        print(cell, {t: round(r[t]["pooled_auc"], 3) for t in ["position", "all", "eegface", "task"]},
              flush=True)
    print(f"CELLS={len(out)}")


if __name__ == "__main__":
    main()
