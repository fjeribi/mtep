"""
run_certify_robustness.py -- is the certified lead a property of the design, or
of the choices made while applying it?

The certificate is a difference of pooled ROC-AUCs, so two questions follow
immediately and the manuscript cannot currently answer either. Would another
estimator certify at the same lead? Would another window length? The
sensitivity grid varies W, g, m and k for the ESTIMATE at a fixed lead; neither
it nor the lead sweep varies anything for the CERTIFICATE.

  A  estimator   lead sweep on all features with the random forest, logistic
                 regression and the SVM, under identical folds and windows
  B  window      lead sweep on all features with the random forest at
                 W = 5, 10 and 15, with g, m and k at their defaults

Both report D(L) = AUC(L-1) - AUC(L) with a participant-clustered interval and
the lead certified by the magnitude rule.

Usage: python code/run_certify_robustness.py [--force]
Outputs -> results/certify_robustness.json
"""
import sys, os, time
import numpy as np

import common as C
import run_casecontrol as CC
import simlib as S

DELTA_STAR = 0.10
LEADS = [0, 1, 2, 3]
MODELS = ["rf", "logreg", "svm"]
WINDOWS = [5, 10, 15]
FS = "all"


def sweep(X, y_event, meta, cols, model, w):
    runs, design = {}, {}
    for L in LEADS:
        Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
            X, y_event, meta, np.arange(X.shape[1]), w=w, lead=L)
        runs[L] = S.lean_loso(Xw[:, :, cols], yw, uw, model)
        design[L] = dict(n_cases=info["n_cases"], n_controls=info["n_controls"],
                         n_participants=info["n_participants"])
    steps = [S.paired_delta_ci(runs[L], runs[L + 1]) for L in LEADS[:-1]]
    cert = next((L for L in range(len(steps)) if steps[L]["delta"] < DELTA_STAR),
                len(steps))
    return dict(auc=[runs[L]["auc"] for L in LEADS],
                steps=[dict(from_lead=L, to_lead=L + 1, **steps[L])
                       for L in range(len(steps))],
                certified_lead=cert,
                design={str(k): v for k, v in design.items()})


def main():
    dest = C.rpath("certify_robustness.json")
    if os.path.exists(dest) and "--force" not in sys.argv:
        print("[exists, use --force]"); return
    X, y_event, meta, names, groups = C.load()
    cols = CC.feature_sets(names, groups)[FS]
    out = {"delta_star": DELTA_STAR, "feature_set": FS,
           "estimator": {}, "window": {}}

    print("A  estimator (W = 10)")
    for m in MODELS:
        t0 = time.time()
        r = sweep(X, y_event, meta, cols, m, CC.W)
        out["estimator"][m] = r
        print(f"   {m:7s} AUC " + " ".join(f"{a:.3f}" for a in r["auc"])
              + "   " + "  ".join(f"D({s['from_lead']}->{s['to_lead']})="
                                  f"{s['delta']:+.3f}" for s in r["steps"])
              + f"   certified L={r['certified_lead']}  [{time.time()-t0:.0f}s]",
              flush=True)

    print("\nB  window length (random forest)")
    for w in WINDOWS:
        t0 = time.time()
        r = sweep(X, y_event, meta, cols, "rf", w)
        out["window"][str(w)] = r
        print(f"   W={w:<3d}  cases " +
              " ".join(f"{r['design'][str(L)]['n_cases']:3d}" for L in LEADS)
              + "   AUC " + " ".join(f"{a:.3f}" for a in r["auc"])
              + "   " + "  ".join(f"D({s['from_lead']}->{s['to_lead']})="
                                  f"{s['delta']:+.3f}" for s in r["steps"])
              + f"   certified L={r['certified_lead']}  [{time.time()-t0:.0f}s]",
              flush=True)

    leads = ([out["estimator"][m]["certified_lead"] for m in MODELS]
             + [out["window"][str(w)]["certified_lead"] for w in WINDOWS])
    out["agree"] = bool(len(set(leads)) == 1)
    out["certified_leads"] = leads
    print(f"\ncertified leads across all six runs: {leads}  "
          f"{'unanimous' if out['agree'] else 'NOT unanimous'}")
    C.save_json(out, "certify_robustness.json")


if __name__ == "__main__":
    main()
