"""
run_permutation.py -- label-permutation null for the controlled design.

The null permutes case status WITHIN each matched set (a case and its own
controls), which preserves the matched structure, the participant and
session composition, and the position balance. The test statistic is the
pooled out-of-fold ROC-AUC of the SAME model as the reported result (random
forest), recomputed with the full leave-one-participant-out procedure.
p = (1 + #null >= observed) / (1 + n_perm).

Resumable: progress is saved every 10 permutations.
Usage: python code/run_permutation.py [n_perm=200] [--force]
Outputs -> results/permutation.json
"""
import sys, os
import numpy as np
import common as C
import run_casecontrol as CC

TESTS = [("all", "rf", "All features (RF)"), ("eegface", "rf", "EEG + facial (RF)")]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_perm = int(args[0]) if args else 200
    path = C.rpath("permutation.json")
    state = {} if ("--force" in sys.argv or not os.path.exists(path)) else C.load_json("permutation.json")
    X, y_event, meta, names, groups = C.load()
    FS = CC.feature_sets(names, groups)
    Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(X, y_event, meta, np.arange(X.shape[1]))
    for fs, kind, label in TESTS:
        st = state.get(fs)
        if st is None:
            r = C.loso(Xw[:, :, FS[fs]], yw, uw, kind)
            st = dict(label=label, model=kind, observed=r["pooled"]["roc_auc"], null=[])
        rng = np.random.RandomState(C.SEED + 1)
        for _ in range(len(st["null"])):          # advance RNG past completed draws
            for s in np.unique(sets):
                rng.permutation(np.where(sets == s)[0])
        while len(st["null"]) < n_perm:
            yp = yw.copy()
            for s in np.unique(sets):
                m = np.where(sets == s)[0]
                yp[m] = yw[rng.permutation(m)]
            st["null"].append(C.loso(Xw[:, :, FS[fs]], yp, uw, kind)["pooled"]["roc_auc"])
            if len(st["null"]) % 10 == 0:
                state[fs] = st; C.save_json(state, "permutation.json")
                print(f"{label}: {len(st['null'])}/{n_perm}", flush=True)
        nul = np.array(st["null"])
        st.update(null_mean=float(nul.mean()), null_p95=float(np.percentile(nul, 95)),
                  p_value=float((1 + np.sum(nul >= st["observed"])) / (1 + len(nul))),
                  n_perm=len(nul))
        state[fs] = st
        C.save_json(state, "permutation.json")
        print(f"{label}: observed={st['observed']:.3f} null_mean={st['null_mean']:.3f} "
              f"p95={st['null_p95']:.3f} p={st['p_value']:.4f}", flush=True)


if __name__ == "__main__":
    main()
