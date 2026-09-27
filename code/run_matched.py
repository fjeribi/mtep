"""
run_matched.py -- the two matching attempts on the cumulative-state label
reported in Section 4.3. Both fail: the position-only model stays far above
chance, because within a session position determines the state label.

  corpus      : positives and negatives balanced within each session-position
                decile across the whole corpus
  participant : balanced within each (participant, decile) cell

Usage: python code/run_matched.py [--force]
Outputs -> results/matched_state.json
"""
import sys, os
import numpy as np
import common as C
import run_experiment as E

NBINS = 10


def build(X, y_event, meta, level):
    lab = E.make_labels(y_event, meta, "state")
    Xw, yw, uw = E.make_windows(X, lab, meta, np.arange(X.shape[1]))
    frac = []
    for _, m in C.session_order(meta):
        n = max(meta[m[0], 4], 1)
        frac += [meta[m[e], 3] / n for e in range(E.W - 1, len(m))]
    bins = np.clip((np.asarray(frac) * NBINS).astype(int), 0, NBINS - 1)
    rng = np.random.RandomState(C.SEED)
    keep = []
    cells = ([(None, b) for b in range(NBINS)] if level == "corpus"
             else [(u, b) for u in np.unique(uw) for b in range(NBINS)])
    for u, b in cells:
        sel = (bins == b) if u is None else ((uw == u) & (bins == b))
        idx = np.where(sel)[0]
        pos, neg = idx[yw[idx] == 1], idx[yw[idx] == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep += rng.choice(pos, k, replace=False).tolist()
        keep += rng.choice(neg, k, replace=False).tolist()
    keep = np.sort(keep)
    return Xw[keep], yw[keep], uw[keep]


def main():
    dest = C.rpath("matched_state.json")
    if os.path.exists(dest) and "--force" not in sys.argv:
        print("[exists, use --force]"); return
    X, y_event, meta, names, groups = C.load()
    pos = np.array([names.index("perf_round_norm")])
    out = {}
    for level in ["corpus", "participant"]:
        Xw, yw, uw = build(X, y_event, meta, level)
        out[level] = {}
        for tag, cols, kind in [("position", pos, "logreg"),
                                ("all", np.arange(X.shape[1]), "logreg")]:
            r = C.loso(Xw[:, :, cols], yw, uw, kind)
            out[level][tag] = dict(pooled_auc=r["pooled"]["roc_auc"],
                                   ci95=r["pooled"]["auc_ci95"],
                                   fold_auc=r["fold_mean"]["roc_auc"]["mean"],
                                   fold_sd=r["fold_mean"]["roc_auc"]["sd"],
                                   n_windows=r["n_windows"],
                                   folds=r["n_participants_scored"])
        print(level, {t: round(v["pooled_auc"], 3) for t, v in out[level].items()}, flush=True)
    C.save_json(out, "matched_state.json")


if __name__ == "__main__":
    main()
