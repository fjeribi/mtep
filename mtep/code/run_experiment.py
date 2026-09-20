"""
run_experiment.py -- Study 1: the conventional sliding-window protocol.

Windows of W consecutive rounds within a session (stride 1); the label of the
final round is the target. Grouped leave-one-participant-out CV (common.loso).

Targets
  state      : participant has pressed at least once at or before this round
  event_d{D} : a new press occurs within the next D rounds (D=0: this round)

Participants whose held-out windows contain one class only cannot yield a
fold-wise AUC; they still contribute to the pooled out-of-fold estimate.
The number of scored participants is stored with every result.

Usage: python code/run_experiment.py [job_index] [--force]
Outputs -> results/study1/job*.json
"""
import sys, os, time
import numpy as np
import common as C

W = 10


def make_labels(y_event, meta, task):
    lab = np.zeros(len(y_event), dtype=int)
    for _, m in C.session_order(meta):
        ev = y_event[m]
        if task == "state":
            lab[m] = (np.cumsum(ev) > 0).astype(int)
        else:
            D = int(task.split("_d")[1])
            lab[m] = [int(ev[i:i + D + 1].sum() > 0) for i in range(len(ev))]
    return lab


def make_windows(X, lab, meta, cols, w=W):
    Xs, ys, us = [], [], []
    for _, m in C.session_order(meta):
        Xi, yi = X[m][:, cols], lab[m]
        for e in range(w - 1, len(m)):
            Xs.append(Xi[e - w + 1:e + 1]); ys.append(yi[e]); us.append(meta[m[0], 1])
    return np.asarray(Xs, np.float32), np.asarray(ys, int), np.asarray(us, int)


def jobs(X, y_event, meta, names, groups):
    allc = np.arange(X.shape[1])
    pos = np.array([names.index("perf_round_norm")])
    ELAPSED = X.shape[1] - 1                 # appended last column, see main()
    ef = np.sort(np.concatenate([groups["eeg"], groups["face"]]))
    J = [("state", allc, "state|all|transformer", "transformer"),
         ("state", allc, "state|all|logreg", "logreg"),
         ("state", allc, "state|all|rf", "rf"),
         ("state", allc, "state|all|svm", "svm"),
         ("state", pos, "state|position_normalised_only|logreg", "logreg"),
         ("state", np.array([ELAPSED]), "state|elapsed_rounds_only|logreg", "logreg"),
         ("state", groups["eeg"], "state|eeg_only|transformer", "transformer"),
         ("state", groups["face"], "state|face_only|transformer", "transformer"),
         ("state", groups["perf"], "state|task_only|transformer", "transformer"),
         ("state", ef, "state|eeg+face|transformer", "transformer")]
    for D in [0, 1, 3, 5]:
        J.append((f"event_d{D}", allc, f"event_d{D}|all|transformer", "transformer"))
    return J


def main():
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = int(args[0]) if args else None
    X, y_event, meta, names, groups = C.load()
    # elapsed rounds since session start: a clock usable at deployment time
    X = np.concatenate([X, meta[:, 3:4].astype(np.float32)], axis=1)
    J = jobs(X, y_event, meta, names, groups)
    for ji, (task, cols, name, kind) in enumerate(J):
        if only is not None and ji != only:
            continue
        dest = C.rpath("study1", f"job{ji:02d}.json")
        if os.path.exists(dest) and not force:
            print(f"[exists, use --force to overwrite] {name}"); continue
        t0 = time.time()
        lab = make_labels(y_event, meta, task)
        Xw, yw, uw = make_windows(X, lab, meta, cols)
        r = C.loso(Xw, yw, uw, kind)
        r.update(name=name, task=task, job_index=ji, window=W,
                 seconds=round(time.time() - t0, 1))
        C.save_json(r, "study1", f"job{ji:02d}.json")
        fm = r["fold_mean"]
        print(f"{name:42s} pooledAUC={r['pooled']['roc_auc']:.3f} "
              f"foldAUC={fm['roc_auc']['mean']:.3f}±{fm['roc_auc']['sd']:.3f} "
              f"folds={r['n_participants_scored']}/{r['n_participants_total']} "
              f"[{r['seconds']}s]", flush=True)
    print(f"NJOBS={len(J)}")


if __name__ == "__main__":
    main()
