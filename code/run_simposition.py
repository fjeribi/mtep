"""
run_simposition.py -- MTEP step 6d: characterisation of the position check.

The position check asks whether a window's position within the session, on its
own, predicts the label.  It fails when the matched sets are position-imbalanced
-- when controls sit systematically on one side of their case -- because then a
classifier can score the label from the clock instead of from physiology.  On a
single corpus the check either passes or fails and its power is unknown.  Here
the imbalance is dialled in: with probability gamma a control is taken from the
nearest remaining candidate BEFORE the case instead of by the balancing rule, so
gamma = 0 is the protocol's own sampler and gamma = 1 is a sampler that always
draws the nearest earlier window.

For each (gamma, replicate) the real matched design is rebuilt at L = 1, a
logistic regression is fitted on the single normalised-position feature under
the same leave-one-participant-out folds, and the check fires when the
participant-clustered bootstrap interval for its pooled AUC excludes 0.5.
gamma = 0 gives the false-alarm rate; gamma > 0 gives the sensitivity.

Usage:  python code/run_simposition.py [--reps N] [--budget SECONDS] [--workers K]
Outputs -> results/sim_position.json
"""
import sys, os, json, time
import numpy as np

import common as C
import run_casecontrol as CC
import simlib as S

N_REPS = 40
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEST = os.path.join(C.RES, "sim_position.json")


def biased_picker(gamma):
    """_pick_balanced, except that each pick is forced to the nearest earlier
    candidate with probability gamma."""
    def pick(avail, ce, k, rng):
        before = sorted([c for c in avail if c < ce], key=lambda c: ce - c)
        after = sorted([c for c in avail if c > ce], key=lambda c: c - ce)
        take, total = [], 0
        for _ in range(k):
            if not before and not after:
                break
            if rng.rand() < gamma and before:
                c = before.pop(0)
            else:
                if total > 0:
                    side = before if before else after
                elif total < 0:
                    side = after if after else before
                else:
                    side = ([before, after][rng.randint(2)]
                            if (before and after) else (before or after))
                c = side.pop(rng.randint(min(3, len(side))))
            take.append(int(c)); total += c - ce
        return np.array(take, int)
    return pick


def one_cell(job):
    gamma, rep = job
    seed = 7000 + rep
    X, y_event, meta, names, groups = C.load()
    FS = CC.feature_sets(names, groups)
    Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
        X, y_event, meta, np.arange(X.shape[1]), lead=1, seed=seed,
        picker=biased_picker(gamma))
    r = S.lean_loso(Xw[:, :, FS["position"]], yw, uw, "logreg", seed=C.SEED)
    ci = S.auc_ci(r)
    co = offs[yw == 0]
    return dict(gamma=gamma, rep=rep, seed=seed, auc=ci["auc"],
                ci95=ci["ci95"], fires=ci["fires"],
                mean_signed_offset=float(co.mean()),
                mean_abs_offset=float(np.abs(co).mean()),
                frac_after=float((co > 0).mean()),
                n_cases=info["n_cases"], n_controls=info["n_controls"])


def main():
    args = sys.argv[1:]

    def opt(name, default, cast=int):
        return cast(args[args.index(name) + 1]) if name in args else default

    reps = opt("--reps", N_REPS)
    workers = opt("--workers", 2)
    done = json.load(open(DEST)) if os.path.exists(DEST) else []
    have = {(d["gamma"], d["rep"]) for d in done}
    jobs = [(g, r) for g in GAMMAS for r in range(reps) if (g, r) not in have]
    print(f"{len(have)} done, {len(jobs)} pending", flush=True)
    if not jobs:
        return
    t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        for i, d in enumerate(pool.imap_unordered(one_cell, jobs), 1):
            done.append(d)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}  {round(time.time()-t0)}s", flush=True)
    done.sort(key=lambda d: (d["gamma"], d["rep"]))
    with open(DEST, "w") as fh:
        json.dump(done, fh, indent=1)
    for g in GAMMAS:
        rows = [d for d in done if d["gamma"] == g]
        if rows:
            print(f"gamma={g}  fires {sum(d['fires'] for d in rows)}/{len(rows)}  "
                  f"AUC {np.mean([d['auc'] for d in rows]):.3f}  "
                  f"signed offset {np.mean([d['mean_signed_offset'] for d in rows]):+.2f}  "
                  f"frac after {np.mean([d['frac_after'] for d in rows]):.2f}")


if __name__ == "__main__":
    main()
