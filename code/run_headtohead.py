"""
run_headtohead.py -- do the four designs reach the right conclusion?

An evaluation design cannot be shown to be better by scoring higher: a design
that removes inflation scores LOWER. What can be shown is whether it reaches the
correct conclusion on data whose truth is known, which the simulation supplies.

Four designs, each adding one property to the one above it, each returning a
single verdict -- does the evidence support state detection, yes or no?

  M1  random folds, unbalanced controls, window ends on the report   P1
  M2  grouped leave-one-participant-out, unbalanced, on the report   P1 P2
  M3  grouped LOSO, position-balanced controls, on the report        P1 P2 P3
  M4  MTEP: grouped LOSO, balanced, read at the certified lead       P1 P2 P3 P4

Four worlds, built on synthetic events so that the truth is known by
construction:

  none      nothing injected                      truth: no state to detect
  act       an act-shaped component at the event  truth: no state to detect
  state     a smooth state around the event       truth: a state is present
  both      act plus state                        truth: a state is present

A design is counted correct when it says yes exactly in the worlds where a state
is present. The verdict rule is the same for all four: the 95% interval for the
pooled ROC-AUC excludes 0.5. M1 resamples rows, since an analyst using random
folds would not cluster; M2 to M4 resample participants. M4 first locates the
certified lead and reports there, and returns no when nothing certifies.

Usage: python code/run_headtohead.py [--reps N] [--workers K]
Outputs -> results/headtohead/rep*.json
"""
import sys, os, json, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import common as C
import run_casecontrol as CC
import simlib as S
from run_simposition import biased_picker

N_REPS = 20
ALPHA_ACT = 0.1          # an act-shaped component the certificate detects in 18/20
ALPHA_STATE = 1.0        # a state that lifts pooled AUC at L = 1 to about 0.714
LEADS = [0, 1, 2]
DELTA_STAR = 0.10
WORLDS = ["none", "act", "state", "both"]
TRUTH = {"none": False, "act": False, "state": True, "both": True}
OUT = os.path.join(C.RES, "headtohead")


def random_fold_eval(Xw, yw, seed, n_folds=5):
    """Folds formed over rows, ignoring participants: what a random split does."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(yw))
    oof = np.full(len(yw), np.nan)
    for f in range(n_folds):
        te = idx[f::n_folds]
        tr = np.setdiff1d(idx, te)
        if len(np.unique(yw[tr])) < 2:
            continue
        sc = C.fit_scaler(Xw[tr])
        clf = RandomForestClassifier(random_state=seed, **C.RF_PARAMS)
        clf.fit(C.summarise(C.apply_scaler(Xw[tr], sc)), yw[tr])
        oof[te] = clf.predict_proba(C.summarise(C.apply_scaler(Xw[te], sc)))[:, 1]
    ok = ~np.isnan(oof)
    return dict(y=yw[ok], p=oof[ok], u=np.arange(ok.sum()),
                auc=float(roc_auc_score(yw[ok], oof[ok])))


def row_bootstrap_ci(r, n_boot=C.N_BOOT, seed=C.SEED):
    """Rows resampled with replacement: no clustering, as the analyst intends."""
    rng = np.random.RandomState(seed)
    n, vals = len(r["y"]), []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if len(np.unique(r["y"][i])) > 1:
            vals.append(roc_auc_score(r["y"][i], r["p"][i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(auc=r["auc"], ci95=[float(lo), float(hi)], says_yes=bool(lo > 0.5))


def clustered(r, seed=C.SEED):
    c = S.auc_ci(r, seed=seed)
    return dict(auc=c["auc"], ci95=c["ci95"], says_yes=bool(c["ci95"][0] > 0.5))


def one_rep(rep):
    seed = 4000 + rep
    dest = os.path.join(OUT, f"rep{rep:03d}.json")
    if os.path.exists(dest):
        return dest, 0.0
    t0 = time.time()
    X, y_event, meta, names, groups = C.load()
    cols = np.arange(X.shape[1])
    sd, elig = S.feature_sd(X, meta, w=CC.W)
    res = S.resolution(X)
    rng = np.random.RandomState(seed)
    y_ps, blocked, ev = S.pseudo_events(y_event, meta, rng, w=CC.W, gap=CC.GAP)
    icols, signs = S.draw_targets(elig, rng)

    def world(name):
        Xi = X
        if name in ("act", "both"):
            Xi = S.inject(Xi, y_ps, meta, "spike", ALPHA_ACT, icols, signs, sd,
                          np.random.RandomState(seed + 601), res=res)
        if name in ("state", "both"):
            Xi = S.inject(Xi, y_ps, meta, "state", ALPHA_STATE, icols, signs, sd,
                          np.random.RandomState(seed + 602), res=res)
        return Xi

    def design(Xi, lead, balanced):
        pick = None if balanced else biased_picker(1.0)
        Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
            Xi, y_ps, meta, cols, lead=lead, seed=seed, blocked=blocked,
            picker=pick)
        return Xw, yw, uw

    out = dict(rep=rep, seed=seed, alpha_act=ALPHA_ACT, alpha_state=ALPHA_STATE,
               events=ev, worlds={})
    for w in WORLDS:
        Xi = world(w)
        # M1: random folds, unbalanced, on the report
        Xw, yw, uw = design(Xi, 0, balanced=False)
        m1 = row_bootstrap_ci(random_fold_eval(Xw, yw, seed))
        # M2: grouped, unbalanced, on the report
        m2 = clustered(S.lean_loso(Xw, yw, uw, "rf", seed=C.SEED))
        # M3: grouped, balanced, on the report
        Xb0, yb0, ub0 = design(Xi, 0, balanced=True)
        r0 = S.lean_loso(Xb0, yb0, ub0, "rf", seed=C.SEED)
        m3 = clustered(r0)
        # M4: MTEP -- certify, then read at the certified lead
        runs = {0: r0}
        for L in LEADS[1:]:
            Xb, yb, ub = design(Xi, L, balanced=True)
            runs[L] = S.lean_loso(Xb, yb, ub, "rf", seed=C.SEED)
        steps = {L: S.paired_delta_ci(runs[L - 1], runs[L]) for L in LEADS[1:]}
        cert = next((L for L in LEADS[:-1]
                     if steps[L + 1]["delta"] < DELTA_STAR), None)
        if cert is None:
            m4 = dict(auc=None, ci95=None, says_yes=False, certified_lead=None)
        else:
            m4 = dict(clustered(runs[cert]), certified_lead=cert)
        out["worlds"][w] = dict(
            truth=TRUTH[w], M1=m1, M2=m2, M3=m3, M4=m4,
            steps={str(L): steps[L]["delta"] for L in LEADS[1:]})
        print(f"  rep{rep:03d} {w:6s} truth={str(TRUTH[w]):5s} "
              + "  ".join(f"{k}={'Y' if out['worlds'][w][k]['says_yes'] else 'n'}"
                          f"({out['worlds'][w][k]['auc'] or float('nan'):.3f})"
                          for k in ("M1", "M2", "M3", "M4"))
              + f"  certL={m4.get('certified_lead')}", flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, dest)
    return dest, out["seconds"]


def main():
    args = sys.argv[1:]

    def opt(n, d):
        return int(args[args.index(n) + 1]) if n in args else d

    reps, workers = opt("--reps", N_REPS), opt("--workers", 2)
    os.makedirs(OUT, exist_ok=True)
    pending = [r for r in range(reps)
               if not os.path.exists(os.path.join(OUT, f"rep{r:03d}.json"))]
    print(f"{reps - len(pending)}/{reps} done, {len(pending)} pending", flush=True)
    if not pending:
        return
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(one_rep, pending):
            print("wrote", os.path.basename(res[0]), res[1], "s", flush=True)


if __name__ == "__main__":
    main()
