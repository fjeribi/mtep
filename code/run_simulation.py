"""
run_simulation.py -- MTEP step 6c: characterisation of the reporting-act check.

Each replicate rebuilds the matched design around synthetic events placed away
from every real press (simlib.pseudo_events), so the null world contains no
reporting act while retaining the real feature matrix, its covariance and its
participant structure.  A component of known amplitude is then injected at
those events in one of two shapes:

  spike   confined to the event round        -> a reporting-act artefact
  state   smooth Gaussian bump around it     -> a genuine cumulative state

and the lead-sweep check is applied: L = 0 against L = 1, with a
participant-clustered bootstrap interval for the difference in pooled AUC.  The
check fires when that interval excludes zero.

Amplitude 0 is the null and gives the false-alarm rate.  Under "spike" the
L = 1 run cannot see the injected component -- no L = 1 case window and no
control window contains an event round -- so one L = 1 run is shared by every
amplitude in that arm.  This is verified numerically in each replicate
(lead1_invariance_max_abs_diff) rather than assumed.

Usage:  python code/run_simulation.py [--reps N] [--budget SECONDS] [--workers K]
Outputs -> results/sim_lead/rep*.json
"""
import sys, os, json, time
import numpy as np

import common as C
import run_casecontrol as CC
import simlib as S

N_REPS = 20
SPIKE_ALPHAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
STATE_ALPHAS = [0.25, 0.5, 1.0]
OUT = os.path.join(C.RES, "sim_lead")


def one_rep(rep):
    seed = 9000 + rep
    dest = os.path.join(OUT, f"rep{rep:03d}.json")
    if os.path.exists(dest):
        return dest, 0.0
    t0 = time.time()
    X, y_event, meta, names, groups = C.load()
    cols_all = np.arange(X.shape[1])
    sd, eligible = S.feature_sd(X, meta, w=CC.W)
    res = S.resolution(X)
    rng = np.random.RandomState(seed)
    y_ps, blocked, ev_info = S.pseudo_events(y_event, meta, rng,
                                             w=CC.W, gap=CC.GAP)
    cols, signs = S.draw_targets(eligible, rng)

    def run(Xin, lead):
        Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
            Xin, y_ps, meta, cols_all, lead=lead, seed=seed, blocked=blocked)
        r = S.lean_loso(Xw, yw, uw, "rf", seed=C.SEED)
        r["design"] = dict(n_cases=info["n_cases"], n_controls=info["n_controls"],
                           frac_after=info["frac_controls_after_case"],
                           mean_abs_offset=info["mean_abs_offset"])
        return r

    out = dict(rep=rep, seed=seed, events=ev_info, n_eligible=int(len(eligible)),
               inject_cols=[names[c] for c in cols], inject_signs=signs.tolist(),
               spike=[], state=[])

    # ---- shared L = 1 run: a spike at the event round is invisible to it -----
    r1_null = run(X, 1)
    out["auc_lead1_null"] = r1_null["auc"]
    out["design_lead1"] = r1_null["design"]

    Xchk = S.inject(X, y_ps, meta, "spike", 50.0, cols, signs, sd,
                    np.random.RandomState(seed + 501), res=res)
    a = CC.build_case_control(Xchk, y_ps, meta, cols_all, lead=1, seed=seed,
                              blocked=blocked)[0]
    b = CC.build_case_control(X, y_ps, meta, cols_all, lead=1, seed=seed,
                              blocked=blocked)[0]
    out["lead1_invariance_max_abs_diff"] = float(np.nanmax(np.abs(a - b)))

    for a_ in SPIKE_ALPHAS:
        # the same noise draws at every amplitude, so the dose-response is paired
        Xi = S.inject(X, y_ps, meta, "spike", a_, cols, signs, sd,
                      np.random.RandomState(seed + 501), res=res)
        r0 = run(Xi, 0)
        d = S.paired_delta_ci(r0, r1_null)
        out["spike"].append(dict(alpha=a_, auc_lead0=r0["auc"],
                                 auc_lead1=r1_null["auc"], **d,
                                 design0=r0["design"]))
        print(f"  rep{rep:03d} spike a={a_:<5} L0={r0['auc']:.3f} "
              f"L1={r1_null['auc']:.3f} d={d['delta']:+.3f} "
              f"[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] fires={d['fires']}",
              flush=True)

    for a_ in STATE_ALPHAS:
        Xi = S.inject(X, y_ps, meta, "state", a_, cols, signs, sd,
                      np.random.RandomState(seed + 502), res=res)
        r0, r1 = run(Xi, 0), run(Xi, 1)
        d = S.paired_delta_ci(r0, r1)
        c1 = S.auc_ci(r1)
        out["state"].append(dict(alpha=a_, auc_lead0=r0["auc"], auc_lead1=r1["auc"],
                                 lead1_ci95=c1["ci95"], **d,
                                 design0=r0["design"], design1=r1["design"]))
        print(f"  rep{rep:03d} state a={a_:<5} L0={r0['auc']:.3f} "
              f"L1={r1['auc']:.3f} d={d['delta']:+.3f} "
              f"[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] fires={d['fires']}",
              flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, dest)
    return dest, out["seconds"]


def main():
    args = sys.argv[1:]

    def opt(name, default, cast=int):
        return cast(args[args.index(name) + 1]) if name in args else default

    reps = opt("--reps", N_REPS)
    budget = opt("--budget", 10 ** 9, float)
    workers = opt("--workers", 2)
    os.makedirs(OUT, exist_ok=True)
    pending = [r for r in range(reps)
               if not os.path.exists(os.path.join(OUT, f"rep{r:03d}.json"))]
    print(f"{reps - len(pending)}/{reps} done, {len(pending)} pending", flush=True)
    if not pending:
        return
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for res in pool.imap_unordered(one_rep, pending):
                print("wrote", os.path.basename(res[0]), res[1], "s", flush=True)
                if time.time() - t0 > budget:
                    pool.terminate()
                    break
    else:
        for r in pending:
            print("wrote", *one_rep(r), flush=True)
            if time.time() - t0 > budget:
                break
    left = [r for r in range(reps)
            if not os.path.exists(os.path.join(OUT, f"rep{r:03d}.json"))]
    print(f"remaining: {len(left)}", flush=True)


if __name__ == "__main__":
    main()
