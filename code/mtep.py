"""
mtep.py -- the MTEP design as a corpus-agnostic function.

Everything else in this release is the CogBeacon application. This module is the
design itself, taking arrays rather than the bundled dataset, so that the four
properties and the two certificates can be obtained for any corpus organised as
recording units nested within participants.

    from mtep import run_mtep
    out = run_mtep(X, y_event, session, participant, unit_index)
    print(out["verdict"])

Input contract
--------------
X            (n_units, n_features) float array, one row per unit, in any order.
y_event      (n_units,) 0/1, where 1 marks a unit at which a report was made.
session      (n_units,) integer id of the recording unit (a session, a shift).
participant  (n_units,) integer id of the person. Folds are formed on this.
unit_index   (n_units,) 1-based position of the unit within its session.
position     (n_units,) optional; normalised position used by the position
             certificate. Defaults to unit_index divided by the session length,
             which is what P3 is stated in terms of.

Returns a dict with the design, both certificates, the estimate at the certified
lead, and a one-line verdict. Nothing is written to disk.

The defaults reproduce the manuscript's analysis when given the bundled
CogBeacon arrays; see `selftest()` at the bottom of this file.
"""
import numpy as np

import common as C
import run_casecontrol as CC
import simlib as S

DELTA_STAR = 0.10          # materiality threshold, calibrated in Section 5.2
LEADS = (0, 1, 2, 3)


def _meta(session, participant, unit_index):
    """Assemble the meta array the windowing code expects."""
    session = np.asarray(session, int)
    participant = np.asarray(participant, int)
    unit_index = np.asarray(unit_index, int)
    n = np.zeros(len(session), int)
    for s in np.unique(session):
        m = session == s
        n[m] = m.sum()
    return np.column_stack([session, participant,
                            np.zeros(len(session), int), unit_index, n])


def _position(meta):
    return (meta[:, 3] / np.maximum(meta[:, 4], 1)).astype(np.float32)


def run_mtep(X, y_event, session, participant, unit_index, position=None,
             W=CC.W, gap=CC.GAP, maxoff=CC.MAXOFF, k=CC.N_CTRL,
             leads=LEADS, model="rf", delta_star=DELTA_STAR, seed=C.SEED,
             feature_names=None, verbose=True):
    """Build the matched design, obtain both certificates, and report the
    estimate at the certified lead."""
    X = np.asarray(X, np.float32)
    y_event = np.asarray(y_event, int)
    meta = _meta(session, participant, unit_index)
    if position is None:
        position = _position(meta)
    pos_col = X.shape[1]
    Xp = np.column_stack([X, np.asarray(position, np.float32)])
    all_cols = np.arange(X.shape[1])

    # ---- P1: the position variable is a diagnostic, never an input ----------
    if feature_names is not None:
        bad = [n for n in feature_names
               if any(t in n.lower() for t in ("label", "report", "press",
                                               "fatigue", "target"))]
        if bad:
            raise ValueError(f"P1 violated: feature names refer to the label: {bad}")

    out = {"design": {}, "leads": {}, "delta_star": float(delta_star)}
    runs = {}
    for L in leads:
        Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
            Xp, y_event, meta, np.arange(Xp.shape[1]), w=W, gap=gap,
            maxoff=maxoff, k=k, lead=L, seed=seed)
        if info["n_cases"] == 0:
            raise ValueError(f"no cases survive at lead {L}; the sessions may be "
                             f"shorter than W + lead")
        runs[L] = dict(
            model=S.lean_loso(Xw[:, :, all_cols], yw, uw, model, seed=seed),
            position=S.lean_loso(Xw[:, :, [pos_col]], yw, uw, "logreg", seed=seed),
            info=info)
        out["leads"][L] = dict(
            auc=runs[L]["model"]["auc"],
            position_auc=runs[L]["position"]["auc"],
            n_cases=info["n_cases"], n_controls=info["n_controls"])
        if verbose:
            print(f"  L={L}  AUC={runs[L]['model']['auc']:.3f}  "
                  f"position={runs[L]['position']['auc']:.3f}  "
                  f"cases={info['n_cases']}", flush=True)

    # ---- P4: displacement steps and the certified lead ----------------------
    steps = {}
    ordered = sorted(leads)
    for a, b in zip(ordered, ordered[1:]):
        steps[b] = dict(from_lead=a, to_lead=b,
                        **S.paired_delta_ci(runs[a]["model"], runs[b]["model"],
                                            seed=seed))
    certified = None
    for L in ordered[:-1]:
        nxt = steps.get(L + 1)
        if nxt is not None and nxt["delta"] < delta_star:
            certified = L
            break
    out["report_independence"] = dict(
        steps={str(kk): vv for kk, vv in steps.items()},
        certified_lead=certified,
        granted=certified is not None)

    # ---- P3: the position certificate at the certified lead -----------------
    L = certified if certified is not None else ordered[-1]
    pc = S.auc_ci(runs[L]["position"], seed=seed)
    out["position_balance"] = dict(lead=L, auc=pc["auc"], ci95=pc["ci95"],
                                   granted=not pc["fires"])

    # ---- the estimate the design delivers -----------------------------------
    est = S.auc_ci(runs[L]["model"], seed=seed)
    out["estimate"] = dict(lead=L, auc=est["auc"], ci95=est["ci95"],
                           n_cases=runs[L]["info"]["n_cases"],
                           n_controls=runs[L]["info"]["n_controls"],
                           n_participants=runs[L]["info"]["n_participants"])
    out["design"] = {kk: runs[ordered[0]]["info"][kk] for kk in
                     ("W", "gap", "maxoff", "k", "presses", "n_sessions",
                      "n_participants")}

    ok_p3 = out["position_balance"]["granted"]
    ok_p4 = out["report_independence"]["granted"]
    if ok_p3 and ok_p4:
        out["verdict"] = (f"certified at lead {L}: pooled AUC {est['auc']:.3f} "
                          f"[{est['ci95'][0]:.3f}, {est['ci95'][1]:.3f}]")
    elif not ok_p4:
        out["verdict"] = ("not certified: the estimate has not stabilised by the "
                          f"largest lead tried ({ordered[-1]}); extend `leads`")
    else:
        out["verdict"] = (f"not certified: position balance fails at lead {L} "
                          f"(position-only AUC {pc['auc']:.3f}, 95% CI "
                          f"{pc['ci95'][0]:.3f}-{pc['ci95'][1]:.3f})")
    if verbose:
        print(out["verdict"], flush=True)
    return out


def selftest():
    """run_mtep on the bundled arrays must reproduce the manuscript."""
    X, y_event, meta, names, groups = C.load()
    out = run_mtep(X, y_event, meta[:, 0], meta[:, 1], meta[:, 3],
                   position=X[:, names.index("perf_round_norm")], verbose=False)
    assert out["report_independence"]["certified_lead"] == 1, out
    assert abs(out["estimate"]["auc"] - 0.6593) < 1e-3, out["estimate"]
    assert abs(out["position_balance"]["auc"] - 0.4934) < 1e-3, out["position_balance"]
    assert out["position_balance"]["granted"]
    print("selftest OK:", out["verdict"])
    return out


if __name__ == "__main__":
    selftest()
