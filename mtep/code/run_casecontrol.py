"""
run_casecontrol.py -- Study 2: MTEP within-session matched case-control design.

Case    : window of W rounds ending L rounds BEFORE a round with a new press
          (L = lead >= 1), so the round containing the press -- and with it any
          motor, gaze or timing trace of the act of pressing -- is never inside
          a case window. Case windows that contain any press round are dropped.
Control : window from the SAME session whose end lies between GAP and MAXOFF
          rounds from every press, and whose span contains no press round.
          For each case, K controls are drawn so that they fall on both sides
          of the case where possible (the running signed offset is steered
          towards zero; each pick is random among the 3 nearest candidates on
          the chosen side). A control window is used at most once.
Matched set: a case and its controls. Matched sets are disjoint.

Lead L = 0 reproduces the original (reporting-act contaminated) definition and
is used only as a diagnostic in run_leadsweep.py.

Usage: python code/run_casecontrol.py [job_index] [--force]
Outputs -> results/study2/job*.json, results/study2/design.json
"""
import sys, os, time
import numpy as np
import common as C

W, GAP, MAXOFF, N_CTRL, LEAD = 10, 5, 30, 3, 1


def _pick_balanced(avail, ce, k, rng):
    """Pick k control ends, balancing before/after so the mean signed offset
    from the case stays near zero (position matching inside the set).
    Each pick is drawn at random from the 3 nearest remaining candidates on
    the side that moves the running offset sum towards zero; if that side is
    empty, the other side is used."""
    before = sorted([c for c in avail if c < ce], key=lambda c: ce - c)
    after = sorted([c for c in avail if c > ce], key=lambda c: c - ce)
    take, total = [], 0
    for _ in range(k):
        if not before and not after:
            break
        if total > 0:
            side = before if before else after
        elif total < 0:
            side = after if after else before
        else:
            side = [before, after][rng.randint(2)] if (before and after) else (before or after)
        j = rng.randint(min(3, len(side)))
        c = side.pop(j)
        take.append(int(c)); total += c - ce
    return np.array(take, int)


def build_case_control(X, y_event, meta, cols, w=W, gap=GAP, maxoff=MAXOFF,
                       k=N_CTRL, lead=LEAD, seed=C.SEED):
    rng = np.random.RandomState(seed)
    Xs, ys, us, ss, sets, offs = [], [], [], [], [], []
    info = dict(presses=int(y_event.sum()), cases_dropped_early=0,
                cases_dropped_overlap=0, cases_without_controls=0,
                controls_short=0)
    set_id = 0
    for s, m in C.session_order(meta):
        ev = y_event[m]
        if ev.sum() == 0:
            continue
        Xi = X[m][:, cols]
        press = np.where(ev == 1)[0]
        ends = np.arange(w - 1, len(m))
        dist = np.array([np.min(np.abs(press - e)) for e in ends])
        spans_press = np.array([np.any((press >= e - w + 1) & (press <= e)) for e in ends])
        pool = list(ends[(dist >= gap) & (dist <= maxoff) & ~spans_press])
        used = set()
        for p in press:
            ce = p - lead
            if ce < w - 1:
                info["cases_dropped_early"] += 1; continue
            if lead > 0 and np.any((press >= ce - w + 1) & (press <= ce)):
                info["cases_dropped_overlap"] += 1; continue
            avail = [c for c in pool if c not in used]
            if not avail:
                info["cases_without_controls"] += 1; continue
            avail = np.array(avail)
            take = _pick_balanced(avail, ce, k, rng)
            if len(take) < k:
                info["controls_short"] += 1
            Xs.append(Xi[ce - w + 1:ce + 1]); ys.append(1)
            us.append(meta[m[0], 1]); ss.append(s); sets.append(set_id); offs.append(0)
            for c in take:
                used.add(int(c))
                Xs.append(Xi[c - w + 1:c + 1]); ys.append(0)
                us.append(meta[m[0], 1]); ss.append(s); sets.append(set_id)
                offs.append(int(c - ce))
            set_id += 1
    out = (np.asarray(Xs, np.float32), np.asarray(ys, int), np.asarray(us, int),
           np.asarray(ss, int), np.asarray(sets, int), np.asarray(offs, int))
    yw, offs_a = out[1], out[5]
    info.update(n_cases=int(yw.sum()), n_controls=int((yw == 0).sum()),
                n_sessions=int(len(np.unique(out[3]))),
                n_participants=int(len(np.unique(out[2]))),
                mean_abs_offset=float(np.abs(offs_a[yw == 0]).mean()) if len(yw) else None,
                frac_controls_after_case=float((offs_a[yw == 0] > 0).mean()) if len(yw) else None,
                W=w, gap=gap, maxoff=maxoff, k=k, lead=lead, seed=seed)
    return out + (info,)


def feature_sets(names, groups):
    allc = np.arange(len(names))
    return dict(
        position=np.array([names.index("perf_round_norm")]),
        all=allc,
        eegface=np.sort(np.concatenate([groups["eeg"], groups["face"]])),
        eeg=groups["eeg"], face=groups["face"], task=groups["perf"])


JOBS = [("position", "logreg", "Position only (LR) [validity check]"),
        ("all", "logreg", "All features (LR)"),
        ("all", "rf", "All features (RF)"),
        ("eegface", "logreg", "EEG + facial (LR)"),
        ("eegface", "rf", "EEG + facial (RF)"),
        ("eeg", "rf", "EEG only (RF)"),
        ("face", "rf", "Facial only (RF)"),
        ("task", "rf", "Task/timing only (RF)"),
        ("all", "transformer", "All features (Transformer)")]


def main():
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = int(args[0]) if args else None
    X, y_event, meta, names, groups = C.load()
    FS = feature_sets(names, groups)
    Xw, yw, uw, sw, sets, offs, info = build_case_control(X, y_event, meta, np.arange(X.shape[1]))
    C.save_json(dict(info, offsets=offs.tolist(), y=yw.tolist(), user=uw.tolist(),
                     set=sets.tolist()), "study2", "design.json")
    for ji, (fs, kind, label) in enumerate(JOBS):
        if only is not None and ji != only:
            continue
        dest = C.rpath("study2", f"job{ji:02d}.json")
        if os.path.exists(dest) and not force:
            print(f"[exists, use --force to overwrite] {label}"); continue
        t0 = time.time()
        r = C.loso(Xw[:, :, FS[fs]], yw, uw, kind)
        r.update(name=label, feature_set=fs, job_index=ji, design=info,
                 seconds=round(time.time() - t0, 1))
        C.save_json(r, "study2", f"job{ji:02d}.json")
        fm, po = r["fold_mean"], r["pooled"]
        print(f"{label:40s} pooledAUC={po['roc_auc']:.3f} "
              f"[{po['auc_ci95'][0]:.3f},{po['auc_ci95'][1]:.3f}] "
              f"foldAUC={fm['roc_auc']['mean']:.3f}  [{r['seconds']}s]", flush=True)
    print(f"cases={info['n_cases']} controls={info['n_controls']} "
          f"sessions={info['n_sessions']} participants={info['n_participants']}")


if __name__ == "__main__":
    main()
