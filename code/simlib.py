"""
simlib.py -- shared machinery for the detector-characterisation simulation.

The simulation asks a question about the two MTEP validity checks that cannot
be answered from a single corpus: given a contamination of KNOWN strength, how
often does the check fire, and how often does it fire when there is none?

Three ingredients:

  pseudo_events()  builds the null world.  The real corpus already contains a
                reporting-act signature, so it cannot serve as a null; and
                editing the press rounds out of the feature matrix introduces a
                discontinuity that is itself detectable.  Instead the matched
                design is rebuilt around SYNTHETIC events: in each session, as
                many rounds are drawn as there are real presses, each at least
                W + GAP rounds from every real press and from every other
                synthetic event.  The real press rounds are passed to
                build_case_control as blocked rounds, so no case or control
                window can touch one.  Nothing in the feature matrix is
                altered, so the real covariance, drift and participant
                structure are all preserved, and by construction no synthetic
                event carries a reporting act.

  inject()      adds a synthetic component of known amplitude at the synthetic
                events, in one of two shapes:

                  "spike"  a discontinuity confined to the press round alone
                           -- the signature of a reporting act.
                  "state"  a smooth Gaussian bump of width TAU rounds centred
                           on the press round -- the signature of a genuine
                           cumulative state that rises before the report and
                           is what a fatigue detector is supposed to find.

                Amplitude is in units of each injected feature's pooled SD.
                M_INJECT features are drawn at random per replicate, with a
                random sign each, so results are averaged over feature subsets
                rather than tied to one hand-picked set.

  lean_loso()   the same grouped leave-one-participant-out loop, estimator and
                hyperparameters as common.loso(), returning only what the
                simulation needs (out-of-fold scores).  It reproduces
                common.loso()'s pooled AUC exactly.

paired_delta_ci() is the decision rule for the lead-sweep check: resample
participants with replacement, recompute the L = 0 and L = 1 pooled AUCs on the
resampled participants, and take the difference.  The check fires when the
resulting 95% interval for the difference excludes zero.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

import common as C

M_INJECT = 10      # features carrying the injected component, per replicate
TAU = 6.0          # width (rounds) of the smooth "state" bump
REL_NOISE = 1.0   # event-to-event scatter of the artefact, relative to alpha
MIN_SEP = 15       # W + GAP: separation required from real presses and each other
MAX_LEAD = 3       # the largest lead the sweep uses, so a case exists for all of them


# --------------------------------------------------------------------------- #
#  null world
# --------------------------------------------------------------------------- #
def pseudo_events(y_event, meta, rng, w=10, gap=5):
    """Synthetic events that carry no reporting act, plus the rounds to block.

    Returns (y_pseudo, blocked): y_pseudo marks the synthetic events used to
    define cases; blocked marks the real press rounds, which are kept out of
    every window.  Per session the number of synthetic events equals the number
    of real presses where the separation constraints allow it.
    """
    sep = w + gap
    y_ps = np.zeros_like(y_event)
    blocked = (y_event == 1).astype(int)
    n_want = n_got = 0
    for _, m in C.session_order(meta):
        press = np.where(y_event[m] == 1)[0]
        if len(press) == 0:
            continue
        n_want += len(press)
        cand = np.array([r for r in range(w - 1 + MAX_LEAD, len(m))
                         if np.min(np.abs(press - r)) >= sep])
        rng.shuffle(cand)
        chosen = []
        for c in cand:
            if len(chosen) == len(press):
                break
            if all(abs(c - q) >= sep for q in chosen):
                chosen.append(int(c))
        for c in chosen:
            y_ps[m[c]] = 1
        n_got += len(chosen)
    return y_ps, blocked, dict(events_requested=int(n_want), events_placed=int(n_got))


# --------------------------------------------------------------------------- #
#  injection
# --------------------------------------------------------------------------- #
def draw_targets(eligible, rng, m=M_INJECT):
    cols = rng.choice(eligible, size=min(m, len(eligible)), replace=False)
    signs = rng.choice([-1.0, 1.0], size=len(cols))
    return cols, signs


def resolution(X):
    """Per-feature recording resolution: the granularity of the stored values.

    Several CogBeacon features are counts (pixels, frames), so an injected
    offset that is not a whole multiple of their step would mark the event round
    exactly, at any amplitude, and the simulation would measure that arithmetic
    tell rather than the detector.  Injected offsets are therefore snapped to
    this grid, which also encodes the physical fact that a perturbation finer
    than the recording resolution leaves no trace.
    """
    out = np.ones(X.shape[1])
    for j in range(X.shape[1]):
        v = np.unique(X[~np.isnan(X[:, j]), j])
        if len(v) < 3:
            continue
        d = np.diff(v)
        d = d[d > 0]
        if len(d):
            out[j] = float(np.percentile(d, 5))
    return out


def feature_sd(X, meta, w=10):
    """Local SD per feature, and which features may carry an injected artefact.

    The yardstick has to be the variability the artefact must hide in, which is
    the round-to-round scatter inside a single window -- not the pooled SD
    across the corpus, which is dominated by between-participant offsets, and
    not the within-session SD either.  Some CogBeacon channels (signal-quality
    flags, for instance) are piecewise constant, so inside a ten-round window
    they usually do not move at all; an offset placed on one of those is
    detectable at any amplitude however small, which says nothing about the
    detector.  Features whose median local SD is zero are therefore not
    eligible for injection, and the amplitudes reported are multiples of the
    median local SD of the features that are.
    """
    sds = [[] for _ in range(X.shape[1])]
    for _, m in C.session_order(meta):
        A = X[m, :]
        for e in range(w - 1, len(m)):
            s = np.nanstd(A[e - w + 1:e + 1, :], axis=0)
            for j in range(X.shape[1]):
                if np.isfinite(s[j]):
                    sds[j].append(s[j])
    med = np.array([np.median(v) if v else 0.0 for v in sds])
    eligible = np.where(med > 0)[0]
    return np.where(med > 0, med, 1.0), eligible


def inject(Xr, y_event, meta, shape, alpha, cols, signs, sd, rng,
           rel_noise=REL_NOISE, res=None):
    """Add a component of standardised amplitude alpha to Xr.

    Each event draws one latent magnitude z = alpha * (1 + rel_noise * e),
    e ~ N(0, 1), in units of feature SD: the artefact has mean amplitude alpha
    and an event-to-event scatter of rel_noise * alpha, so a real act whose
    magnitude varies from one occurrence to the next is represented, and
    alpha = 0 is exactly no injection.  The latent loads on the chosen features
    with feature-specific signs and scales, as a single physical act would.

    shape = "spike": placed on the event round only -- a discontinuity at the
                     report, which is what a reporting act produces.
    shape = "state": spread over rounds with weight exp(-((r-p)/TAU)**2) summed
                     over the session's events and capped at 1 -- a smooth
                     latent that rises before the report and is what a genuine
                     cumulative state looks like.
    """
    if alpha == 0.0:
        return Xr
    Xi = Xr.copy()
    load = sd[cols] * signs
    for _, m in C.session_order(meta):
        press = np.where(y_event[m] == 1)[0]
        if len(press) == 0:
            continue
        for p in press:
            z = alpha * (1.0 + rel_noise * rng.randn())
            if shape == "spike":
                w = np.zeros(len(m))
                w[p] = 1.0
            elif shape == "state":
                w = np.exp(-((np.arange(len(m)) - p) / TAU) ** 2)
            else:
                raise ValueError(shape)
            hit = np.where(w > 1e-3)[0]
            delta = z * w[hit, None] * load
            if res is not None:
                q = res[cols][None, :]
                delta = np.round(delta / q) * q
            Xi[np.ix_(m[hit], cols)] += delta
    return Xi


# --------------------------------------------------------------------------- #
#  lean cross-validation
# --------------------------------------------------------------------------- #
def lean_loso(Xw, yw, uw, kind="rf", seed=C.SEED):
    """Out-of-fold scores from the same LOSO loop and estimator as common.loso."""
    users = np.unique(uw)
    rng = np.random.RandomState(seed)
    oof = np.full(len(yw), np.nan)
    for u in users:
        te = uw == u
        tr_users = np.setdiff1d(users, [u])
        va_users = rng.choice(tr_users, size=max(1, len(tr_users) // 8), replace=False)
        tr = (~te) & (~np.isin(uw, va_users))
        if len(np.unique(yw[tr])) < 2:
            continue
        sc = C.fit_scaler(Xw[tr])
        Xtr = C.summarise(C.apply_scaler(Xw[tr], sc))
        Xte = C.summarise(C.apply_scaler(Xw[te], sc))
        if kind == "rf":
            clf = RandomForestClassifier(random_state=seed, **C.RF_PARAMS)
        elif kind == "logreg":
            clf = LogisticRegression(**C.LR_PARAMS)
        elif kind == "svm":
            clf = SVC(random_state=seed, **C.SVM_PARAMS)
        else:
            raise ValueError(kind)
        clf.fit(Xtr, yw[tr])
        if kind == "svm":
            # the same logistic link on the decision function as common.loso
            oof[te] = 1.0 / (1.0 + np.exp(-clf.decision_function(Xte)))
        else:
            oof[te] = clf.predict_proba(Xte)[:, 1]
    ok = ~np.isnan(oof)
    return dict(y=yw[ok], p=oof[ok], u=uw[ok],
                auc=float(roc_auc_score(yw[ok], oof[ok])))


# --------------------------------------------------------------------------- #
#  decision rules
# --------------------------------------------------------------------------- #
def _by_user(r):
    return {u: np.where(r["u"] == u)[0] for u in np.unique(r["u"])}


def paired_delta_ci(r0, r1, n_boot=C.N_BOOT, seed=C.SEED):
    """Participant-clustered bootstrap CI for AUC(L=0) - AUC(L=1).

    The two runs share participants but not windows, so the resampling unit is
    the participant: each replicate draws participants with replacement and
    recomputes both pooled AUCs from that participant multiset.
    """
    rng = np.random.RandomState(seed)
    i0, i1 = _by_user(r0), _by_user(r1)
    users = np.array(sorted(set(i0) & set(i1)))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(users, size=len(users), replace=True)
        a = np.concatenate([i0[u] for u in pick])
        b = np.concatenate([i1[u] for u in pick])
        if len(np.unique(r0["y"][a])) > 1 and len(np.unique(r1["y"][b])) > 1:
            vals.append(roc_auc_score(r0["y"][a], r0["p"][a])
                        - roc_auc_score(r1["y"][b], r1["p"][b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(delta=r0["auc"] - r1["auc"], ci95=[float(lo), float(hi)],
                fires=bool(lo > 0.0), n_boot_used=len(vals))


def auc_ci(r, n_boot=C.N_BOOT, seed=C.SEED):
    """Participant-clustered bootstrap CI for one pooled AUC."""
    rng = np.random.RandomState(seed)
    idx = _by_user(r)
    users = np.array(sorted(idx))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(users, size=len(users), replace=True)
        a = np.concatenate([idx[u] for u in pick])
        if len(np.unique(r["y"][a])) > 1:
            vals.append(roc_auc_score(r["y"][a], r["p"][a]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(auc=r["auc"], ci95=[float(lo), float(hi)],
                fires=bool(lo > 0.5 or hi < 0.5), n_boot_used=len(vals))
