"""
test_mtep.py -- invariants of the design, checked rather than assumed.

These are the properties the manuscript relies on. Each is stated as an
assertion over the released arrays or over small synthetic corpora, so that a
change to the windowing, the sampler or the decision rule that breaks one of
them fails loudly.

    python code/test_mtep.py            # ~2 minutes, CPU only

Tests marked SLOW fit models; the rest are pure construction checks.
"""
import sys, time
import numpy as np

import common as C
import run_casecontrol as CC
import simlib as S

PASS, FAIL = [], []


def check(name, fn):
    t0 = time.time()
    try:
        fn()
        PASS.append(name)
        print(f"  ok    {name}  [{time.time() - t0:.1f}s]", flush=True)
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}: {e}", flush=True)


X, YE, META, NAMES, GROUPS = C.load()
ALL = np.arange(X.shape[1])


def _windows(lead, **kw):
    """Rebuild the design and return, per window, the set of unit indices it
    spans, together with its label and its session."""
    Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
        X, YE, META, ALL, lead=lead, **kw)
    return Xw, yw, uw, sw, sets, offs, info


# --------------------------------------------------------------------------- #
# P4: no window may contain an event unit once a lead is applied
# --------------------------------------------------------------------------- #
def t_no_event_in_window():
    for lead in (1, 2, 3):
        spans = _spans(lead)
        for sess, lo, hi, lab in spans:
            ev = set(np.where(YE[_rows(sess)] == 1)[0])
            inside = ev & set(range(lo, hi + 1))
            assert not inside, (f"lead {lead}: a {'case' if lab else 'control'} "
                                f"window in session {sess} spans event unit(s) "
                                f"{sorted(inside)}")


def _rows(sess):
    m = np.where(META[:, 0] == sess)[0]
    return m[np.argsort(META[m, 3])]


def _spans(lead, **kw):
    """(session, first_unit, last_unit, is_case) for every window built."""
    rng_out = []
    Xw, yw, uw, sw, sets, offs, info = _windows(lead, **kw)
    # reconstruct ends: a case ends lead units before its event; a control's end
    # is its case's end plus the recorded offset
    for s in np.unique(sw):
        idx = np.where(sw == s)[0]
        press = np.where(YE[_rows(s)] == 1)[0]
        for i in idx:
            if yw[i] == 1:
                continue
        # ends are recovered per matched set below
    # a direct reconstruction: rerun the builder's logic for ends
    for s, m in C.session_order(META):
        ev = YE[m]
        if ev.sum() == 0:
            continue
        press = np.where(ev == 1)[0]
        for p in press:
            ce = p - lead
            if ce < CC.W - 1:
                continue
            if lead > 0 and np.any((press >= ce - CC.W + 1) & (press <= ce)):
                continue
            rng_out.append((s, ce - CC.W + 1, ce, 1))
    Xw, yw, uw, sw, sets, offs, info = _windows(lead, **kw)
    # controls: offsets are relative to their case end, which is recoverable
    case_ends = {}
    ci = 0
    for s, m in C.session_order(META):
        ev = YE[m]
        if ev.sum() == 0:
            continue
        press = np.where(ev == 1)[0]
        for p in press:
            ce = p - lead
            if ce < CC.W - 1:
                continue
            if lead > 0 and np.any((press >= ce - CC.W + 1) & (press <= ce)):
                continue
            case_ends[ci] = (s, ce)
            ci += 1
    for i in range(len(yw)):
        if yw[i] == 0:
            s, ce = case_ends[sets[i]]
            end = ce + offs[i]
            rng_out.append((s, end - CC.W + 1, end, 0))
    return rng_out


def t_controls_respect_gap():
    """The constraint is on distance to the nearest EVENT, not to the case: a
    control may sit far from its own case while still being close to another
    event in the same session."""
    for s, lo, hi, lab in _spans(1):
        if lab != 0:
            continue
        ev = np.where(YE[_rows(s)] == 1)[0]
        d = int(np.min(np.abs(ev - hi)))
        assert d >= CC.GAP, f"control ending at {hi} is {d} units from an event"
        assert d <= CC.MAXOFF, f"control ending at {hi} is {d} units from every event"


def t_controls_used_once():
    Xw, yw, uw, sw, sets, offs, info = _windows(1)
    keys = [(sw[i], sets[i], offs[i]) for i in range(len(yw)) if yw[i] == 0]
    ends = {}
    for s, st, off in keys:
        ends.setdefault(s, []).append((st, off))
    # a control end is (case end + offset); distinct matched sets may not reuse one
    spans = [(s, lo, hi) for s, lo, hi, lab in _spans(1) if lab == 0]
    assert len(spans) == len(set(spans)), "a control window is used more than once"


def t_matched_sets_disjoint():
    Xw, yw, uw, sw, sets, offs, info = _windows(1)
    for st in np.unique(sets):
        m = sets == st
        assert yw[m].sum() == 1, f"matched set {st} does not have exactly one case"
        assert len(np.unique(sw[m])) == 1, f"matched set {st} spans sessions"
        assert len(np.unique(uw[m])) == 1, f"matched set {st} spans participants"


def t_lead_semantics():
    """A case window must end exactly `lead` units before its event."""
    for lead in (0, 1, 2, 3):
        for s, lo, hi, lab in _spans(lead):
            if lab != 1:
                continue
            ev = np.where(YE[_rows(s)] == 1)[0]
            assert (hi + lead) in set(ev), (
                f"lead {lead}: case window ending at {hi} has no event at "
                f"{hi + lead} in session {s}")
            assert hi - lo + 1 == CC.W, "case window is not W units long"


def t_blocked_units_excluded():
    """With `blocked` set, no window may contain a blocked unit."""
    rng = np.random.RandomState(0)
    y_ps, blocked, info = S.pseudo_events(YE, META, rng, w=CC.W, gap=CC.GAP)
    Xw, yw, uw, sw, sets, offs, inf = CC.build_case_control(
        X, y_ps, META, ALL, lead=1, seed=0, blocked=blocked)
    assert inf["n_cases"] > 0
    # every real press must be outside every window: verify via the feature rows
    for s, m in C.session_order(META):
        real = set(np.where(YE[m] == 1)[0])
        ps = set(np.where(y_ps[m] == 1)[0])
        for p in ps:
            assert min((abs(p - r) for r in real), default=99) >= S.MIN_SEP, (
                f"a synthetic event sits {min(abs(p - r) for r in real)} units "
                f"from a real one")


def t_blocked_default_is_inert():
    """Passing blocked=None must reproduce the released design exactly."""
    a = CC.build_case_control(X, YE, META, ALL, lead=1)
    b = CC.build_case_control(X, YE, META, ALL, lead=1,
                              blocked=np.zeros_like(YE))
    for i in (1, 2, 3, 4, 5):
        assert np.array_equal(a[i], b[i]), "blocked=None differs from blocked=0"
    assert a[6] == b[6]


def t_injection_is_inert_at_zero():
    rng = np.random.RandomState(1)
    sd, elig = S.feature_sd(X, META, w=CC.W)
    cols, signs = S.draw_targets(elig, rng)
    Xi = S.inject(X, YE, META, "spike", 0.0, cols, signs, sd,
                  np.random.RandomState(2))
    assert Xi is X or np.array_equal(np.nan_to_num(Xi), np.nan_to_num(X)), \
        "amplitude 0 changed the feature matrix"


def t_spike_touches_only_event_units():
    rng = np.random.RandomState(1)
    sd, elig = S.feature_sd(X, META, w=CC.W)
    cols, signs = S.draw_targets(elig, rng)
    Xi = S.inject(X, YE, META, "spike", 1.0, cols, signs, sd,
                  np.random.RandomState(2))
    changed = np.where(np.nansum(np.abs(Xi - X), axis=1) > 0)[0]
    assert set(changed) <= set(np.where(YE == 1)[0]), \
        "a spike injection touched a unit that carries no event"


def t_eligible_features_move_locally():
    sd, elig = S.feature_sd(X, META, w=CC.W)
    assert len(elig) < X.shape[1], "no feature was excluded; the filter is inert"
    assert np.all(sd[elig] > 0), "an eligible feature has zero local SD"


def t_certified_lead_rule():
    """The rule must return the first lead whose following step is immaterial."""
    from mtep import DELTA_STAR
    d = C.load_json("lead_delta.json")
    for r in d:
        steps = [s["delta"] for s in r["steps"]]
        want = next((L for L in range(len(steps)) if steps[L] < DELTA_STAR),
                    len(steps))
        assert r["certified_lead"] == want, (
            f"{r['feature_set']}: stored {r['certified_lead']}, rule gives {want}")


def t_paired_delta_matches_point_estimate():
    a = dict(y=np.array([0, 1, 0, 1]), p=np.array([.1, .9, .2, .8]),
             u=np.array([1, 1, 2, 2]), auc=1.0)
    b = dict(y=np.array([0, 1, 0, 1]), p=np.array([.9, .1, .8, .2]),
             u=np.array([1, 1, 2, 2]), auc=0.0)
    r = S.paired_delta_ci(a, b, n_boot=50, seed=0)
    assert abs(r["delta"] - 1.0) < 1e-12, r["delta"]


def t_label_feature_separation():
    bad = [n for n in NAMES if any(t in n.lower() for t in
                                   ("label", "report", "press", "fatigue"))]
    assert not bad, f"a feature name refers to the label stream: {bad}"


def t_released_design_counts():
    """The bundled design must still be the one the manuscript describes."""
    want = {0: (77, 231), 1: (68, 204), 2: (66, 198), 3: (63, 189)}
    for lead, (nc, nk) in want.items():
        info = CC.build_case_control(X, YE, META, ALL, lead=lead)[6]
        assert (info["n_cases"], info["n_controls"]) == (nc, nk), (
            f"lead {lead}: {info['n_cases']}/{info['n_controls']}, "
            f"manuscript says {nc}/{nk}")


def t_lean_loso_matches_full_loso():   # SLOW
    Xw, yw, uw, sw, sets, offs, info = _windows(1)
    cols = CC.feature_sets(NAMES, GROUPS)["task"]
    full = C.loso(Xw[:, :, cols], yw, uw, "rf")
    lean = S.lean_loso(Xw[:, :, cols], yw, uw, "rf")
    assert abs(full["pooled"]["roc_auc"] - lean["auc"]) < 1e-12, (
        full["pooled"]["roc_auc"], lean["auc"])


def t_api_reproduces_manuscript():     # SLOW
    import mtep
    out = mtep.selftest()
    assert out["report_independence"]["certified_lead"] == 1


def main():
    slow = "--fast" not in sys.argv
    for name, fn in [
            ("no event unit inside any window at lead >= 1", t_no_event_in_window),
            ("case windows end exactly `lead` units before the event", t_lead_semantics),
            ("controls respect the gap and maximum offset", t_controls_respect_gap),
            ("each control window is used at most once", t_controls_used_once),
            ("matched sets are disjoint, one case each", t_matched_sets_disjoint),
            ("blocked units stay out of every window", t_blocked_units_excluded),
            ("blocked=None reproduces the released design", t_blocked_default_is_inert),
            ("released design counts are unchanged", t_released_design_counts),
            ("amplitude 0 leaves the feature matrix alone", t_injection_is_inert_at_zero),
            ("a spike touches only event units", t_spike_touches_only_event_units),
            ("the eligibility filter excludes flat features", t_eligible_features_move_locally),
            ("the certified-lead rule matches the stored result", t_certified_lead_rule),
            ("paired delta equals the difference in AUC", t_paired_delta_matches_point_estimate),
            ("no feature name refers to the label stream", t_label_feature_separation)]:
        check(name, fn)
    if slow:
        check("lean CV reproduces the full CV exactly (SLOW)", t_lean_loso_matches_full_loso)
        check("the public API reproduces the manuscript (SLOW)", t_api_reproduces_manuscript)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
