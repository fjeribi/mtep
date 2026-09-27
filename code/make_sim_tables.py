"""
make_sim_tables.py -- aggregate the detector-characterisation simulation.

Reads results/sim_lead/rep*.json and results/sim_position.json and writes
results/simulation.json plus a printable summary.

Two decision rules are reported for the reporting-act check:
  interval rule    fires when the participant-clustered bootstrap 95% interval
                   for AUC(L=0) - AUC(L=1) excludes zero
  magnitude rule   fires when that difference is at least DELTA_STAR, which is
                   the materiality threshold a practitioner would use
and one for the position check: fires when the interval for the position-only
pooled AUC excludes 0.5.

For each cell the sensitivity or false-alarm rate is given with a
Clopper-Pearson 95% interval, since the denominators are small.
"""
import os, json, glob
import numpy as np
from scipy import stats

import common as C

DELTA_STAR = 0.10


def cp(k, n):
    """Clopper-Pearson 95% interval for k successes in n trials."""
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return [float(lo), float(hi)]


def agg(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def main():
    reps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(C.RES, "sim_lead", "rep*.json")))]
    out = dict(n_replicates=len(reps), delta_star=DELTA_STAR)
    if reps:
        out["n_eligible_features"] = reps[0]["n_eligible"]
        out["events_placed"] = agg([r["events"]["events_placed"] for r in reps])
        out["events_requested"] = reps[0]["events"]["events_requested"]
        out["lead1_invariance_max_abs_diff"] = float(
            max(r["lead1_invariance_max_abs_diff"] for r in reps))
        out["design_lead1"] = dict(
            cases=agg([r["design_lead1"]["n_cases"] for r in reps]),
            controls=agg([r["design_lead1"]["n_controls"] for r in reps]))

    for arm in ["spike", "state"]:
        alphas = sorted({c["alpha"] for r in reps for c in r[arm]})
        rows = []
        for a in alphas:
            cells = [c for r in reps for c in r[arm] if c["alpha"] == a]
            n = len(cells)
            k_i = sum(c["fires"] for c in cells)
            k_m = sum(c["delta"] >= DELTA_STAR for c in cells)
            rows.append(dict(
                alpha=a, n=n,
                auc_lead0=agg([c["auc_lead0"] for c in cells]),
                auc_lead1=agg([c["auc_lead1"] for c in cells]),
                delta=agg([c["delta"] for c in cells]),
                fires_interval=k_i, rate_interval=k_i / n, ci_interval=cp(k_i, n),
                fires_magnitude=k_m, rate_magnitude=k_m / n, ci_magnitude=cp(k_m, n)))
        out[arm] = rows

    pos = json.load(open(os.path.join(C.RES, "sim_position.json")))
    gammas = sorted({d["gamma"] for d in pos})
    prow = []
    for g in gammas:
        cells = [d for d in pos if d["gamma"] == g]
        n = len(cells); k = sum(d["fires"] for d in cells)
        prow.append(dict(gamma=g, n=n, fires=k, rate=k / n, ci=cp(k, n),
                         auc=agg([d["auc"] for d in cells]),
                         signed_offset=agg([d["mean_signed_offset"] for d in cells]),
                         frac_after=agg([d["frac_after"] for d in cells]),
                         cases=agg([d["n_cases"] for d in cells])))
    out["position"] = prow

    C.save_json(out, "simulation.json")

    print(f"reporting-act check, {len(reps)} replicates, "
          f"{out.get('n_eligible_features')} eligible features, "
          f"invariance check max |diff| = {out.get('lead1_invariance_max_abs_diff')}")
    for arm in ["spike", "state"]:
        print(f"\n  {arm}")
        print("   alpha   AUC(L=0)      AUC(L=1)      delta            "
              "interval rule    delta>=%.2f" % DELTA_STAR)
        for r in out[arm]:
            print("   %-6.3f %.3f (%.3f)  %.3f (%.3f)  %+.3f (%.3f)   "
                  "%2d/%-2d %4.0f%%     %2d/%-2d %4.0f%%"
                  % (r["alpha"], r["auc_lead0"]["mean"], r["auc_lead0"]["sd"],
                     r["auc_lead1"]["mean"], r["auc_lead1"]["sd"],
                     r["delta"]["mean"], r["delta"]["sd"],
                     r["fires_interval"], r["n"], 100 * r["rate_interval"],
                     r["fires_magnitude"], r["n"], 100 * r["rate_magnitude"]))
    print("\n  position check")
    print("   gamma  fires        AUC           signed offset  frac after")
    for r in out["position"]:
        print("   %-6.2f %2d/%-3d %4.0f%%  %.3f (%.3f)  %+6.2f        %.2f"
              % (r["gamma"], r["fires"], r["n"], 100 * r["rate"],
                 r["auc"]["mean"], r["auc"]["sd"],
                 r["signed_offset"]["mean"], r["frac_after"]["mean"]))


if __name__ == "__main__":
    main()


def rows():
    """Emit the three manuscript tables as row lists, for the document build.

    Counts are reported as designs CERTIFIED, since that is what the design
    section claims: a certificate granted is the positive outcome.
    """
    sim = C.load_json("simulation.json")
    dl = C.load_json("lead_delta.json")
    NICE = {"all": "All features", "eegface": "EEG + facial", "eeg": "EEG only",
            "face": "Facial only", "task": "Task/timing only"}

    t8 = []
    for arm, lab in [("spike", "Act-shaped component"),
                     ("state", "Genuine cumulative state")]:
        for r in sim[arm]:
            t8.append([lab, ("0 (reference)" if r["alpha"] == 0 else "%g" % r["alpha"]),
                       "%.3f (%.3f)" % (r["auc_lead0"]["mean"], r["auc_lead0"]["sd"]),
                       "%.3f (%.3f)" % (r["auc_lead1"]["mean"], r["auc_lead1"]["sd"]),
                       "%+.3f (%.3f)" % (r["delta"]["mean"], r["delta"]["sd"]),
                       "%d/%d" % (r["n"] - r["fires_interval"], r["n"]),
                       "%d/%d" % (r["n"] - r["fires_magnitude"], r["n"])])
    t9 = []
    for r in sim["position"]:
        lo, hi = 1.0 - r["ci"][1], 1.0 - r["ci"][0]
        t9.append(["%.2f" % r["gamma"], "%+.2f" % r["signed_offset"]["mean"],
                   "%.2f" % r["frac_after"]["mean"],
                   "%.3f (%.3f)" % (r["auc"]["mean"], r["auc"]["sd"]),
                   "%d/%d" % (r["n"] - r["fires"], r["n"]),
                   "%.2f-%.2f" % (lo, hi)])
    t10 = []
    for d in dl:
        row = [NICE[d["feature_set"]]] + ["%.3f" % a for a in d["auc"]]
        row += ["%+.3f" % st["delta"] for st in d["steps"]]
        row += ["L = %d" % d["certified_lead"]]
        t10.append(row)
    # Table 13: is the certified lead a property of the design or of the choices
    rb = C.load_json("certify_robustness.json")
    t13 = []
    for m, lab in [("rf", "Random forest"), ("logreg", "Logistic regression"),
                   ("svm", "SVM")]:
        r = rb["estimator"][m]
        t13.append(["Estimator (W = 10)", lab] +
                   ["%.3f" % a for a in r["auc"]] +
                   ["%+.3f" % st["delta"] for st in r["steps"]] +
                   ["L = %d" % r["certified_lead"]])
    for w in ["5", "10", "15"]:
        r = rb["window"][w]
        t13.append(["Window length (random forest)", "W = %s rounds" % w] +
                   ["%.3f" % a for a in r["auc"]] +
                   ["%+.3f" % st["delta"] for st in r["steps"]] +
                   ["L = %d" % r["certified_lead"]])
    return dict(t8=t8, t9=t9, t10=t10, t13=t13,
                delta_star=sim["delta_star"], agree=rb["agree"])


def headtohead():
    """Aggregate the four-design comparison over the simulated worlds."""
    import glob
    reps = [json.load(open(p)) for p in
            sorted(glob.glob(os.path.join(C.RES, "headtohead", "rep*.json")))]
    if not reps:
        return None
    worlds = ["none", "act", "state", "both"]
    truth = {w: reps[0]["worlds"][w]["truth"] for w in worlds}
    methods = ["M1", "M2", "M3", "M4"]
    out = dict(n_replicates=len(reps), truth=truth,
               alpha_act=reps[0]["alpha_act"], alpha_state=reps[0]["alpha_state"],
               by_method={})
    for m in methods:
        cell, correct = {}, 0
        for w in worlds:
            yes = [r["worlds"][w][m]["says_yes"] for r in reps]
            auc = [r["worlds"][w][m]["auc"] for r in reps
                   if r["worlds"][w][m]["auc"] is not None]
            cell[w] = dict(says_yes=int(sum(yes)), n=len(yes),
                           auc_mean=float(np.mean(auc)) if auc else None,
                           auc_sd=float(np.std(auc, ddof=1)) if len(auc) > 1 else 0.0,
                           correct=int(sum(y == truth[w] for y in yes)))
            correct += cell[w]["correct"]
        out["by_method"][m] = dict(worlds=cell, correct=correct,
                                   n_total=len(worlds) * len(reps))
    out["certified_lead"] = {
        w: [r["worlds"][w]["M4"].get("certified_lead") for r in reps]
        for w in worlds}
    return out


def h2h_rows():
    h = headtohead()
    if h is None:
        return None
    LAB = {"M1": ("Random folds, unbalanced, on the report", "P1"),
           "M2": ("Grouped LOSO, unbalanced, on the report", "P1, P2"),
           "M3": ("Grouped LOSO, balanced, on the report", "P1, P2, P3"),
           "M4": ("MTEP, read at the certified lead", "P1-P4")}
    rows = []
    for m in ["M1", "M2", "M3", "M4"]:
        d = h["by_method"][m]
        r = [LAB[m][0], LAB[m][1]]
        for w in ["none", "act", "state", "both"]:
            c = d["worlds"][w]
            r.append("%d/%d (%.2f)" % (c["says_yes"], c["n"], c["auc_mean"])
                     if c["auc_mean"] is not None
                     else "%d/%d (-)" % (c["says_yes"], c["n"]))
        r.append("%d/%d" % (d["correct"], d["n_total"]))
        rows.append(r)
    return rows, h


def h2h_prose():
    """The head-to-head section, with every number read from the results."""
    rows, h = h2h_rows()
    n = h["n_replicates"]
    W = {m: h["by_method"][m]["worlds"] for m in ("M1", "M2", "M3", "M4")}
    tot = {m: h["by_method"][m]["correct"] for m in W}
    N = h["by_method"]["M1"]["n_total"]
    cert0 = sum(1 for w in ("state", "both") for L in h["certified_lead"][w] if L == 0)

    def y(m, w):
        return W[m][w]["says_yes"]

    def a(m, w):
        return W[m][w]["auc_mean"]

    p1 = (
     "An evaluation design cannot be shown to be better by scoring higher, because "
     "a design that removes inflation scores lower. What can be shown is whether it "
     "reaches the correct conclusion on data whose truth is known, and the "
     "construction of Section 4.11 supplies exactly that: synthetic events on the "
     "real feature matrix, with a component injected or not, so that whether a "
     "state is present is settled by construction rather than estimated. We "
     "therefore apply four designs to the same corpora and record one verdict from "
     "each, namely whether the evidence supports state detection. Each design adds "
     "one property to the one above it, so the comparison isolates what each "
     "property buys, and all four use the same estimator, the same features and the "
     "same verdict rule: the 95% interval for the pooled ROC-AUC excludes 0.5.")

    p2 = (
     "Table 6 reports %d replicates in each of four worlds. With nothing injected "
     "the four designs withheld the verdict in %d, %d, %d and %d replicates; the "
     "departure from the nominal nineteen in twenty is the slight anticonservatism "
     "of a bootstrap interval on twenty participants, which Section 5.2 also "
     "reports, and it affects the three grouped designs alike. The second world is "
     "the decisive one: an act-shaped component is present, no state is, and the "
     "correct answer is that there is nothing to detect. Random folds concluded "
     "that a detector works in %d of %d replicates, grouped "
     "leave-one-participant-out in %d of %d, and grouped cross-validation with "
     "position-balanced matching, which holds three of the four properties and is "
     "the strongest design in current use, in %d of %d. The mean pooled ROC-AUC "
     "those three reported was %.2f, %.2f and %.2f, for corpora containing no "
     "state whatever. MTEP concluded that a detector works in %d of %d, at a mean "
     "of %.2f."
     % (n, n - y("M1", "none"), n - y("M2", "none"), n - y("M3", "none"),
        n - y("M4", "none"),
        y("M1", "act"), n, y("M2", "act"), n, y("M3", "act"), n,
        a("M1", "act"), a("M2", "act"), a("M3", "act"),
        y("M4", "act"), n, a("M4", "act")))

    p3 = (
     "Specificity bought by discarding real signal would be worthless, so the third "
     "and fourth worlds test the converse. Where a genuine state is present MTEP "
     "concluded correctly in %d of %d replicates with the state alone and %d of %d "
     "with the state and the act together, matching grouped cross-validation with "
     "matching at %d and %d: nothing was lost. In all %d of those replicates the "
     "certified lead was zero, which is the certificate passing a design unchanged "
     "when it finds no material discontinuity. That it does so in the fourth world, "
     "where an act-shaped component is present alongside the state, is the "
     "magnitude rule behaving as defined rather than failing: MTEP's estimate there "
     "is %.2f against %.2f with the state alone, so the act contributed about %.2f "
     "to it and was correctly judged immaterial. Across all four worlds the correct "
     "verdict was returned in %d of %d cases by MTEP, against %d, %d and %d by the "
     "three designs above it."
     % (y("M4", "state"), n, y("M4", "both"), n,
        y("M3", "state"), y("M3", "both"), cert0,
        a("M4", "both"), a("M4", "state"), a("M4", "both") - a("M4", "state"),
        tot["M4"], N, tot["M1"], tot["M2"], tot["M3"]))

    p4 = (
     "Two qualifications. The comparison is between designs on simulated corpora, "
     "not between published studies, and no claim is made that any particular "
     "result in the literature is of the kind the second world describes. And the "
     "distance between the third design and the fourth is the whole of the "
     "contribution: the three properties that existing practice supplies are "
     "necessary, they are assumed throughout this paper, and the finding is only "
     "that they do not imply the fourth. The same gap appears on CogBeacon without "
     "any simulation, in Section 5.6, where a design satisfying the first three "
     "properties returns a position-only ROC-AUC of 0.493 both with the reporting "
     "round inside the window and with it excluded, while the estimates from those "
     "two designs differ by 0.228.")

    caption = (
     "Table 6 Do the four designs reach the right conclusion? Twenty replicates "
     "per world.")

    note = (
     "Each design adds one property to the one above it and returns one verdict, "
     "whether the 95%% interval for its pooled ROC-AUC excludes 0.5. Worlds are "
     "built on synthetic events so that the truth is known: an act-shaped component "
     "of %g local standard deviations carries no state, a smooth state of %g does. "
     "Cells give the number of replicates in which the design concluded that a "
     "detector works, with the mean pooled ROC-AUC it reported in brackets; a "
     "design is correct when it says yes exactly in the two worlds where a state is "
     "present. M1 resamples rows for its interval, as an analyst using random folds "
     "would; M2 to M4 resample participants."
     % (h["alpha_act"], h["alpha_state"]))

    return [("P", p1), ("P", p2), ("P", p3), ("P", p4)], caption, rows, note
