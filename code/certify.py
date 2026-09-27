"""
certify.py -- apply the MTEP design to a corpus of your own.

    python code/certify.py yourdata.npz [--out report.json] [--lead-max 4]

The .npz must contain:

    X            (n_units, n_features)  features, one row per unit
    y_event      (n_units,)             1 at a unit where a report was made
    session      (n_units,)             recording-unit id
    participant  (n_units,)             person id; folds are formed on this
    unit_index   (n_units,)             1-based position within the session

and may contain:

    position     (n_units,)             normalised position, if the default
                                        (unit_index / session length) is wrong
    feature_names (n_features,)         checked against P1

It prints the two certificates and the estimate at the certified lead, and
writes the full result as JSON when --out is given. Nothing about CogBeacon is
assumed; the defaults are the manuscript's, and every one of them can be
overridden on the command line.
"""
import argparse, json, sys
import numpy as np

import run_casecontrol as CC
import common as C
from mtep import run_mtep, DELTA_STAR

REQUIRED = ["X", "y_event", "session", "participant", "unit_index"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", help="write the full result here as JSON")
    ap.add_argument("--window", type=int, default=CC.W, help="W, units per window")
    ap.add_argument("--gap", type=int, default=CC.GAP,
                    help="minimum distance from a control to every event")
    ap.add_argument("--maxoff", type=int, default=CC.MAXOFF,
                    help="maximum distance from a control to its case")
    ap.add_argument("--controls", type=int, default=CC.N_CTRL, help="k per case")
    ap.add_argument("--lead-max", type=int, default=3,
                    help="largest lead to try; raise it if nothing certifies")
    ap.add_argument("--model", default="rf", choices=["rf", "logreg"])
    ap.add_argument("--delta-star", type=float, default=DELTA_STAR,
                    help="materiality threshold for the displacement statistic")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        sys.exit(f"{a.npz} is missing: {', '.join(missing)}\n\n{__doc__}")
    names = [str(x) for x in d["feature_names"]] if "feature_names" in d else None

    out = run_mtep(d["X"], d["y_event"], d["session"], d["participant"],
                   d["unit_index"],
                   position=d["position"] if "position" in d else None,
                   W=a.window, gap=a.gap, maxoff=a.maxoff, k=a.controls,
                   leads=tuple(range(a.lead_max + 1)), model=a.model,
                   delta_star=a.delta_star, seed=a.seed, feature_names=names)

    p3, p4 = out["position_balance"], out["report_independence"]
    print()
    print(f"  P3 position balance      {'granted' if p3['granted'] else 'REFUSED'}"
          f"   position-only AUC {p3['auc']:.3f} "
          f"[{p3['ci95'][0]:.3f}, {p3['ci95'][1]:.3f}]")
    print(f"  P4 report-independence   "
          f"{'granted' if p4['granted'] else 'REFUSED'}"
          f"   certified lead {p4['certified_lead']}")
    for k, s in sorted(p4["steps"].items(), key=lambda kv: int(kv[0])):
        here = s["from_lead"] == p4["certified_lead"]
        print(f"       D({s['from_lead']} to {s['to_lead']}) = {s['delta']:+.3f} "
              f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
              f"{'   <- certifies at lead %d' % s['from_lead'] if here else ''}")
    print(f"\n  {out['verdict']}")
    print("\n  The specificity and sensitivity of these certificates were measured "
          "on CogBeacon\n  (Section 5.2). To measure them for this corpus, point "
          "MTEP_DATA at this file and\n  run code/run_simulation.py and "
          "code/run_simposition.py.")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
