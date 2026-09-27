"""
run_leaddelta.py -- the reporting-act check applied to CogBeacon, under exactly
the decision rule the simulation characterises.

For each feature set, fits the matched design at L = 0, 1 and 2 and reports the
step from L = 0 to L = 1 and the following step from L = 1 to L = 2, each with a
participant-clustered bootstrap interval. The design is certified at the
smallest lead beyond which removing one further unit changes nothing material,
read against the operating characteristics in results/simulation.json.

Usage: python code/run_leaddelta.py [--force]
Outputs -> results/lead_delta.json
"""
import sys, os
import numpy as np

import common as C
import run_casecontrol as CC
import simlib as S

SETS = ["all", "eegface", "eeg", "face", "task"]
DELTA_STAR = 0.10
MODEL = {"all": "rf", "eegface": "rf", "eeg": "rf", "face": "rf", "task": "rf"}


def main():
    dest = C.rpath("lead_delta.json")
    if os.path.exists(dest) and "--force" not in sys.argv:
        print("[exists, use --force]"); return
    X, y_event, meta, names, groups = C.load()
    FS = CC.feature_sets(names, groups)
    runs = {}
    for lead in [0, 1, 2, 3]:
        Xw, yw, uw, sw, sets, offs, info = CC.build_case_control(
            X, y_event, meta, np.arange(X.shape[1]), lead=lead)
        for fs in SETS:
            runs[(fs, lead)] = S.lean_loso(Xw[:, :, FS[fs]], yw, uw, MODEL[fs])
            print(f"L={lead} {fs:8s} AUC={runs[(fs, lead)]['auc']:.4f}", flush=True)
    out = []
    for fs in SETS:
        steps = [S.paired_delta_ci(runs[(fs, L)], runs[(fs, L + 1)])
                 for L in range(3)]
        # certified at the smallest lead beyond which one more unit is immaterial
        cert = next((L for L in range(3) if steps[L]["delta"] < DELTA_STAR), 3)
        out.append(dict(feature_set=fs, model=MODEL[fs],
                        auc=[runs[(fs, L)]["auc"] for L in range(4)],
                        steps=[dict(from_lead=L, to_lead=L + 1, **steps[L])
                               for L in range(3)],
                        certified_lead=cert,
                        auc_lead0=runs[(fs, 0)]["auc"],
                        auc_lead1=runs[(fs, 1)]["auc"],
                        **steps[0]))
        print(f"{fs:8s} " + "  ".join(
            f"d({L}->{L+1})={steps[L]['delta']:+.3f}" for L in range(3))
            + f"   certified at L={cert}")
    C.save_json(out, "lead_delta.json")


if __name__ == "__main__":
    main()
