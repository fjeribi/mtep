"""
run_stats.py -- paired comparisons between Study 2 configurations.

Folds are paired by participant ID (not by list position). Test: two-sided
Wilcoxon signed-rank on fold-wise ROC-AUC, zero differences discarded
(zero_method="wilcox"); the exact null distribution is used when there are no
zero or tied differences, otherwise the normal approximation. Effect size:
matched-pairs rank-biserial correlation over non-zero differences. Wins,
ties and losses are reported. Holm-Bonferroni correction across the family.

Usage: python code/run_stats.py
Outputs -> results/stats.json
"""
import glob, json
import numpy as np
from scipy.stats import wilcoxon, rankdata
import common as C

PAIRS = [
    ("All features (RF)", "All features (Transformer)"),
    ("All features (RF)", "All features (LR)"),
    ("All features (RF)", "EEG + facial (RF)"),
    ("EEG + facial (RF)", "EEG only (RF)"),
    ("EEG + facial (RF)", "Facial only (RF)"),
    ("All features (RF)", "Task/timing only (RF)"),
    ("All features (RF)", "Position only (LR) [validity check]"),
    ("EEG + facial (RF)", "Position only (LR) [validity check]"),
    ("Task/timing only (RF)", "Position only (LR) [validity check]"),
    ("All features (Transformer)", "All features (LR)"),
]


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def main():
    folds = {}
    for f in sorted(glob.glob(C.rpath("study2", "job*.json"))):
        r = json.load(open(f))
        folds[r["name"]] = {x["participant"]: x["roc_auc"] for x in r["per_fold"]}
    rows = []
    for a, b in PAIRS:
        ids = sorted(set(folds[a]) & set(folds[b]))
        va = np.array([folds[a][i] for i in ids]); vb = np.array([folds[b][i] for i in ids])
        d = va - vb
        nz = d[d != 0]
        exact = len(nz) == len(d) and len(np.unique(np.abs(nz))) == len(nz)
        stat, p = wilcoxon(va, vb, zero_method="wilcox",
                           method="exact" if exact else "approx")
        r = rankdata(np.abs(nz))
        rb = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()) if len(nz) else 0.0
        rows.append(dict(a=a, b=b, n_folds=len(ids), mean_a=float(va.mean()),
                         mean_b=float(vb.mean()), median_diff=float(np.median(d)),
                         W=float(stat), p_raw=float(p), method="exact" if exact else "normal approx.",
                         rank_biserial=rb, wins=int((d > 0).sum()), ties=int((d == 0).sum()),
                         losses=int((d < 0).sum())))
    for r, adj in zip(rows, holm([r["p_raw"] for r in rows])):
        r["p_holm"] = float(adj); r["significant"] = bool(adj < 0.05)
    C.save_json(rows, "stats.json")
    for r in rows:
        print(f"{r['a'][:22]:22s} vs {r['b'][:22]:22s} {r['mean_a']:.3f} {r['mean_b']:.3f} "
              f"W/T/L={r['wins']}/{r['ties']}/{r['losses']} p={r['p_raw']:.4f} "
              f"holm={r['p_holm']:.4f} rb={r['rank_biserial']:+.2f}")


if __name__ == "__main__":
    main()
