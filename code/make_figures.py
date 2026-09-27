"""
make_figures.py -- Fig. S1 (per-participant ROC-AUC), drawn only from results/.

Every value is read from the Study 2 JSONs, which store the per-fold results.
Figures 1-4 and Fig. S2 are drawn by make_protocol_figure.py, make_sim_figure.py
and make_merged_figures.py.

Usage: python code/make_figures.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

os.makedirs(C.FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.dpi": 300})
GREY, BLUE, RED, GREEN, ORANGE, PURPLE = "#666666", "#2c5f8a", "#b03a2e", "#1e7a5a", "#c07a10", "#6a4c93"


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(C.rpath(d, "job*.json"))):
        r = json.load(open(f)); out[r["name"]] = r
    return out


S2 = load_dir("study2")


def save(fig, name):
    fig.savefig(os.path.join(C.FIG, name + ".png")); plt.close(fig); print("wrote", name)


# ---- Fig. S1: per participant
keys = [("Position only (LR) [validity check]", "Position\n(LR)"), ("EEG only (RF)", "EEG\n(RF)"),
        ("Facial only (RF)", "Facial\n(RF)"), ("EEG + facial (RF)", "EEG+face\n(RF)"),
        ("Task/timing only (RF)", "Task/timing\n(RF)"), ("All features (RF)", "All\n(RF)"),
        ("All features (Transformer)", "All\n(Transf.)")]
fig, ax = plt.subplots(figsize=(5.4, 2.9))
for i, (k, nm) in enumerate(keys):
    pf = S2[k]["per_fold"]; v = np.array([f["roc_auc"] for f in pf]); sz = np.array([f["n_pos"] for f in pf])
    ax.scatter(np.full(len(v), i) + np.random.RandomState(i).uniform(-.13, .13, len(v)), v,
               s=6 + 5 * sz, color=RED if i == 0 else BLUE, alpha=.55, edgecolors="none")
    ax.plot([i - .25, i + .25], [v.mean()] * 2, color="black", lw=1.3)
ax.axhline(.5, color=GREY, ls=":", lw=.8)
ax.set_xticks(range(len(keys))); ax.set_xticklabels([n for _, n in keys], fontsize=7)
ax.set_ylabel("ROC-AUC, held-out participant"); ax.set_ylim(0, 1.05)
ax.set_title(f"Per-participant variability (n = {len(pf)}; marker area ∝ cases)", loc="left")
save(fig, "fig8_per_participant")
