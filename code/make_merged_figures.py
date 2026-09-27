"""
make_merged_figures.py -- the consolidated manuscript figures.

Three figures replace six. Every panel is the same panel that stood alone
before, drawn from the same results JSON; nothing is redrawn by hand and no
number is typed here.

  fig3_confound_single  Fig. 3   the position confound alone (the fold-wise
                                 panel it used to carry is Table 7)
  fig4_temporal         Fig. 4   (a) conventional target against horizon D
                                 (b) matched design against lead L
  fig5_design           Fig. 5   (a) control offsets  (b) ROC  (c) PR

Usage: python code/make_merged_figures.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score)
import common as C
import run_experiment as E

os.makedirs(C.FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.dpi": 300})
GREY, BLUE, RED = "#666666", "#2c5f8a", "#b03a2e"
GREEN, ORANGE, PURPLE = "#1e7a5a", "#c07a10", "#6a4c93"


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(C.rpath(d, "job*.json"))):
        r = json.load(open(f)); out[r["name"]] = r
    return out


S1, S2 = load_dir("study1"), load_dir("study2")
LEAD = C.load_json("leadsweep.json")
DES = C.load_json("study2", "design.json")


def save(fig, name):
    fig.savefig(os.path.join(C.FIG, name + ".png"))
    plt.close(fig)
    print("wrote", name)


# ---- Fig. 3: the position confound, single panel ---------------------------
X, y_event, meta, names, groups = C.load()
lab = E.make_labels(y_event, meta, "state")
frac = meta[:, 3] / np.maximum(meta[:, 4], 1)
fig, ax = plt.subplots(figsize=(4.2, 2.8))
bins = np.linspace(0, 1, 21)
ax.hist(frac[lab == 0], bins=bins, color=BLUE, alpha=.75,
        label="Not yet reported", edgecolor="white", linewidth=.4)
ax.hist(frac[lab == 1], bins=bins, color=RED, alpha=.75,
        label="Fatigue reported", edgecolor="white", linewidth=.4)
ax.set_xlabel("Normalised position in session")
ax.set_ylabel("Rounds")
ax.legend(frameon=False)
ax.set_title("The state label is a function of time", loc="left")
save(fig, "fig3_confound_single")


# ---- Fig. 4: two temporal sweeps in one figure -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

Ds = [0, 1, 3, 5]
rs = [S1[f"event_d{D}|all|transformer"] for D in Ds]
ax[0].plot(Ds, [r["pooled"]["roc_auc"] for r in rs], marker="o", color=BLUE,
           lw=1.2, ms=4, label="Pooled ROC-AUC")
ax[0].fill_between(Ds, [r["pooled"]["auc_ci95"][0] for r in rs],
                   [r["pooled"]["auc_ci95"][1] for r in rs],
                   color=BLUE, alpha=.15, lw=0)
ax[0].plot(Ds, [r["pooled"]["pr_auc"] for r in rs], marker="s", color=ORANGE,
           lw=1.2, ms=4, label="Pooled PR-AUC")
ax[0].plot(Ds, [r["prevalence"] for r in rs], ls="--", color=GREY, lw=.9,
           marker=".", label="Prevalence")
ax[0].set_xlabel("Horizon D (rounds)"); ax[0].set_ylabel("Score")
ax[0].set_ylim(0, 1); ax[0].set_xticks(Ds); ax[0].legend(frameon=False)
ax[0].set_title("(a) Conventional target, unmatched", loc="left")

leads = [r["lead"] for r in LEAD]
for key, lab_, c, ls in [("all", "All features (RF)", BLUE, "-"),
                         ("task", "Task/timing (RF)", PURPLE, "-"),
                         ("face", "Facial (RF)", GREEN, "-"),
                         ("eegface", "EEG + facial (RF)", ORANGE, "-"),
                         ("eeg", "EEG (RF)", GREY, "-"),
                         ("position", "Position only (LR)", RED, "--")]:
    v = [r[key]["pooled_auc"] for r in LEAD]
    lo = [r[key]["ci95"][0] for r in LEAD]
    hi = [r[key]["ci95"][1] for r in LEAD]
    ax[1].plot(leads, v, color=c, ls=ls, marker="o", ms=3.5, lw=1.2, label=lab_)
    if key in ("all", "task"):
        ax[1].fill_between(leads, lo, hi, color=c, alpha=.12, lw=0)
ax[1].axhline(.5, color=GREY, ls=":", lw=.8)
ax[1].set_xticks(leads)
ax[1].set_xlabel("Lead L (rounds)")
ax[1].set_ylabel("Pooled ROC-AUC"); ax[1].set_ylim(.35, 1.0)
ax[1].legend(frameon=False, fontsize=6.2, ncol=2, loc="upper right",
             columnspacing=.8, handlelength=1.4)
ax[1].set_title("(b) Certifying report-independence", loc="left")
save(fig, "fig4_temporal")


# ---- Fig. 5: design, ROC and PR in one figure ------------------------------
specs = [("Position only (LR) [validity check]", "Position only (LR)", RED, "--"),
         ("All features (RF)", "All features (RF)", BLUE, "-"),
         ("Task/timing only (RF)", "Task/timing (RF)", PURPLE, "-"),
         ("EEG + facial (RF)", "EEG + facial (RF)", ORANGE, "-"),
         ("All features (Transformer)", "All features (Transf.)", GREEN, "-")]
yv = np.array(S2["All features (RF)"]["y"])

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
off = np.array(DES["offsets"])[np.array(DES["y"]) == 0]
ax[0].hist(off, bins=np.arange(-32.5, 33, 1), color=BLUE, alpha=.8,
           edgecolor="white", linewidth=.3)
ax[0].axvline(0, color=RED, lw=1.2, label="Case window end")
ax[0].set_xlabel("Control end relative to case (rounds)")
ax[0].set_ylabel("Controls")
ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].legend(frameon=False, fontsize=6.6, loc="upper left")
ax[0].set_title("(a) Controls on both sides", loc="left")

for key, name, c, ls in specs:
    r = S2[key]; p = np.array(r["oof"]); y = np.array(r["y"])
    fpr, tpr, _ = roc_curve(y, p)
    ax[1].plot(fpr, tpr, color=c, ls=ls, lw=1.2,
               label=f"{name} ({roc_auc_score(y, p):.3f})")
ax[1].plot([0, 1], [0, 1], color=GREY, ls=":", lw=.8)
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].legend(frameon=False, loc="lower right", fontsize=5.8)
ax[1].set_title("(b) Pooled ROC (L = 1)", loc="left")

for key, name, c, ls in specs:
    r = S2[key]; p = np.array(r["oof"]); y = np.array(r["y"])
    pr, rc, _ = precision_recall_curve(y, p)
    ax[2].plot(rc, pr, color=c, ls=ls, lw=1.2,
               label=f"{name} ({average_precision_score(y, p):.3f})")
ax[2].axhline(yv.mean(), color=GREY, ls=":", lw=.8)
ax[2].text(.02, yv.mean() + .02, f"prevalence = {yv.mean():.2f}", fontsize=6.4,
           color=GREY)
ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision"); ax[2].set_ylim(0, 1.02)
ax[2].legend(frameon=False, loc="upper right", fontsize=5.8)
ax[2].set_title("(c) Pooled precision-recall", loc="left")
save(fig, "fig5_design")
