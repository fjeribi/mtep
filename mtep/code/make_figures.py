"""
make_figures.py -- all eight manuscript figures, drawn only from results/.

No figure contains a hand-typed number: every value is read from the study
JSONs (which store out-of-fold predictions), leadsweep.json,
permutation.json or sensitivity.json.

Usage: python code/make_figures.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import common as C
import run_experiment as E

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


S1, S2 = load_dir("study1"), load_dir("study2")
LEAD = C.load_json("leadsweep.json")
PERM = C.load_json("permutation.json") if os.path.exists(C.rpath("permutation.json")) else {}
PERM = {k: v for k, v in PERM.items() if "p_value" in v}
SENS = C.load_json("sensitivity.json")
DES = C.load_json("study2", "design.json")


def save(fig, name):
    fig.savefig(os.path.join(C.FIG, name + ".png")); plt.close(fig); print("wrote", name)


# ---- Figure 1: the state-label confound
X, y_event, meta, names, groups = C.load()
lab = E.make_labels(y_event, meta, "state")
frac = meta[:, 3] / np.maximum(meta[:, 4], 1)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
bins = np.linspace(0, 1, 21)
ax[0].hist(frac[lab == 0], bins=bins, color=BLUE, alpha=.75, label="Not yet reported",
           edgecolor="white", linewidth=.4)
ax[0].hist(frac[lab == 1], bins=bins, color=RED, alpha=.75, label="Fatigue reported",
           edgecolor="white", linewidth=.4)
ax[0].set_xlabel("Normalised position in session"); ax[0].set_ylabel("Rounds")
ax[0].legend(frameon=False); ax[0].set_title("(a) The state label is a function of time", loc="left")
order = ["state|position_normalised_only|logreg", "state|elapsed_rounds_only|logreg",
         "state|all|logreg", "state|all|rf", "state|all|svm", "state|all|transformer",
         "state|task_only|transformer", "state|eeg+face|transformer",
         "state|eeg_only|transformer", "state|face_only|transformer"]
short = ["Position (norm.)", "Elapsed rounds", "All, LR", "All, RF", "All, SVM", "All, Transf.",
         "Task/timing, Tr.", "EEG+face, Tr.", "EEG, Tr.", "Face, Tr."]
for i, k in enumerate(order):
    v = [f["roc_auc"] for f in S1[k]["per_fold"]]
    c = RED if i < 2 else BLUE
    ax[1].scatter(np.full(len(v), i) + np.random.RandomState(i).uniform(-.12, .12, len(v)), v,
                  s=8, color=c, alpha=.55, edgecolors="none")
    ax[1].plot([i - .25, i + .25], [np.mean(v)] * 2, color="black", lw=1.2)
ax[1].axhline(.5, color=GREY, ls=":", lw=.8)
ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels(short, rotation=55, ha="right")
ax[1].set_ylabel("Fold-wise ROC-AUC"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("(b) A position-only model wins", loc="left")
save(fig, "fig1_confound")

# ---- Figure 2: Study 1 event horizon
Ds = [0, 1, 3, 5]
rs = [S1[f"event_d{D}|all|transformer"] for D in Ds]
fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.plot(Ds, [r["pooled"]["roc_auc"] for r in rs], marker="o", color=BLUE, lw=1.2, ms=4,
        label="Pooled ROC-AUC")
ax.fill_between(Ds, [r["pooled"]["auc_ci95"][0] for r in rs],
                [r["pooled"]["auc_ci95"][1] for r in rs], color=BLUE, alpha=.15, lw=0)
ax.plot(Ds, [r["pooled"]["pr_auc"] for r in rs], marker="s", color=ORANGE, lw=1.2, ms=4,
        label="Pooled PR-AUC")
ax.plot(Ds, [r["prevalence"] for r in rs], ls="--", color=GREY, lw=.9, marker=".", label="Prevalence")
ax.set_xlabel("Horizon D (rounds)"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.set_xticks(Ds); ax.legend(frameon=False)
ax.set_title("Conventional event target vs horizon", loc="left")
save(fig, "fig2_horizon")

# ---- Figure 3: lead sweep (reporting-act diagnostic)
fig, ax = plt.subplots(figsize=(4.6, 3.0))
leads = [r["lead"] for r in LEAD]
for key, lab_, c, ls in [("all", "All features (RF)", BLUE, "-"), ("task", "Task/timing (RF)", PURPLE, "-"),
                         ("face", "Facial (RF)", GREEN, "-"), ("eegface", "EEG + facial (RF)", ORANGE, "-"),
                         ("eeg", "EEG (RF)", GREY, "-"), ("position", "Position only (LR)", RED, "--")]:
    v = [r[key]["pooled_auc"] for r in LEAD]
    lo = [r[key]["ci95"][0] for r in LEAD]; hi = [r[key]["ci95"][1] for r in LEAD]
    ax.plot(leads, v, color=c, ls=ls, marker="o", ms=3.5, lw=1.2, label=lab_)
    if key in ("all", "task"):
        ax.fill_between(leads, lo, hi, color=c, alpha=.12, lw=0)
ax.axhline(.5, color=GREY, ls=":", lw=.8)
ax.set_xticks(leads); ax.set_xticklabels(["0\n(press round\nincluded)", "1", "2", "3"])
ax.set_xlabel("Lead L: rounds between case window end and press")
ax.set_ylabel("Pooled ROC-AUC"); ax.set_ylim(.35, 1.0)
ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
ax.set_title("Reporting-act diagnostic (MTEP step 6b)", loc="left")
save(fig, "fig3_leadsweep")

# ---- Figure 4: design + ROC
yv = np.array(S2["All features (RF)"]["y"])
specs = [("Position only (LR) [validity check]", "Position only (LR)", RED, "--"),
         ("All features (RF)", "All features (RF)", BLUE, "-"),
         ("Task/timing only (RF)", "Task/timing (RF)", PURPLE, "-"),
         ("EEG + facial (RF)", "EEG + facial (RF)", ORANGE, "-"),
         ("All features (Transformer)", "All features (Transf.)", GREEN, "-")]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
off = np.array(DES["offsets"])[np.array(DES["y"]) == 0]
ax[0].hist(off, bins=np.arange(-32.5, 33, 1), color=BLUE, alpha=.8, edgecolor="white", linewidth=.3)
ax[0].axvline(0, color=RED, lw=1.2, label="Case window end")
ax[0].set_xlabel("Control window end relative to case (rounds)"); ax[0].set_ylabel("Controls")
ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].legend(frameon=False); ax[0].set_title("(a) Controls on both sides of each case", loc="left")
for key, name, c, ls in specs:
    r = S2[key]; p = np.array(r["oof"]); y = np.array(r["y"])
    fpr, tpr, _ = roc_curve(y, p)
    ax[1].plot(fpr, tpr, color=c, ls=ls, lw=1.2, label=f"{name} ({roc_auc_score(y, p):.3f})")
ax[1].plot([0, 1], [0, 1], color=GREY, ls=":", lw=.8)
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].legend(frameon=False, loc="lower right", fontsize=6.8)
ax[1].set_title("(b) Pooled ROC, controlled design (L = 1)", loc="left")
save(fig, "fig4_design_roc")

# ---- Figure 5: PR
fig, ax = plt.subplots(figsize=(3.8, 2.9))
for key, name, c, ls in specs:
    r = S2[key]; p = np.array(r["oof"]); y = np.array(r["y"])
    pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, color=c, ls=ls, lw=1.2, label=f"{name} ({average_precision_score(y, p):.3f})")
ax.axhline(yv.mean(), color=GREY, ls=":", lw=.8)
ax.text(.02, yv.mean() + .02, f"prevalence = {yv.mean():.2f}", fontsize=7, color=GREY)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc="upper right", fontsize=6.8)
ax.set_title("Pooled precision–recall, controlled design", loc="left")
save(fig, "fig5_pr")

# ---- Figure 6: per participant
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
save(fig, "fig6_per_participant")

# ---- Figure 7: permutation null
if PERM:
  fig, ax = plt.subplots(1, len(PERM), figsize=(7.0, 2.6))
  for i, (k, st) in enumerate(PERM.items()):
      nul = np.array(st["null"])
      ax[i].hist(nul, bins=30, color=GREY, edgecolor="white", linewidth=.3)
      ax[i].axvline(st["observed"], color=RED, lw=1.5, label=f"Observed = {st['observed']:.3f}")
      ax[i].axvline(np.percentile(nul, 95), color=BLUE, ls="--", lw=1,
                    label=f"Null 95th pct = {np.percentile(nul, 95):.3f}")
      ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
      ax[i].set_xlabel("Pooled ROC-AUC under permuted labels")
      if i == 0: ax[i].set_ylabel("Permutations")
      ax[i].set_xlim(min(nul.min(), .35) - .02, max(st["observed"], nul.max()) + .05)
      ax[i].set_ylim(0, ax[i].get_ylim()[1] * 1.35)
      ax[i].legend(frameon=False, fontsize=7, loc="upper left"); ax[i].set_title(f"({'ab'[i]}) {st['label']}", loc="left")
  save(fig, "fig7_permutation")

# ---- Figure 8: sensitivity
fig, ax = plt.subplots(figsize=(4.8, 2.9))
tags = [("position", "Position\nonly (LR)", RED), ("eegface", "EEG + facial\n(RF)", ORANGE),
        ("task", "Task/timing\n(RF)", PURPLE), ("all", "All features\n(RF)", BLUE)]
for i, (t, nm, c) in enumerate(tags):
    v = np.array([r[t]["pooled_auc"] for r in SENS])
    ax.scatter(np.full(len(v), i) + np.random.RandomState(i).uniform(-.13, .13, len(v)), v,
               s=10, color=c, alpha=.6, edgecolors="none")
    ax.plot([i - .25, i + .25], [np.median(v)] * 2, color="black", lw=1.3)
    ax.text(i, .30, f"{v.min():.3f}–{v.max():.3f}", ha="center", fontsize=7, color=GREY)
ax.axhline(.5, color=GREY, ls=":", lw=.8)
ax.set_xticks(range(len(tags))); ax.set_xticklabels([n for _, n, _ in tags], fontsize=7.5)
ax.set_ylabel("Pooled ROC-AUC"); ax.set_ylim(.25, 1.0)
ax.set_title(f"Sensitivity across {len(SENS)} design configurations (L = 1)", loc="left")
save(fig, "fig8_sensitivity")

# ---- Figure 0 (manuscript Figure 1): the MTEP protocol schematic
fig, ax = plt.subplots(figsize=(6.6, 3.9))
ax.axis("off")
def box(x, y, w, h, text, fc="white", ec=BLUE, fs=7.5, lw=1.0, ls="-"):
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor=fc, edgecolor=ec,
                               linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)
def arrow(x1, y1, x2, y2, c=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, shrinkA=0, shrinkB=0))
box(0.02, 0.72, 0.22, 0.20, "Corpus\nsessions within\nparticipants", ec=GREY)
box(0.30, 0.72, 0.30, 0.20, "Step 1: separate label\nand feature construction\n(name check)")
box(0.66, 0.72, 0.32, 0.20, "Step 2: position-only\ndiagnostic baseline\n(is the target a clock?)")
box(0.30, 0.42, 0.30, 0.20, "Steps 3-5: matched sets\ncases at lead L >= 1,\ncontrols from same session")
box(0.66, 0.42, 0.32, 0.20, "Step 6a: position check\nCI of diagnostic AUC\nmust include 0.5", ec=RED, ls="--")
box(0.66, 0.12, 0.32, 0.20, "Step 6b: event-unit check\nlead sweep L = 0..3", ec=RED, ls="--")
box(0.30, 0.12, 0.30, 0.20, "Steps 7-10: grouped CV,\npermutation null,\nsensitivity, reporting", ec=GREEN)
box(0.68, -0.14, 0.28, 0.14, "FAIL: stop and diagnose", ec=RED, fc="#f7e9e7", fs=7)
arrow(0.24, 0.82, 0.30, 0.82); arrow(0.60, 0.82, 0.66, 0.82)
arrow(0.82, 0.72, 0.82, 0.62); arrow(0.66, 0.52, 0.60, 0.52)
arrow(0.45, 0.72, 0.45, 0.62); arrow(0.82, 0.42, 0.82, 0.32)
arrow(0.66, 0.22, 0.60, 0.22); arrow(0.82, 0.12, 0.82, 0.01)
ax.text(0.63, 0.25, "pass", fontsize=6.5, color=GREEN, ha="center")
ax.text(0.845, 0.06, "fail", fontsize=6.5, color=RED, ha="left")
ax.set_xlim(0, 1); ax.set_ylim(-0.18, 1.0)
ax.set_title("MTEP: protocol flow (Algorithm 1)", loc="left")
save(fig, "fig0_protocol")
