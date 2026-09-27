"""
make_sim_figure.py -- Fig. 2, calibration of the two validity certificates.

Every value is read from results/simulation.json, results/sim_lead/rep*.json and
results/lead_delta.json.  Nothing is typed by hand.

Usage: python code/make_sim_figure.py
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
GREY, BLUE, RED, GREEN, ORANGE = "#666666", "#2c5f8a", "#b03a2e", "#1e7a5a", "#c07a10"

SIM = C.load_json("simulation.json")
REPS = [json.load(open(p)) for p in sorted(glob.glob(C.rpath("sim_lead", "rep*.json")))]
DELTA = C.load_json("lead_delta.json")
OBS = [d for d in DELTA if d["feature_set"] == "all"][0]
DSTAR = SIM["delta_star"]

fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.1))

# ---- (a) detection rate of the reporting-act check --------------------------
a = ax[0]
sp = SIM["spike"]
x = np.arange(len(sp))
for rows, key, col, lab, mk in [
        (sp, "rate_magnitude", RED, r"magnitude rule ($\Delta \geq %.2f$)" % DSTAR, "o"),
        (sp, "rate_interval", BLUE, "interval rule (CI excludes 0)", "s")]:
    y = [r[key] for r in rows]
    lo = [r[key] - r["ci_" + key.split("_")[1]][0] for r in rows]
    hi = [r["ci_" + key.split("_")[1]][1] - r[key] for r in rows]
    a.errorbar(x, y, yerr=[lo, hi], color=col, marker=mk, ms=4, lw=1.4,
               capsize=2.5, label=lab)
a.set_xticks(x)
a.set_xticklabels([("0" if r["alpha"] == 0 else "%g" % r["alpha"]) for r in sp])
a.set_xlabel(r"act-shaped component $\alpha$ (local SD)")
a.set_ylabel("certificate refused (proportion)")
a.set_ylim(-0.03, 1.03)
a.axhline(0.05, color=GREY, ls=":", lw=0.9)
a.text(0.72, 0.075, "5%", transform=a.get_yaxis_transform(), color=GREY, fontsize=7)
a.legend(loc="lower right", frameon=False)
a.set_title("(a) report-independence certificate", loc="left")

# ---- (b) the statistic itself, by condition --------------------------------
b = ax[1]
conds, labs, cols = [], [], []
for cell_alpha in [c["alpha"] for c in SIM["spike"]]:
    conds.append([c["delta"] for r in REPS for c in r["spike"] if c["alpha"] == cell_alpha])
    labs.append("0" if cell_alpha == 0 else "%g" % cell_alpha)
    cols.append(GREY if cell_alpha == 0 else RED)
for cell_alpha in [c["alpha"] for c in SIM["state"]]:
    conds.append([c["delta"] for r in REPS for c in r["state"] if c["alpha"] == cell_alpha])
    labs.append("%g" % cell_alpha)
    cols.append(GREEN)
rng = np.random.RandomState(C.SEED)
for i, (v, col) in enumerate(zip(conds, cols)):
    b.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=7, color=col, alpha=0.55,
              linewidths=0)
    b.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=col, lw=1.8)
b.axhline(0, color=GREY, lw=0.8)
b.axhline(DSTAR, color=RED, ls="--", lw=1.0)
b.axhline(OBS["delta"], color="black", ls="-", lw=1.2)
b.text(len(conds) - 0.4, OBS["delta"] + 0.012, "CogBeacon (%.3f)" % OBS["delta"],
       ha="right", fontsize=7.5)
b.text(-0.4, DSTAR + 0.012, r"$\delta^\ast$", color=RED, fontsize=8)
b.set_xticks(range(len(conds)))
b.set_xticklabels(labs, fontsize=7)
b.set_xlabel(r"act-shaped $\alpha$ (red)          genuine state $\alpha$ (green)")
b.set_ylabel(r"$\Delta = \mathrm{AUC}(L{=}0) - \mathrm{AUC}(L{=}1)$")
b.set_title("(b) displacement statistic, by injected condition", loc="left")

# ---- (c) position check ----------------------------------------------------
c = ax[2]
po = SIM["position"]
g = [r["gamma"] for r in po]
y = [r["rate"] for r in po]
lo = [r["rate"] - r["ci"][0] for r in po]
hi = [r["ci"][1] - r["rate"] for r in po]
c.errorbar(g, y, yerr=[lo, hi], color=BLUE, marker="o", ms=4, lw=1.4, capsize=2.5)
c.axhline(0.05, color=GREY, ls=":", lw=0.9)
c.set_xlabel(r"control-side imbalance $\gamma$")
c.set_ylabel("certificate refused (proportion)", color=BLUE)
c.tick_params(axis="y", colors=BLUE)
c.set_ylim(-0.03, 1.03)
c2 = c.twinx()
c2.plot(g, [r["signed_offset"]["mean"] for r in po], color=ORANGE, marker="^", ms=4,
        lw=1.2, ls="--", label="mean signed offset")
c2.set_ylabel("mean control offset (rounds)", color=ORANGE)
c2.tick_params(axis="y", colors=ORANGE)
c2.spines["right"].set_visible(True)
c2.spines["right"].set_color(ORANGE)
c.set_title("(c) position certificate", loc="left")

fig.tight_layout()
fig.savefig(os.path.join(C.FIG, "fig2_calibration.png"))
fig.savefig(os.path.join(C.FIG, "fig2_calibration.pdf"))
print("wrote fig2_calibration.{png,pdf}")
