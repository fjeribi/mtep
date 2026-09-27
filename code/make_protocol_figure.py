"""
make_protocol_figure.py -- Fig. 1, the MTEP design as a flow diagram.

Solid boxes are construction steps; dashed boxes are the two certificates.
The step numbers are those of Algorithm 1.

Usage: python code/make_protocol_figure.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import common as C

os.makedirs(C.FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.6,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.dpi": 300})
INK, BLUE, RED = "#222222", "#2c5f8a", "#b03a2e"

# (x, y, w, h, text, kind)  kind: "build" | "certify" | "out"
BOXES = [
    (0.01, 0.775, 0.305, 0.165,
     "1  extract\nfeatures and labels from\ndisjoint streams; assert that\nno feature names the label", "build"),
    (0.355, 0.775, 0.29, 0.165,
     "2  diagnostic\nfit a position-only model\non the whole corpus", "certify"),
    (0.01, 0.545, 0.305, 0.165,
     "3  cases\nwindows of W units\nending Lead units\nbefore an event", "build"),
    (0.355, 0.545, 0.29, 0.165,
     "4  controls\nk per case from the\nsame recording unit,\nbalanced before and after", "build"),
    (0.69, 0.545, 0.30, 0.165,
     "5  matched sets\ndisjoint sets of a case\nand its controls", "build"),
    (0.01, 0.305, 0.305, 0.18,
     "6a  certificate: P3\nposition-only AUC on\nthe matched sample;\ngrant if its 95% CI\nincludes 0.5", "certify"),
    (0.355, 0.305, 0.29, 0.18,
     "6b  certificate: P4\nD(L) = AUC(L-1) - AUC(L);\ncertify at the smallest L\nwith D(L+1) below\nthe threshold", "certify"),
    (0.69, 0.305, 0.30, 0.18,
     "7  evaluate\ngrouped leave-one-\nparticipant-out; pooled AUC,\nparticipant-clustered CI", "build"),
    (0.01, 0.075, 0.305, 0.155,
     "8  permutation null\npermute case status\nwithin matched sets", "build"),
    (0.355, 0.075, 0.29, 0.155,
     "9  sensitivity\nrepeat over a grid of\nW, g, m, k and seed", "build"),
    (0.69, 0.075, 0.30, 0.155,
     "10  report\npooled and fold-wise\nresults, paired tests,\ngrid range", "out"),
]

ARROWS = [
    ((0.163, 0.775), (0.163, 0.712)),   # 1 -> 3
    ((0.316, 0.858), (0.355, 0.858)),   # 1 -> 2
    ((0.316, 0.628), (0.355, 0.628)),   # 3 -> 4
    ((0.646, 0.628), (0.690, 0.628)),   # 4 -> 5
    ((0.840, 0.545), (0.840, 0.487)),   # 5 -> 7
    ((0.690, 0.395), (0.646, 0.395)),   # 7 -> 6b
    ((0.355, 0.395), (0.316, 0.395)),   # 6b -> 6a
    ((0.163, 0.545), (0.163, 0.487)),   # 3 -> 6a
    ((0.840, 0.305), (0.840, 0.232)),   # 7 -> 10
    ((0.163, 0.305), (0.163, 0.232)),   # 6a -> 8
    ((0.316, 0.152), (0.355, 0.152)),   # 8 -> 9
    ((0.646, 0.152), (0.690, 0.152)),   # 9 -> 10
]

fig, ax = plt.subplots(figsize=(7.2, 4.9))
ax.set_xlim(0, 1); ax.set_ylim(0.03, 1.0); ax.axis("off")

for x, y, w, h, text, kind in BOXES:
    dashed = kind == "certify"
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.3 if dashed else 1.0,
        linestyle=(0, (4, 2)) if dashed else "solid",
        edgecolor=RED if dashed else INK,
        facecolor="#fbf4f3" if dashed else ("#f2f2f2" if kind == "out" else "white"),
        zorder=2))
    head, *rest = text.split("\n")
    ax.text(x + w / 2, y + h - 0.030, head, ha="center", va="center",
            fontsize=7.8, fontweight="bold",
            color=RED if dashed else INK, zorder=3)
    ax.text(x + w / 2, y + (h - 0.052) / 2, "\n".join(rest), ha="center",
            va="center", fontsize=6.7, color=INK, zorder=3, linespacing=1.45)

for (x0, y0), (x1, y1) in ARROWS:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.9,
                                 color="#555555", zorder=1,
                                 shrinkA=1, shrinkB=1))

ax.text(0.50, 0.978,
        "solid: construction steps        dashed: the two certificates",
        ha="center", va="center", fontsize=7.6, color="#555555")

fig.savefig(os.path.join(C.FIG, "fig1_protocol.png"))
fig.savefig(os.path.join(C.FIG, "fig1_protocol.pdf"))
print("wrote fig1_protocol.{png,pdf}")
