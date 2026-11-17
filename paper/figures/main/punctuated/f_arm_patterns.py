"""punctuated figure (main), panel f — recurrent ccPG chromosome-ARM programmes as a bubble heatmap (involvement x timing).
Dot SIZE = % of a type's ccPG events involving the arm; dot COLOUR = mean event molecular time
(viridis; missing-timing dots grey). Reads the derived arm-membership table staged in figures/data/<label>/
by paper/figures/data/punctuated/build.py (per PGD event: recovered autosome-arm set + event time — not carried in tau_events).
`make()` returns the figure and writes PNG+SVG next to this script.
Ported from dev/pcawg/FigWGDPGD/panel_E_arm_patterns.py.
"""
import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _wgdpgd_common import save_panel, plt, MM, TXT

R = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(R))
from paper.config import cache_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

MIN_TYPE = 15
MIN_CELL = 2     # >= this many events to colour a (type, arm) dot by timing
ACROCENTRIC_P = {"13p", "14p", "15p", "21p", "22p"}   # negligible true content -> no row
ARMS = [f"{c}{a}" for c in range(1, 23) for a in "pq" if f"{c}{a}" not in ACROCENTRIC_P]
TCMAP = plt.get_cmap("viridis")
LAB = {"CNS.GBM": "CNS GBM", "CNS.Medullo": "Medulloblastoma", "Liver.HCC": "Liver",
       "Kidney.RCC": "Kidney", "Panc.AdenoCA": "Pancreatic", "Panc.Endocrine": "Panc. endocrine",
       "Breast.AdenoCA": "Breast", "Lymph.BNHL": "Lymphoma B-NHL", "Head.SCC": "Head/neck SCC",
       "Stomach.AdenoCA": "Stomach", "ColoRect.AdenoCA": "Colorectal", "Eso.AdenoCA": "Esophageal"}


def make(save=True):
    df = pd.read_csv(cache_dir_of(__file__) / "pgd_arm_membership.tsv", sep="\t")
    df["arms"] = df["arms"].apply(ast.literal_eval)
    df["time"] = df["time"].astype(float)
    n = len(df)

    inv = Counter()
    for a in df.arms:
        inv.update(a)
    arms = [a for a in ARMS if inv.get(a, 0) > 0]

    vc = df.tumor_type.value_counts()
    types = [t for t in vc.index if vc[t] >= MIN_TYPE and t != "Unknown"]

    pct = np.zeros((len(types), len(arms)))          # % involvement -> dot size
    tim = np.full((len(types), len(arms)), np.nan)   # mean timing -> dot colour
    for ti, t in enumerate(types):
        sub = df[df.tumor_type == t]; c = Counter(); times = defaultdict(list)
        for _, r in sub.iterrows():
            for arm in r["arms"]:
                c[arm] += 1; times[arm].append(r["time"])
        for ai, arm in enumerate(arms):
            pct[ti, ai] = c.get(arm, 0) / len(sub) * 100
            if len(times[arm]) >= MIN_CELL:
                tim[ti, ai] = float(np.mean(times[arm]))

    # row order by dominant arm position, then its value
    ordr = sorted(range(len(types)), key=lambda i: (int(np.argmax(pct[i])), -pct[i].max()))
    types = [types[i] for i in ordr]; pct = pct[ordr]; tim = tim[ordr]

    # ── layout as in the assembled figure: tumour types as COLUMNS (labels rotated 45 deg above),
    # chromosome arms as ROWS (1p at the top), size/colour legends stacked at the right ──
    types = types[::-1]; pct = pct[::-1]; tim = tim[::-1]        # left-to-right = the PDF's order
    W_MM, H_MM = 72.0, 96.0
    fig, ax = plt.subplots(figsize=(W_MM * MM, H_MM * MM))
    fig.subplots_adjust(left=9.5 / W_MM, right=49.0 / W_MM, top=1 - 25.0 / H_MM, bottom=5.7 / H_MM)   # dot grid 39.4 x 65.3 mm

    xx, yy = np.meshgrid(np.arange(len(types)), np.arange(len(arms)))   # x = type, y = arm
    m = pct.T > 0
    sizes = (pct.T[m] / 100) * 40 + 2                # marker area scaled by involvement
    xm, ym, tm = xx[m], yy[m], tim.T[m]
    nan = np.isnan(tm)                               # missing-timing dots -> fixed grey (not a viridis colour)
    ax.scatter(xm[nan], ym[nan], s=sizes[nan], c="#bbbbbb", edgecolor="#3a3a3a", lw=0.3, zorder=2)
    sc = ax.scatter(xm[~nan], ym[~nan], s=sizes[~nan], c=tm[~nan], cmap=TCMAP, vmin=0, vmax=1,
                    edgecolor="#3a3a3a", lw=0.3, zorder=3)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{LAB.get(t, t)} ({vc[t]})" for t in types], fontsize=S.tick, color=TXT,
                       rotation=45, ha="left", va="bottom", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=S.tick)
    # arm labels alternate black/grey by chromosome, with a thin rule between chromosomes
    chrom_of = [a[:-1] for a in arms]
    for lab, c in zip(ax.get_yticklabels(), chrom_of):
        lab.set_color(TXT if int(c) % 2 else "#8a8a8a")
    for i in range(1, len(arms)):
        if chrom_of[i] != chrom_of[i - 1]:
            ax.axhline(i - 0.5, color="#c8c8c8", lw=0.4, zorder=1)
    ax.set_ylabel("Chromosome arm", fontsize=S.label, color=TXT)
    fig.text(1.5 / W_MM, 1 - 3.0 / H_MM, "ccPG arm programmes", fontsize=S.title, color=TXT, ha="left", va="top")
    ax.set_xlim(-0.6, len(types) - 0.4); ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.invert_yaxis(); ax.tick_params(length=0, pad=1.5)
    ax.set_axisbelow(True); ax.grid(True, axis="x", color="#eeeeee", lw=0.3)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # size legend (right, top) then colourbar (right, below it)
    hands = [Line2D([0], [0], marker="o", ls="", mfc="#bbbbbb", mec="#3a3a3a", mew=0.3,
                    ms=np.sqrt((p / 100) * 40 + 2), label=f"{p}%") for p in (20, 50, 80)]
    ax.legend(handles=hands, title="% involvement", fontsize=S.title, title_fontsize=S.legend, frameon=False,
              loc="upper left", bbox_to_anchor=(1.04, 1.0), ncol=1, handletextpad=0.6, labelspacing=0.9,
              borderaxespad=0.0)
    cax = fig.add_axes([53.0 / W_MM, 1 - 79.0 / H_MM, 2.2 / W_MM, 26.0 / H_MM])
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=S.tick, length=0); cb.outline.set_visible(False)
    cb.ax.yaxis.set_label_position("left")             # label on the LEFT of the bar, ticks on the right
    cb.set_label("mean timing", fontsize=S.label, color=TXT, labelpad=3)
    cb.set_ticks([0, 0.5, 1.0]); cb.set_ticklabels(["0 early", "0.5", "1 late"])

    out = save_panel(fig, "f_arm_patterns", save=save, exact=True)
    print(f"  {n} PGD events, {len(types)} types, {len(arms)} arms")
    return out


if __name__ == "__main__":
    make(); print("f_arm_patterns: PNG+SVG written")
