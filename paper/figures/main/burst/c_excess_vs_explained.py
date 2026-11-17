"""burst figure (main), panel c — EXCESS vs WGD-EXPLAINED burst. Two (4,2) segments contrasted against the doubling budget
D = ⌊M/2⌋+⌊m/2⌋ = 3: excess (all 4 gains co-time at WGD — one more than a single doubling makes) vs
WGD-explained (1 early gain + 3 at WGD = the doubling itself). Each drawn from the WGD-collapse polytope
solution. Table-driven: tau_gains + shipped route topology + tau_events, via _trees.py. Mirrors dev
panel_c_excess_vs_explained.py. make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.labelsize": 7, "axes.linewidth": 0.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "svg.fonttype": "none", "figure.dpi": 300})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import _trees
from _trees import fetch, plot_traj, build_real_tree, draw_real_tree, GAIN_DOT, NATREV, WGD_COL, TXT, MM, fix_svg_fonts

PANELS = panels_dir_of(__file__)
GREY = NATREV["grey"]
PICKS = [
    {"prefix": "f37de4fa", "chrom": "7", "start": 145845038, "title": "Excess burst", "color": NATREV["teal"][5], "n_burst": 4},
    {"prefix": "b60392fb", "chrom": "7", "start": 61700000, "title": "WGD-explained", "color": NATREV["teal"][2], "n_burst": 3},
]


def make(save=True):
    fig = plt.figure(figsize=(35 * MM, 44 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.5], left=0.184, right=0.972,
                          top=0.66, bottom=0.21, hspace=0.50, wspace=0.40)
    axes = []
    for ci, p in enumerate(PICKS):
        s = fetch(p["prefix"], p["chrom"], p["start"])
        axg = fig.add_subplot(gs[1, ci]); axes.append(axg)
        plot_traj(axg, s["draws"], s["ne"], s["wgd"], tick_fs=6, scale=0.85)
        axt = fig.add_subplot(gs[0, ci])
        maj_root, min_root, M, m, times = build_real_tree(p["prefix"], p["chrom"], p["start"],
                                                          collapse_wgd=s["wgd"], n_burst=p["n_burst"])
        rep = np.sort(times); yy = np.arange(1, s["ne"] + 1)
        axg.plot(rep, yy, color=TXT, lw=0.9, zorder=5)
        axg.scatter(rep, yy, **GAIN_DOT)
        axg.set_xlim(-0.02, 1.02); axg.set_xticks([0, 0.5, 1.0]); axg.set_xticklabels(["0", "0.5", "1"])
        draw_real_tree(axt, maj_root, min_root, None, show_minor=(m > 0), show_band=False,
                       gain_marker="none", show_allele_labels=False)
        axt.set_xlim(-0.02, 1.02)
        if s["wgd"] is not None:
            axg.add_artist(ConnectionPatch(
                xyA=(s["wgd"], 0.4), coordsA=axg.transData, xyB=(s["wgd"], 1.0), coordsB=axt.transData,
                color=WGD_COL, lw=1.1, ls=(0, (4, 2)), zorder=2))
        pos = axt.get_position()
        loc = f"chr{p['chrom']}:{p['start']/1e6:.1f}–{s['end']/1e6:.1f}Mb"
        fig.text((pos.x0 + pos.x1) / 2, 0.94, p["title"], ha="center", va="bottom", fontsize=S.annot,
                 color=p["color"], fontweight="bold")
        axt.text(0.5, 1.14, f"({s['major']},{s['minor']})\n{s['tumor_type']}\n{loc}\n{p['prefix']}",
                 transform=axt.transAxes, ha="center", va="bottom", fontsize=S.tiny, color=GREY[4], linespacing=1.1)

    axes[0].set_ylabel("gain index", fontsize=S.label, color=TXT, labelpad=2)
    fig.text(0.55, 0.125, "molecular time", ha="center", fontsize=S.label, color=TXT)
    fig.legend(handles=[Line2D([0], [0], color=GREY[5], lw=1.2, label="major"),
                        Line2D([0], [0], color=GREY[5], lw=1.2, ls=(0, (2.2, 1.4)), label="minor"),
                        Line2D([0], [0], color=TXT, marker="o", ls="none", ms=4, markeredgecolor="white",
                               markeredgewidth=0.6, label="gain"),
                        Line2D([0], [0], color=WGD_COL, lw=1.1, ls=(0, (4, 2)), label="WGD")],
               loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=4, frameon=False, fontsize=S.annot,
               handlelength=1.4, columnspacing=1.0)
    if save:
        fig.savefig(PANELS / "c_excess_vs_explained.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "c_excess_vs_explained.svg", bbox_inches="tight")
        fix_svg_fonts(PANELS / "c_excess_vs_explained.svg")
    return fig


if __name__ == "__main__":
    make(); print("b_excess_vs_explained: PNG+SVG written")
