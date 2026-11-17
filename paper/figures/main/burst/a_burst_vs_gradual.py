"""burst figure (main), panel a — BURST vs GRADUAL exemplars. Real high-CN segments side by side, each an allele lineage tree
(top) over its gain trajectory (bottom): WGD-sample bursts whose gains stack at WGD, a non-WGD focal
burst, and gradual segments whose gains spread over molecular time. Table-driven: tau_gains (per-solution
gain times) + shipped route topology (tree geometry) + tau_events (WGD), via _trees.py. Mirrors dev
panel_a_burst_vs_gradual.py. make() -> fig + PNG/SVG.
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
from _trees import fetch, plot_traj, build_real_tree, draw_real_tree, GAIN_DOT, NATREV, TXT, MM, WGD_COL, fix_svg_fonts

PANELS = panels_dir_of(__file__)
TEAL, GREY = NATREV["teal"], NATREV["grey"]
PICKS = [
    dict(prefix="60413de1", chrom="9", start=418935, label="Burst", burst=True),
    dict(prefix="fb9135d9", chrom="X", start=142419843, label="Burst", burst=True, coln=4),
    dict(prefix="3b41cb48", chrom="8", start=115144627, label="Burst (no WGD)", burst=True),
    dict(prefix="8b28f6d2", chrom="4", start=171433714, label="Gradual", burst=False),
    dict(prefix="140d5fa9", chrom="18", start=19000000, label="Gradual", burst=False),
    dict(prefix="e93b0979", chrom="7", start=40386, label="Gradual", burst=False),
]


def make(save=True):
    NP = len(PICKS)
    fig = plt.figure(figsize=(116 * MM, 46 * MM))
    gs = fig.add_gridspec(2, NP, height_ratios=[0.78, 1.5], left=0.065, right=0.99,
                          top=0.66, bottom=0.21, hspace=0.50, wspace=0.40)
    axes = []; taxes = []
    for ci, p in enumerate(PICKS):
        s = fetch(p["prefix"], p["chrom"], p["start"])
        if p.get("coln"):
            maj_root, min_root, M, m, times = build_real_tree(
                p["prefix"], p["chrom"], p["start"], collapse_wgd=s["wgd"], n_burst=p["coln"])
        else:
            maj_root, min_root, M, m, times = build_real_tree(
                p["prefix"], p["chrom"], p["start"], bursty=p["burst"])
        dots = np.sort(times)
        axg = fig.add_subplot(gs[1, ci]); axes.append(axg)
        plot_traj(axg, s["draws"], s["ne"], s["wgd"], tick_fs=6, scale=0.85)
        yy = np.arange(1, s["ne"] + 1)
        axg.plot(dots, yy, color=TXT, lw=0.9, zorder=5)
        axg.scatter(dots, yy, **GAIN_DOT)
        axg.set_xlim(-0.02, 1.02); axg.set_xticks([0, 0.5, 1.0]); axg.set_xticklabels(["0", "0.5", "1"])
        axt = fig.add_subplot(gs[0, ci]); taxes.append(axt)
        mspan = (0.99, 0.45) if s["major"] >= 6 else (0.94, 0.55)
        draw_real_tree(axt, maj_root, min_root, None, show_minor=(m > 0), gain_marker="none",
                       show_allele_labels=False, maj_span=mspan)
        axt.set_xlim(-0.02, 1.02)
        if s["wgd"] is not None:
            axg.add_artist(ConnectionPatch(
                xyA=(s["wgd"], 0.4), coordsA=axg.transData, xyB=(s["wgd"], 1.0), coordsB=axt.transData,
                color=WGD_COL, lw=1.1, ls=(0, (4, 2)), zorder=2))
        note = " (no WGD)" if (p["burst"] and s["wgd"] is None) else ""
        loc = f"chr{p['chrom']}:{p['start']/1e6:.1f}–{s['end']/1e6:.1f}Mb"
        axt.text(0.5, 1.16, f"({s['major']},{s['minor']}){note}\n{s['tumor_type']}\n{loc}\n{p['prefix']}",
                 transform=axt.transAxes, ha="center", va="bottom", fontsize=S.tiny, color=GREY[4], linespacing=1.1)

    axes[0].set_ylabel("gain index", fontsize=S.label, color=TXT, labelpad=2)
    fig.text(0.55, 0.085, "molecular time", ha="center", fontsize=S.label, color=TXT)

    yb = max(a.get_position().y1 for a in taxes) + 0.20

    def _umbrella(axlist, label, color):
        x0 = min(a.get_position().x0 for a in axlist); x1 = max(a.get_position().x1 for a in axlist)
        fig.add_artist(Line2D([x0, x1], [yb, yb], color=color, lw=1.1, transform=fig.transFigure, zorder=9))
        for xe in (x0, x1):
            fig.add_artist(Line2D([xe, xe], [yb, yb - 0.022], color=color, lw=1.1, transform=fig.transFigure, zorder=9))
        fig.text((x0 + x1) / 2, yb + 0.03, label, ha="center", va="bottom", fontsize=S.label, fontweight="bold", color=color)
    _umbrella(taxes[:3], "Burst", TEAL[5])
    _umbrella(taxes[3:], "Gradual", GREY[4])

    fig.legend(handles=[Line2D([0], [0], color=GREY[5], lw=1.2, label="major"),
                        Line2D([0], [0], color=GREY[5], lw=1.2, ls=(0, (2.2, 1.4)), label="minor"),
                        Line2D([0], [0], color=TXT, marker="o", ls="none", ms=4, markeredgecolor="white",
                               markeredgewidth=0.6, label="gain"),
                        Line2D([0], [0], color=WGD_COL, lw=1.1, ls=(0, (4, 2)), label="WGD")],
               loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=4, frameon=False, fontsize=S.annot,
               handlelength=1.4, columnspacing=1.1)
    if save:
        fig.savefig(PANELS / "a_burst_vs_gradual.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "a_burst_vs_gradual.svg", bbox_inches="tight")
        fix_svg_fonts(PANELS / "a_burst_vs_gradual.svg")
    return fig


if __name__ == "__main__":
    make(); print("a_burst_vs_gradual: PNG+SVG written")
