"""burst figure (main), ALTERNATIVE to g (superseded by g_seglen.py, the excess-only histogram in the paper) — length distribution WITH COMPARATORS.

the excess-burst length profile on its own has nothing to compare against, so "arm-scale"
is unfalsifiable. This adds the two obvious reference groups from the same table:

    excess burst      a burst larger than the doubling budget D        (is_excess)
    WGD-explained     a burst, but no larger than D                    (burst class, not excess)
    gradual           no burst at all                                  (classification == gradual)

Curves are normalised WITHIN group (fraction of that group's segments per bin) — gradual outnumbers
excess 3:1, so raw counts would say nothing about shape. Medians are marked on the top axis.

Table-driven: fig6_bursts.tsv. make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.linewidth": 0.5, "svg.fonttype": "none"})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import _wgd
from _wgd import NATREV, BURST_CATS, MM

PANELS = panels_dir_of(__file__)
TEAL, GREY, TXT = NATREV["teal"], NATREV["grey"], "#273246"

GROUPS = [("excess burst",  TEAL[5], 1.3),
          ("WGD-explained", TEAL[2], 1.0),
          ("gradual",       GREY[3], 1.0)]


def _split(df):
    burst = df["classification"].isin(BURST_CATS)
    L = ((df["end_bp"] - df["start_bp"]) / 1e6)
    ok = L > 0
    return {"excess burst":  L[ok & burst & df["is_excess"]].values,
            "WGD-explained": L[ok & burst & ~df["is_excess"]].values,
            "gradual":       L[ok & (df["classification"] == "gradual")].values}


def make(save=True):
    parts = _split(_wgd.load_bursts())
    allL = np.concatenate(list(parts.values()))
    bins = np.logspace(np.log10(0.2), np.log10(allL.max()), 26)

    fig, ax = plt.subplots(figsize=(52 * MM, 27 * MM))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.80, bottom=0.24)
    handles = []
    for name, col, lw in GROUPS:
        L = parts[name]
        h, _ = np.histogram(L, bins=bins)
        frac = h / h.sum()
        if name == "excess burst":
            ax.stairs(frac, bins, fill=True, color=col, alpha=0.30, linewidth=0)
        ax.stairs(frac, bins, color=col, linewidth=lw, zorder=4)
        med = float(np.median(L))
        ax.plot([med], [1.02], marker="v", ms=3, color=col, transform=ax.get_xaxis_transform(),
                clip_on=False, zorder=6)
        handles.append(Line2D([0], [0], color=col, lw=lw,
                              label=f"{name}  (n={len(L):,}, {med:.0f} Mb)"))

    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 5, 10, 50, 100]); ax.set_xticklabels(["0.5", "1", "5", "10", "50", "100"])
    ax.set_xlabel("segment length (Mb)", fontsize=S.label, color=TXT)
    ax.set_ylabel("fraction of group", fontsize=S.label, color=TXT)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.55)          # headroom for the median markers + legend
    ax.set_title("Excess bursts are arm-scale, like every other gain",
                 fontsize=S.annot, loc="left", color=TXT, pad=6)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2)
    ax.legend(handles=handles, fontsize=S.legend, frameon=False, loc="upper left",
              handlelength=1.2, handletextpad=0.5, labelspacing=0.25, borderpad=0.1)
    if save:
        fig.savefig(PANELS / "alt_seglen_groups.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "alt_seglen_groups.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    from scipy.stats import mannwhitneyu
    p = _split(_wgd.load_bursts())
    for a, b in [("excess burst", "WGD-explained"), ("excess burst", "gradual")]:
        u = mannwhitneyu(p[a], p[b])
        print(f"{a} (med {np.median(p[a]):.1f} Mb, n={len(p[a])}) vs "
              f"{b} (med {np.median(p[b]):.1f} Mb, n={len(p[b])}): p={u.pvalue:.2e}")
    make(); print("seglen: PNG+SVG written")
