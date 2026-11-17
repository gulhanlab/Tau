"""burst figure (main), panel g — genomic length distribution of excess-burst segments (a single histogram of excess bursts on a log length axis, counts on y, dashed line and label at
the median). The three-group comparison (excess / WGD-explained / gradual) is kept as alt_seglen_groups.py.

Table-driven: fig6_bursts.tsv (best-route excess bursts). make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[3]))
import _wgd  # noqa: E402
from _wgd import BURST_CATS, NATREV, TXT, MM  # noqa: E402
from paper.config import panels_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

PANELS = panels_dir_of(__file__)
COL = NATREV["teal"][3]


def excess_lengths():
    df = _wgd.load_bursts()
    m = df["classification"].isin(BURST_CATS) & df["is_excess"]
    L = ((df["end_bp"] - df["start_bp"]) / 1e6)[m]
    return L[L > 0].values


def make(save=True):
    L = excess_lengths()
    bins = np.logspace(np.log10(0.2), np.log10(L.max()), 26)
    fig, ax = plt.subplots(figsize=(52 * MM, 30 * MM))
    fig.subplots_adjust(left=0.20, right=0.985, top=0.95, bottom=0.26)
    ax.hist(L, bins=bins, color=COL, edgecolor="white", linewidth=0.3, zorder=3)
    med = float(np.median(L))
    ax.axvline(med, color=TXT, ls="--", lw=0.8, zorder=4)
    ax.text(med * 1.15, ax.get_ylim()[1] * 0.93, f"median {med:.0f} Mb", fontsize=S.annot, color=TXT, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 5, 10, 50, 100]); ax.set_xticklabels(["0.5", "1", "5", "10", "50", "100"])
    ax.set_xlabel("excess-burst segment length (Mb)", fontsize=S.label, color=TXT)
    ax.set_ylabel("segments", fontsize=S.label, color=TXT)
    ax.tick_params(labelsize=S.tick, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    if save:
        fig.savefig(PANELS / "g_seglen.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "g_seglen.svg", bbox_inches="tight")
    print(f"  {len(L):,} excess bursts, median {med:.1f} Mb")
    return fig


if __name__ == "__main__":
    make(); print("g_seglen: PNG+SVG written")
