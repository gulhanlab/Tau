"""benchmarking_segment figure (supp), panel c — mutation count per segment vs per pooled-segment cluster (PCAWG).

Tau pools segments sharing a copy-number state and gain route, then re-times the pooled unit. This is
what the pooling buys in power: 76.4% of amplified segments end up ASSIGNED TO a cluster carrying more
than 200 SBS1/SBS5-weighted mutations, against only 3.7% that reach 200 on their own.

Both bars are SEGMENT-weighted, i.e. one count per segment in each series, so the comparison is
like-for-like. Counting one row per cluster instead would give 23.2%, because 60% of clusters are
singletons (holding just 12% of segments) while the 6% of clusters with >20 members hold 46% of
segments and all clear 200. The segment-weighted view is the relevant one: the question is how much of
the DATA gets well-powered timing, not how many bookkeeping units exist.

Ported from the right-hand half of fig3/panel_scripts/e_clustering_mutcount.py. Table-driven from
bench_pcawg_mutcount (per-segment `mut_unclustered` vs its cluster's `mut_clustered`).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
GREY, NAVY = "#8a93a3", "#2C7BB6"
BINS = [0, 50, 100, 200, np.inf]
LABS = ["<50", "50–100", "100–200", "≥200"]


def make(save=True):
    d = load_bench("pcawg_mutcount")
    fr_u = pd.cut(d.mut_unclustered, BINS, labels=LABS, right=False).value_counts(normalize=True).reindex(LABS)
    fr_c = pd.cut(d.mut_clustered, BINS, labels=LABS, right=False).value_counts(normalize=True).reindex(LABS)

    x = np.arange(len(LABS)); bw = 0.34
    fig, ax = plt.subplots(figsize=(55.0 * MM, 58.0 * MM))
    ax.bar(x - bw / 2, fr_u.values, bw, color=GREY, label="individual segment", zorder=3)
    ax.bar(x + bw / 2, fr_c.values, bw, color=NAVY, label="pooled-segment cluster", zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(LABS)
    ax.set_ylim(0, 0.92)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("SBS1/SBS5-weighted mutations")
    ax.set_ylabel("fraction of amplified segments")
    ax.set_title("Mutation count before\nand after pooling", fontweight="bold", pad=3, fontsize=S.title)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, labelspacing=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.3, color="0.92"); ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.2000, right=0.9455, top=0.8707, bottom=0.1638)
    if save:
        _save_panel(fig, PANELS, "c_cluster_mutcount")
    return fig


if __name__ == "__main__":
    make(); print(f"benchmarking_segment/c: written -> {PANELS}")
