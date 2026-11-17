"""benchmarking_segment figure (supp), panel a — the ESS>=10 timing threshold against the PCAWG segment distribution.

Tau times any amplified segment carrying at least 10 SBS1/SBS5-weighted mutations. Most amplified
segments are small and fall below that, but SEGMENT LENGTH is what determines how much of the genome
gets timed, and length concentrates in the large segments that clear the cut comfortably — so the
threshold retains 79.7% of amplified segment length.

The histogram is therefore weighted by base pairs, not segment count: counting segments
over-represents the many tiny ones and understates what the threshold actually keeps.

Table-driven from tau_segments (ess, seg_len_bp, major_cn).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_segments

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
CUT = 10
BAR, MARK = "#8fa8c8", "#b0402f"


def make(save=True):
    s = load_segments()
    amp = s[(s.major_cn >= 2) & s.ess.notna() & (s.ess > 0)].copy()
    kept = amp.loc[amp.ess >= CUT, "seg_len_bp"].sum() / amp.seg_len_bp.sum()

    fig, ax = plt.subplots(figsize=(55 * MM, 58 * MM))
    bins = np.logspace(np.log10(max(amp.ess.min(), 0.1)), np.log10(amp.ess.max()), 45)
    ax.hist(amp.ess, bins=bins, weights=amp.seg_len_bp / 1e9, color=BAR, edgecolor="none")
    ax.set_xscale("log")
    ax.axvline(CUT, color=MARK, ls="--", lw=1.0, zorder=5)

    ax.set_xlabel("SBS1/SBS5-weighted mutations\nper amplified segment")
    ax.set_ylabel("amplified sequence (Gb)")
    ax.set_title("Mutation-count threshold\nfor timing", fontweight="bold", pad=3, fontsize=S.title)
    # plain decade labels: the log formatter's 10^n superscript renders below the 5 pt floor
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ymax = ax.get_ylim()[1]
    ax.annotate(f"{kept:.1%} of amplified\nlength at ≥ {CUT} mutations",
                xy=(CUT, ymax * 0.60), xytext=(CUT * 3.0, ymax * 0.92), fontsize=S.tiny, color=MARK,
                ha="left", va="top", linespacing=1.4, zorder=8,
                arrowprops=dict(arrowstyle="->", color=MARK, lw=0.6, shrinkA=1, shrinkB=2))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.2000, right=0.9455, top=0.8707, bottom=0.1638)
    if save:
        _save_panel(fig, PANELS, "a_mutcount_threshold")
    return fig


if __name__ == "__main__":
    make(); print(f"benchmarking_segment/a: written -> {PANELS}")
