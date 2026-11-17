"""wgd_dynamics figure (main), panel c — median molecular time to WGD by tumour type: full cohort vs WGD+
samples (Weibull time-to-event fits), plus the raw median time to the first and to the second WGD.

The model is the model code in `wgd_timing_model/` (e2 -> e7 -> e8 -> e9 -> e10; see its README); this
script only DRAWS, from `e10.compute()`'s per-type fit table, in the layout of the assembled figure:
  - rows = the tumour types of panel b, in panel b's order, restricted to those with >= 3 WGD events (24 of 29;
    the WGD-poor types at the bottom of h are dropped, as in the paper); NO row labels of its own, no "ALL POOLED" row
    (that number is quoted in the text instead);
  - four series per row: grey square = fitted time to first WGD, full cohort (0->1);
    teal triangle = fitted time to a WGD in samples with WGD, all transitions pooled (0->1, 1->2, ...);
    purple x = raw median time to first WGD; pink + = raw median 1->2 gap;
  - x from 0 to 2 (values beyond are pinned at the edge), title and legend as in the figure.
`make()` returns the figure and writes PNG+SVG next to this script.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "wgd_timing_model"))
import e10_weibull_wgd_timing_model as e10  # noqa: E402
from _wgdpgd_common import save_panel, plt, MM, TXT  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import b_prevalence_timing as h  # noqa: E402

STEM = "c_wgd_weibull"
MIN_WGD = 3          # rows need >= this many WGD events: keeps Prostate (4), drops Oligo (1) and the WGD-free types = the 24 rows of the paper
XMAX = 2.0
C_FULL, C_TRANS, C_FIRST, C_SECOND = "#8d8c8c", "#116a79", "#ac4b9c", "#f59db6"   # as in the assembled figure
FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = S.title, S.title, S.tick, S.label


def make(save=True, force_recompute=False, width_mm=60.0):
    fits = e10.compute(force=force_recompute)["fits"].set_index("tumor_type")
    # panel b's order, but only the types that HAVE a WGD fit (as in the assembled figure: 24 rows;
    # the WGD-free types at the bottom of h -- Prostate, MPN, CLL, Thyroid, PiloAstro -- are not drawn)
    order = [t for t in h.row_order() if t in fits.index and np.isfinite(fits.loc[t, "all_half_time"])
             and int(fits.loc[t, "n_events"]) >= MIN_WGD]
    shown = fits.reindex(order)
    print(f"  [c] {len(order)} rows with a WGD fit (of {len(h.row_order())} in panel b)")

    n = len(order)
    # row pitch as in the assembled figure (2.1 mm per row), title above, legend below
    ROW_MM, TOP_MM, LEGEND_MM, BOTTOM_AX_MM = 53.6 / 24, 14.0, 24.0, 9.0   # 53.6 mm axes for 24 rows (her figure)
    ax_h = ROW_MM * n
    H = TOP_MM + ax_h + BOTTOM_AX_MM + LEGEND_MM
    fig, ax = plt.subplots(figsize=(width_mm * MM, H * MM))
    # axes 30 mm wide at the left of the canvas; the legend below it is wider and runs to the right
    fig.subplots_adjust(left=6.0 / width_mm, right=34.0 / width_mm, top=1 - TOP_MM / H,   # axes 28 mm wide
                        bottom=(LEGEND_MM + BOTTOM_AX_MM) / H)

    def _err(yi, pt, lo, hi, color, marker, ms, mew=0.8):
        if not np.isfinite(pt):
            return
        pt_c = min(pt, XMAX * 0.985)
        if not (np.isfinite(lo) or np.isfinite(hi)):
            ax.plot([pt_c], [yi], marker=marker, color=color, ms=ms, mew=mew, ls="", zorder=3)
            return
        lo_c = min(max(lo, 0.0) if np.isfinite(lo) else 0.0, pt_c)
        hi_c = max(min(hi, XMAX) if np.isfinite(hi) else XMAX, pt_c)
        ax.errorbar([pt_c], [yi], xerr=[[pt_c - lo_c], [hi_c - pt_c]], fmt=marker, color=color, ms=ms,
                    mew=mew, capsize=1.5, lw=0.6, zorder=3)

    for i, (tt, r) in enumerate(shown.iterrows()):
        yi = n - 1 - i                                          # h draws row 0 at the bottom
        _err(yi, r["all_half_time"], r["all_lo"], r["all_hi"], C_FULL, "s", 3.0)
        _err(yi, r["trans_half_time"], r["trans_lo"], r["trans_hi"], C_TRANS, "^", 3.5)
        if np.isfinite(r["raw_median"]):
            ax.plot([min(r["raw_median"], XMAX * 0.985)], [yi], marker="x", color=C_FIRST, ms=3.5, mew=0.9,
                    ls="", zorder=4)
        if np.isfinite(r["raw_median_12"]):
            ax.plot([min(r["raw_median_12"], XMAX * 0.985)], [yi], marker="+", color=C_SECOND, ms=3.5,
                    mew=0.9, ls="", zorder=4)

    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n)); ax.set_yticklabels([]); ax.tick_params(axis="y", length=2, width=0.5)
    ax.set_xlim(0, XMAX); ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel("Median time to WGD\n(molecular-time units)", fontsize=FS_LABEL, color=TXT, linespacing=1.2)
    ax.set_title("Increasing rate\nof WGD in samples\nmanifesting it", fontsize=FS_TITLE, color=TXT,
                 loc="center", pad=4, linespacing=1.2)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([], [], marker="s", ls="", color=C_FULL, ms=3.5, label="Time to first WGD, full cohort\nfitted (0→1)"),
        Line2D([], [], marker="^", ls="", color=C_TRANS, ms=4, label="Time to a WGD in samples with WGD\nfitted (0→1, 1→2, 2→3 ...)"),
        Line2D([], [], marker="x", ls="", color=C_FIRST, ms=3.5, mew=0.9, label="Median time to first WGD (0→1)"),
        Line2D([], [], marker="+", ls="", color=C_SECOND, ms=3.5, mew=0.9, label="Median time to 2nd WGD (1→2)"),
    ]
    fig.legend(handles=handles, fontsize=FS_LEG, frameon=False, loc="upper left",
               bbox_to_anchor=(0.0, (LEGEND_MM - 1.0) / H), ncol=1, handletextpad=0.5, labelspacing=0.5,
               borderaxespad=0.0)

    out = save_panel(fig, STEM, save=save, exact=True)
    return out


if __name__ == "__main__":
    make(); print(f"{STEM}: PNG+SVG written")
