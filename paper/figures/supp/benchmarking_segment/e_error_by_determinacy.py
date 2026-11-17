"""benchmarking_segment figure (supp), panel e — timing error for determined vs underdetermined gains, >200 mutations.

Being underdetermined does not make a gain less accurately timed. At >200 SBS1/SBS5-weighted mutations
the median distance-to-valid-set is 0.009 for underdetermined gains and 0.028 for determined ones, and
81% vs 73% fall within 0.05.

That is not a paradox and it is not a free lunch: an underdetermined gain is scored against its whole
solution interval, so the error is 0 whenever the truth lands inside the band — which happens for 40%
of them. The band is wider, so it is easier to contain the truth. Underdetermined timing is therefore
no less ACCURATE, only less PRECISE, and panel g quantifies the precision cost as the interval width.
Reading this panel without panel g would overstate the case.

Table-driven from bench_seg_gains (per-gain `dist_to_set`, `underdetermined`, `truth_in_band`).
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
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
MC_CUT = 200
XMAX, XMARK = 0.5, 0.10          # shared with panels d and f
C_DET, C_UND = "#4c72b0", "#dd8452"


def _ecdf(a):
    a = np.sort(np.asarray(a, float)); a = a[~np.isnan(a)]
    return a, np.arange(1, len(a) + 1) / max(len(a), 1)


def make(save=True):
    g = load_bench("seg_gains")
    hi = g[g.mut_count > MC_CUT]
    fig, ax = plt.subplots(figsize=(55.0 * MM, 58.0 * MM))

    for flag, col, lab in [(False, C_DET, "determined"), (True, C_UND, "underdetermined")]:
        sub = hi[hi.underdetermined == flag]
        x, y = _ecdf(sub.dist_to_set.values)
        ax.plot(x, y, color=col, lw=1.6, zorder=4,
                label=f"{lab}\nmedian {np.median(sub.dist_to_set):.3f}   n={len(sub):,}")

    ax.axvline(XMARK, ls=":", lw=0.8, color="#999")
    ax.text(XMARK + 0.008, 0.03, f"{XMARK:.2f}", fontsize=S.tiny, color="#999")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("max-gain timing error")
    ax.set_ylabel("fraction of gains below error")
    ax.set_title("Max timing error per segment\nby determinacy (>200 mut)", fontweight="bold", pad=3, fontsize=S.title)
    ax.legend(frameon=False, loc="lower right", handlelength=1.4, labelspacing=0.85)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(lw=0.3, color="0.93"); ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.2000, right=0.9455, top=0.8707, bottom=0.1638)
    if save:
        _save_panel(fig, PANELS, "e_error_by_determinacy")
    return fig


if __name__ == "__main__":
    make(); print(f"benchmarking_segment/e: written -> {PANELS}")
