"""COMPANION (superseded) — the max-gain version of panel f. Kept so the metric change is visible.

Do not use as the paper panel: max-gain error per segment correlates with gains-per-segment at
rho = +0.95, and gains-per-segment = major + minor - 1, so the state ranking it shows is close to a
restatement of total copy number. It ranks (2,2) best and (4,4) worst; under a per-gain point solution
that ordering inverts. See f_error_by_cn.py.

Supp (benchmarking_segment) f — timing accuracy by copy-number state, at >200 mutations.

Same construction and axes as panels d and e: the ECDF of each segment's MAX-gain distance to the
valid solution set. Restricted to segments with >200 SBS1/SBS5-weighted mutations, so coverage is not
the limiting factor and what remains is the copy-number state itself.

States are coloured on a ramp ordered by median error, worst first in the legend. The spread is 13-fold
— (2,2) at 0.017 to (4,4) at 0.220 — and it tracks how many amplification routes the state admits
rather than its total copy number: single-route states ((2,1), (2,2), (3,1)) sit at the left edge
regardless of copy number, while the many-route balanced states trail right.

Table-driven from bench_route_selection (`acc_chosen` = the route Tau picks). NOT bench_route_accuracy,
whose `acc_bestguess` merges tied routes and inflates exactly these states (see d_tree_accuracy).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
MC_CUT, MIN_N = 200, 50
XMAX, XMARK = 0.5, 0.10          # shared with panels d and e


def _ecdf(a):
    a = np.sort(np.asarray(a, float)); a = a[~np.isnan(a)]
    return a, np.arange(1, len(a) + 1) / max(len(a), 1)


def make(save=True):
    s = load_bench("route_selection")
    h = s[s.mc > MC_CUT].copy()
    h["cn"] = h.major.astype(str) + "_" + h.minor.astype(str)
    med = h.groupby("cn").acc_chosen.agg(["size", "median"])
    med = med[med["size"] >= MIN_N].sort_values("median", ascending=False)

    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=-0.15, vmax=1.05)          # keep both ends off the extremes of the ramp
    fig, ax = plt.subplots(figsize=(89.0 * MM, 58.0 * MM))
    for i, (cn, row) in enumerate(med.iterrows()):
        x, y = _ecdf(h[h.cn == cn].acc_chosen.values)
        maj, mino = cn.split("_")
        ax.plot(x, y, color=cmap(norm(i / max(len(med) - 1, 1))), lw=1.2, zorder=4,
                label=f"({maj},{mino})  {row['median']:.3f}")

    ax.axvline(XMARK, ls=":", c="#999", lw=0.8)
    ax.text(XMARK + 0.008, 0.03, f"{XMARK:.2f}", fontsize=S.tiny, color="#999")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("max-gain timing error")
    ax.set_ylabel("fraction of segments below error")
    ax.set_title("Max timing error per segment\nby copy-number state (>200 mut)", fontweight="bold", pad=3, fontsize=S.title)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
              handlelength=1.0, labelspacing=0.16, handletextpad=0.3, fontsize=S.tiny,
              title="state   median", title_fontsize=S.legend)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(lw=0.3, color="0.93"); ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.1236, right=0.5843, top=0.8707, bottom=0.1638)
    if save:
        _save_panel(fig, PANELS, "f_error_by_cn_maxgain")
    return fig


if __name__ == "__main__":
    make(); print(f"benchmarking_segment/f: written -> {PANELS}")
