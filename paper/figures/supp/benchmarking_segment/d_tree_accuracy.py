"""benchmarking_segment figure (supp), panel d — timing accuracy before and after route clustering, by coverage tier.

Each segment is scored by its MAX-gain distance to the valid solution set (worst-timed gain;
underdetermined bands score 0 when the truth falls inside). Within each mutation-count tier the solid
curve is the segment timed on its own and the dashed curve is the same segment after Tau pools it with
others sharing its copy-number state and gain route.

Pooling shifts every tier left: median error falls 0.111 -> 0.052 (<50 mut), 0.068 -> 0.026, 0.048 ->
0.013, 0.028 -> 0.013 (>=200) — a 53-72% reduction per tier and 58.6% overall. The lift is largest in
the middle tiers: sparse segments gain most from pooling, while the best-covered ones already sit near
the floor set by their copy-number state (panel f).

Table-driven from bench_oracle_lift (`acc_base` = per segment, `acc_prod` = after clustering).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
TIERS = [("<50", 0, 50), ("50–100", 50, 100), ("100–200", 100, 200), ("≥200", 200, 1e12)]
XMAX, XMARK = 0.5, 0.10          # shared with panels e and f
DASH = (0, (3, 1.6))


def _ecdf(a):
    a = np.sort(np.asarray(a, float)); a = a[~np.isnan(a)]
    return a, np.arange(1, len(a) + 1) / max(len(a), 1)


def make(save=True):
    df = load_bench("oracle_lift").dropna(subset=["mut_count", "acc_base", "acc_prod"])
    fig, ax = plt.subplots(figsize=(55.0 * MM, 58.0 * MM))
    cmap = plt.cm.viridis(np.linspace(0.08, 0.85, len(TIERS)))

    for (lab, lo, hi), col in zip(TIERS, cmap):
        t = df[(df.mut_count >= lo) & (df.mut_count < hi)]
        if len(t) < 30:
            continue
        x1, y1 = _ecdf(t.acc_base.values)
        x2, y2 = _ecdf(t.acc_prod.values)
        ax.plot(x1, y1, color=col, lw=1.4, ls="-", zorder=4)
        ax.plot(x2, y2, color=col, lw=1.4, ls=DASH, zorder=5)

    ax.axvline(XMARK, ls=":", c="#999", lw=0.8)
    ax.text(XMARK + 0.008, 0.03, f"{XMARK:.2f}", fontsize=S.tiny, color="#999")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("max-gain timing error")
    ax.set_ylabel("fraction of segments below error")
    ax.set_title("Max timing error per segment\nby mutation count", fontweight="bold", pad=3, fontsize=S.title)

    style_h = [Line2D([0], [0], color="#555", lw=1.4, ls="-", label="per segment"),
               Line2D([0], [0], color="#555", lw=1.4, ls=DASH, label="after clustering")]
    tier_h = [Line2D([0], [0], color=c, lw=1.6, label=f"{lab} mut")
              for (lab, _, _), c in zip(TIERS, cmap)]
    leg1 = ax.legend(handles=style_h, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.44),
                     handlelength=1.7, labelspacing=0.2, handletextpad=0.35, fontsize=S.tiny)
    ax.add_artist(leg1)
    ax.legend(handles=tier_h, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0),
              title="coverage (mut)", title_fontsize=S.legend, fontsize=S.title,
              handlelength=1.1, labelspacing=0.18, handletextpad=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(lw=0.3, color="0.93"); ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.2000, right=0.9455, top=0.8707, bottom=0.1638)
    if save:
        _save_panel(fig, PANELS, "d_tree_accuracy")
    return fig


if __name__ == "__main__":
    make(); print(f"benchmarking_segment/d: written -> {PANELS}")
