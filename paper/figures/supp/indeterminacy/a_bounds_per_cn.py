"""indeterminacy figure (supp), panel a — how tightly are UNDERDETERMINED gains bounded, per CN state?
For every gain of an underdetermined route the solution set gives an interval [time_lo, time_hi] rather than
a point; its WIDTH is the identifiability of that gain. Shows the width distribution (median, IQR, 10–90%)
per CN state. Table-driven from bench_seg_gains.
Note a route being underdetermined does NOT make every one of its gains unidentifiable: a large share of
gains inside underdetermined routes are pinned to a point (width ~ 0) by the constraints.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE, DOUBLE, HALF
_apply_style()          # full double-column figure: standard 6-7 pt scale, not compact
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
MIN_N = 200            # CN states with fewer underdetermined gains than this are not shown
PINNED = 0.10          # "effectively pinned" width threshold
BAR, PIN = "#4c72b0", "#55a868"



def _stats(by="cn"):
    """by='cn'   -> one row per copy-number state, ordered by median width (identifiability ranking)
       by='gain' -> one row per GAIN INDEX (gains are time-sorted, so 1 = earliest), natural order"""
    df = load_bench("seg_gains")
    u = df[df.underdetermined].copy()
    u["width"] = u.time_hi - u.time_lo
    keys = ["major", "minor"] if by == "cn" else ["gain_index"]
    g = (u.groupby(keys)
           .agg(n=("width", "size"), p10=("width", lambda x: x.quantile(0.10)),
                q1=("width", lambda x: x.quantile(0.25)), w_med=("width", "median"),
                q3=("width", lambda x: x.quantile(0.75)), p90=("width", lambda x: x.quantile(0.90)),
                pinned=("width", lambda x: (x < PINNED).mean()))
           .reset_index())
    g = g[g.n >= MIN_N].copy()
    if by == "cn":
        # least identifiable at the top, best-determined at the bottom
        g = g.sort_values("w_med", ascending=True)
        g["label"] = [f"({m},{n})" for m, n in zip(g["major"], g["minor"])]
    else:
        g = g.sort_values("gain_index", ascending=False)     # gain 1 at the top
        g["label"] = [f"gain {i}" for i in g["gain_index"]]
    return u, g.reset_index(drop=True)


def make(by="cn", save=True):
    u, g = _stats(by)
    y = np.arange(len(g))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 88 * MM), sharey=True,
                                  gridspec_kw={"width_ratios": [2.1, 1], "wspace": 0.07})

    ax.hlines(y, g["p10"], g["p90"], color=BAR, lw=0.8, alpha=0.55)          # 10-90%
    ax.hlines(y, g["q1"], g["q3"], color=BAR, lw=3.4, alpha=0.85)            # IQR
    ax.plot(g["w_med"], y, "o", ms=3.4, color="white", mec=BAR, mew=1.0, zorder=5)
    ax.axvline(PINNED, ls=":", lw=0.8, color="0.45")
    ax.text(PINNED + 0.006, len(g) - 0.4, f"{PINNED:g}", fontsize=S.annot, color="0.45", va="top")

    ax.set_xlim(-0.01, 0.62); ax.set_xlabel("underdetermined interval width (molecular time)", fontsize=S.label)
    ax.set_yticks(y); ax.set_yticklabels(g["label"], fontsize=S.tick)
    ax.set_ylabel("copy-number state (major, minor)" if by == "cn"
                  else "gain index (1 = earliest gain in the tree)", fontsize=S.label)
    for i, (nn, p90) in enumerate(zip(g["n"], g["p90"])):   # count sits just past its own whisker
        ax.text(p90 + 0.012, i, f"n={nn:,}", fontsize=S.annot, color="0.45", ha="left", va="center")

    # ---- right: how often the constraints pin a gain to (almost) a point ----
    ax2.barh(y, g["pinned"], height=0.6, color=PIN, alpha=0.9)
    ax2.set_xlim(0, 1.12); ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xlabel(f"fraction of gains with width < {PINNED:g}", fontsize=S.label)

    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_ylim(-0.7, len(g) - 0.3)
        a.grid(axis="x", lw=0.3, color="0.85", zorder=0); a.set_axisbelow(True)
    h = [plt.Line2D([], [], color=BAR, lw=3.4, alpha=0.85, label="IQR"),
         plt.Line2D([], [], color=BAR, lw=0.8, alpha=0.55, label="10–90%"),
         plt.Line2D([], [], color="white", mec=BAR, mew=1.0, marker="o", ms=3.4, ls="", label="median")]
    ax.legend(handles=h, frameon=False, loc="lower right", handlelength=1.6, ncol=1, fontsize=S.legend)
    # explicit margins rather than tight_layout: the panel is saved at its exact canvas size, so the
    # axis labels have to be budgeted for here
    fig.subplots_adjust(left=0.095, right=0.995, top=0.975, bottom=0.135, wspace=0.07)
    if save:
        stem = "a_bounds_per_cn" if by == "cn" else "a_bounds_per_gain"
        _save_panel(fig, PANELS, stem)
    return fig


if __name__ == "__main__":
    make("cn"); print(f"identifiability/a: written -> {PANELS}")
