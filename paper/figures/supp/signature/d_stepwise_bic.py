"""signature figure (supp), panel d — the stepwise-change model and how BIC picks the number of transitions.

Top row, three worked fits. Grey bars are the OBSERVED intervals: each spans the molecular time it
covers, positioned at its log2 activity relative to the sample's own baseline, with thickness set by the
signature mutations supporting it. They overlap, and they disagree — which is the problem the model
exists to solve. The red step function is the fitted state sequence, shaded with its standard error.

Because the intervals are windows rather than points, the model deconvolves: each observation is treated
as the overlap-weighted mean of the states it spans, and the states are solved for. A changepoint may
only be placed at an observed interval boundary.

Bottom row, the criterion. BIC = n log(RSS/n) + S log(n), evaluated along the greedy search path — the
sequence of models the algorithm actually compared, adding at each step the changepoint that most
reduced the weighted residual. The search stops when the next state stops paying for itself, marked in
red. This is forward selection, so it is not an exhaustive scan over S.

The three examples are chosen to show the selection doing work in both directions: a sample where no
transition is supported despite 26 usable intervals, one with a single transition, one with two.

Two constraints matter as much as the criterion and are invisible here: no state may be narrower than
0.05 molecular time, and a state's level must lie within the range of log-ratios that at least 10
mutations actually support. Without the second, overlapping windows can imply a level no interval
attests, and the model invents transitions.

Cache-driven from paper/figures/data/signature/bic_examples_{windows,fits,trace}.tsv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of, cache_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, DOUBLE, S
_apply_style(compact=True)

CACHE = cache_dir_of(__file__)
PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
OBS, FIT, SEL = "#9aa5b1", "#b2182b", "#b2182b"
ORDER = ["Fig 2b Eso — declining", "Fig 2a Stomach — rising", "no transition supported"]


def make(save=True):
    win = pd.read_csv(CACHE / "bic_examples_windows.tsv", sep="\t")
    fit = pd.read_csv(CACHE / "bic_examples_fits.tsv", sep="\t")
    tr = pd.read_csv(CACHE / "bic_examples_trace.tsv", sep="\t")

    fig, axes = plt.subplots(2, len(ORDER), figsize=(DOUBLE, 68 * MM),
                             gridspec_kw={"height_ratios": [1.9, 1], "hspace": 0.55, "wspace": 0.26})
    for c, ex in enumerate(ORDER):
        w, f, t = win[win.example == ex], fit[fit.example == ex], tr[tr.example == ex]
        ax, bx = axes[0, c], axes[1, c]

        # observed windows: line width encodes the mutations behind each interval
        lw = 0.4 + 2.2 * (w.w / w.w.max())
        for r, L in zip(w.itertuples(), lw):
            ax.plot([r.lo, r.hi], [r.z, r.z], color=OBS, lw=L, alpha=0.75,
                    solid_capstyle="butt", zorder=2)
        # fitted step function with SE band
        for r in f.itertuples():
            ax.fill_between([r.lo, r.hi], r.log2_fold - r.log2_fold_se, r.log2_fold + r.log2_fold_se,
                            color=FIT, alpha=0.18, lw=0, zorder=3)
            ax.plot([r.lo, r.hi], [r.log2_fold] * 2, color=FIT, lw=1.5, solid_capstyle="butt", zorder=4)
        for a, b in zip(f.itertuples(), list(f.itertuples())[1:]):
            ax.plot([a.hi, a.hi], [a.log2_fold, b.log2_fold], color=FIT, lw=1.0, zorder=4)
        ax.axhline(0, color="0.6", lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("molecular time", fontsize=S.label)
        if c == 0:
            ax.set_ylabel("log2 activity\nrelative to sample baseline", fontsize=S.label)
        r0 = w.iloc[0]
        ax.set_title(f"{r0.tumor_type}  ·  {r0.signature}\n{len(w)} intervals: "
                     f"{int(f.n_states.iloc[0])} state" + ("s" if f.n_states.iloc[0] > 1 else ""),
                     fontsize=S.annot, fontweight="bold", pad=3, linespacing=1.4)

        # BIC along the greedy path
        bx.plot(t.n_states, t.bic, "-o", color="0.45", lw=0.9, ms=3.0, zorder=3)
        s = t[t.selected]
        if len(s):
            bx.plot(s.n_states, s.bic, "o", color=SEL, ms=4.6, zorder=5)
            # above the marker: below it collides with the tick labels, and at the left-hand
            # example the selected point sits on the axis edge
            bx.annotate("selected", (s.n_states.iloc[0], s.bic.iloc[0]),
                        textcoords="offset points", xytext=(6, 7), ha="left",
                        fontsize=S.tiny, color=SEL, fontweight="bold")
        bx.set_xticks(sorted(t.n_states.unique()))
        bx.set_xlabel("number of states", fontsize=S.label)
        if c == 0:
            bx.set_ylabel("BIC", fontsize=S.label)
        bx.margins(y=0.42, x=0.14)
        for a in (ax, bx):
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(labelsize=S.tick, length=1.4, pad=1.2)

    fig.text(0.005, 0.012, "Grey: observed intervals, width ∝ signature mutations. "
             "Red: fitted states ± s.e.  BIC is evaluated along the greedy search path.",
             fontsize=S.tiny, color="0.45")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.145)
    if save:
        _save_panel(fig, PANELS, "d_stepwise_bic")
    return fig


if __name__ == "__main__":
    make(); print("d_stepwise_bic: PNG+SVG written")
