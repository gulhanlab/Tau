"""burst_characteristics figure (supp) — EXCESS bursts, real vs a matched simulation null.

Rebuilt on the excess definition the rest of the figure uses: max_burst_size > D, D = floor(M/2) +
floor(m/2), computed over TIED-BEST routes on both sides. The previous version scored `complete_burst`
(p90_span < theta) and swept theta — a different statistic from the one the figure reports, and a null
on a different statistic is not a null. The sweep is gone because max_burst_size is defined at a fixed
0.05 window; there is no theta to vary without re-solving every replicate at every theta.

THE NULL. Each real segment is re-simulated through its own route and mutation count with uniform
Dirichlet interval fractions, then run through Tau (simulate_null_perrep.py). It therefore asks what
Tau OUTPUTS when gains are NOT clustered in time, on exactly these segments — not what an idealised
geometric sampler would produce. One replicate = one null segment.

Both sides also now take max_burst_size over tied-best routes only. The null used to pool every route
at dist_cut = 0.5, looser than Tau's own 0.20 and far looser than the real side; pooling extra routes
can only RAISE a max, so the old null was biased burst-ward.

READ IT BY STATE, NOT POOLED
----------------------------
D is a far weaker bar in some states than others, so the null excess rate runs from 2.1% at (6,2) to
78.1% at (5,5). Pooling those into one number (real 22.1% vs null 10.4%, "2.1x") averages a
near-impossible bar with an almost-free one, so the pooled figure is stated in the footnote and the
per-state view is the headline:

    strong:  (6,2) 7.8x   (3,2) 4.5x   (6,1) 3.8x   (5,2) 3.8x   (4,1) 2.8x
    absent:  (5,4) 1.1x   (5,5) 1.1x   (6,5) 1.0x

Panel b gives the mechanism: the null rate climbs steeply with gain count, because more gains mean both
a wider polytope and more windows in which some solution can place D+1 of them together. Excess
therefore carries information in the low-minor, few-gain states — which is exactly where the
hemizygous-burst story lives — and carries almost none in balanced high-CN states.

Cache-driven from paper/figures/data/burst_characteristics/burst_null_excess_{by_state,by_ngain}.tsv (paper/figures/data/burst_characteristics/build/build_burst_null.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of, cache_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)

CACHE = cache_dir_of(__file__)
PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
C_REAL, C_NULL = "#b23a2f", "#e8a33d"
MIN_EVENTS_LABEL = 10   # under the conservative definition the null is ~0, so a fold is
                        # unbounded and uninformative; label counts instead below this n


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def make(save=True, ax=None, fs=1.0):
    """ax: optional axes to draw into, for composing this panel into an assembled figure.

    When supplied, no figure is created and nothing is saved — the caller owns the canvas. Drawing
    code stays here rather than being copied into the page script, so a page and the standalone panel
    cannot drift apart.
    """
    # ONE comparison, everything pooled: how often the call fires on real data vs how often it fires
    # when the answer is known to be no. Scored conservatively (a burst counts only if it holds under
    # every fit) — the permissive reading has a false-positive rate near its own signal and is not
    # plotted. The determined/underdetermined split lives in burst_null_pooled.tsv for later.
    st = pd.read_csv(CACHE / "burst_null_excess_by_state.tsv", sep="\t")
    real_n, real_tot = int(st.real_n.sum()), int(st.real_total.sum())
    null_n, null_tot = int(st.null_n.sum()), int(st.null_total.sum())
    vals = [real_n / real_tot, null_n / null_tot]
    cis = [_wilson(real_n, real_tot), _wilson(null_n, null_tot)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(SINGLE, 52 * MM))
    else:
        fig, save = ax.figure, False

    x = np.arange(2)
    # "(gains spread at random)" on one line is wider than the category slot once the panel is sized
    # for a three-across row, and runs into "(PCAWG)". Wrapped rather than abbreviated — the phrase is
    # what defines the null, so it should not be shortened to fit.
    labs = [f"real\n(PCAWG)\nn={real_tot:,}",
            f"simulated null\n(gains spread\nat random)\nn={null_tot:,}"]
    for k, (v, (lo, hi), c) in enumerate(zip(vals, cis, (C_REAL, C_NULL))):
        ax.bar(k, 100 * v, 0.55, color=c, zorder=3)
        ax.errorbar(k, 100 * v, yerr=[[max(0, 100 * (v - lo))], [max(0, 100 * (hi - v))]],
                    fmt="none", ecolor="0.35", elinewidth=0.5, capsize=1.5, zorder=4)
        ax.text(k, 100 * v + 0.5, f"{100*v:.1f}%", ha="center", va="bottom",
                fontsize=S.label * fs, fontweight="bold", color=c)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=S.tick * fs, linespacing=1.35)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, 13)
    ax.set_ylabel("% of segments called\nan excess burst", fontsize=S.label * fs)
    ax.tick_params(axis="y", labelsize=S.tick * fs)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if save:
        fig.subplots_adjust(left=0.205, right=0.98, top=0.97, bottom=0.30)
    if save:
        _save_panel(fig, PANELS, "alt_burst_null")
    return fig


if __name__ == "__main__":
    make(); print("a_burst_null: rebuilt on the EXCESS definition")
