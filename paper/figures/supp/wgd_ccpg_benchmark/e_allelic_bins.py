"""wgd_ccpg_benchmark figure (supp), panel e — the same measurement as panel d, binned.

COMPANION to d_allelic.py, not an alternative to it: the two are shown side by side. d plots biallelic
fraction against event genome fraction, which shows that the WGD signal scales with how much genome the
event covers. This one drops the genome-fraction axis and shows the two DISTRIBUTIONS side by side,
which makes the separation itself the point:

    bin        WGD    ccPG
    0          16%     83%
    0-5%       22%     15%
    5-10%      20%      3%
    10-20%     23%      0%
    20-30%      9%      0%
    >30%        9%      0%

Not one of 116 ccPG events reaches 10% of the timed genome biallelic (the largest is 7.3%); 41% of WGD
events do. That is a cleaner sentence than either median, and it does not depend on where the medians
happen to fall.

DENOMINATOR — the same as panel d, which is what lets the two sit together: biallelic length over the
sample's whole timed genome (`bi_frac_timed`), not over the length the event itself gained. That puts a
small ccPG and a genome-wide WGD on one absolute scale, so an event that is biallelic across a handful
of segments cannot score as highly as one that doubled the genome. (An earlier revision of d used the
gained-length denominator and the two panels were NOT comparable; they are now.)

Exactly zero is drawn as its own bin rather than folded into 0-20%, because the two classes reach it
for different reasons. For ccPG, zero is the expected value: its segments mostly have no minor allele
to gain at all (median frac_capable 0.000), so the event is monoallelic trivially. For WGD, the 16%
at zero is an anomaly sitting apart from an otherwise broad, near-flat 0-100% spread, not the low end
of a ramp — this is the group behind the WGD_with_loss calls the paper discusses.

The 60-80% bin is included even though it is empty for ccPG: it holds 13% of WGD events, and it is
one of the two bins where the ccPG count is exactly zero, which is the comparison being made.

Bars are percentages WITHIN each class, since n differs by ~5x (558 vs 116) and raw counts would show
only that WGD is commoner.

Method, filters and the containment argument are identical to e_allelic.py — see its docstring.
Cache-driven from paper/figures/data/wgd_ccpg_benchmark/event_alleles.tsv (paper/figures/data/wgd_ccpg_benchmark/build/build_event_alleles.py).
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
_apply_style()

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
SRC = cache_dir_of(__file__) / "event_alleles.tsv"
WGD_COL, CCPG_COL = "#0b6979", "#cd9f2c"
MIN_SEG = 3
EDGES = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 1.0001)]
LABELS = ["0", "0–5", "5–10", "10–20", "20–30", ">30"]
EPS = 1e-9          # "exactly zero" — bi is a ratio of summed lengths, so test with a tolerance
# Geometry matched to d_allelic.py, which sits BESIDE this panel: the pair tiles to 88 mm, inside a
# SINGLE column, at the 42 mm row height b_recall and c_falsepos already use. Keep the two figsizes
# equal — a mismatch would force one to be rescaled on assembly, which is what makes type disagree
# across a figure. Margins are in mm, not figure fractions, so resizing the panel does not silently
# rescale them too.
W, H = 44 * MM, 42 * MM
MAR_L, MAR_R, MAR_T, MAR_B = 8.4 * MM, 1.0 * MM, 3.4 * MM, 10.6 * MM


def _binned(x):
    """-> share of events in [exact 0, then the open-lower/closed-upper bins of EDGES]."""
    out = [float((x <= EPS).mean())]
    out += [float(((x > max(lo, EPS)) & (x <= hi)).mean()) for lo, hi in EDGES]
    return np.array(out)


def make(save=True):
    t = pd.read_csv(SRC, sep="\t")
    t = t[t.n_seg >= MIN_SEG].copy()
    series = {c: t.loc[t.classification == c, "bi_frac_timed"].dropna().to_numpy(float)
              for c in ("WGD", "ccPG")}
    frac = {c: _binned(v) for c, v in series.items()}

    fig, ax = plt.subplots(figsize=(W, H))
    x = np.arange(len(LABELS)); w = 0.38
    for (c, col), off in zip(((("WGD"), WGD_COL), ("ccPG", CCPG_COL)), (-w / 2, w / 2)):
        v = 100 * frac[c]
        ax.bar(x + off, v, w, color=col, zorder=3,
               label=f"{c}  (n={len(series[c])})")

    # rotated: six bin labels across a 34 mm plot box overlap when horizontal
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=35, ha="right")
    # wording matched to panel d's y axis — same quantity, and the long form ran to the canvas edge
    ax.set_xlabel("biallelic genome fraction (%)")
    ax.set_ylabel("% of events")
    ax.set_ylim(0, 95)
    ax.legend(loc="upper right", handlelength=0.8, handletextpad=0.4, labelspacing=0.3,
              borderpad=0.2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", lw=0.3, color="0.92", zorder=0); ax.set_axisbelow(True)
    # the "segments with a gain within +/-0.10 of the event; exact 0 shown separately" note that used
    # to sit here is caption material now: at 44 mm it needed 4.6 pt, below the allowed 5 pt floor
    fig.subplots_adjust(left=MAR_L / W, right=1 - MAR_R / W, top=1 - MAR_T / H, bottom=MAR_B / H)
    if save:
        _save_panel(fig, PANELS, "e_allelic_bins")
    for c in ("WGD", "ccPG"):
        print(f"  {c:>5} n={len(series[c]):>4}  " +
              "  ".join(f"{l}={100*v:.0f}%" for l, v in zip(LABELS, frac[c])))
    return fig


if __name__ == "__main__":
    make(); print("e_allelic_bins: PNG+SVG+PDF written (cache-driven from event_alleles.tsv)")
