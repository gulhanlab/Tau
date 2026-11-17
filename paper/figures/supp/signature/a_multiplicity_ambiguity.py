"""signature figure (supp), panel a — why a mutation's multiplicity does not fix its interval.

The premise of signature timing is that multiplicity places a mutation between two gains. It constrains
that placement but does not determine it, and this panel is the counterexample the text refers to.

A (2,1) segment has one gain, dividing molecular time into two intervals. A mutation ends at
multiplicity 2 only if it arose on the major allele BEFORE the gain, so it was carried onto the new
copy as well. Multiplicity 1 has two distinct origins:

    the minor allele, at any time            (never duplicated)
    the major allele, after the gain         (present on only one of the two copies)

So multiplicity-1 mutations are drawn from BOTH intervals. They are enriched for post-gain mutations —
the minor allele's entire pre-gain history also lands there — but they are not "late" mutations.
Assigning each mutation to an interval on the basis of its multiplicity alone would therefore mis-time
exactly the mutations the analysis depends on.

Tau does not make that assignment. The multiplicity COUNTS enter the route's system of linear equations
and the interval durations are solved for; the ambiguity is carried by the model rather than resolved
by fiat.

The tree is drawn by tau.trees.plot_route on route 2_1.1, i.e. by the same code as every other tree in
the paper, so the topology is Tau's own: a gain leaves the original lineage running uninterrupted and
sends a new copy off to the side. It is not a symmetric bifurcation — amplification produces one
retained original plus one new copy, and which branch is which is exactly what determines where the
multiplicity-2 mutations can sit.

Mutation positions are illustrative; no data is read.
"""
import copy
import sys
from pathlib import Path

import numpy as np
from paper.config import panels_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)
from tau.trees import build_allele_trees, plot_route
from tau.tree_polytope import _events_for_route
from _arrows import dim_arrow

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
MAJ, MIN_ = "#2f4858", "#7a8b99"
# Red / blue for the mutations. They are the SUBJECT of this panel and kept losing out to the tree
# they sit on, so both are fully saturated. The blue is deliberately BRIGHTER and bluer than the
# slate lineages (#2f4858) and the teal doubling line (#0b6979) it has to sit against — a mid-blue
# would have gone muddy on the trunk. Amber/crimson and magenta/green were both tried first; amber
# additionally collided with the PGD gold used elsewhere, which carries a meaning it lacks here.
M1, M2 = "#e01b24", "#1e6fff"          # ends at multiplicity 1 / 2
# ...and a darker red wherever M1 has to work as TEXT rather than as a filled shape
M1_TXT = "#b3141b"
T_GAIN = 0.42
RNG = np.random.default_rng(7)


def _verticals(ax):
    """The drawn tree's vertical lines as (x, y_lo, y_hi), which is where mutations can sit."""
    out = []
    for ln in ax.get_lines():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(x) == 2 and np.isclose(x[0], x[1]) and not np.isclose(y[0], y[1]):
            out.append((float(x[0]), float(min(y)), float(max(y))))
    return sorted(out)


def _muts(ax, x, y0, y1, n, colour):
    # Stratified, not uniform: one mutation per equal sub-interval, jittered within it. A plain
    # uniform draw clumps, and an evenly spaced grid lines the triangles up across lineages so they
    # read as simultaneous. This keeps them spread but out of register between branches.
    pad = 0.07 * (y1 - y0)
    lo, hi = y0 + pad, y1 - pad
    edges = np.linspace(lo, hi, n + 1)
    ys = edges[:-1] + (edges[1] - edges[0]) * RNG.uniform(0.15, 0.85, n)
    ax.scatter(np.full(n, x), ys, marker="^", s=26, color=colour, edgecolors="black",
               linewidths=0.35, zorder=6, clip_on=False)


def make(save=True):
    ev = copy.deepcopy(_events_for_route("2_1.1"))
    for e in ev:
        e["time"] = T_GAIN
    maj, mn = build_allele_trees(ev, 2, 1)

    # 86 mm, matching panel b so the two sit level on the figure's top row. The extra height
    # goes into the lineages, which spreads the mutation triangles instead of stretching type.
    fig, ax = plt.subplots(figsize=(SINGLE, 86 * MM))
    fig.subplots_adjust(left=0.115, right=0.995, top=0.97, bottom=0.06)
    # minor allele dashed, the convention throughout the paper. An explicit dash pattern rather than
    # plot_route's default "--", which scales with linewidth and at 0.55 draws only a few long dashes
    # down a full-height trunk.
    plot_route(ax, maj, mn, major_color=MAJ, minor_color=MIN_,
               linewidth_scale=0.55, spacing=0.75, minor_dashes=(2.4, 1.4))

    v = _verticals(ax)
    # the minor allele is the right-most lineage; of the two major lineages the ORIGINAL spans the
    # full height and the OFFSHOOT starts at the gain
    minor = v[-1]
    major = [s for s in v[:-1]]
    original = max(major, key=lambda s: s[2] - s[1])
    offshoot = min(major, key=lambda s: s[2] - s[1])

    _muts(ax, original[0], 0.0, T_GAIN, 4, M2)          # pre-gain on the original -> copied -> mult 2
    _muts(ax, original[0], T_GAIN, 1.0, 3, M1)          # post-gain on the original -> mult 1
    _muts(ax, offshoot[0], T_GAIN, 1.0, 3, M1)          # on the new copy -> mult 1
    _muts(ax, minor[0], 0.0, T_GAIN, 3, M1)             # minor, either interval -> mult 1
    _muts(ax, minor[0], T_GAIN, 1.0, 3, M1)

    # no gain rule or label: the tree already shows the gain as the point the new lineage splits off,
    # and the two interval brackets already meet there
    xs = [s[0] for s in v]
    lo, hi = min(xs) - 0.3, max(xs) + 0.3

    # limits before the arrows: dim_arrow sizes its heads in points and reads transData to do it
    ax.set_xlim(lo - 0.15, hi + 3.5); ax.set_ylim(-0.03, 1.03)

    # the same arrow as panel c (shared _arrows.py), and spanning the interval EXACTLY — the previous
    # annotate version was inset 0.02 at each end, so the brackets stopped short of the gain and of
    # the axis ends, which is the one thing they are there to delimit
    for y0, y1, lab in ((0.0, T_GAIN, "interval 1"), (T_GAIN, 1.0, "interval 2")):
        dim_arrow(ax, hi + 0.25, y0, y1, "#1a1a1a", lw=0.6)
        ax.text(hi + 0.38, (y0 + y1) / 2, lab, fontsize=S.tiny, va="center", color="0.35")
    ax.text(hi + 1.30, 0.5, "multiplicity-1 mutations\narise in BOTH intervals",
            fontsize=S.annot, va="center", ha="left", color=M1_TXT, fontweight="bold")

    ax.set_xticks([])                               # the tree topology is self-explanatory
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([0, T_GAIN, 1]); ax.set_yticklabels(["0", "", "1"])
    ax.set_ylabel("molecular time", fontsize=S.label)
    ax.tick_params(length=1.4, pad=1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Two legends rather than one four-entry block: the triangles and the lineages are different kinds
    # of thing, and a single "mutations" title over all four would mislabel the alleles.
    h = [mlines.Line2D([], [], color=M2, marker="^", ls="none", ms=5.2, mec="black", mew=0.35,
                       label="multiplicity 2"),
         mlines.Line2D([], [], color=M1, marker="^", ls="none", ms=5.2, mec="black", mew=0.35,
                       label="multiplicity 1")]
    lg = ax.legend(handles=h[::-1], loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=S.legend,
                   handletextpad=0.3, labelspacing=0.25, borderpad=0.25, title="mutations")
    lg.get_title().set_fontsize(5.2); lg.get_title().set_fontweight("bold")
    ax.add_artist(lg)                                   # keeps the first legend when a second is set

    ha = [mlines.Line2D([], [], color=MAJ, lw=1.1, label="major"),
          mlines.Line2D([], [], color=MIN_, lw=1.1, ls=(0, (2.4, 1.4)), label="minor")]
    lg2 = ax.legend(handles=ha, loc="upper right", bbox_to_anchor=(1.0, 0.885), fontsize=S.legend,
                    handlelength=1.6, handletextpad=0.3, labelspacing=0.25, borderpad=0.25,
                    title="alleles")
    lg2.get_title().set_fontsize(5.2); lg2.get_title().set_fontweight("bold")
    lg.get_title().set_fontweight("bold")
    if save:
        _save_panel(fig, PANELS, "a_multiplicity_ambiguity")
    return fig


if __name__ == "__main__":
    make(); print("a_multiplicity_ambiguity: PNG+SVG written")
