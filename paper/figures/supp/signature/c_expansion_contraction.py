"""signature figure (supp), panel c — what interval expansion and contraction mean.

Timing the same pooled cluster twice — once weighting mutations by the SBS1/SBS5 clock, once by a
signature — gives two versions of the same amplification history. Same route, same gains, same order;
only the ruler changes. Each timeline is normalised to sum to 1, so an interval occupying a LARGER
share under the signature is one in which that signature accumulated mutations faster than the clock.
That is expansion; contraction is the converse.

Each example is shown twice over. LEFT, the comparison: both trees drawn by tau.trees.plot_route into
one axes, the second placed with plot_route's own `x_offset` so the two timings share a coordinate
system. Each dashed line runs from the actual junction where a copy is gained on the clock tree to that
same junction on the signature tree — the displacement of one specific amplification, not an abstract
tick. The shading between consecutive dashed lines is the interval they bound: red where it expands
under the signature, blue where it contracts, labelled with the fold change.

RIGHT, the synthesis the comparison exists to justify: ONE tree, at its ordinary SBS1/SBS5 clock
timing — the timing used everywhere else in the paper — with each stretch of branch coloured by the
signature's log2 activity ratio in the interval it lies in. This is the form the cohort analysis uses,
and the point of the pair is that it needs no second timeline: the two-tree comparison on the left and
the single coloured tree on the right carry the same information, but only the latter scales past one
segment. Front-loading reads as a red trunk turning blue; back-loading as the reverse.

Two examples, both fully determined (clock polytope range 0.000) and well powered, so the comparison is
not confounded with polytope width:

    Eso.AdenoCA  (3,0), route 3_0.1, 25 pooled segments,   952 clock mutations — SBS17 front-loaded
    Lung.AdenoCA (4,1), route 4_1.1, 45 pooled segments, 1,347 clock mutations — APOBEC back-loaded

Different copy-number states, so the effect is not read as a property of one topology, and the (4,1)
shows three gains against the (3,0)'s two. Illustrative examples, not a claim about typical behaviour;
the cohort picture is Fig. 7b,c.

Cache-driven from paper/figures/data/signature/sig_interval_examples.tsv (paper/figures/data/signature/build/build_examples.py).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of, cache_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, DOUBLE, S
_apply_style(compact=True)
from tau.trees import build_allele_trees, plot_route, sort_nodes
from tau.tree_polytope import _events_for_route
from _arrows import dim_arrow, dims as _dims

CACHE = cache_dir_of(__file__)
PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
# the trend colours of the signature figures (plot_cluster_pancancer_arrows.R)
UP, DOWN = "#c0392b", "#2a6fb0"
# both timings drawn in the same ink: they are the same tree, and greying one of them implied a
# difference in status that is not there. They are told apart by position and by their labels.
CLOCK_C = SIG_C = "#1a1a1a"
ORDER = ["front", "back"]
SP = 0.5             # plot_route lineage spacing
GAP = 3.2            # clear space between the two trees, in lineage-spacing units
HEIGHT = 104 * MM    # tall canvas: the late intervals are the narrow ones, and they need the room
TOP, BOT = 0.885, 0.185
YLO, YHI = -0.03, 1.03
# points of canvas per unit of molecular time — the label-crowding tests below are written in molecular
# time, so they have to be derived from the canvas rather than tuned by hand at one figure size
PT_PER_T = HEIGHT * 72 * (TOP - BOT) / (YHI - YLO)
LAB_PT = 11.0        # a rotated 4-character 5 pt label is ~11 pt TALL, not ~5
MIN_LABEL = 0.05     # narrower intervals get an arrow but no width label
TREE_LW = 1.5        # tree stroke width in points (plot_route uses 2 * linewidth_scale)
DASH_MIN = (0, (2.4, 1.4))   # minor allele, dashed — the convention throughout the paper
# The synthesis colouring: the same two trend colours as the bands, so a red branch and a red band
# mean the same thing. Symmetric and clipped at +/-2 (4x either way) — wide enough that only the one
# 5.4x interval saturates, tight enough that the milder Eso example still reads as coloured. The
# neutral is a light GREY rather than white: at log2 ~ 0 an almost-white branch reads as a break in
# the trunk rather than as an interval where the signature simply tracked the clock.
LR_CMAP = LinearSegmentedColormap.from_list("lr", [DOWN, "#d8d8d8", UP])
LR_NORM = Normalize(-2.0, 2.0)


def _gains(v):
    """interval fractions -> cumulative gain times (drop the trailing 1.0)."""
    return np.cumsum(v / v.sum())[:-1]


def _tree(ax, route, cn, times, colour, x_offset=0.0):
    """Draw one timing of the route and return its junction points, in tree order.

    A junction is a node with a parent: where a new copy splits off its parent, at the gain time.
    plot_route lays nodes out by `sort_nodes`, which is topology-driven rather than time-driven, so the
    k-th junction is the same amplification under either timing — that is what makes the trees linkable.
    """
    evs = copy.deepcopy(_events_for_route(route))
    for j, e in enumerate(evs):
        e["time"] = float(np.clip(times[j], 0, 1))
    maj, mn = build_allele_trees(evs, cn[0], cn[1])
    n0 = len(ax.get_lines())
    plot_route(ax, maj, mn, x_offset=x_offset, major_color=colour, minor_color=colour,
               linewidth_scale=TREE_LW / 2, spacing=SP, minor_dashes=DASH_MIN[1])

    # plot_route strokes each gain's horizontal connector CENTRED on the gain time, so the stroke
    # straddles the boundary and eats half a linewidth out of the intervals on either side. At a
    # 0.017-wide interval (~3 pt) two such strokes swallow most of the gap and the interval arrow's
    # tips end up buried inside them, reading as if the arrow fell short. Nudging each connector up by
    # half a linewidth puts its LOWER edge on the gain time — the copy exists from there upward — so
    # an arrow drawn tip-to-tip on the gain times lands on visible edges. Verticals get butt caps for
    # the same reason: matplotlib's default projecting cap overhangs the gain by half a linewidth.
    shift = 0.5 * TREE_LW / PT_PER_T
    for ln in ax.get_lines()[n0:]:
        y = np.asarray(ln.get_ydata(), dtype=float)
        if len(y) == 2 and y[0] == y[1]:
            ln.set_ydata(y + shift)
        else:
            ln.set_solid_capstyle("butt")
    # plot_route assigns node.index as it draws, so read it back rather than re-deriving the layout.
    # A junction spans the horizontal connector from parent to child; both ends are returned so the
    # caller can link from whichever end FACES the other tree, and the dashes stay in the gap instead
    # of cutting back across the lineages.
    js = [(n.index + x_offset,
           min(n.index, n.parent.index) + x_offset,
           max(n.index, n.parent.index) + x_offset, n.time)
          for root in (maj, mn) for n in sort_nodes(root)
          if n.parent is not None and n.mult != 0]
    return js, maj.index + x_offset          # ...and the trunk, which the first interval belongs to


def _logratio_tree(ax, route, cn, times, bnds, lr):
    """The SYNTHESIS: the tree at its ordinary clock timing, branches coloured by the signature's
    log2 activity ratio in whichever interval each stretch of branch lies in.

    Drawn by letting plot_route lay the tree out and then replacing each of its lines with the pieces
    that fall in each interval, so the geometry is byte-for-byte the layout used everywhere else — no
    second implementation of the tree to drift out of step. A vertical branch spans several intervals
    and so is cut at their boundaries; a horizontal connector sits at one gain and takes the colour of
    the interval that gain OPENS.
    """
    evs = copy.deepcopy(_events_for_route(route))
    for j, e in enumerate(evs):
        e["time"] = float(np.clip(times[j], 0, 1))
    maj, mn = build_allele_trees(evs, cn[0], cn[1])
    n0 = len(ax.get_lines())
    plot_route(ax, maj, mn, major_color="k", minor_color="k",
               linewidth_scale=TREE_LW / 2, spacing=SP, minor_dashes=DASH_MIN[1])
    shift = 0.5 * TREE_LW / PT_PER_T
    # plot_route lays the minor nodes out AFTER the major ones, so x separates them. The dash cannot be
    # read back off the artist (Line2D reports a tuple dash as plain "--"), and it has to be re-applied
    # by hand because every line here is replaced by its per-interval pieces.
    x_min_minor = SP * (len(sort_nodes(maj)) + 0.5)

    def col_at(y):
        i = int(np.clip(np.searchsorted(bnds, y, side="right") - 1, 0, len(lr) - 1))
        return LR_CMAP(LR_NORM(lr[i]))

    for ln in list(ax.get_lines()[n0:]):
        x = np.asarray(ln.get_xdata(), dtype=float)
        y = np.asarray(ln.get_ydata(), dtype=float)
        ln.remove()
        ls = DASH_MIN if float(min(x)) > x_min_minor else "-"
        if y[0] == y[1]:
            ax.plot(x, y + shift, color=col_at(y[0]), lw=TREE_LW, ls=ls,
                    solid_capstyle="butt", zorder=2)
            continue
        lo, hi = float(min(y)), float(max(y))
        cuts = np.unique(np.clip(np.concatenate([[lo, hi], bnds]), lo, hi))
        for a, b in zip(cuts[:-1], cuts[1:]):
            ax.plot([x[0], x[0]], [a, b], color=col_at(0.5 * (a + b)), lw=TREE_LW, ls=ls,
                    solid_capstyle="butt", zorder=2)
    return SP * (cn[0] + cn[1])


def make(save=True):
    t = pd.read_csv(CACHE / "sig_interval_examples.tsv", sep="\t")
    fig = plt.figure(figsize=(DOUBLE, HEIGHT))
    # before the axes exist, so their transData is already the final geometry when dim_arrow and
    # _logratio_tree convert points to molecular time
    fig.subplots_adjust(left=0.055, right=0.985, top=TOP, bottom=BOT)
    outer = fig.add_gridspec(1, 2, wspace=0.10)
    axes, raxes = [], []

    for col, kind in enumerate(ORDER):
        inner = outer[col].subgridspec(1, 3, width_ratios=[1, 0.15, 0.40], wspace=0.0)
        ax = fig.add_subplot(inner[0])
        axa = fig.add_subplot(inner[1])                     # the arrow between the two halves
        axr = fig.add_subplot(inner[2], sharey=ax)
        axes.append(ax); raxes.append(axr)
        d = t[t.example == kind].sort_values("time_interval")
        sig, route = d.signature.iloc[0], d.route.iloc[0]
        maj, mnr = (int(x) for x in d.cn_state.iloc[0].split("_"))
        gc, gs_ = _gains(d.clock_norm.to_numpy(float)), _gains(d.sig_norm.to_numpy(float))

        jc, rc = _tree(ax, route, (maj, mnr), gc, CLOCK_C)
        xmax = max([x for _, _, x, _ in jc] + [SP * (maj + mnr)])
        dx = xmax + GAP * SP
        js, rs = _tree(ax, route, (maj, mnr), gs_, SIG_C, x_offset=dx)
        # limits first: dim_arrow reads transData to size its heads in points
        ax.set_xlim(SP - 1.10, dx + xmax + 1.10); ax.set_ylim(YLO, YHI)

        # pair junctions by tree position (same amplification), then order them along time; each pair
        # links the INNER end of the clock junction to the inner end of the signature one
        pairs = sorted([((a[2], a[3]), (b[1], b[3])) for a, b in zip(jc, js)], key=lambda p: p[0][1])
        g_lo, g_hi = xmax + 0.32, dx + SP - 0.32          # the clear gap between the two trees

        def yat(pair, g):
            """Where a junction-to-junction line crosses the gap at x = g."""
            (x0, y0), (x1, y1) = pair
            return y0 + (y1 - y0) * (g - x0) / (x1 - x0)

        # bands: each interval is bounded by the two dashed lines around it (by 0 and 1 at the ends),
        # so the shading is literally the space between the events rather than a separate schematic
        bnd = [None] + list(pairs) + [None]
        bc = np.concatenate([[0.0], gc, [1.0]])
        bs = np.concatenate([[0.0], gs_, [1.0]])
        for i in range(len(bnd) - 1):
            lo, hi = bnd[i], bnd[i + 1]
            ylo = (0.0, 0.0) if lo is None else (yat(lo, g_lo), yat(lo, g_hi))
            yhi = (1.0, 1.0) if hi is None else (yat(hi, g_lo), yat(hi, g_hi))
            wc, ws = bc[i + 1] - bc[i], bs[i + 1] - bs[i]
            # same colormap, same scale, same numbers as the synthesis tree on the right — the band
            # and the branch it becomes are literally the same colour, so the arrow between them is
            # a restatement rather than a new encoding to learn
            v = np.log2(ws / wc) if wc > 0 and ws > 0 else 0.0
            fill = LR_CMAP(LR_NORM(v))
            ax.fill([g_lo, g_hi, g_hi, g_lo], [ylo[0], ylo[1], yhi[1], yhi[0]],
                    color=fill, lw=0, zorder=0)
            if wc > 0 and (wc + ws) / 2 > 5.5 / PT_PER_T:   # room for one 5 pt line
                lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
                ax.text(0.5 * (g_lo + g_hi), 0.25 * (ylo[0] + ylo[1] + yhi[0] + yhi[1]),
                        f"{v:+.2f}", ha="center", va="center", fontsize=S.tiny,
                        color=("white" if lum < 0.55 else "#1a1a1a"),
                        fontweight="bold", zorder=5)

        for (x0, y0), (x1, y1) in pairs:
            ax.plot([x0, x1], [y0, y1], color="0.25", lw=0.7, ls=(0, (2.2, 1.6)), zorder=4)
            ax.plot([x0, x1], [y0, y1], ls="none", marker="o", ms=2.2, mfc="0.25",
                    mec="none", zorder=6)

        # the intervals themselves, marked against the outside edge of each tree: what the fold change
        # in the middle is a ratio OF. Each column sums to 1 by construction.
        for xa, xt, bnds, col in ((SP - 0.20, SP - 0.42, bc, CLOCK_C),
                                  (dx + xmax + 0.20, dx + xmax + 0.42, bs, SIG_C)):
            sgn = -1.0 if xt < SP else 1.0            # which way is "further from the tree"
            last = []                                 # last labelled midpoint in each column
            for i in range(len(bnds) - 1):
                lo, hi = bnds[i], bnds[i + 1]
                dim_arrow(ax, xa, lo, hi, col)
                # Widths under 0.05 go unlabelled: at the 5 pt journal floor the label is taller than
                # the interval it sits in, and nothing is lost — the arrow still marks the span and
                # the band in the middle still gives that interval's fold change.
                if hi - lo < MIN_LABEL:
                    continue
                # a label that would still collide with the last one steps out to the next column
                mid = 0.5 * (lo + hi)
                k = next((j for j, y in enumerate(last) if mid - y >= LAB_PT / PT_PER_T),
                         len(last))
                ax.text(xt + sgn * 0.22 * k, mid, f"{hi - lo:.2f}", rotation=90, fontsize=S.tiny,
                        color=col, ha="center", va="center")
                last[k:k + 1] = [mid]

        # the two trees name themselves through the x ticks — no room for two x labels on one axes
        ax.set_xticks([0.5 * (SP + xmax), dx + 0.5 * (SP + xmax)])
        ax.set_xticklabels(["clock\n(SBS1/SBS5)", sig], fontsize=S.tick)
        for lab in ax.get_xticklabels():
            lab.set_color(SIG_C); lab.set_fontweight("bold")
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", labelsize=S.tick, length=1.4, pad=1.2)
        ax.set_yticks([0, 0.5, 1])
        for s_ in ("top", "right", "bottom"):
            ax.spines[s_].set_visible(False)
        # named the way the sample's own overview figure names it — "(3,0) Cluster 1" and the same
        # length-over-hg19 percentage — so a reader can find this exact cluster in that figure. The
        # route is dropped: it is implied by the tree drawn directly beneath.
        ax.set_title(f"{d.tumor_type.iloc[0]}  {d['sample'].iloc[0][:8]}\n"
                     f"{d.cluster_label.iloc[0]}, {d.pct_genome.iloc[0]:.1f}% of genome, "
                     f"{int(d.n_segments.iloc[0])} segments\n"
                     f"{sig} {'front' if kind == 'front' else 'back'}-loaded",
                     fontsize=S.annot, fontweight="bold", pad=4, linespacing=1.4)

        # ---- the synthesis: one tree, ordinary clock timing, coloured by the log2 ratio
        lr = np.log2(np.diff(bs) / np.diff(bc))
        rmax = _logratio_tree(axr, route, (maj, mnr), gc, bc, lr)
        axr.set_xlim(SP - 0.45, rmax + 0.45)
        axr.set_xticks([0.5 * (SP + rmax)])
        axr.set_xticklabels([f"{sig} activity\nrelative to clock"], fontsize=S.tick, fontweight="bold")
        axr.tick_params(axis="x", length=0, pad=2)
        plt.setp(axr.get_yticklabels(), visible=False); axr.tick_params(left=False)
        for s_ in ("top", "right", "bottom", "left"):
            axr.spines[s_].set_visible(False)

        axa.set_axis_off()
        axa.annotate("", xy=(0.86, 0.55), xytext=(0.14, 0.55), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.8,
                                     mutation_scale=5, shrinkA=0, shrinkB=0))

    axes[0].set_ylabel("normalised molecular time", fontsize=S.label)

    # the two flat swatches are gone: the colourbar now explains the bands and the branches at once
    h = [mlines.Line2D([], [], color="0.25", lw=0.7, ls=(0, (2.2, 1.6)), marker="o", ms=1.8,
                       mfc="0.25", mec="none", label="same amplification, both timings")]
    fig.legend(handles=h, fontsize=S.legend, loc="lower center", bbox_to_anchor=(0.5, -0.004),
               ncol=1, frameon=False, handlelength=1.8, columnspacing=1.6)
    for x, s_, ha in ((0.392, "contraction", "right"), (0.608, "expansion", "left")):
        fig.text(x, 0.0885, s_, fontsize=S.tiny, color="0.45", ha=ha, va="center")
    cax = fig.add_axes([0.40, 0.082, 0.20, 0.013])
    cb = fig.colorbar(ScalarMappable(norm=LR_NORM, cmap=LR_CMAP), cax=cax, orientation="horizontal", extend="both")
    cb.set_label("log2 signature activity relative to clock", fontsize=S.label, labelpad=1.5)
    cb.set_ticks([-2, -1, 0, 1, 2]); cb.ax.tick_params(labelsize=S.tick, length=1.4, pad=1.0)
    cb.outline.set_linewidth(0.4)
    if save:
        _save_panel(fig, PANELS, "c_expansion_contraction")
    return fig


if __name__ == "__main__":
    make(); print("c_expansion_contraction: PNG+SVG written")
