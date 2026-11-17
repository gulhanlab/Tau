"""signatures figure (main), panel a (right half) — every pooled cluster of one genome, coloured by signature activity.
Used by a_sbs17_overview_trees.py via tree_rows(); make() draws the strip on its own (alt_).

Panel c makes the argument on two hand-picked clusters: time a cluster twice, once by the clock and
once by a signature, and an interval that takes a larger share under the signature is one where that
signature ran faster. Its right-hand half then collapses that into a single tree at ordinary clock
timing with the branches coloured by the ratio. This panel is that collapsed form applied to a WHOLE
genome — the same sample Figures 2b, 2c and 3a use — which is the point of the collapse: it needs only
one timeline per cluster, so it scales past an exemplar.

    z_i = log2( ratio_i / baseline ),   baseline = sum_i raw_sig_i / sum_i raw_clock_i

`ratio_i` is the raw (un-normalised) interval ratio (ws/wc)(D_sig/D_clock) from the cache. The
baseline is summed over every interval of every cluster in the sample, so z measures an interval
against THIS genome's own average rather than an absolute scale — a signature that is simply abundant
does not colour the whole figure red. Red = the signature outran the clock during that interval
(expansion), blue = it fell behind (contraction). Both are the trend colours used by panel c and by
the cohort figures, so a red branch here means what a red band means there.

TIMING IS ALWAYS THE CLOCK TIMING. The signature timeline is never drawn.

WHICH clock timing, for an underdetermined route, is a choice. The gain times are a polytope, and the
trees here are drawn at the WGD-collapse solution — the admissible point that puts gains on a detected
doubling — which is the convention tau.trees.plot_cluster_trees uses, so these trees match the
overview's. It matters for (2,2) Cluster 1 and (4,0) Cluster 1, whose polytopes are 0.24 wide: the
collapse shows both allele gains at the WGD rather than an arbitrary split across it.

Colour survives the collapse intact, and not by luck. The intervals the collapse closes are exactly
the polytope's degenerate directions, and the ratio cannot distinguish them either — (2,2) Cluster 1
reads +0.13 / -0.63 / -0.63 and (4,0) Cluster 1 reads +1.00 / -0.26 / -0.61 / -0.61, so the pair being
merged already carried one value. z is indexed by which two gains an interval lies between, which the
collapse does not change; it only moves where those gains sit.

Read with the caveat that the cache prints and this panel does not repeat: clusters whose route is
underdetermined carry a clock polytope of non-zero width, so their interval boundaries are one
admissible solution rather than a measurement. For this sample that is (2,2) Cluster 1 and (4,0)
Cluster 1, at 0.24 — and the first of those is the largest cluster on the figure. It belongs in the
caption.

INPUTS ARE STANDARD TAU OUTPUTS, both from paper/data_tables/per_sample/<sample>/:

    <sample>.tau_sig_intervals.tsv     what `tau run --include-signatures` writes; carries
                                       clocklike_cumulative_* and, per signature, <SIG>_ratio
    <sample>.clustered_genome.pkl.gz   the pooled units, for route, polytope draws and unit size

There is no figure-specific cache. paper/figures/data/signatures/build/build_cluster_trees.py exists only to RUN the
signature step for a sample whose clock run predates it, writing that same standard file.

Unit selection and ordering are tau/overview.py's, not _get_final_segment_clusters': every pooled unit
with a drawable route above MIN_ESS, in CN order. The two rules give different sets — the
_get_final_segment_clusters gate (num_segs > 2 AND dist_rel <= 0.05) drops single-segment high-CN
units, which is 13 trees here against the overview's 16.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of, PER_SAMPLE  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, DOUBLE, S
_apply_style(compact=True)
from tau.trees import build_allele_trees, plot_route, sort_nodes
from tau.tree_polytope import _events_for_route

SAMPLE = "0250e6f7-2538-4d1f-a8ff-ca34a7057dde"      # the Fig. 2b / Fig. 7a esophageal adenocarcinoma
PER = PER_SAMPLE / SAMPLE
MIN_ESS = 30           # = tau/overview.py MIN_ESS; below it the overview draws no tree either
PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
SIG = "SBS17"          # the dominant process in this esophageal adenocarcinoma
CLIP = 2.0             # log2 colour limit
# The Methods' own floor for an interval to carry a usable relative rate. It matters here rather than
# cosmetically: a cluster with a near-zero clock interval divides by ~0 and the ratio explodes —
# (3,1) Cluster 1 produced +13.9 and -inf on 56 clock mutations spread over three intervals. A tree
# with one undefined interval cannot be coloured honestly, so the whole cluster is dropped.
# 0 = draw every final cluster, so the tree set matches Figure 2b's overview exactly. Raise it to
# exclude under-powered clusters instead: at 5.0 (the Methods' floor for an interval to carry a usable
# relative rate) this sample loses (3,2) Cluster 1 and (4,2) Cluster 1. Their thinnest intervals hold
# 2.8 and 0.3 clock mutations, so those intervals' colours are noise, not measurement — the console
# names them on every run and the caption should too.
MIN_CLOCK_MUT = 0.0
THIN_WARN = 5.0        # report, but still draw
# the trend colours of panel c. Neutral is a light GREY, not white: at z ~ 0 an almost-white branch
# reads as a break in the trunk rather than as an interval where the signature tracked the clock.
UP, DOWN = "#c0392b", "#2a6fb0"
LR_CMAP = LinearSegmentedColormap.from_list("lr", [DOWN, "#d8d8d8", UP])
SP = 0.5               # plot_route lineage spacing
TREE_LW = 1.5          # tree stroke width in points (plot_route uses 2 * linewidth_scale)
DASH_MIN = (0, (2.4, 1.4))    # minor allele dashed — the convention throughout the paper
HEIGHT = 64 * MM
TOP, BOT = 0.86, 0.055
YLO, YHI = -0.03, 1.03
PT_PER_T = HEIGHT * 72 * (TOP - BOT) / (YHI - YLO)


def _logratio_tree(ax, route, cn, times, bnds, z, norm):
    """The tree at clock timing, each stretch of branch coloured by z in the interval it lies in.

    Same construction as c_expansion_contraction._logratio_tree: plot_route lays the tree out and its
    lines are then replaced by their per-interval pieces, so the geometry is the layout used
    everywhere else rather than a second implementation that can drift out of step.
    """
    evs = copy.deepcopy(_events_for_route(route))
    for j, e in enumerate(evs):
        e["time"] = float(np.clip(times[j], 0, 1))
    maj, mn = build_allele_trees(evs, cn[0], cn[1])
    n0 = len(ax.get_lines())
    plot_route(ax, maj, mn, major_color="k", minor_color="k",
               linewidth_scale=TREE_LW / 2, spacing=SP, minor_dashes=DASH_MIN[1])
    # plot_route strokes each gain's horizontal connector CENTRED on the gain time, so it straddles
    # the boundary and eats half a linewidth from the intervals either side; nudging it up puts its
    # LOWER edge on the gain. Verticals get butt caps for the same reason.
    shift = 0.5 * TREE_LW / PT_PER_T
    # minor nodes are laid out after the major ones, so x separates them. The dash cannot be read back
    # off the artist (Line2D reports a tuple dash as plain "--") and must be re-applied by hand.
    x_min_minor = SP * (len(sort_nodes(maj)) + 0.5)

    def col_at(y):
        return LR_CMAP(norm(z[int(np.clip(np.searchsorted(bnds, y, side="right") - 1,
                                          0, len(z) - 1))]))

    for ln in list(ax.get_lines()[n0:]):
        x = np.asarray(ln.get_xdata(), float)
        y = np.asarray(ln.get_ydata(), float)
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


def _units():
    """(seg_id -> identity) for the units tau/overview.py would draw, in its order."""
    import gzip, pickle
    from tau.utils import pick_best_key
    from tau.trees import _pick_draw_for_tree
    from tau.clustering import cluster_times_bottomup
    with gzip.open(PER / f"{SAMPLE}.clustered_genome.pkl.gz", "rb") as f:
        cg = pickle.load(f)
    # An un-clustered segment (DBSCAN label -1) keeps its raw coordinate seg_id on the clustered
    # genome, but the signature step promotes it to a one-member cluster and so refers to it as
    # merged_{M}_{m}_{k}. Rebuild that mapping — it is deterministic from segment_clusters.tsv, the
    # same file the promotion reads — or those units silently fail to join and their trees vanish.
    scd = pd.read_csv(PER / f"{SAMPLE}.segment_clusters.tsv", sep="\t")
    lone = scd.cluster_label == -1
    promoted = {}
    if lone.any():
        nxt = scd[~lone].groupby(["major_cn", "minor_cn"]).cluster_label.max().to_dict()
        for _, r in scd[lone].iterrows():
            cn = (r.major_cn, r.minor_cn)
            nxt[cn] = int(nxt.get(cn, -1)) + 1
            promoted[r.seg_id] = f"merged_{int(cn[0])}_{int(cn[1])}_{nxt[cn]}"

    ess = lambda x: float(np.sum(x.N_counts) if x.N_counts is not None else 0.0)
    keep = [x for x in cg.segments if pick_best_key(x) is not None and ess(x) >= MIN_ESS]
    keep.sort(key=lambda x: (int(x.major_cn), int(x.minor_cn), -getattr(x, "num_segs", 1)))

    # gains are drawn at the WGD-collapse solution, the convention plot_cluster_trees uses; only the
    # LARGEST (2,2) collapses onto one doubling, a secondary one may split across two
    wgd = pd.read_csv(PER / f"{SAMPLE}.tau_events.tsv", sep="\t")
    wgd_times = [float(x) for x in wgd.loc[wgd.classification == "WGD", "event_time_frac"]]
    s22 = [x for x in keep if (int(x.major_cn), int(x.minor_cn)) == (2, 2)]
    dom22 = max(s22, key=lambda x: getattr(x, "num_segs", 1)).seg_id if s22 else None

    out, counts = {}, {}
    for seg in keep:
        cn = (int(seg.major_cn), int(seg.minor_cn))
        counts[cn] = counts.get(cn, 0) + 1
        ev, _i = _pick_draw_for_tree(seg, wgd_times, collapse_22=(seg.seg_id == dom22))
        tl = getattr(seg, "total_length", seg.end - seg.start)
        key = promoted.get(seg.seg_id, seg.seg_id)
        out[key] = dict(label=f"({cn[0]},{cn[1]}) #{counts[cn]}", route=pick_best_key(seg),
                               cn=cn, pct=100.0 * tl / 3_095_677_412,
                               gains=None if ev is None else np.asarray(ev, float))
    return out


def tree_rows(verbose=False):
    """The pooled units with their clock-timed gains and per-interval log2 ratio z (see module docstring)."""
    t = pd.read_csv(PER / f"{SAMPLE}.tau_sig_intervals.tsv", sep="\t")
    units = _units()
    missing = sorted(set(units) - set(t.cluster))
    if missing:
        print(f"  NOT IN THE SIGNATURE TABLE (drawn by the overview, absent here): {missing}")
    t = t[t.cluster.isin(units)]
    rcol, ncol = f"{SIG}_raw_interval", f"{SIG}_ratio"
    ok = t.dropna(subset=[rcol])
    baseline = ok[rcol].sum() / ok.clocklike_raw_interval.sum()

    rows, dropped, thin_note = [], [], []
    for cl, g in t.groupby("cluster", sort=False):
        g = g.sort_values("time_interval")
        if g[ncol].isna().any():
            dropped.append((units[cl]["label"], f"no {SIG} mutations"))
            continue
        thin = float(g.clocklike_raw_interval.min())
        if thin < MIN_CLOCK_MUT:
            dropped.append((units[cl]["label"],
                            f"thinnest interval carries {thin:.1f} clock mutations"))
            continue
        if thin < THIN_WARN:
            thin_note.append((units[cl]["label"], thin))
        u = units[cl]
        rep = np.sort(g.clocklike_cumulative_end.to_numpy(float))
        use = (np.concatenate([[0.0], np.sort(u["gains"]), [1.0]])
               if u["gains"] is not None and len(u["gains"]) == len(rep) - 1
               else np.concatenate([[0.0], rep]))
        rows.append(dict(
            collapsed=bool(not np.allclose(use[1:], rep)),
            label=u["label"], route=str(u["route"]), cn=u["cn"], pct=u["pct"],
            clock=float(g.clocklike_raw_interval.sum()),
            rng=float(g.clocklike_range.max()),
            bnds=use,
            z=np.clip(np.nan_to_num(np.log2(g[ncol].to_numpy(float) / baseline),
                                     nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)))
    rows.sort(key=lambda r: (r["cn"][0], r["cn"][1], r["label"]))    # the overview's own order
    if not rows:
        sys.exit("no unit had usable signature intervals")
    for lab, why in dropped:
        print(f"  dropped {lab:<16} ({why})")
    for lab, thin in thin_note:
        print(f"  UNDER-POWERED {lab:<16} thinnest interval {thin:.1f} clock mutations "
              f"— that interval's colour is noise")
    return rows


def make(save=True):
    """Standalone strip of the trees (the former supp panel); the paper's panel a composes these with
    the ratio-coloured overview in a_sbs17_overview_trees.py."""
    rows = tree_rows()
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE, HEIGHT), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.075, right=0.865, top=TOP, bottom=BOT, wspace=0.35)
    norm = Normalize(-CLIP, CLIP)

    print(f"{'cluster':<16}{'route':<8}{'%gen':>6}{'clock':>8}{'range':>7}  log2 by interval")
    for ax, r in zip(axes, rows):
        _logratio_tree(ax, r["route"], r["cn"], r["bnds"][1:-1], r["bnds"], r["z"], norm)
        ax.set_xlim(SP - 0.55, SP * (r["cn"][0] + r["cn"][1]) + 0.55)
        ax.set_ylim(YLO, YHI); ax.set_xticks([])
        # same title as the overview's trees: CN state over genome fraction
        ax.set_title(f"{r['label']}\n{r['pct']:.1f}%", fontsize=S.title, pad=2.5)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        if ax is not axes[0]:              # one time axis for the row, not n identical copies
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        print(f"{r['label']:<16}{r['route']:<8}{r['pct']:6.1f}{r['clock']:8.0f}{r['rng']:7.2f}"
              f"{'  collapsed' if r['collapsed'] else '          '}  "
              + "  ".join(f"{v:+.2f}" for v in r["z"]))

    axes[0].set_ylabel("molecular time", fontsize=S.label)
    axes[0].set_yticks([0, 0.5, 1.0]); axes[0].tick_params(length=1.4, pad=1.2)
    cax = fig.add_axes([0.888, 0.22, 0.014, 0.46])
    zmax = np.abs(np.concatenate([r["z"] for r in rows])).max()
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=LR_CMAP), cax=cax,
                      extend="both" if zmax > CLIP else "neither")
    cb.set_label(f"log$_2$ {SIG} activity relative to clock", fontsize=S.label)
    cb.ax.tick_params(labelsize=S.tick, length=1.4, pad=1.2)
    cb.outline.set_linewidth(0.4)

    if save:
        _save_panel(fig, PANELS, "alt_cluster_trees_strip")
    return fig


if __name__ == "__main__":
    make(); print("alt_cluster_trees_strip: PNG+SVG+PDF written")
