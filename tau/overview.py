#!/usr/bin/env python3
"""tau/overview.py — the per-sample two-level overview figure.

    [ overview (individual segment timing) | merged allele-trees (pooled timing) | timing histogram ]

Tau produces timing at two levels and they answer different questions. The left panel is the raw
per-segment timing: every timed segment as a stacked band, coloured by copy-number state, so you see
the genome-wide spread. The trees are the POOLED units — segments sharing a CN state and a timing
cluster merged into one higher-power unit — each drawn as the allele tree Tau solved for it. The
histogram on the right is the marginal of exactly the solutions the trees display.

Individual events (`tau_events`, from the raw genome) are drawn as DASHED lines across the overview;
merged events (`tau_merged_events`, pooled) as DOTTED lines across the trees and histogram. So you can
watch the individual per-segment timings on the left resolve into the merged event times on the right.

Ported from `paper/batch_scripts/proto_fig2_twolevel.py` (itself adapted from
`dev/pcawg/figure2_signature/make_overview_trees.py`). The numeric behaviour is unchanged; what changed
is that the sample now arrives as objects rather than being loaded from a hard-coded cohort directory,
so `tau run` can emit this for any sample.

Usage from the pipeline::

    from tau.overview import save_overview
    save_overview(genome, clustered_genome, tables["events"], tables["merged_events"], "out/S1")
"""
from __future__ import annotations

import copy
import csv
import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.legend as _mleg

from tau.plotting import (compute_offsets, plot_tau_stack, segment_cluster_plotting,
                          _cmap_for_major, _colors_for_segment, extract_mat)
from tau.trees import _pick_draw_for_tree, build_allele_trees, plot_route
from tau.tree_polytope import _events_for_route
from tau.utils import pick_best_key, order_t

HG19_TOTAL = 3_095_677_412
FIG_W_IN, FIG_H = 183 / 25.4, 60 / 25.4
W_OVERVIEW, W_HIST, W_GAP, W_TREE, TREE_SPACING, FS = 9.0, 0.55, 1.05, 0.62, 0.32, 6

MIN_ESS = 30   # pooled units below this are not drawn: too little evidence to show, and a tiny unit
               # reads with the same visual weight as a well-powered one. No member-count rule — at
               # high CN a state is often represented by a single segment, and excluding those removed
               # every (5,x)/(6,x)/(7,x) tree from the Stomach sample.

# tree line weight carries the pooled mutation count, so a well-powered cluster reads as heavier than
# a sparse one at a glance. log scale: ESS spans ~10 to ~3500 across units, unusable linearly.
LW_MIN, LW_MAX = 0.30, 1.10
LW_ESS_LO, LW_ESS_HI = 10.0, 2000.0

SNAP_TOL  = 0.01   # simultaneity: the block's interior intervals must collapse to under this
MATCH_TOL = 0.15   # a doubling this close to a detected WGD counts as explaining it
SPLIT_TOL = 0.15   # fallback only: mean per-gain distance to the nearest detected WGD must beat this


def _data_path(name):
    import importlib.resources as _r
    try:
        with _r.files("tau.data").joinpath(name) as p:
            return Path(p)
    except (AttributeError, TypeError):       # older Python
        import pkg_resources
        return Path(pkg_resources.resource_filename("tau", f"data/{name}"))


@lru_cache(maxsize=1)
def _doubling_blocks():
    """route_key -> [(start, length, loss), ...]: the WGD-consistent forms each route can express."""
    p = _data_path("doubling_blocks.tsv")
    out = {}
    if not p.exists():
        return out
    with open(p) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("blocks"):
                out[r["route_key"]] = [tuple(int(v) for v in b.replace("/", "+").split("+"))
                                       for b in r["blocks"].split(";")]
    return out


def _lw_for_ess(ess):
    if not ess or ess <= 0:
        return LW_MIN
    f = (np.log10(max(ess, LW_ESS_LO)) - np.log10(LW_ESS_LO)) / (
        np.log10(LW_ESS_HI) - np.log10(LW_ESS_LO))
    return LW_MIN + (LW_MAX - LW_MIN) * float(np.clip(f, 0.0, 1.0))


def _row_color(cls):   # chrom_specific = dark red, distinct from grey unassigned timing
    return {"WGD": "#0b6979", "ccPG": "#cd9f2c", "PGD": "#cd9f2c",
            "chrom_specific": "#9b2226"}.get(cls, "#8c8c8c")


def _ev_label(r, prefix=""):   # concise: chrom_specific -> chrN, PGD shown as ccPG, else WGD; + time
    cls = r["classification"]
    if cls == "chrom_specific" and pd.notna(r.get("chrom")):
        return f"{prefix}chr{int(float(r['chrom']))} {r['event_time_frac']:.2f}"
    disp = "ccPG" if cls == "PGD" else cls   # legacy outputs
    return f"{prefix}{disp} {r['event_time_frac']:.2f}"


def _ess(seg):
    return float(np.sum(seg.N_counts)) if getattr(seg, "N_counts", None) is not None else 0.0


def _pick_draw_sim_span(seg, wgd_times, tol=MATCH_TOL):
    """DEFAULT rule. Pick the tightest solution; escalate to the doublings only if it explains none.

    Step 1 is `sim_span` — of the polytope's solutions, take the one whose gains span the least time.
    It is the simultaneity prior stated directly, it uses no event times, and it is the rule benchmarked
    against simulation truth on v10 (underdetermined per-gain error 0.151 -> 0.035, parity with GRITIC).
    It needs no route metadata: the polytope does the work. A (4,4) unit in a double-WGD sample is NOT
    collapsed by it, because no feasible solution puts all six gains together — the tightest available
    span is already ~0.73, and the solution it picks is two gains at WGD1 and four at WGD2.

    Step 2 exists for one situation. Where a segment's gains sit on DIFFERENT doublings, the collapsed
    reading is genuinely more simultaneous, so span alone prefers it: a secondary (2,2) cluster in a
    double-WGD sample scores span 0 at t=0.51 — a time at which no doubling occurred — over one gain at
    each real doubling. No simultaneity rule can fix that, structural or otherwise; only the detected
    event times distinguish the two. So when the tightest solution lands near NO detected doubling, and
    only then, re-pick by minimising total distance from each gain to its nearest doubling.

    That escalation is CIRCULAR by construction — such a tree is positioned by the detected times, not
    purely by its own data, and cannot be used as independent evidence that the trees converge on those
    times. It is reported separately in `info["modes"]` for exactly that reason. On the canonical
    double-WGD sample it fires on 1 of 15 pooled units.

    Returns (event_times, draw_idx, mode).
    """
    key = pick_best_key(seg)
    if key is None:
        return None, None, "none"
    draws = [d for d in seg.timing_result[key].get("draws", []) if d.get("boot_id", 0) == 0]
    if not draws:
        return None, None, "none"
    cums = np.clip(np.array([np.cumsum(order_t(d["t"]))[:-1] for d in draws]), 0.0, 1.0)
    if len(draws) == 1:
        return cums[0], 0, "determined"
    j = int(np.argmin(cums.max(axis=1) - cums.min(axis=1)))       # sim_span
    W = list(wgd_times) if wgd_times is not None else []
    if not W or any(min(abs(t - w) for w in W) <= tol for t in cums[j]):
        return cums[j], j, "sim_span"
    d = np.array([sum(min(abs(t - w) for w in W) for t in row) for row in cums])
    i = int(np.argmin(d))
    return cums[i], i, "anchored"


def _pick_draw_wgd_preset(seg, n_wgd, wgd_times, tol=SNAP_TOL, min_ess=MIN_ESS):
    """Pick the displayed solution from the detected event count, then the route, then the event times.

    The rule, in order:

      1. how many doublings does the sample show?  0 -> no preset at all
      2. enumerate the WGD-consistent forms with up to that many doublings
      3. keep the ones THIS SEGMENT'S ROUTE can express, and whose simultaneity the polytope can
         actually realise (interior intervals collapsing to within `tol`)
      4. if more than one survives, take the form whose doubling times sit closest to the sample's
         detected WGD times

    No parsimony ordering: a form is not preferred for being shorter or assuming less loss. Steps 1-3
    are structural and use no event times, so where they leave a single form the displayed timing is
    an output. Step 4 uses the detected times, but only to choose among forms that are already valid —
    it cannot move where any of them land.

    Returns (event_times, draw_idx, mode).
    """
    key = pick_best_key(seg)
    if key is None:
        return None, None, "none"
    draws = [d for d in seg.timing_result[key].get("draws", []) if d.get("boot_id", 0) == 0]
    if not draws:
        return None, None, "none"
    cums = np.array([np.cumsum(order_t(d["t"]))[:-1] for d in draws])
    if len(draws) == 1:
        return cums[0], 0, "determined"
    if _ess(seg) < min_ess:      # guard 1 (redundant now the filter is at selection, kept as a net)
        med = len(draws) // 2
        return cums[med], med, "neutral"
    if n_wgd < 1:
        med = len(draws) // 2
        return cums[med], med, "neutral"

    ts = np.array([order_t(d["t"]) for d in draws])
    blocks = [b for b in _doubling_blocks().get(key, []) if b[1] >= 2]   # forms this route expresses
    fits = []
    for r in range(1, n_wgd + 1):
        for combo in itertools.combinations(blocks, r):
            iv = sorted((s_, s_ + L - 1) for s_, L, _ in combo)
            if any(iv[i][1] >= iv[i + 1][0] for i in range(len(iv) - 1)):
                continue                                        # doublings cannot overlap
            score = np.zeros(len(draws))
            for (s_, L, _) in combo:
                score = score + ts[:, s_:s_ + L - 1].sum(axis=1)
            i = int(np.argmin(score))
            if score[i] > tol:                                  # the polytope cannot realise it
                continue
            got = [float(cums[i][s_ - 1]) for (s_, L, _) in combo]
            if wgd_times is not None and len(wgd_times) and not any(
                    min(abs(g - w) for w in wgd_times) <= MATCH_TOL for g in got):
                continue                                        # guard 2: explains no detected event
            fits.append((combo, i))
    if not fits:
        # No structural form survives. Before giving up, try placing the gains against the detected
        # events directly: score each solution by the mean distance from each gain to its NEAREST
        # detected WGD. This is the distance-minimisation the structural machinery deliberately avoids,
        # so it is a LAST RESORT and labelled separately — a tree placed this way is positioned by the
        # event times, not by the data, and cannot count toward any convergence claim.
        if wgd_times is not None and len(wgd_times):
            wtl = list(wgd_times)
            sc = np.array([np.mean([min(abs(float(g) - w) for w in wtl) for g in row]) for row in cums])
            j = int(np.argmin(sc))
            if sc[j] <= SPLIT_TOL:
                return cums[j], j, "event-aligned"
        med = len(draws) // 2
        return cums[med], med, "neutral"
    if len(fits) == 1:
        combo, i = fits[0]
        return cums[i], i, f"wgd x{len(combo)}"

    # Several valid forms -> rank by FIT to the detected events, on two keys:
    #   1. how many detected WGDs the form actually accounts for (a doubling within MATCH_TOL)
    #   2. how close those doublings sit, averaged over the form's doublings
    # Both are needed. Summing distance over the detected events is degenerate for a one-doubling
    # form: |w1 - t| + |w2 - t| is constant for any t between the two events, so every single-doubling
    # candidate ties and the choice falls to list order. Counting explained events first fixes that,
    # and the mean keeps forms of different size comparable. Loss breaks any remaining tie only to
    # keep the choice deterministic.
    wt = list(wgd_times) if wgd_times is not None else []

    def rank(ci):
        combo, i = ci
        got = [float(cums[i][s_ - 1]) for (s_, L, _) in combo]
        # gains the doublings account for. A gain belonging to no doubling is an extra event needing
        # its own explanation, and being unconstrained it drifts to the polytope wall. In a 2-WGD
        # sample (4,4) is (1,1)->WGD->(2,2)->WGD->(4,4): every gain belongs to a doubling, so full
        # coverage is the correct reading and should outrank a marginally closer fit.
        covered = sum(L for _, L, _ in combo)
        if not wt:
            return (0, -covered, 0.0, sum(l for _, _, l in combo))
        explained = sum(1 for w in wt if min(abs(w - g) for g in got) <= MATCH_TOL)
        mean_d = float(np.mean([min(abs(g - w) for w in wt) for g in got]))
        return (-explained, -covered, mean_d, sum(l for _, _, l in combo))

    combo, i = min(fits, key=rank)
    return cums[i], i, f"wgd x{len(combo)} (tie-broken)"


def _cn_legend_above_trees(fig, tree_axes, states, nbands, fontsize=4.6, h_frac=0.235, aspect=3.0):
    """CN-state colour key as ONE self-contained landscape block, floated above the tree area.

    `tau.plotting._add_cn_timing_legend` draws it portrait and to the LEFT of the overview: one row per
    CN state, time running left->right. Here it is transposed — one COLUMN per colour group, time
    running UPWARDS so the key reads the same way round as the figure's y axis — and sized to a fixed
    ~3:1 block rather than stretched across the trees, so it stays a discrete key.
    """
    from matplotlib.patches import Rectangle as Rect
    groups = {}
    for (maj, mn) in sorted(states):
        K = int((nbands or {}).get((maj, mn)) or max(int(maj), 1))
        groups.setdefault((maj, K), []).append(mn)

    def _lab(maj, mns):
        ms = sorted(set(mns))
        return f"({maj},{ms[0]})" if len(ms) == 1 else f"({maj},{'/'.join(map(str, ms))})"

    cols = [((maj, K), _lab(maj, mns)) for (maj, K), mns in sorted(groups.items())]
    n = len(cols)
    if not n or not tree_axes:
        return None
    fw, fh = fig.get_size_inches()
    w_frac = aspect * h_frac * fh / fw          # fixed display aspect, independent of the tree block
    x0 = tree_axes[0].get_position().x0
    y0 = max(a.get_position().y1 for a in tree_axes) + 0.235
    lg = fig.add_axes([x0, y0, w_frac, h_frac])
    lg.set_xlim(-0.5, n - 0.5); lg.set_ylim(0, 1)
    lg.set_xticks([]); lg.set_yticks([])
    for sp in lg.spines.values():
        sp.set_visible(False)
    for i, ((maj, K), lab) in enumerate(cols):
        shades = _colors_for_segment(int(maj), int(K), _cmap_for_major(maj))
        for j, c in enumerate(shades):                      # bottom = early, top = late
            lg.add_patch(Rect((i - 0.46, j / len(shades)), 0.92, 1.0 / len(shades),
                              facecolor=c, edgecolor="none"))
        lg.text(i, -0.10, lab, ha="center", va="top", fontsize=fontsize, rotation=90, clip_on=False)
    lg.text(-0.85, 0.0, "$t_{early}$", ha="right", va="bottom", fontsize=fontsize,
            color="0.35", clip_on=False)
    lg.text(-0.85, 1.0, "$t_{late}$", ha="right", va="top", fontsize=fontsize,
            color="0.35", clip_on=False)
    return lg


def _cluster_fill_key(fig, anchor_ax, n_clusters, fontsize=4.4, pad=0.020):
    """Fill-pattern key for the pooled-segment clusters, beside the CN-timing block.

    Within one CN state the first cluster is drawn solid and later ones striped (see HATCHES in
    `tau.plotting.segment_cluster_plotting`). Swatches here are explicit VECTOR lines clipped to a grey
    box, not a matplotlib hatch: hatch exports as an SVG <pattern> that Illustrator drops, leaving a
    blank swatch.
    """
    from matplotlib.patches import Rectangle as Rect
    from matplotlib.lines import Line2D
    n = max(int(n_clusters), 1)
    pos = anchor_ax.get_position()
    ax = fig.add_axes([pos.x1 + pad, pos.y0, 0.035, pos.height])
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(n):
        y = n - 1 - i                       # cluster 1 on top
        box = Rect((0.0, y - 0.34), 0.55, 0.68, facecolor="grey", edgecolor="none")
        ax.add_patch(box)
        slopes = ([1] if i >= 1 else []) + ([-1] if i >= 2 else [])
        for sl in slopes:
            for k in range(-4, 8):
                x0 = 0.0 + k * 0.14
                y0_, y1_ = (y - 0.34, y + 0.34) if sl > 0 else (y + 0.34, y - 0.34)
                ln = Line2D([x0, x0 + 0.28], [y0_, y1_], color="white", lw=0.45)
                ln.set_clip_path(box); ax.add_line(ln)
        ax.text(0.68, y, f"Cluster {i + 1}", ha="left", va="center", fontsize=fontsize, clip_on=False)
    return ax


def build_timing_hist(ax, tree_times, flip=False):
    """ESS-weighted gain-time histogram of the REPRESENTATIVE solution shown in each tree.

    Binning every polytope draw over-reads the open end of the solution space: the last gain of an
    underdetermined route is bounded below but NOT above, so its band runs to t=1 by construction, and
    binning all draws piles weight above t=0.9 as an artefact of the polytope's upper wall. Binning
    only the draw each tree displays removes that — one gain time per gain per cluster, weighted by the
    cluster's mutation count — so the histogram is the marginal of exactly what is drawn beside it and
    the two halves of the figure cannot disagree.
    """
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    vals, wts = [], []
    for et, ess in tree_times:
        for t in np.asarray(et, float):
            vals.append(float(np.clip(t, 0.0, 1.0))); wts.append(ess)
    if not vals:
        return
    ax.hist(vals, bins=np.linspace(0, 1, 26), weights=wts,
            orientation="horizontal", color="#9e9e9e", edgecolor="white", linewidth=0.3)
    ax.set_xlim(left=0)
    ax.spines["left"].set_visible(True); ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_linewidth(0.5)
    if flip:
        ax.invert_xaxis()


def overview_figure(genome, clustered_genome, events, merged_events, *, n_trees=None,
                    show_chrom_specific=False, blur=False, min_ess=MIN_ESS, preset="sim",
                    verbose=False):
    """Build the two-level overview. Returns (fig, info).

    genome / clustered_genome : the raw and pooled `tau.core.Genome` objects.
    events / merged_events    : the `tau_events` / `tau_merged_events` tables (DataFrames).
    n_trees                   : cap on the number of pooled units drawn (None = all that qualify).
    preset                    : which solution each tree displays --
                                "sim"    tightest gain span, escalating to the doublings only where
                                         that explains none (:func:`_pick_draw_sim_span`). Default.
                                "blocks" structural doubling-block enumeration
                                         (:func:`_pick_draw_wgd_preset`).
                                The two agree on 15/15 pooled units of the canonical double-WGD
                                sample; "sim" is kept as the default because it is the rule that was
                                benchmarked against simulation truth, and it needs no route metadata.
    """
    ev_ind = events.sort_values("event_time_frac").copy()
    ev_mrg = merged_events.sort_values("event_time_frac").copy()
    if not show_chrom_specific:      # hide chrom-specific events (keep WGD/PGD) unless toggled on
        ev_ind = ev_ind[ev_ind.classification != "chrom_specific"]
        ev_mrg = ev_mrg[ev_mrg.classification != "chrom_specific"]
    wgd_times = ev_ind[ev_ind.classification == "WGD"]["event_time_frac"].tolist()

    g, cg = genome, clustered_genome

    # ---- pooled units with a drawable route (top-N by size only if n_trees), shown in CN order ----
    allsegs = [s for s in cg.segments if pick_best_key(s) is not None and _ess(s) >= min_ess]
    allsegs.sort(key=lambda s: -getattr(s, "num_segs", 1))
    if n_trees:
        top = allsegs[:n_trees]
        top_ids = {id(s) for s in top}
        top_states = {(int(s.major_cn), int(s.minor_cn)) for s in top}
        # always keep high-CN units in the showcase range (major 5-7) even if below the size cut — but
        # only ONE (best) per STATE, and only for states not already in the top-N. Chosen by ESS, not
        # member count: at high CN a state is often a single segment.
        hi_by_state = {}
        for s in sorted(allsegs, key=lambda x: -_ess(x)):
            cn5 = (int(s.major_cn), int(s.minor_cn))
            if (5 <= cn5[0] <= 7 and id(s) not in top_ids
                    and cn5 not in hi_by_state and cn5 not in top_states):
                hi_by_state[cn5] = s
        allsegs = top + list(hi_by_state.values())
    allsegs.sort(key=lambda s: (int(s.major_cn), int(s.minor_cn), -getattr(s, "num_segs", 1)))
    fc, counts = {}, {}
    for s in allsegs:
        cn = (int(s.major_cn), int(s.minor_cn)); counts[cn] = counts.get(cn, 0) + 1
        fc[f"({cn[0]},{cn[1]}) #{counts[cn]}"] = s
    labels = list(fc.keys()); ntree = len(labels)

    with mpl.rc_context({"font.size": FS, "axes.linewidth": 0.5, "svg.fonttype": "none"}):
        # layout: overview | gap | trees... | hist(pooled). The segment-level histogram is gone — the
        # pooled-cluster histogram is the one that matters, and dropping it buys width for the trees.
        width_ratios = [W_OVERVIEW, W_GAP] + [W_TREE] * ntree + [W_HIST]
        fig_w = max(FIG_W_IN, sum(width_ratios) * 0.52)   # scale width with #trees
        fig = plt.figure(figsize=(fig_w, FIG_H))
        gs = fig.add_gridspec(1, 2 + ntree + 1, width_ratios=width_ratios, wspace=0.0)
        fig.subplots_adjust(left=0.055, right=0.945, bottom=0.10, top=0.76)

        # ---- overview (individual segment timing) ----
        ax_ov = fig.add_subplot(gs[0]); ax_ov.set_ylim(0, 1)
        segs_ord, offsets = compute_offsets(g.segments)
        plot_tau_stack(ax_ov, segs_ord, offsets, ess_thresh=0, cluster_times=[],
                       segment_cluster_ids={}, color_by_cluster=False)
        segment_cluster_plotting(ax_ov, cg, segs_ord, offsets, final_clusters=fc)
        states = sorted({(int(s.major_cn), int(s.minor_cn)) for s in g.segments
                         if pick_best_key(s) is not None and int(s.major_cn) != 1})
        nbands = {}
        for (maj, mn) in states:
            seg = next((s for s in g.segments if int(s.major_cn) == maj and int(s.minor_cn) == mn
                        and pick_best_key(s) is not None), None)
            m = extract_mat(seg, pick_best_key(seg), boot_id=0) if seg is not None else None
            nbands[(maj, mn)] = int(m.shape[1]) if (m is not None and m.size) else max(maj, 1)
        # drop the internal "Cluster N"/timepoint legends (they live on nested inset axes)
        for _lg in fig.findobj(_mleg.Legend):
            _lg.remove()
        ax_ov.set_ylabel("Molecular time (early → late)", fontsize=6)

        # ---- trees (merged / pooled) ----
        # the dominant (2,2) unit (most segments) collapses both gains onto one WGD; secondary (2,2)
        # clusters in a double-WGD sample may split one gain per doubling (collapse_22=False).
        _segs22 = [(lab, s) for lab, s in fc.items() if (int(s.major_cn), int(s.minor_cn)) == (2, 2)]
        dominant_22 = max(_segs22, key=lambda ls: getattr(ls[1], "num_segs", 1))[0] if _segs22 else None
        tree_axes, tree_times, modes = [], [], []
        max_wgd = int((ev_mrg.classification == "WGD").sum())   # detected count = upper bound
        for ci, label in enumerate(labels):
            seg = fc[label]; ax_t = fig.add_subplot(gs[2 + ci], sharey=ax_ov); tree_axes.append(ax_t)
            key = pick_best_key(seg); cn = (int(seg.major_cn), int(seg.minor_cn))
            _lw = _lw_for_ess(_ess(seg))
            ev_tpl = _events_for_route(key)
            collapse_22 = (label == dominant_22)
            if preset == "blocks":
                et, _di, mode = _pick_draw_wgd_preset(seg, max_wgd, wgd_times, min_ess=min_ess)
            else:
                et, _di, mode = _pick_draw_sim_span(seg, wgd_times)
            if et is None:   # fall back to the legacy WGD-distance heuristic only if the preset can't decide
                et, _ = _pick_draw_for_tree(seg, wgd_times, events_tpl=ev_tpl, collapse_22=collapse_22)
                mode = "legacy"
            modes.append((label, mode))
            et = np.clip(np.asarray(et, float), 0.0, 1.0)   # cumsum can hit 1.0000000002
            tree_times.append((et, _ess(seg)))
            evs = copy.deepcopy(ev_tpl)
            for j, ev in enumerate(evs):
                ev["time"] = float(et[j])
            maj_root, min_root = build_allele_trees(evs, cn[0], cn[1])
            color = _cmap_for_major(cn[0])(0.6)
            if blur:   # DensiTree overlay: every polytope solution faintly behind the representative
                draws = [d for d in seg.timing_result[key].get("draws", []) if d.get("boot_id", 0) == 0]
                if len(draws) > 1:
                    a = max(0.03, min(0.18, 3.0 / len(draws)))
                    for d in draws:
                        cs = np.clip(np.cumsum(order_t(d["t"]))[:-1], 0.0, 1.0)
                        evs_b = copy.deepcopy(ev_tpl)
                        for j, ev in enumerate(evs_b):
                            ev["time"] = float(cs[j])
                        mb, nb = build_allele_trees(evs_b, cn[0], cn[1])
                        plot_route(ax_t, mb, nb, major_color=color, minor_color=color,
                                   linewidth_scale=_lw * 0.9, spacing=TREE_SPACING, alpha=a, zorder=2)
            plot_route(ax_t, maj_root, min_root, major_color=color, minor_color=color,
                       linewidth_scale=_lw, spacing=TREE_SPACING, zorder=5)
            xd = [x for ln in ax_t.get_lines() for x in ln.get_xdata()]
            if xd:
                ax_t.set_xlim(min(xd) - 0.35, max(xd) + 0.35)
            ax_t.set_xticks([]); ax_t.set_yticks([])
            tl = getattr(seg, "total_length", seg.end - seg.start)
            ax_t.set_title(f"({cn[0]},{cn[1]})\n{tl/HG19_TOTAL*100:.1f}%", fontsize=4.5, pad=2)
            for sp in ("top", "right", "bottom", "left"):
                ax_t.spines[sp].set_visible(False)

        fig.canvas.draw()   # positions must be final before the key is anchored to the tree block
        lg = _cn_legend_above_trees(fig, tree_axes, states, nbands)
        if lg is not None:
            _cluster_fill_key(fig, lg, max(counts.values()) if counts else 1)

        # ---- pooled timing histogram, to the RIGHT of the trees ----
        ax_h2 = fig.add_subplot(gs[-1], sharey=ax_ov)
        build_timing_hist(ax_h2, tree_times, flip=False)

        # Event lines are SINGLE figure-spanning artists so the dash/dot pattern stays continuous and
        # evenly spaced across the panels they cross (a per-axis axhline restarts it at each boundary).
        fig.canvas.draw()

        def _span_line(left_ax, right_ax, et_, color, ls, lw):
            p0 = left_ax.get_position(); p1 = right_ax.get_position()
            y = p0.y0 + et_ * p0.height          # all panels share the molecular-time y-axis (0..1)
            fig.add_artist(mlines.Line2D([p0.x0, p1.x1], [y, y], transform=fig.transFigure,
                                         color=color, ls=ls, lw=lw, alpha=0.9, zorder=6))

        for _, r in ev_ind.iterrows():      # individual events (dashed) across the overview
            c = _row_color(r["classification"])
            _span_line(ax_ov, ax_ov, float(r["event_time_frac"]), c, "--", 0.9)
            ax_ov.text(1.02, r["event_time_frac"], _ev_label(r),
                       transform=ax_ov.get_yaxis_transform(), fontsize=4.5, va="center",
                       ha="left", color=c)

        if tree_axes:
            for _, r in ev_mrg.iterrows():  # merged events (dotted) across trees + pooled histogram
                c = _row_color(r["classification"])
                _span_line(tree_axes[0], ax_h2, float(r["event_time_frac"]), c, ":", 1.1)
                ax_h2.text(1.04, r["event_time_frac"], _ev_label(r, "merged "),
                           transform=ax_h2.get_yaxis_transform(), fontsize=4.5, va="center",
                           ha="left", color=c)

        h = [mlines.Line2D([], [], color="dimgrey", lw=1.2, label="major allele"),
             mlines.Line2D([], [], color="dimgrey", lw=1.2, ls="--", label="minor allele"),
             mlines.Line2D([], [], color="#333", lw=0.9, ls="--", label="individual event (raw)"),
             mlines.Line2D([], [], color="#333", lw=1.1, ls=":", label="merged event (pooled)")]
        fig.legend(handles=h, fontsize=5, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4,
                   frameon=False, handlelength=1.6, columnspacing=1.2, handletextpad=0.4)

    from collections import Counter
    info = dict(n_trees=ntree, n_individual_events=len(ev_ind), n_merged_events=len(ev_mrg),
                max_wgd=max_wgd, preset=preset, modes=dict(Counter(m for _, m in modes)))
    if verbose:
        print(f"  overview: {ntree} pooled units, {len(ev_ind)} individual / {len(ev_mrg)} merged "
              f"events, draw selection {info['modes']}")
    return fig, info


def save_overview(genome, clustered_genome, events, merged_events, out_stem, **kw):
    """Write `<out_stem>.png` and `<out_stem>.svg`. Returns the paths."""
    fig, info = overview_figure(genome, clustered_genome, events, merged_events, **kw)
    png, svg = Path(f"{out_stem}.png"), Path(f"{out_stem}.svg")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return [png, svg], info
