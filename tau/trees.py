#!/usr/bin/env python3
"""Allele tree construction and visualization for CN segment evolutionary histories."""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl

from tau.utils import pick_best_key, extract_mat, order_t


# ---------------------------------------------------------------------------
# Node / Tree data structures
# ---------------------------------------------------------------------------

class Node:
    """A node in an allele copy-number tree."""

    def __init__(self, time=0.0, alignment=None, parent=None, mult=None):
        self.time = float(time)
        self.branches = []
        self.alignment = alignment  # 'L' or 'R' for plotting
        self.parent = parent
        self.mults = [] if mult is None else [int(mult)]

    @property
    def mult(self):
        return self.mults[-1] if self.mults else None

    def add_branch(self, t, alignment=None, mult=None):
        t = float(t)
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be in [0,1]")
        if t < self.time:
            raise ValueError("split time must be >= parent branch birth time")
        child = Node(time=t, alignment=alignment, parent=self, mult=mult)
        self.branches.append(child)
        self.branches.sort(key=lambda x: x.time)
        return child


class Tree:
    def __init__(self):
        self.trunk = Node(0.0)


# ---------------------------------------------------------------------------
# Tree building helpers
# ---------------------------------------------------------------------------

def sort_nodes(node):
    """Depth-first left/right ordering of nodes for plotting."""
    if len(node.branches) == 0:
        return [node]
    lefthand = [
        b for branch in node.branches if branch.alignment == "L"
        for b in sort_nodes(branch)
    ]
    righthand = [
        b for branch in node.branches if branch.alignment == "R"
        for b in sort_nodes(branch)
    ]
    return [*lefthand, node, *righthand]


def _find_parent_with_mult(root, parent_mult, align="L"):
    nodes = sort_nodes(root)
    if align == "R":
        nodes = list(reversed(nodes))
    for n in nodes:
        if n.mult == parent_mult:
            return n
    return None


def _apply_event_to_root(root, parent_mult, split, t, align="L"):
    k, l = map(int, split)
    parent = _find_parent_with_mult(root, parent_mult, align=align)
    if parent is None:
        raise ValueError(
            f"No active branch with multiplicity {parent_mult} found at time {t}"
        )
    cont_mult, split_mult = min(k, l), max(k, l)
    parent.mults.append(int(cont_mult))
    non_align = "R" if align == "L" else "L"
    child_align = (
        non_align
        if parent.branches and parent.branches[-1].alignment == align
        else align
    )
    parent.add_branch(t, alignment=child_align, mult=int(split_mult))


def build_allele_trees(events, start_major, start_minor, times=None):
    """Build major/minor allele trees from an ordered list of events.

    Parameters
    ----------
    events : list[dict]
        Each dict has keys ``allele``, ``parent_mult``, ``split``.
    start_major, start_minor : int
        Starting copy numbers for the two alleles.
    times : list[float], optional
        If provided, events consume times from this list in order.

    Returns
    -------
    maj_root, min_root : Node
    """
    maj_root = Node(0.0, mult=int(start_major))
    min_root = Node(0.0, mult=int(start_minor))

    times_list = list(times) if times is not None else None
    for e in events:
        allele = e["allele"]
        parent_mult = e["parent_mult"]
        split = e["split"]
        t = float(times_list.pop(0)) if times_list is not None else float(e["time"])
        if allele == "major":
            root, align = maj_root, "L"
        else:
            root, align = min_root, "R"
        _apply_event_to_root(root, parent_mult, split, t, align=align)

    return maj_root, min_root


# ---------------------------------------------------------------------------
# Single-tree plotting (ported from notebook)
# ---------------------------------------------------------------------------

def plot_route(ax, maj_node, min_node, x_offset=0,
               major_highlights=None, minor_highlights=None,
               major_color="crimson", minor_color="royalblue",
               linewidth_scale=1.0, spacing=1.0, alpha=1.0, zorder=None,
               minor_dashes=None):
    """Plot an allele tree pair on *ax*.

    Parameters
    ----------
    spacing : float
        Horizontal distance between adjacent vertical lines (default 1.0).
        Use smaller values (e.g. 0.5) to pack lines closer together.
    alpha : float
        Line opacity (default 1.0). Use a small value (e.g. 0.06) to overlay
        many polytope solutions as a DensiTree-style blur for underdetermined
        routes; the node x-layout is topology-driven (identical across draws),
        so only the event-time (y) positions vary across overlaid draws.
    minor_dashes : tuple, optional
        Explicit (on, off) dash lengths in points for the minor-allele lines.
        The default ``"--"`` scales with linewidth, so short branches fit only
        one or two dashes; pass e.g. ``(2.4, 1.4)`` for a finer, denser pattern.

    Minor trunks are drawn from y=1 DOWN to the gain time so their dash phase is
    anchored at the top. That matters when overlaying many polytope solutions:
    drawing upward from a per-draw gain time gives each draw a different dash
    phase, and the overlaid gaps fill in until a dashed trunk looks solid.
    """
    sorted_maj = sort_nodes(maj_node)
    sorted_min = sort_nodes(min_node)
    lw = 2 * linewidth_scale
    zk = {} if zorder is None else {"zorder": zorder}
    # None -> dashed (legacy default); False -> solid; a tuple -> that dash pattern
    mls = "--" if minor_dashes is None else ("-" if minor_dashes is False else (0, tuple(minor_dashes)))

    major_times = sorted([n.time for n in sorted_maj])
    minor_times = sorted([n.time for n in sorted_min])

    for i, node in enumerate(sorted_maj, 1):
        node.index = i * spacing
    for i, node in enumerate(sorted_min, 1):
        node.index = (i + len(sorted_maj)) * spacing

    for node in sorted_maj:
        gain = node.time
        idx = node.index + x_offset
        mtime_index = major_times.index(gain)
        if major_highlights is not None and mtime_index in major_highlights:
            ax.plot(idx, gain, marker="o", color="black", zorder=10, markersize=6)
        ax.plot([idx, idx], [gain, 1], lw=lw, alpha=alpha, color=major_color, **zk)
        if node.parent is not None:
            pidx = node.parent.index + x_offset
            ax.plot([idx, pidx], [gain, gain], lw=lw, alpha=alpha, color=major_color, **zk)

    for node in sorted_min:
        if node.mult == 0:
            continue  # no minor allele (e.g. (M,0) states) — draw no trunk/branch
        gain = node.time
        idx = node.index + x_offset
        mtime_index = minor_times.index(gain)
        if minor_highlights is not None and mtime_index in minor_highlights:
            ax.plot(idx, gain, marker="o", color="black", zorder=10, markersize=6)
        ax.plot([idx, idx], [1, gain], lw=lw, alpha=alpha, color=minor_color, linestyle=mls, **zk)
        if node.parent is not None:
            pidx = node.parent.index + x_offset
            ax.plot([idx, pidx], [gain, gain], lw=lw, alpha=alpha, color=minor_color, linestyle=mls, **zk)


# ---------------------------------------------------------------------------
# Cluster identification + tree building from route_metadata
# ---------------------------------------------------------------------------

def _get_final_segment_clusters(clustered_genome):
    """Identify major segment clusters from a clustered genome."""
    big_clusters = {}
    for major in range(1, 8):
        for minor in range(0, major + 1):
            segs = [
                seg for seg in clustered_genome.segments
                if seg.major_cn == major and seg.minor_cn == minor
            ]
            for seg in segs:
                if getattr(seg, "num_segs", 1) > 2:
                    best_key = pick_best_key(seg)
                    if best_key is None:
                        continue
                    if seg.timing_result[best_key]["qc"][0]["dist_rel"] > 0.05:
                        continue
                    big_clusters[seg.seg_id] = seg

    big_clusters_sorted = dict(sorted(
        big_clusters.items(),
        key=lambda x: (x[1].major_cn, x[1].minor_cn, -getattr(x[1], "num_segs", 0)),
    ))
    final = {}
    counts = {}
    for cluster_id, seg in big_clusters_sorted.items():
        major, minor = seg.major_cn, seg.minor_cn
        counts[(major, minor)] = counts.get((major, minor), 0) + 1
        final[f"({major},{minor}) Cluster {counts[(major, minor)]}"] = seg
    return final


def _pick_draw_for_tree(cluster_seg, wgd_times, events_tpl=None, collapse_22=True):
    """Choose the draw whose event times best represent a WGD collapse.

    Priority when WGD times are supplied:

    1. **(2,2), collapse_22=True** (the dominant (2,2) cluster) — maximise
       ``cumsum[0]`` → enforces t₂=0 (both allele gains simultaneous at the
       single WGD time = N₂/2).
    1b. **(2,2), collapse_22=False** with ≥2 WGD times (a secondary (2,2)
       cluster in a double-WGD sample) — snap EACH gain to its nearest WGD via
       a per-event nearest-WGD score, so one allele gain lands on WGD1 and the
       other on WGD2 (the "split across the two doublings" history). Note the
       general heuristic in (2) minimises distance of ALL events to a SINGLE
       nearest WGD, so it would collapse rather than split — hence the
       dedicated per-event score here.
    2. **Other underdetermined states** — score each draw by proximity of
       events to t_wgd *plus* the size of any adjacent different-allele gap
       straddling t_wgd.  Minimising this score collapses the cross-allele gap
       closest to the WGD time (zero-enforcement for simultaneous gains).
    3. **Fallback** — median draw (no WGD context or single draw).

    Returns ``(event_times_array, draw_idx)`` or ``(None, None)`` on failure.
    """
    key = pick_best_key(cluster_seg)
    if key is None:
        return None, None
    draws = [d for d in cluster_seg.timing_result[key].get("draws", [])
             if d.get("boot_id", 0) == 0]
    if not draws:
        return None, None

    all_cumsums = np.array([np.cumsum(order_t(d["t"]))[:-1] for d in draws])

    if len(draws) == 1:
        return all_cumsums[0], 0

    if not wgd_times:
        med = len(draws) // 2
        return all_cumsums[med], med

    maj = int(cluster_seg.major_cn)
    mn  = int(cluster_seg.minor_cn)

    # (2,2) in a double-WGD sample: dominant cluster collapses both gains onto ONE WGD (t₂=0);
    # a secondary cluster snaps each gain to its nearest WGD (one gain per doubling).
    if maj == 2 and mn == 2:
        if collapse_22 or len(wgd_times) < 2:
            idx = int(np.argmax(all_cumsums[:, 0]))          # t₂=0, both events at N₂/2
            return all_cumsums[idx], idx
        scores = np.array([                                   # each event -> nearest WGD, summed
            sum(min((float(e) - wt) ** 2 for wt in wgd_times) for e in cs)
            for cs in all_cumsums
        ])
        best = int(np.argmin(scores))
        return all_cumsums[best], best

    # Score each draw by mean squared distance of ALL events from WGD —
    # pure MSD, no base/gap terms.  Selects the draw where WGD-associated
    # gains are most balanced around the WGD time.  For mixed-history
    # segments (e.g. (4,2) with 1 PGD + 3 WGD events) the PGD event adds
    # a large but constant penalty across draws so selection is still driven
    # by the WGD-adjacent events.
    scores = np.array([
        min(float(np.mean((cs - wt) ** 2)) for wt in wgd_times)
        for cs in all_cumsums
    ])

    best = int(np.argmin(scores))
    return all_cumsums[best], best


def _build_tree_for_cluster(seg, route_metadata, draw_idx=None):
    """Build allele trees for a single clustered segment.

    Parameters
    ----------
    draw_idx : int or None
        Which draw to use for event times.  ``None`` (default) uses the
        median across all draws; an integer index selects a specific draw
        (negative indexing supported, e.g. ``-1`` for the last draw).

    Returns ``(maj_root, min_root)`` or ``None``.
    """
    best_key = pick_best_key(seg)
    if best_key is None:
        return None

    cn_key = (int(seg.major_cn), int(seg.minor_cn))
    if cn_key not in route_metadata:
        return None

    route_dict = route_metadata[cn_key]
    if best_key not in route_dict:
        return None

    route_items = route_dict[best_key]
    route_info = route_items[0] if isinstance(route_items, list) else route_items

    events = route_info.get("events", [])
    if not events:
        return None

    mat = extract_mat(seg, best_key, boot_id=0)
    if mat is None or mat.size == 0:
        return None

    cum = np.cumsum(mat, axis=1)
    if draw_idx is None:
        event_times = list(np.median(cum, axis=0)[:-1])
    else:
        event_times = list(cum[draw_idx][:-1])

    if len(event_times) != len(events):
        return None

    try:
        return build_allele_trees(
            events, int(seg.major_cn), int(seg.minor_cn), times=event_times
        )
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Main plotting entry point
# ---------------------------------------------------------------------------

def _draw_wgd_split(seg, two_wgd):
    """Pick the draw whose first two gain times best match the two distinct WGD times
    (one gain at the first WGD, the other at the second) instead of collapsing both to one.
    Returns the draw's event-time list, or None."""
    key = pick_best_key(seg)
    if key is None:
        return None
    draws = [d for d in seg.timing_result[key].get("draws", []) if d.get("boot_id", 0) == 0]
    if not draws:
        return None
    cums = np.array([np.cumsum(order_t(d["t"]))[:-1] for d in draws])
    if cums.shape[1] < 2:
        return None
    w = np.array(sorted(two_wgd)[:2])
    scores = np.sum((np.sort(cums[:, :2], axis=1) - w) ** 2, axis=1)
    return list(cums[int(np.argmin(scores))])


def plot_cluster_trees(clustered_genome, route_metadata,
                       cluster_times=None, output_file=None,
                       cluster_summary_df=None, save_svg=False, draw_overrides=None):
    """Plot allele trees for all segment clusters side by side.

    Parameters
    ----------
    clustered_genome : Genome
        Clustered genome (from ``make_clustered_genome``).
    route_metadata : dict
        Route metadata keyed by ``(major_cn, minor_cn)``.
    cluster_times : array-like, optional
        Cluster event times (legacy; superseded by *cluster_summary_df*).
    output_file : str, optional
        If given, save PDF (and SVG if *save_svg*) to this path.
    cluster_summary_df : pd.DataFrame, optional
        Output of ``cluster_times_bottomup``.  When supplied, event times,
        chrom-specific names, and genome fractions are taken from this table
        and drawn as reference lines with a matching legend.  WGD enforcement
        (zero cross-allele gap collapse) is also applied to draw selection.
    save_svg : bool
        Also save an SVG alongside the PDF.  Sets SVG-friendly font rcParams.
    """
    from tau.plotting import OVERVIEW_SADJ, OVERVIEW_FIG_HEIGHT, _event_label, _event_color, _cmap_for_major

    HG19_TOTAL = 3_095_677_412

    final_clusters = _get_final_segment_clusters(clustered_genome)

    # Determine WGD times for draw selection
    wgd_times = []
    if cluster_summary_df is not None and not cluster_summary_df.empty:
        wgd_rows = cluster_summary_df[cluster_summary_df["classification"] == "WGD"]
        wgd_times = wgd_rows["time"].tolist()

    # Build trees using WGD-aware draw selection
    tree_data = []   # (label, seg, maj_root, min_root, pct_hg19, pct_trees, color)
    trees_total_len = sum(
        getattr(s, "total_length", s.end - s.start) for s in final_clusters.values()
    )
    cn_counts = {}
    for label, seg in final_clusters.items():
        key = pick_best_key(seg)
        if key is None:
            continue
        cn_key = (int(seg.major_cn), int(seg.minor_cn))
        route_dict = route_metadata.get(cn_key, {})
        if key not in route_dict:
            continue
        variations  = route_dict[key]
        event_set   = variations[0] if isinstance(variations, list) else variations
        events_tpl  = event_set.get("events", [])
        if not events_tpl:
            continue

        ov = (draw_overrides or {}).get(label)
        event_times = None
        if ov == "wgd_split" and len(wgd_times) >= 2:
            event_times = _draw_wgd_split(seg, wgd_times)
        if event_times is None:
            event_times, _ = _pick_draw_for_tree(seg, wgd_times, events_tpl=events_tpl)
        if event_times is None or len(event_times) != len(events_tpl):
            # Fall back to legacy _build_tree_for_cluster
            result = _build_tree_for_cluster(seg, route_metadata)
            if result is None:
                continue
            maj_root, min_root = result
        else:
            events = copy.deepcopy(events_tpl)
            for j, ev in enumerate(events):
                ev["time"] = float(event_times[j])
            try:
                maj_root, min_root = build_allele_trees(
                    events, int(seg.major_cn), int(seg.minor_cn)
                )
            except (ValueError, IndexError):
                continue

        tl        = getattr(seg, "total_length", seg.end - seg.start)
        pct_hg19  = tl / HG19_TOTAL * 100
        pct_trees = tl / max(trees_total_len, 1) * 100

        cn_counts[cn_key] = cn_counts.get(cn_key, 0) + 1
        cmap  = _cmap_for_major(cn_key[0])
        color = cmap(0.6)

        tree_data.append((label, seg, maj_root, min_root, pct_hg19, pct_trees, color))

    if not tree_data:
        print("No trees could be built.")
        return None

    n = len(tree_data)
    n_muts_arr = np.array([sum(d[1].N_counts) for d in tree_data], float)
    lw_scales  = 0.4 + 2.0 * (n_muts_arr / max(n_muts_arr.max(), 1.0))

    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, n, figsize=(max(1.5 * n, 7.0), OVERVIEW_FIG_HEIGHT), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (label, seg, maj_root, min_root, pct_hg19, pct_trees, color) in enumerate(tree_data):
        ax = axes[i]
        # y=0 at bottom (early), y=1 at top (late) — matches overview orientation
        plot_route(ax, maj_root, min_root, x_offset=0,
                   major_color=color, minor_color=color,
                   linewidth_scale=lw_scales[i], spacing=0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])

        x_data = [x for ln in ax.get_lines() for x in ln.get_xdata()]
        if x_data:
            ax.set_xlim(min(x_data) - 0.4, max(x_data) + 0.4)

        cn_label = label.split(" Cluster ")[0]  # e.g. "(4,2)"
        ax.set_title(f"{cn_label}\n{pct_hg19:.1f}% of genome",
                     fontsize=7, pad=3)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])

    axes[0].set_ylabel("Time (early → late)", fontsize=7)

    # Align bottom/top with overview so 0-1 y-axes line up side by side
    tree_sadj = dict(left=0.08, right=OVERVIEW_SADJ["right"],
                     bottom=OVERVIEW_SADJ["bottom"], top=OVERVIEW_SADJ["top"],
                     wspace=0.10)
    fig.subplots_adjust(**tree_sadj)
    fig.suptitle("Inferred Trees", fontsize=9, fontweight="bold", y=1.06)
    fig.canvas.draw()

    # Draw event reference lines across the subplot region
    ax0, axN = axes[0], axes[-1]
    x0_fig = fig.transFigure.inverted().transform(ax0.transAxes.transform((0, 0)))[0]
    x1_fig = fig.transFigure.inverted().transform(axN.transAxes.transform((1, 0)))[0]
    y0_d   = ax0.transData.transform((0, 0))[1]
    y1_d   = ax0.transData.transform((0, 1))[1]
    y0_fig = fig.transFigure.inverted().transform((0, y0_d))[1]
    y1_fig = fig.transFigure.inverted().transform((0, y1_d))[1]

    ev_rows = (cluster_summary_df.sort_values("time").iterrows()
               if cluster_summary_df is not None and not cluster_summary_df.empty
               else enumerate([{"time": ct, "classification": f"Event {i}",
                                 "gf": 0, "pct_of_theoretical_genome": 0}
                                for i, ct in enumerate(cluster_times or [])]))
    for idx, row in ev_rows:
        c     = _event_color(row)
        t     = row["time"]
        y_fig = y0_fig + t * (y1_fig - y0_fig)
        fig.add_artist(mpl.lines.Line2D(
            [x0_fig, x1_fig], [y_fig, y_fig],
            transform=fig.transFigure,
            color=c, linestyle="--", linewidth=1.5, alpha=0.8,
        ))
        pct = row.get("pct_of_theoretical_genome", row.get("gf", 0) * 100)
        fig.text(
            x1_fig + 0.01, y_fig,
            f"t={t:.2f}  ({pct:.1f}%)",
            transform=fig.transFigure,
            fontsize=7, va="center", ha="left", color=c,
        )

    # Legend: allele style only, no event lines
    maj_h = mlines.Line2D([], [], color="dimgrey", lw=1.5, label="Major allele (solid)")
    min_h = mlines.Line2D([], [], color="dimgrey", lw=1.5, ls="--", label="Minor allele (dashed)")
    axN.legend(handles=[maj_h, min_h], fontsize=7,
               loc="upper left", bbox_to_anchor=(1.02, 1.0),
               bbox_transform=axN.transAxes, framealpha=0.85, borderaxespad=0)

    if output_file:
        mpl.rcParams.update({"svg.fonttype": "none", "text.usetex": False})
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        if save_svg:
            svg_path = str(output_file).rsplit(".", 1)[0] + ".svg"
            fig.savefig(svg_path, dpi=300, bbox_inches="tight", transparent=True)
    else:
        plt.show()
    return fig
