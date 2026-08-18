#!/usr/bin/env python3
"""Visualization functions for Tau timing analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from tau.core import Segment, Genome
from tau.utils import pick_best_key, extract_mat

HG19_ORDER = [f"{i}" for i in range(1, 23)] + ["X", "Y"]
HG19_SIZES = {
    "1": 249250621,
    "2": 243199373,
    "3": 198022430,
    "4": 191154276,
    "5": 180915260,
    "6": 171115067,
    "7": 159138663,
    "8": 146364022,
    "9": 141213431,
    "10": 135534747,
    "11": 135006516,
    "12": 133851895,
    "13": 115169878,
    "14": 107349540,
    "15": 102531392,
    "16": 90354753,
    "17": 81195210,
    "18": 78077248,
    "19": 59128983,
    "20": 63025520,
    "21": 48129895,
    "22": 51304566,
}


def normalize_chrom(chrom):
    """Normalize chromosome name by stripping 'chr' prefix if present."""
    chrom_str = str(chrom)
    if chrom_str.startswith("chr"):
        return chrom_str[3:]
    return chrom_str


def compute_offsets(segments, chrom_sizes=HG19_SIZES, chrom_order=HG19_ORDER):
    """Compute x-axis offsets for genome-wide plotting."""
    order_index = {c: i for i, c in enumerate(chrom_order)}
    segs = sorted(
        segments,
        key=lambda s: (
            order_index.get(normalize_chrom(s.chrom), 10**9),
            int(s.start),
            int(s.end),
        ),
    )

    offsets = {}
    prev = None
    offset = 0
    for seg in segs:
        chrom = normalize_chrom(seg.chrom)
        if chrom != prev and prev is not None:
            offset += chrom_sizes.get(prev, 0)
        offsets[seg.seg_id] = offset
        prev = chrom
    return segs, offsets


def seg_signature(idx_arr):
    """Convert index array to tuple signature."""
    return tuple(int(x) for x in idx_arr)


COLORS = [
    plt.cm.Greys,
    plt.cm.Greens,
    plt.cm.Oranges,
    plt.cm.Purples,
    plt.cm.Blues,
    plt.cm.Reds,
    plt.cm.Greys,    # major 7 → grays
]

_COLORS_BY_MAJ = dict(zip(range(1, len(COLORS) + 1), COLORS))

GREY_RGBA = (0.70, 0.70, 0.70, 1.0)


def _cmap_for_major(maj: int):
    """Get colormap for a given major copy number."""
    if maj in _COLORS_BY_MAJ:
        return _COLORS_BY_MAJ[maj]
    # cycle through known maps for maj >= 6
    keys = sorted(_COLORS_BY_MAJ.keys())
    return _COLORS_BY_MAJ[keys[(maj - 1) % len(keys)]]


def _gradated_colors(n, cmap):
    """Generate n gradations from a single color family."""
    if n <= 0:
        return []
    return [cmap(i / max(n, 1)) for i in range(1, n + 1)]


def _colors_for_segment(maj: int, n: int, cmap):
    """
    Return the per-interval colors for a segment.
    If major CN == 1, force a flat gray (no gradient).
    Otherwise, return the usual gradated colors from the family cmap.
    """
    if n <= 0:
        return []
    if maj == 1:
        return [GREY_RGBA] * n
    # default: gradient
    return [cmap(i / max(n, 1)) for i in range(1, n + 1)]


order_index = {c: i for i, c in enumerate(HG19_ORDER)}


def plot_tau_stack(
    ax,
    segments,
    offsets,
    ess_thresh=10,
    cluster_times=[],
    segment_cluster_ids={},
    output_file=None,
    color_by_cluster=True,
):
    """Plot stacked timing distribution across genome."""
    num_clusters = len(cluster_times)

    # cluster colors should be from a distinct colormap
    cluster_cmap = plt.cm.tab10
    cluster_colors = [cluster_cmap(i) for i in range(num_clusters)]

    prev_chrom = None
    chrom_start_x = None
    chrom_offsets = {}
    last_autosome_seg = None
    for seg in segments:
        chrom = normalize_chrom(seg.chrom)
        xoff = offsets[seg.seg_id]
        chrom_offsets[chrom] = xoff
        x0 = int(seg.start) + xoff
        x1 = int(seg.end) + xoff

        if chrom not in [str(i) for i in range(1, 23)]:
            continue

        last_autosome_seg = seg

        if chrom != prev_chrom:
            # draw boundary line (except first chromosome)
            if prev_chrom is not None:
                ax.axvline(x0, color="black", ls="--", lw=0.5, zorder=6)

                # label previous chromosome
                mid = (chrom_start_x + x0) / 2
                ax.text(
                    mid,
                    -0.02,
                    prev_chrom,
                    ha="center",
                    va="top",
                    fontsize=4,
                    clip_on=False,
                )

            prev_chrom = chrom
            chrom_start_x = x0

        ess = float(np.sum(getattr(seg, "N_counts", np.zeros(0))))
        if ess < ess_thresh:
            continue

        key = pick_best_key(seg)
        if key is None:
            continue

        mat = extract_mat(seg, key, boot_id=0)
        if mat is None or mat.size == 0:
            continue

        xoff = offsets[seg.seg_id]
        x0, x1 = int(seg.start) + xoff, int(seg.end) + xoff
        K = mat.shape[1]
        cmap = _cmap_for_major(int(seg.major_cn))
        cols = _colors_for_segment(int(seg.major_cn), K, cmap)

        x_vec = np.linspace(x0, x1, max(2, mat.shape[0]))
        ax.stackplot(x_vec, *mat.T, colors=cols, baseline="zero", zorder=2)

        cum = np.cumsum(mat, axis=1)  # shape (n_x, K)

        # cluster ids per boundary (length K-1)
        cids = segment_cluster_ids.get(seg.seg_id, None)
        if not color_by_cluster:
            continue

        for b in range(K - 1):
            y = cum[:, b]

            if cids is None:
                continue

            cid = int(cids[b])
            color = "black" if cid == -1 else cluster_colors[cid]

            if mat.shape[0] == 1:
                # single x-point: draw as a segment across the region
                ax.plot(
                    [x0, x1],
                    [float(y[0]), float(y[0])],
                    color=color,
                    lw=2.0,
                    alpha=0.95,
                    zorder=5,
                )
            else:
                # many x-points: build matching x_vec and draw curve
                x_line = np.linspace(x0, x1, mat.shape[0])
                ax.plot(x_line, y, color=color, lw=1, alpha=0.95, zorder=5)

    if prev_chrom is not None:
        last_seg = last_autosome_seg if last_autosome_seg is not None else segments[-1]
        end_x = int(last_seg.end) + offsets[last_seg.seg_id]
        mid = (chrom_start_x + end_x) / 2
        ax.text(
            mid, -0.02, prev_chrom, ha="center", va="top", fontsize=4, clip_on=False
        )

    # add vertical lines for cluster times
    for cid, ct in enumerate(cluster_times):
        ax.axhline(
            ct, color=cluster_colors[cid], linestyle="--", linewidth=3, alpha=0.8, zorder=5
        )

    ax.margins(x=0)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Normalised time", fontsize=8)
    if output_file:
        plt.savefig(output_file)


def timepoint_cluster_plotting(
    fig, ax, g, cluster_times, segment_cluster_ids, original_times, cluster_definitions
):
    """Plot timepoint cluster distributions."""
    per_cluster_times = {}
    per_cluster_weights = {}
    for seg_id, (cluster_assignments, og_times) in zip(
        segment_cluster_ids.keys(),
        zip(segment_cluster_ids.values(), original_times.values()),
    ):
        seg = next(seg for seg in g.segments if seg.seg_id == seg_id)
        for c, t in zip(cluster_assignments, og_times):
            if c not in per_cluster_times:
                per_cluster_times[c] = []
                per_cluster_weights[c] = []
            per_cluster_times[c].append(t)
            num_mutations = sum(seg.N_counts)
            seg_length = seg.end - seg.start
            per_cluster_weights[c].append(num_mutations * seg_length)

    num_clusters = len(cluster_times)
    cluster_cmap = plt.cm.tab10
    num_mutations_per_seg = {seg.seg_id: sum(seg.N_counts) for seg in g.segments}
    seg_length = {seg.seg_id: seg.end - seg.start for seg in g.segments}
    # Build color dict keyed by actual cluster IDs (handles non-sequential IDs)
    all_pos_cids = sorted(cid for cid in per_cluster_times if cid >= 0)
    cluster_colors = {cid: cluster_cmap(i % 10) for i, cid in enumerate(all_pos_cids)}
    # add timepoint density plots for each cluster on the right side of the main figure
    inset_width = 0.15
    fig_inset = ax.inset_axes([1, 0, inset_width, 1])

    for cid, times in per_cluster_times.items():
        if cid < 0:
            continue
        # make histplot using seaborn
        weights = per_cluster_weights[cid]
        sns.histplot(
            y=times,
            weights=weights,
            bins=7,
            alpha=0.7,
            color=cluster_colors[cid],
            fill=True,
            label=f"Cluster {cid}",
            ax=fig_inset,
        )

    fig_inset.set_xlim(0, None)
    fig_inset.set_ylim(0, 1)
    fig_inset.set_yticks([])
    fig_inset.set_xticks([])
    fig_inset.set_xlabel("")
    # get rid of frame
    for spine in fig_inset.spines.values():
        spine.set_visible(False)
    fig_inset.set_title("Timepoint Clusters", fontsize=18)

    # add legend for timepoint cluster colors
    # cluster_times may be a numpy array (indexed 0..N-1) or a dict {cid: time}
    if hasattr(cluster_times, "keys"):
        legend_cids = sorted(cluster_times.keys())
        cluster_time_lookup = cluster_times
    else:
        legend_cids = list(range(len(cluster_times)))
        cluster_time_lookup = {i: t for i, t in enumerate(cluster_times)}
    handles = [
        plt.Line2D([0], [0], color=cluster_colors.get(cid, cluster_cmap(i % 10)), alpha=0.7, lw=4)
        for i, cid in enumerate(legend_cids)
    ]
    labels = [
        f"{cluster_definitions.get(cid, str(cid))}, time = {cluster_time_lookup[cid]:.2f}"
        for cid in legend_cids
    ]
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1, 0.65),
        fontsize=14,
        frameon=False,
    )


def segment_cluster_plotting(ax, clustered_genome, segments_ordered, offsets,
                             final_clusters=None):
    """Plot segment cluster assignments above the main axis.

    Each row is a unique CN state (major, minor).  Within a row, different
    clusters are distinguished by FILL pattern (no border, so the colour reads
    cleanly): Cluster 1 = solid, Cluster 2 = diagonal hatch, Cluster 3 = cross,
    etc. Hatch lines are drawn thin and white.

    final_clusters: optional {name: merged_segment} to display instead of the
    default _get_final_segment_clusters — pass the tree-panel selection so the
    track rows match exactly the CN states drawn as trees (e.g. 2-seg (5,4)/(6,1)
    merges that the stricter default filter would drop).
    """
    import matplotlib as _mpl
    from matplotlib.patches import Rectangle
    from tau.trees import _get_final_segment_clusters

    _mpl.rcParams["hatch.linewidth"] = 0.4   # thin hatch lines

    final_segment_clusters = (final_clusters if final_clusters is not None
                              else _get_final_segment_clusters(clustered_genome))

    # Map original segment ids back to their cluster name
    seg_by_cluster = {}
    for cluster_name, seg in final_segment_clusters.items():
        for original_seg_id in getattr(seg, "seg_ids", []):
            seg_by_cluster[original_seg_id] = cluster_name

    # Collect unique CN states (in display order) and cluster index within each
    cn_states = []  # ordered list of (major, minor)
    cn_cluster_num = {}  # cluster_name -> 1-based index within its CN state
    seen_cn = {}
    for name, seg in final_segment_clusters.items():
        cn = (int(seg.major_cn), int(seg.minor_cn))
        if cn not in seen_cn:
            seen_cn[cn] = 0
            cn_states.append(cn)
        seen_cn[cn] += 1
        cn_cluster_num[name] = seen_cn[cn]

    if not cn_cluster_num:
        return  # nothing timed in any cluster — skip inset

    cn_row = {cn: i for i, cn in enumerate(cn_states)}
    n_rows = len(cn_states)

    # Fill patterns by cluster index: solid, then increasingly distinct hatches
    HATCHES = ["", "////////", "........", "xxxxxxxx", "--------"]

    def _color_for_cn(major):
        cmap = _cmap_for_major(major)
        return cmap(0.6)

    # Create inset above the main figure
    inset_height = 0.06 * n_rows  # scale with number of CN states
    fig_inset = ax.inset_axes([0, 1.02, 1, inset_height])

    for seg in segments_ordered:
        xoff = offsets[seg.seg_id]
        x0 = int(seg.start) + xoff
        x1 = int(seg.end) + xoff

        if seg.seg_id not in seg_by_cluster:
            continue

        cluster_name = seg_by_cluster[seg.seg_id]
        cn = (int(seg.major_cn), int(seg.minor_cn))
        row = cn_row[cn]
        cluster_idx = cn_cluster_num[cluster_name]  # 1-based
        hatch = HATCHES[min(cluster_idx - 1, len(HATCHES) - 1)]
        color = _color_for_cn(cn[0])

        fig_inset.add_patch(Rectangle(
            (x0, row - 0.4), x1 - x0, 0.8,
            facecolor=color,
            edgecolor="white",   # hatch line colour; no border (linewidth=0)
            hatch=hatch,
            linewidth=0,
            rasterized=True,     # bars are sub-point wide: SVG can't tile a hatch pattern onto a
                                 # sliver (renders blank). Rasterise this strip so SVG == PNG.
        ))

    fig_inset.set_ylim(-0.5, n_rows - 0.5)
    fig_inset.set_xticks([])
    fig_inset.set_xlim(ax.get_xlim())

    # Label rows with (major, minor) on the right side, outside the track
    fig_inset.set_yticks(list(range(n_rows)))
    fig_inset.set_yticklabels(
        [f"({maj},{mn})" for maj, mn in cn_states],
        fontsize=7,
    )
    fig_inset.yaxis.set_ticks_position("left")
    fig_inset.yaxis.set_label_position("left")

    # Thin separators between rows
    for y in range(n_rows - 1):
        fig_inset.axhline(y + 0.5, color="grey", linestyle="-", linewidth=0.5)

    for spine in fig_inset.spines.values():
        spine.set_visible(False)

    # Legend: solid swatch = Cluster 1, diagonally striped = Cluster 2 (cross = 3+). Drawn as
    # explicit VECTOR lines via a custom handler — NOT a matplotlib hatch, because hatch becomes an
    # SVG <pattern> that Illustrator drops (blank swatch), and rasterising the tiny swatch is blocky.
    from matplotlib.legend_handler import HandlerBase
    import matplotlib.lines as _mlines

    class _ClSwatch:
        def __init__(self, idx): self.idx = idx

    class _ClHandler(HandlerBase):
        def create_artists(self, legend, orig, xd, yd, w, h, fontsize, trans):
            bg = Rectangle((xd, yd), w, h, facecolor="grey", edgecolor="none", transform=trans)
            arts = [bg]
            slopes = []
            if orig.idx >= 2:
                slopes.append(+1)              # diagonal "/"
            if orig.idx >= 3:
                slopes.append(-1)              # add "\" → cross
            for sl in slopes:
                step = w / 5.0
                for k in range(-5, 7):
                    x0 = xd + k * step
                    y0, y1 = (yd, yd + h) if sl > 0 else (yd + h, yd)
                    ln = _mlines.Line2D([x0, x0 + h], [y0, y1], color="white", lw=0.5, transform=trans)
                    ln.set_clip_path(bg)        # clip stripes to the swatch box
                    arts.append(ln)
            return arts

    max_cluster_idx = max(cn_cluster_num.values())
    handles = [_ClSwatch(ci) for ci in range(1, max_cluster_idx + 1)]
    labels = [f"Cluster {ci}" for ci in range(1, max_cluster_idx + 1)]
    fig_inset.legend(
        handles, labels, handler_map={_ClSwatch: _ClHandler()},
        loc="lower left", fontsize=7, frameon=False, ncol=max_cluster_idx,
        bbox_to_anchor=(0.0, 1.02),
    )



def _add_cn_timing_legend(ax, major_cns, fontsize=7, states=None, nbands=None):
    """CN colour legend left of the main axis; vertical t_early (bottom, light) →
    t_late (top, dark) gradient.

    Default (states=None): one column per major CN, 5 discrete shades.
    Per-state (states given): one column per distinct COLOUR SET. A state's colours
    depend only on (major → palette, K = #bands → #shades), so colour-identical
    states merge into one column "(maj, m1/m2)" drawn with the EXACT K colours via
    ``_colors_for_segment`` — matching the interactive viewer's legend.

    Positioned to the left of the main axis. Requires subplots_adjust left≥0.22.
    """
    from matplotlib.patches import Rectangle as Rect

    if states:
        nbands = nbands or {}
        groups = {}                      # (major, K) -> [minors]
        for (maj, mn) in sorted(states):
            K = int(nbands.get((maj, mn)) or max(int(maj), 1))
            groups.setdefault((maj, K), []).append(mn)

        def _lab(maj, mns):
            ms = sorted(set(mns))
            return f"({maj},{ms[0]})" if len(ms) == 1 else f"({maj},{'/'.join(map(str, ms))})"

        # HORIZONTAL layout: one ROW per CN-state group (stacked), time gradient left→right within a row,
        # horizontal labels on the left. Same inset footprint as before, just rotated internally.
        rows = [((maj, K), _lab(maj, mns)) for (maj, K), mns in sorted(groups.items())]
        n = len(rows)
        inset = ax.inset_axes([-0.28, 0.05, 0.13, 0.90])   # ~1/3 narrower gradient strip
        inset.set_xlim(0, 1)
        inset.set_ylim(-0.5, n - 0.5)
        for i, ((maj, K), _l) in enumerate(rows):
            cmap = _cmap_for_major(maj)
            for j, c in enumerate(_colors_for_segment(int(maj), int(K), cmap)):
                inset.add_patch(Rect((j / K, i - 0.42), 1.0 / K, 0.84,
                                     facecolor=c, edgecolor="none"))
        inset.set_yticks(range(n))
        inset.set_yticklabels([l for _k, l in rows], fontsize=fontsize)
        inset.set_xticks([0.08, 0.92])
        inset.set_xticklabels(["$t_{early}$", "$t_{late}$"], fontsize=fontsize)
        inset.xaxis.set_ticks_position("bottom")
        inset.yaxis.set_ticks_position("left")
        inset.tick_params(axis="both", length=0)
        for spine in inset.spines.values():
            spine.set_visible(False)
        return

    N = 5  # discrete steps light → dark
    major_cns = sorted(major_cns)
    n = len(major_cns)

    # Negative x offset places the inset to the left of the main axis
    inset = ax.inset_axes([-0.32, 0.05, 0.20, 0.90])
    inset.set_xlim(-0.5, n - 0.5)
    inset.set_ylim(-0.5, N - 0.5)

    for i, maj in enumerate(major_cns):
        cmap = _cmap_for_major(maj)
        for j in range(N):
            shade = (j + 1) / N
            inset.add_patch(Rect(
                (i - 0.4, j - 0.5), 0.8, 1,
                facecolor=cmap(shade), edgecolor="none",
            ))

    inset.set_xticks(range(n))
    inset.set_xticklabels([f"Major CN = {maj}" for maj in major_cns], fontsize=fontsize, rotation=45, ha="right")
    inset.xaxis.set_ticks_position("bottom")
    inset.set_yticks([0, N - 1])
    inset.set_yticklabels(["$t_{early}$", "$t_{late}$"], fontsize=fontsize)
    inset.yaxis.tick_right()
    inset.tick_params(axis="both", length=0)
    for spine in inset.spines.values():
        spine.set_visible(False)


# Shared layout constants — imported by tau/trees.py to ensure the 0-1 y-axis
# is the same physical height in both the overview and tree plots.
OVERVIEW_FIG_HEIGHT = 2.8   # inches

# Shared subplots_adjust parameters — keep in sync with tau/trees.py so that
# overview and tree plots align on the 0-1 time axis when placed side by side.
OVERVIEW_SADJ = dict(left=0.18, right=0.80, bottom=0.12, top=0.92)
# right=0.92 used when no cluster_summary_df legend (legacy behaviour)
_OVERVIEW_SADJ_LEGACY = dict(left=0.18, right=0.92, bottom=0.12, top=0.92)


# Event colours matching figure 4 (Nature palette)
WGD_COLOR = "#0b6979"   # teal
PGD_COLOR = "#cd9f2c"   # gold

def _event_color(row):
    """Return WGD/PGD colour for an event row."""
    cls = row.get("classification", "") if hasattr(row, "get") else ""
    return WGD_COLOR if cls == "WGD" else PGD_COLOR


def _event_label(row):
    """Human-readable label: just timing + genome fraction, no classification text."""
    t   = row["time"]
    pct = row.get("pct_of_theoretical_genome", row.get("gf", 0) * 100)
    return f"t={t:.3f}  ({pct:.1f}% genome)"


def _add_timing_histogram(ax_main, timing_df, df_events):
    """Add ESS-weighted timing histogram as an inset flush to the right of *ax_main*.

    Parameters
    ----------
    ax_main : matplotlib.axes.Axes
        The main overview axis (returned by ``plot_overview``).
    timing_df : pd.DataFrame
        DataFrame in the format returned by ``make_time_df`` — must contain
        columns: ``bootstrap_id``, ``best_route``, ``segment_id``,
        ``time_point``, ``time_fraction``, ``mutation_count``.
    df_events : pd.DataFrame
        Cluster-summary table from ``cluster_times_bottomup``; must have
        columns ``time``, ``classification``, and either
        ``pct_of_theoretical_genome`` or ``gf``.
    """
    import numpy as np

    df = timing_df[(timing_df["bootstrap_id"] == 0) & (timing_df["best_route"] == True)]
    grp = df.groupby(["segment_id", "time_point"]).agg(
        t_mean=("time_fraction", "mean"),
        t_range=("time_fraction", lambda x: float(x.max() - x.min())),
        ess=("mutation_count", "first"),
    ).reset_index()
    grp = grp[grp["t_range"] < 0.10].copy()
    if grp.empty:
        return

    events_sorted = df_events.sort_values("time").reset_index(drop=True)
    ev_times = events_sorted["time"].values

    def _nearest(t):
        dists = np.abs(ev_times - t)
        idx = dists.argmin()
        return int(idx) if dists[idx] < 0.20 else -1

    grp["event_idx"] = grp["t_mean"].apply(_nearest)

    # Remove the allele-style legend — histogram colours carry the event identity
    leg = ax_main.get_legend()
    if leg is not None:
        leg.remove()

    ax_hist = ax_main.inset_axes([1.0, 0, 0.18, 1.0])
    bins = np.linspace(0, 1, 26)
    for i, row in events_sorted.iterrows():
        sub = grp[grp["event_idx"] == i]
        if sub.empty:
            continue
        color = _event_color(row)
        ax_hist.hist(
            sub["t_mean"], bins=bins, weights=sub["ess"],
            orientation="horizontal", color=color, alpha=0.70,
            edgecolor="white", linewidth=0.3,
        )
        ax_hist.axhline(row["time"], color=color, lw=1.5, linestyle="--", alpha=0.9)
        pct = row.get("pct_of_theoretical_genome", row.get("gf", 0) * 100)
        ax_hist.text(
            0.97, row["time"], f" t={row['time']:.2f}  ({pct:.1f}%)",
            transform=ax_hist.get_yaxis_transform(),
            fontsize=7, va="center", ha="left", color=color,
        )
    ax_hist.set_ylim(0, 1)
    ax_hist.set_xlim(left=0)
    ax_hist.set_yticks([])
    ax_hist.set_xlabel("Timing density", fontsize=7)
    ax_hist.set_title("Timepoint\nclusters", fontsize=7, pad=3)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.spines["left"].set_visible(False)
    ax_hist.tick_params(axis="x", labelsize=7)


def plot_overview(
    g,
    cluster_times=None,
    segment_cluster_ids=None,
    original_times=None,
    clustered_genome=None,
    cluster_definitions=None,
    output_file=None,
    save_svg=False,
    cluster_summary_df=None,
    merged_genome=None,
    timing_df=None,
    cn_legend_states=False,
):
    """Generate genome-wide timing overview plot.

    Parameters
    ----------
    g : Genome
        The (original) genome; used for CN legend, segment track ordering,
        and the stackplot unless *merged_genome* is given.
    cluster_summary_df : pd.DataFrame, optional
        Output of ``cluster_times_bottomup``.  When supplied, event times and
        labels (including chrom-specific chromosome names and genome fractions)
        are drawn from this table and shown in a legend to the right of the
        axis.  Supersedes *cluster_times* / *cluster_definitions* for the
        legend; both can coexist for backward compat.
    merged_genome : Genome, optional
        If provided, segment timings are taken from this genome (cluster
        timings propagated to original segments) for the stackplot, while
        positions and the multiplicity-cluster track still use *g* /
        *clustered_genome*.
    timing_df : pd.DataFrame, optional
        Full per-draw timing table (format returned by ``make_time_df``).
        When provided alongside *cluster_summary_df*, an ESS-weighted
        histogram inset is added to the right of the main axis showing the
        distribution of identifiable segment timings around each event.
    """
    import matplotlib as mpl

    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    segments_ordered, offsets = compute_offsets(g.segments)

    # Determine event times for reference lines
    if cluster_summary_df is not None and not cluster_summary_df.empty:
        ev_times = np.array(sorted(cluster_summary_df["time"].tolist()))
    elif cluster_times is not None:
        ev_times = np.asarray(cluster_times)
    else:
        ev_times = np.array([])

    # Use merged_genome for the stackplot if provided, else original g
    plot_segs = segments_ordered
    if merged_genome is not None:
        # Build a seg_id→merged_seg map for timing lookup, keep original ordering
        merged_map = {s.seg_id: s for s in merged_genome.segments}
        plot_segs = [merged_map.get(s.seg_id, s) for s in segments_ordered]

    plot_tau_stack(
        ax,
        plot_segs,
        offsets,
        ess_thresh=0,
        cluster_times=[],  # event lines drawn separately below with correct colours
        segment_cluster_ids=segment_cluster_ids if segment_cluster_ids is not None else {},
        color_by_cluster=False,
    )

    # CN colour legend (left side)
    unique_major_cns = sorted({
        int(seg.major_cn) for seg in g.segments
        if pick_best_key(seg) is not None and int(seg.major_cn) != 1
    })
    if cn_legend_states:
        # per-CN-STATE legend (one column per distinct colour set), matching the interactive viewer
        states = sorted({(int(s.major_cn), int(s.minor_cn)) for s in g.segments
                         if pick_best_key(s) is not None and int(s.major_cn) != 1})
        nbands = {}
        for (maj, mn) in states:
            seg = next((s for s in g.segments if int(s.major_cn) == maj
                        and int(s.minor_cn) == mn and pick_best_key(s) is not None), None)
            K = None
            if seg is not None:
                m = extract_mat(seg, pick_best_key(seg), boot_id=0)
                if m is not None and m.size:
                    K = int(m.shape[1])
            nbands[(maj, mn)] = K or max(maj, 1)
        _add_cn_timing_legend(ax, unique_major_cns, states=states, nbands=nbands)
    else:
        _add_cn_timing_legend(ax, unique_major_cns)

    # Multiplicity cluster track (inset above main axis)
    if clustered_genome is not None:
        segment_cluster_plotting(ax, clustered_genome, segments_ordered, offsets)

    # ── Event-time legend ────────────────────────────────────────────────────
    if cluster_summary_df is not None and not cluster_summary_df.empty:
        # New-style: genome fractions + chrom-specific chromosome names,
        # placed outside the right edge of the axis.
        for _, row in cluster_summary_df.sort_values("time").iterrows():
            c = _event_color(row)
            ax.axhline(row["time"], color=c, linestyle="--", linewidth=1.5,
                       alpha=0.9, zorder=5)
        # Minimal legend: allele style only
        maj_h = mlines.Line2D([], [], color="dimgrey", lw=1.5, label="Major allele (solid)")
        min_h = mlines.Line2D([], [], color="dimgrey", lw=1.5, ls="--", label="Minor allele (dashed)")
        ax.legend(handles=[maj_h, min_h], fontsize=7, loc="upper left",
                  bbox_to_anchor=(1.22, 1.0), bbox_transform=ax.transAxes,
                  framealpha=0.85, borderaxespad=0)
        fig.subplots_adjust(**OVERVIEW_SADJ)

    elif (cluster_times is not None and segment_cluster_ids is not None
          and original_times is not None and cluster_definitions is not None):
        # Legacy: histogram inset + old-style legend
        timepoint_cluster_plotting(
            fig, ax, g,
            cluster_times, segment_cluster_ids, original_times, cluster_definitions,
        )
        fig.subplots_adjust(**_OVERVIEW_SADJ_LEGACY)

    else:
        fig.subplots_adjust(**_OVERVIEW_SADJ_LEGACY)

    # Timing histogram inset (new-style, requires both timing_df and cluster_summary_df)
    if timing_df is not None and cluster_summary_df is not None and not cluster_summary_df.empty:
        _add_timing_histogram(ax, timing_df, cluster_summary_df)

    if output_file:
        mpl.rcParams.update({"svg.fonttype": "none", "text.usetex": False})
        fig.savefig(f"{output_file}", dpi=400)
        if save_svg:
            svg_path = str(output_file).rsplit(".", 1)[0] + ".svg"
            fig.savefig(svg_path, dpi=400, transparent=True)
    else:
        plt.show()
    return fig, ax
