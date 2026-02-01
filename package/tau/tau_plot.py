#!/usr/bin/env python3
import sys
import tau_time
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tau_multiplicity import Segment, Genome
from tau_utils import pick_best_key, extract_mat

HG19_ORDER = [f"{i}" for i in range(1, 23)] + ["X", "Y"]
HG19_SIZES = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276,
    "5": 180915260, "6": 171115067, "7": 159138663, "8": 146364022,
    "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753,
    "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
    "21": 48129895, "22": 51304566}#, "X": 155270560, "Y": 59373566,
#}

def compute_offsets(segments, chrom_sizes=HG19_SIZES, chrom_order=HG19_ORDER):
    order_index = {c:i for i,c in enumerate(chrom_order)}
    segs = sorted(
        segments,
        key=lambda s: (order_index.get(str(s.chrom), 10**9), int(s.start), int(s.end))
    )

    offsets = {}
    prev = None
    offset = 0
    for seg in segs:
        chrom = str(seg.chrom)
        if chrom != prev and prev is not None:
            offset += chrom_sizes.get(prev, 0)
        offsets[seg.seg_id] = offset
        prev = chrom
    return segs, offsets

def seg_signature(idx_arr):
    # idx_arr is e.g. [-1,0,0,1]
    return tuple(int(x) for x in idx_arr)

COLORS = [plt.cm.Greys, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples, plt.cm.Blues, plt.cm.Reds, plt.cm.GnBu]

_COLORS_BY_MAJ = dict(zip(range(1, len(COLORS)+1), COLORS))

GREY_RGBA = (0.70, 0.70, 0.70, 1.0)

def _cmap_for_major(maj: int):
    if maj in _COLORS_BY_MAJ:
        return _COLORS_BY_MAJ[maj]
    # cycle through known maps for maj >= 6
    keys = sorted(_COLORS_BY_MAJ.keys())
    return _COLORS_BY_MAJ[keys[(maj - 1) % len(keys)]]

def _gradated_colors(n, cmap):
    # n gradations from a single color family
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

order_index = {c:i for i, c in enumerate(HG19_ORDER)}


def plot_tau_stack(ax, segments, offsets, ess_thresh=10, cluster_times=[], segment_cluster_ids={}, output_file=None, color_by_cluster=True):
    num_clusters = len(cluster_times)

    #cluster colors should be from a distinct colormap
    cluster_cmap = plt.cm.tab10
    cluster_colors = [cluster_cmap(i) for i in range(num_clusters)]

    prev_chrom = None
    chrom_start_x = None
    chrom_offsets = {}
    for seg in segments:
        chrom = str(seg.chrom)
        xoff = offsets[seg.seg_id]
        chrom_offsets[chrom] = xoff
        x0 = int(seg.start) + xoff
        x1 = int(seg.end) + xoff

        if chrom not in [str(i) for i in range(1,23)]:
            continue

        if chrom != prev_chrom:
            # draw boundary line (except first chromosome)
            if prev_chrom is not None:
                ax.axvline(x0, color="black", ls="--", lw=1.5, zorder=6)

                # label previous chromosome
                mid = (chrom_start_x + x0) / 2
                ax.text(mid, -0.04, prev_chrom,
                        ha="center", va="top", fontsize=14, clip_on=False)

                #also mark arms using hg19_centromeres
                #p_end = hg19_centromeres.query(f'chrom == "{prev_chrom}"')['p_end'].iloc[0]
                #q_start = hg19_centromeres.query(f'chrom == "{prev_chrom}"')['q_start'].iloc[0]
                #label 'p' and 'q' arms. use chrom_offsets to get correct x-positions
                #cen_start = p_end + chrom_offsets[prev_chrom]
                #cen_end = q_start + chrom_offsets[prev_chrom]
                #mark boundaries
                #ax.axvline(cen_start, color="black", ls=":", lw=1.0, zorder=6)
                #ax.axvline(cen_end, color="black", ls=":", lw=1.0, zorder=6)
                #ax.text((chrom_start_x + cen_start) / 2, -0.01, 'p',
                #        ha="center", va="top", fontsize=8, clip_on=False)
                #ax.text((cen_end + x0) / 2, -0.01, 'q',
                #        ha="center", va="top", fontsize=8, clip_on=False)

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
            color = 'black' if cid == -1 else cluster_colors[cid]

            if mat.shape[0] == 1:
                # single x-point: draw as a segment across the region
                ax.plot([x0, x1], [float(y[0]), float(y[0])],
                        color=color, lw=2.0, alpha=0.95, zorder=5)
            else:
                # many x-points: build matching x_vec and draw curve
                x_line = np.linspace(x0, x1, mat.shape[0])
                ax.plot(x_line, y, color=color, lw=1, alpha=0.95, zorder=5)

    if prev_chrom is not None:
        last_seg = segments[-1]
        end_x = int(last_seg.end) + offsets[last_seg.seg_id]
        mid = (chrom_start_x + end_x) / 2
        ax.text(mid, -0.04, prev_chrom,
                ha="center", va="top", fontsize=14, clip_on=False)
    
    #add vertical lines for cluster times
    for cid, ct in enumerate(cluster_times):
        ax.axhline(ct, color=cluster_colors[cid], linestyle='--', linewidth=3, alpha=0.8, zorder=5)

    ax.margins(x=0)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_ylabel("Normalised time", fontsize=18)
    if output_file:
        plt.savefig(output_file)

def timepoint_cluster_plotting(fig, ax, g, cluster_times, segment_cluster_ids, original_times, cluster_definitions):
    #TIMEPOINT CLUSTER PLOTTING
    #collect corresponding time points of clustered segments, using segment_cluster_times and 
    per_cluster_times = {}
    per_cluster_weights = {}
    for seg_id, (cluster_assignments, og_times) in zip(segment_cluster_ids.keys(), zip(segment_cluster_ids.values(), original_times.values())):
        seg = next(seg for seg in g.segments if seg.seg_id == seg_id)
        #if seg_by_cluster.get(seg.seg_id, None) != '(4,2) Cluster 2':
        #    continue
        #if not (seg.major_cn == 3 and seg.minor_cn == 2):
        #    continue
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
    cluster_colors = [cluster_cmap(i) for i in range(num_clusters)]
    #add timepoint density plots for each cluster on the right side of the main figure
    inset_width = 0.15
    fig_inset = ax.inset_axes([1, 0, inset_width, 1])

    for cid, times in per_cluster_times.items():
        if cid < 0:
            continue
        #make histplot using seaborn
        weights = per_cluster_weights[cid]
        sns.histplot(y=times, weights=weights, bins=7, alpha=0.7, color=cluster_colors[cid], fill=True, label=f'Cluster {cid}', ax=fig_inset)


    fig_inset.set_xlim(0, None)
    fig_inset.set_ylim(0,1)
    fig_inset.set_yticks([])
    fig_inset.set_xticks([])
    fig_inset.set_xlabel('')
    #get rid of frame
    for spine in fig_inset.spines.values():
        spine.set_visible(False)
    fig_inset.set_title('Timepoint Clusters', fontsize=18)

    #add legend for timepoint cluster colors
    handles = [plt.Line2D([0], [0], color=cluster_colors[cid], alpha=0.7, lw=4) for cid in range(num_clusters)]
    labels = [f'{cluster_definitions[cid]}, time = {cluster_times[cid]:.2f}' for cid in range(num_clusters)]
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 0.65), fontsize=14, frameon=False)

def segment_cluster_plotting(ax, clustered_genome, segments_ordered, offsets):
    #SEGMENT CLUSTER PLOTTING
    big_clusters = {}

    def get_segments_by_cn(clustered_g, major_cn, minor_cn):
        segs = []
        for seg in clustered_g.segments:
            if seg.major_cn == major_cn and seg.minor_cn == minor_cn:
                segs.append(seg)
        return segs

    for major in range(1, 7):
        for minor in range(0, major+1):
            cn_key = f"{major}_{minor}"
            segs = get_segments_by_cn(clustered_genome, major, minor)
            #pick segments with more than two segments and add all of these to the big_clusters
            for seg in segs:
                if getattr(seg, 'num_segs', 1) > 2:
                    #check 'qc' of best_key
                    best_key = pick_best_key(seg)
                    if best_key is None:
                        continue
                    #print(seg.seg_id, seg.timing_result[best_key]['qc'][0]['dist_rel'], best_key, sum(seg.N_counts))
                    if seg.timing_result[best_key]['qc'][0]['dist_rel'] > 0.02:
                        continue
                    big_clusters[seg.seg_id] = seg

    #rename genome clusters by major minor cn and then length
    big_clusters_sorted = dict(sorted(big_clusters.items(), key=lambda x: (x[1].major_cn, x[1].minor_cn, -getattr(x[1], 'num_segs', 0))))
    final_segment_clusters = {}
    maj_min_counts = {}
    for cluster_id, seg in big_clusters_sorted.items():
        major, minor = seg.major_cn, seg.minor_cn
        maj_min_counts[(major, minor)] = maj_min_counts.get((major, minor), 0) + 1
        new_name = f"({major},{minor}) Cluster {maj_min_counts[(major, minor)]}"
        final_segment_clusters[new_name] = seg

    seg_by_cluster = {}
    for cluster_id, seg in final_segment_clusters.items():
        for original_seg_id in getattr(seg, 'seg_ids', []):
            seg_by_cluster[original_seg_id] = cluster_id

    cluster_id_order = {cid: i for i, cid in enumerate(final_segment_clusters.keys())}

    #segment cluster plotting
    #add cluster labels in an inset figure ABOVE the main figure (each cluster should be a row)
    #and then shaded in if it corresponds to a cluster
    inset_height = 0.55
    fig_inset = ax.inset_axes([0, 1.05, 1, inset_height])
    for seg in segments_ordered:
        xoff = offsets[seg.seg_id]
        x0 = int(seg.start) + xoff
        x1 = int(seg.end) + xoff

        if seg.seg_id in seg_by_cluster:
            cluster_id = seg_by_cluster[seg.seg_id]
            color = 'black'
            fig_inset.hlines(y=cluster_id_order[cluster_id], xmin=x0, xmax=x1, colors=color, linewidth=10)

    fig_inset.set_ylim(-0.5, len(cluster_id_order) - 0.5)
    fig_inset.set_xticks([])
    fig_inset.set_xlim(ax.get_xlim())

    #label with cluster ids
    fig_inset.set_yticks(list(cluster_id_order.values()), labels=list(cluster_id_order.keys()))
    #add lines between cluster rows
    for y in range(len(cluster_id_order) - 1):
        fig_inset.axhline(y + 0.5, color='black', linestyle='-', linewidth=1)
    fig_inset.set_title('Segment Clusters', fontsize=18)


def plot_overview(g, cluster_times=None, segment_cluster_ids=None, original_times=None, clustered_genome=None, cluster_definitions=None, output_file=None):
    #GENERAL PLOTTING
    fig, ax = plt.subplots(figsize=(25,12))
    segments_ordered, offsets = compute_offsets(g.segments)

    plot_tau_stack(ax, segments_ordered, offsets, 
                ess_thresh=0, cluster_times=cluster_times, 
                segment_cluster_ids=segment_cluster_ids,
                color_by_cluster=False)

    if clustered_genome is not None:
        #SEGMENT CLUSTER PLOTTING
        segment_cluster_plotting(ax, clustered_genome, segments_ordered, offsets)
    if cluster_times is not None and segment_cluster_ids is not None and original_times is not None and cluster_definitions is not None:
        #TIMEPOINT CLUSTER PLOTTING
        timepoint_cluster_plotting(fig, ax, g, cluster_times, segment_cluster_ids, original_times, cluster_definitions)

    fig.tight_layout()
    if output_file:
        fig.savefig(f'{output_file}', dpi=400)
    else:
        plt.show()