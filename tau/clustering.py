#!/usr/bin/env python3
"""Event clustering functions for Tau timing analysis."""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import resources
from scipy.stats import poisson as sp_poisson
from statsmodels.stats.multitest import multipletests

from tau.core import Segment, Genome
from tau.plotting import compute_offsets, plot_tau_stack
from tau.utils import pick_best_key, order_t, extract_mat

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
HG19_SIZES_INT = {int(k): v for k, v in HG19_SIZES.items() if k.isdigit()}
GENOME_SIZE = sum(HG19_SIZES.values())


def _load_centromeres():
    """Load centromere positions from package data."""
    try:
        with resources.files("tau.data").joinpath("hg19.centromeres.tsv").open() as f:
            df = pd.read_csv(f, sep="\t")
    except (FileNotFoundError, TypeError):
        # Fallback for older Python or missing data
        import os
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        df = pd.read_csv(os.path.join(data_dir, "hg19.centromeres.tsv"), sep="\t")
    df["chrom"] = df["chrom"].astype(str)
    return df


# Load centromeres lazily
_hg19_centromeres = None


def _get_centromeres():
    global _hg19_centromeres
    if _hg19_centromeres is None:
        _hg19_centromeres = _load_centromeres()
    return _hg19_centromeres


def cluster_times_bottomup(
    g,
    cluster_output_file=None,
    min_ess=10,
    half_win=0.06,
    identif_range=0.10,
    refine_win=0.10,
    wgd_thresh=0.40,
    wgd_genome_thresh=0.50,
    min_chrom_frac_cand=0.50,
    merge_tol=0.15,
    match_tol=0.15,
):
    """Cluster timing events using the bottom-up sliding-window Poisson method.

    This is the production clustering method. For each chromosome, a sliding
    window (±half_win) tests whether ESS concentration significantly exceeds
    a uniform null via Poisson test with BH FDR correction.  Significant
    windows are merged into per-chromosome candidates, then candidates are
    merged across chromosomes into events (WGD/PGD/chrom_specific).

    For genomes where >wgd_genome_thresh of bases have major_cn >= 2, (2,2)
    segments are treated as fully determined by enforcing t2=0, giving
    t_wgd = N2/2 for each segment.  This makes WGD-timed (2,2) segments
    visible to the Poisson test without relaxing the identif_range threshold.

    Returns the same 4-tuple as the legacy cluster_times() for compatibility:
      (cluster_times_result, segment_cluster_ids, original_times, cluster_summary_df)
    """
    total_genome_len = sum(s.end - s.start for s in g.segments)
    wgd_len = sum(s.end - s.start for s in g.segments
                  if getattr(s, "major_cn", 0) >= 2)
    is_wgd_genome = (wgd_len / total_genome_len > wgd_genome_thresh
                     if total_genome_len > 0 else False)
    print(f"WGD genome: {is_wgd_genome}  "
          f"(major_cn>=2 frac={wgd_len/max(total_genome_len,1):.2f})")

    # Build per-draw rows (for range computation) and original_times (median draw)
    rows_filt = []   # identifiable timepoints → Poisson test
    rows_all  = []   # all timepoints (mean draw) → refinement
    original_times = {}

    for seg in g.segments:
        if seg.timing_result is None:
            continue
        key = pick_best_key(seg)
        if key is None or key not in seg.timing_result:
            continue
        draws = seg.timing_result[key].get("draws", [])
        ess = float(sum(seg.N_counts)) if seg.N_counts is not None else 0
        if ess < min_ess:
            continue
        try:
            chrom = int(seg.chrom)
        except (ValueError, TypeError):
            continue

        pt_draws = [d for d in draws if d.get("boot_id", 0) == 0 and d.get("t")]
        if not pt_draws:
            continue

        all_cumsums = np.array([np.cumsum(order_t(d["t"]))[:-1] for d in pt_draws])
        original_times[seg.seg_id] = np.median(all_cumsums, axis=0)

        base = {"chrom": chrom, "start": seg.start, "end": seg.end,
                "ess": ess, "length": seg.end - seg.start}

        # (2,2) WGD enforcement: treat t2=0 → t_wgd = N2/2 = max(cumsum[:,0])
        if is_wgd_genome and getattr(seg, "major_cn", 0) == 2 and getattr(seg, "minor_cn", 0) == 2:
            t_wgd = float(all_cumsums[:, 0].max())
            rows_filt.append({**base, "time_point": 0, "time_fraction": t_wgd})
            rows_all.append( {**base, "time_point": 0, "time_fraction": t_wgd})
            continue

        rng = all_cumsums.max(0) - all_cumsums.min(0)
        mean_tf = all_cumsums.mean(0)
        for tp_idx in range(all_cumsums.shape[1]):
            row = {**base, "time_point": tp_idx, "time_fraction": float(mean_tf[tp_idx])}
            rows_all.append(row)
            if rng[tp_idx] < identif_range:
                rows_filt.append(row)

    empty_return = (np.array([]),
                    {s.seg_id: np.repeat(-1, len(original_times.get(s.seg_id, np.array([]))))
                     for s in g.segments if s.seg_id in original_times},
                    original_times,
                    pd.DataFrame(columns=["cluster_id","classification","time","n_segments"]))

    if not rows_filt:
        return empty_return

    df_filt = pd.DataFrame(rows_filt)
    df_all  = (pd.DataFrame(rows_all)
               .groupby(["chrom","start","end","time_point","length"])
               .agg(time_fraction=("time_fraction","mean"), ess=("ess","first"))
               .reset_index())

    n_id = df_filt.drop_duplicates(["chrom","start","end"]).shape[0]
    print(f"Identifiable timepoints: {len(df_filt)}  unique segs: {n_id}")

    # ── Per-chromosome Poisson test ───────────────────────────────────────────
    candidates = []
    for chrom, cdf in df_filt.groupby("chrom"):
        cs = HG19_SIZES_INT.get(chrom)
        if cs is None or cdf.empty:
            continue
        grid     = np.arange(0.05, 0.96, 0.025)
        total    = cdf["ess"].sum()
        expected = 2 * half_win * total
        if expected == 0:
            continue
        obs = np.array([cdf.loc[np.abs(cdf["time_fraction"] - c) <= half_win, "ess"].sum()
                        for c in grid])
        pv  = np.array([sp_poisson.sf(o - 1, expected) for o in obs])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rej, _, _, _ = multipletests(pv, method="fdr_bh", alpha=0.05)
        sig = grid[rej]
        chrom_clusters = []
        for c in sig:
            if chrom_clusters and c - chrom_clusters[-1][-1] <= 0.055:
                chrom_clusters[-1].append(c)
            else:
                chrom_clusters.append([c])
        for cl in chrom_clusters:
            center  = float(np.mean(cl))
            near    = cdf[np.abs(cdf["time_fraction"] - center) <= half_win + 0.02]
            near_u  = near.drop_duplicates(["start", "end"])
            cf      = near_u["length"].sum() / cs
            if cf < min_chrom_frac_cand:
                continue
            candidates.append({"chrom": chrom, "time": center, "chrom_frac": cf,
                                "length": near_u["length"].sum(),
                                "ess_near": float(near_u["ess"].sum())})

    print(f"Per-chrom candidates: {len(candidates)} from "
          f"{len({c['chrom'] for c in candidates})} chroms")

    if not candidates:
        return empty_return

    # ── Merge candidates across chromosomes ───────────────────────────────────
    cands_sorted = sorted(candidates, key=lambda x: x["time"])
    merged_clusters = []
    for c in cands_sorted:
        placed = False
        for mc in merged_clusters:
            mc_mean = (sum(x["time"] * x["ess_near"] for x in mc) /
                       sum(x["ess_near"] for x in mc))
            if abs(c["time"] - mc_mean) <= merge_tol:
                mc.append(c); placed = True; break
        if not placed:
            merged_clusters.append([c])

    events = []
    for mc in merged_clusters:
        total_ess   = sum(x["ess_near"] for x in mc)
        center_pre  = float(sum(x["time"] * x["ess_near"] for x in mc) / total_ess)
        cand_chroms = list({x["chrom"] for x in mc})
        n_cand      = len(cand_chroms)

        if n_cand >= 2:
            # Refine centre using all timepoints (incl. underdetermined) within refine_win
            near_all  = df_all[np.abs(df_all["time_fraction"] - center_pre) <= refine_win]
            near_segs = near_all.drop_duplicates(["chrom", "start", "end"])
            if not near_segs.empty:
                center = float(np.average(near_segs["time_fraction"],
                                          weights=near_segs["ess"]))
            else:
                center = center_pre
            gf       = near_segs["length"].sum() / GENOME_SIZE if not near_segs.empty else \
                       sum(x["length"] for x in mc) / GENOME_SIZE
            n_chroms = int(near_segs["chrom"].nunique()) if not near_segs.empty else n_cand
            cls      = "WGD" if gf >= wgd_thresh else "PGD"
        else:
            center   = center_pre
            gf       = sum(x["length"] for x in mc) / GENOME_SIZE
            n_chroms = n_cand
            cls      = "chrom_specific"
            cand_chroms = [str(x["chrom"]) for x in mc]

        events.append({"time": center, "classification": cls,
                        "gf": gf, "n_chroms": n_chroms,
                        "chrom": cand_chroms[0] if cls == "chrom_specific" else None})

    event_times = np.array(sorted(e["time"] for e in events))
    print(f"Detected {len(event_times)} event(s): {np.round(event_times, 3)}")

    # ── Assign segments to nearest event ─────────────────────────────────────
    segment_cluster_ids = {}
    for seg in g.segments:
        if seg.seg_id not in original_times:
            continue
        times   = original_times[seg.seg_id]
        idx_arr = np.full(len(times), -1, dtype=int)
        if len(event_times) > 0:
            for t_idx, t in enumerate(times):
                dists   = np.abs(event_times - t)
                nearest = int(np.argmin(dists))
                if dists[nearest] <= match_tol:
                    idx_arr[t_idx] = nearest
        segment_cluster_ids[seg.seg_id] = idx_arr

    # ── Build cluster summary dataframe ──────────────────────────────────────
    rows_summary = []
    for i, ev in enumerate(sorted(events, key=lambda e: e["time"])):
        n_segs = sum(1 for idx_arr in segment_cluster_ids.values() if i in idx_arr)
        rows_summary.append({
            "cluster_id":               i,
            "time":                     round(ev["time"], 4),
            "classification":           ev["classification"],
            "gf":                       round(ev["gf"], 4),
            "n_chroms":                 ev["n_chroms"],
            "n_segments":               n_segs,
            "pct_of_theoretical_genome": round(ev["gf"] * 100, 2),
            "chrom":                    ev.get("chrom"),  # set for chrom_specific only
        })
    cluster_summary_df = pd.DataFrame(rows_summary) if rows_summary else \
        pd.DataFrame(columns=["cluster_id","time","classification","gf",
                               "n_chroms","n_segments","pct_of_theoretical_genome","chrom"])

    # ── Optional plot ─────────────────────────────────────────────────────────
    if cluster_output_file:
        fig, ax = plt.subplots(figsize=(25, 10))
        segs_ord, offsets = compute_offsets(g.segments)
        plot_tau_stack(ax, segs_ord, offsets, ess_thresh=min_ess,
                       cluster_times=event_times,
                       segment_cluster_ids=segment_cluster_ids,
                       output_file=cluster_output_file)
        plt.close(fig)

    return event_times, segment_cluster_ids, original_times, cluster_summary_df


def assign_chrom_arm(chrom, start, end):
    """Classify segment as p arm, q arm, or centromere based on position."""
    hg19_centromeres = _get_centromeres()
    centromere_row = hg19_centromeres[hg19_centromeres["chrom"] == chrom]
    if centromere_row.empty:
        return chrom
    cent_start = centromere_row["p_end"].values[0]
    cent_end = centromere_row["q_start"].values[0]
    if end < cent_start:
        return "p"
    elif start > cent_end:
        return "q"
    else:
        return "centromere"


def cluster_segment_multiplicities(genome, output_plot_file=None):
    """Cluster segments by multiplicity profiles using PCA and DBSCAN."""
    from sklearn.cluster import DBSCAN

    records = []
    for major_cn in range(2, 7):
        for minor_cn in range(0, major_cn + 1):
            seg_subset = [
                seg
                for seg in genome.segments
                if seg.major_cn == major_cn and seg.minor_cn == minor_cn
            ]
            points = []
            seg_ids = []
            for seg in seg_subset:
                seg_id = seg.seg_id
                N_counts = np.array(seg.N_counts)
                total = N_counts.sum()
                if total < 10:
                    continue

                norm_counts = N_counts / total
                eff_N = total
                points.append(np.concatenate([norm_counts, [eff_N]]))
                seg_ids.append(seg_id)

            if len(points) <= 2:
                continue

            points = np.array(points)
            X = points[:, : len(seg.N_counts) - 1]  # drop last component (sums to 1)
            weights = points[:, -1]
            labels = DBSCAN(eps=0.1, min_samples=5).fit_predict(X, sample_weight=weights)

            for i, seg_id in enumerate(seg_ids):
                seg = next((s for s in genome.segments if s.seg_id == seg_id), None)
                records.append(
                    {
                        "seg_id": seg_id,
                        "major_cn": major_cn,
                        "minor_cn": minor_cn,
                        "cluster_label": labels[i],
                        "x": X[i, 0],
                        "y": X[i, 1] if X.shape[1] > 1 else 0.0,
                        "N_counts": seg.N_counts,
                    }
                )

    seg_cluster_df = pd.DataFrame(records)

    if not output_plot_file or seg_cluster_df.empty:
        return seg_cluster_df

    # plot the PCA results colored by chromosome arm (p or q) and cluster label side by side
    for (major_cn, minor_cn), sub in seg_cluster_df.groupby(["major_cn", "minor_cn"]):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        arms = []
        for _, row in sub.iterrows():
            seg = next((s for s in genome.segments if s.seg_id == row["seg_id"]), None)
            if seg is not None:
                arm = assign_chrom_arm(str(seg.chrom), seg.start, seg.end)
                arms.append(str(seg.chrom) + arm)
            else:
                arms.append("unknown")
        sub = sub.assign(chrom_arm=arms)

        unique_arms = sorted(set(arms))
        colors = sns.color_palette("hsv", len(unique_arms))
        arm_color_map = {arm: colors[i] for i, arm in enumerate(unique_arms)}

        for arm in unique_arms:
            mask = sub["chrom_arm"] == arm
            ax[0].scatter(
                sub.loc[mask, "x"],
                sub.loc[mask, "y"],
                s=20,
                alpha=0.5,
                color=arm_color_map[arm],
                label=f"Arm {arm}",
            )
        ax[0].set_title(
            f"Major CN={major_cn}, Minor CN={minor_cn} colored by Chromosome Arm"
        )
        ax[0].set_xlabel("Proportion (mult 1)")
        ax[0].set_ylabel("Proportion (mult 2)")
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        unique_labels = sorted(sub["cluster_label"].unique())
        colors = sns.color_palette("hsv", len(unique_labels))
        label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            mask = sub["cluster_label"] == label
            ax[1].scatter(
                sub.loc[mask, "x"],
                sub.loc[mask, "y"],
                s=20,
                alpha=0.5,
                color=label_color_map[label],
                label=f"Cluster {label}",
            )
        ax[1].set_title(
            f"Major CN={major_cn}, Minor CN={minor_cn} colored by Cluster Label"
        )
        ax[1].set_xlabel("Proportion (mult 1)")
        ax[1].set_ylabel("Proportion (mult 2)")
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

    plt.savefig(f"{output_plot_file}", dpi=400)

    return seg_cluster_df


def make_clustered_genome(g, seg_cluster_df):
    """Create a new Genome with segments merged by cluster assignment."""
    clustered_genome = Genome()

    # merge segments that have the same cluster label and major_cn and minor_cn
    for (major_cn, minor_cn, cluster_label), group in seg_cluster_df.groupby(
        ["major_cn", "minor_cn", "cluster_label"]
    ):
        if cluster_label == -1:
            # just return standard segment
            for _, row in group.iterrows():
                seg = next((s for s in g.segments if s.seg_id == row["seg_id"]), None)
                if seg is not None:
                    clustered_genome.segments.append(seg)
            continue

        snv_tables = pd.concat(
            [
                s.snv_table
                for s in g.segments
                if s.seg_id in group["seg_id"].values
            ],
            ignore_index=True,
        )

        merged_seg = Segment(
            chrom=",".join(
                group["seg_id"].apply(lambda x: x.split(":")[0])
            ),  # concatenate all together with commas
            start=min(
                next((s.start for s in g.segments if s.seg_id == seg_id), None)
                for seg_id in group["seg_id"]
            ),
            end=max(
                next((s.end for s in g.segments if s.seg_id == seg_id), None)
                for seg_id in group["seg_id"]
            ),
            major_cn=major_cn,
            minor_cn=minor_cn,
            seg_id=f"merged_{major_cn}_{minor_cn}_{cluster_label}",
            purity=g.segments[0].purity,
            snv_table=snv_tables,
        )
        merged_seg.num_segs = len(group)
        merged_seg.seg_ids = group["seg_id"].tolist()
        merged_seg.total_length = sum(
            next((s.end - s.start) for s in g.segments if s.seg_id == seg_id)
            for seg_id in group["seg_id"]
        )

        clustered_genome.segments.append(merged_seg)

    return clustered_genome


def cluster_times(g, env=None, **kwargs):
    """Deprecated — use cluster_times_bottomup() instead.

    The legacy KDE+EM clustering method has been removed. This stub forwards
    compatible keyword arguments to cluster_times_bottomup() and ignores *env*.
    """
    warnings.warn(
        "cluster_times() is deprecated and will be removed in a future version. "
        "Use cluster_times_bottomup() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    valid = {
        "cluster_output_file", "min_ess", "half_win", "identif_range",
        "refine_win", "wgd_thresh", "wgd_genome_thresh", "min_chrom_frac_cand",
        "merge_tol", "match_tol",
    }
    return cluster_times_bottomup(g, **{k: v for k, v in kwargs.items() if k in valid})
