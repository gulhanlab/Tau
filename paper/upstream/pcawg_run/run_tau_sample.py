#!/usr/bin/env python3
"""
Runner script for Tau timing analysis on a single sample.

This script uses the tau package API to run timing analysis.
It is called by the Snakemake workflow for each sample.
"""

import argparse
import copy
import gzip
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({"svg.fonttype": "none", "text.usetex": False})
import matplotlib.pyplot as plt

# Import from tau package
from tau import (
    Segment,
    Genome,
    preprocess_sample,
    load_routes,
    make_time_df,
    cluster_times_bottomup,
    cluster_segment_multiplicities,
    make_clustered_genome,
    plot_overview,
)
from tau import timing



_DEFAULT_ROUTE_META_PKL = os.path.join(os.environ.get("TAU_PCAWG_RESULTS_DIR", "."), "route_metadata.pkl")


def _make_modified_genome(g, cg):
    """Return copy of g with cluster-level timing for clustered segments."""
    seg_to_cluster = {}
    for cseg in cg.segments:
        for sid in getattr(cseg, "seg_ids", []):
            seg_to_cluster[sid] = cseg
    new_segs = []
    for seg in g.segments:
        cseg = seg_to_cluster.get(seg.seg_id)
        if cseg is not None:
            s2 = copy.copy(seg)
            s2.timing_result = cseg.timing_result
            s2.N_counts = cseg.N_counts
            new_segs.append(s2)
        else:
            new_segs.append(seg)
    new_g = copy.copy(g)
    new_g.segments = new_segs
    return new_g


def _save_pdf_svg(fig, stem):
    mpl.rcParams.update({"svg.fonttype": "none", "text.usetex": False})
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight", transparent=True)


def get_default_data_path(filename):
    """Get path to package data file."""
    try:
        from importlib import resources
        with resources.files("tau.data").joinpath(filename) as p:
            return str(p)
    except (FileNotFoundError, TypeError, AttributeError):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tau", "data")
        return os.path.join(data_dir, filename)


# Default data file paths
DEFAULT_ROUTES_SAGE = os.environ.get(
    "TAU_ROUTES_SAGE", get_default_data_path("7_5_solutions_updated.sobj")
)
DEFAULT_MATRICES_H5 = os.environ.get(
    "TAU_MATRICES_H5", get_default_data_path("matrices_7_7.h5")
)


def _generate_improved_plots_no_clusters(g, cluster_summary_df, times_df, args):
    """Fallback when no multiplicity clusters: produce unmerged overview only."""
    gpkl_path = Path(args.output_gpkl)
    sdir = gpkl_path.parent
    sid = args.sample

    stem_merged   = Path(args.output_overview_merged)   if args.output_overview_merged   else sdir / f"{sid}.overview_merged"
    stem_unmerged = Path(args.output_overview_unmerged) if args.output_overview_unmerged else sdir / f"{sid}.overview_unmerged"
    stem_tree     = Path(args.output_tree)               if args.output_tree               else sdir / f"{sid}.tree_overview"

    print("Generating improved overview (no segment clusters)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, _, df_events = cluster_times_bottomup(g)

    fig, ax = plot_overview(g, cluster_summary_df=df_events, timing_df=times_df)
    _save_pdf_svg(fig, stem_unmerged)
    # merged = same as unmerged when no clusters
    _save_pdf_svg(fig, stem_merged)
    plt.close(fig)

    # Touch tree files so Snakemake output requirements are satisfied
    for ext in (".pdf", ".svg"):
        p = Path(str(stem_tree) + ext)
        p.touch()


def _generate_improved_plots(g, clustered_genome, cluster_summary_df, times_df, args):
    """Generate improved overview (merged + unmerged) and tree plots."""
    # Derive output stems from output_gpkl path if not explicitly provided
    gpkl_path = Path(args.output_gpkl)
    sdir = gpkl_path.parent
    sid = args.sample

    stem_merged   = Path(args.output_overview_merged)   if args.output_overview_merged   else sdir / f"{sid}.overview_merged"
    stem_unmerged = Path(args.output_overview_unmerged) if args.output_overview_unmerged else sdir / f"{sid}.overview_unmerged"
    stem_tree     = Path(args.output_tree)               if args.output_tree               else sdir / f"{sid}.tree_overview"

    # Re-cluster original genome for unmerged view
    print("Generating improved unmerged overview...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, _, df_unmerged = cluster_times_bottomup(g)

    fig, ax = plot_overview(
        g, clustered_genome=clustered_genome,
        cluster_summary_df=df_unmerged, timing_df=times_df,
    )
    _save_pdf_svg(fig, stem_unmerged)
    plt.close(fig)

    # Build g_enhanced (cluster-level timing on original segments) for merged view
    g_enhanced = _make_modified_genome(g, clustered_genome)

    # Re-cluster enhanced genome for merged view
    print("Generating improved merged overview...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, _, df_merged = cluster_times_bottomup(g_enhanced)

    fig, ax = plot_overview(
        g,
        clustered_genome=clustered_genome,
        cluster_summary_df=df_merged,
        merged_genome=g_enhanced,
        timing_df=times_df,
    )
    _save_pdf_svg(fig, stem_merged)
    plt.close(fig)

    # Tree plot (requires route_metadata)
    route_meta_path = Path(args.route_metadata)
    if route_meta_path.exists():
        try:
            from tau.trees import plot_cluster_trees
            print("Generating tree overview...")
            with open(route_meta_path, "rb") as f:
                route_metadata = pickle.load(f)
            plot_cluster_trees(
                clustered_genome, route_metadata,
                cluster_summary_df=df_merged,
                output_file=str(stem_tree) + ".pdf",
                save_svg=True,
            )
        except Exception as e:
            print(f"  Warning: tree plot failed: {e}")
    else:
        print(f"  Skipping tree plot (route_metadata not found: {route_meta_path})")
    # Touch tree output files if not created (e.g. "No trees could be built"),
    # so Snakemake's output requirements are satisfied.
    for ext in (".pdf", ".svg"):
        p = Path(str(stem_tree) + ext)
        if not p.exists():
            p.touch()


def main():
    parser = argparse.ArgumentParser(
        description="Run Tau timing analysis on a single sample."
    )
    parser.add_argument("--sample", required=True, help="Sample ID")
    parser.add_argument("--vcf", required=True, help="Path to VCF file")
    parser.add_argument("--cnv", required=True, help="Path to CNV file")
    parser.add_argument("--purity", type=float, default=None, help="Tumor purity (0-1)")
    parser.add_argument("--purity_file", default=None, help="Path to purity file")
    parser.add_argument("--ref_fasta", default=None, help="Reference genome FASTA")
    parser.add_argument("--cosmic_csv", default=None, help="COSMIC signatures CSV")
    parser.add_argument("--exposures", default=None, help="Signature exposures file")
    parser.add_argument(
        "--routes_sage", default=DEFAULT_ROUTES_SAGE, help="Sage solutions file"
    )
    parser.add_argument(
        "--matrices_h5", default=DEFAULT_MATRICES_H5, help="Constraint matrices HDF5"
    )
    parser.add_argument("--signatures", nargs="+", default=["SBS1", "SBS5"],
                        help="Signature names. Pass 'none' for all mutations (no filtering).")
    parser.add_argument("--mode", default="soft", choices=["soft", "hard"],
                        help="Signature weighting mode: soft (continuous) or hard (0/1).")
    parser.add_argument("--detect_min_alt", type=int, default=3)
    parser.add_argument("--bootstrap_B", type=int, default=1)
    parser.add_argument("--sex", type=str, default=None, choices=["male", "female"],
                        help="Sample sex; 'male' sets normal_cn=1 for chrX/Y")
    parser.add_argument("--subclonal_list", default=None,
                        help="Path to file of subclonal mutation IDs (one per line) to exclude from timing")

    # Output files
    parser.add_argument("--output_times", required=True)
    parser.add_argument("--output_fig", required=True)
    parser.add_argument("--output_gpkl", required=True)
    parser.add_argument("--output_cluster_tsv", required=True)
    parser.add_argument("--output_cluster_plot", required=True)
    parser.add_argument("--output_cluster_gpkl", required=True)
    parser.add_argument("--output_time_cluster_plot", required=True)
    parser.add_argument("--output_time_cluster_df", required=True)
    parser.add_argument("--output_time_cluster2_plot", default=None)
    parser.add_argument("--output_time_cluster2_df", default=None)
    # Improved overview outputs (PDF+SVG generated automatically alongside)
    parser.add_argument("--output_overview_merged", default=None,
                        help="Stem path for merged improved overview (produces .pdf and .svg)")
    parser.add_argument("--output_overview_unmerged", default=None,
                        help="Stem path for unmerged improved overview (produces .pdf and .svg)")
    parser.add_argument("--output_tree", default=None,
                        help="Stem path for tree overview (produces .pdf and .svg)")
    parser.add_argument("--route_metadata", default=_DEFAULT_ROUTE_META_PKL,
                        help="Path to route_metadata.pkl for tree plots")

    args = parser.parse_args()

    # Create output directories
    for out_path in [
        args.output_times,
        args.output_fig,
        args.output_gpkl,
        args.output_cluster_tsv,
        args.output_cluster_plot,
        args.output_cluster_gpkl,
        args.output_time_cluster_plot,
        args.output_time_cluster_df,
        args.output_time_cluster2_plot,
        args.output_time_cluster2_df,
    ]:
        if out_path is None:
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Create empty files for Snakemake compatibility
        with open(out_path, "w") as f:
            pass

    # Get purity
    if args.purity is not None:
        purity = args.purity
    elif args.purity_file is not None:
        purity_df = pd.read_csv(args.purity_file, sep="\t")
        purity = float(
            purity_df.loc[purity_df["samplename"] == args.sample, "purity"].iloc[0]
        )
    else:
        raise ValueError("Either --purity or --purity_file must be provided")

    # Resolve signatures ("none" sentinel → None means use all mutations)
    signatures = None if args.signatures == ["none"] else args.signatures

    print(f"Processing sample: {args.sample}")
    print(f"Purity: {purity}")
    print(f"Signatures: {signatures!r}  mode: {args.mode}")

    # Load routes
    print("Loading routes...")
    routes = load_routes(args.routes_sage, args.matrices_h5)

    # Preprocess sample
    print("Preprocessing sample...")
    mut_df = preprocess_sample(
        sample=args.sample,
        vcf_path=args.vcf,
        cnv_tsv=args.cnv,
        purity=purity,
        ref_fasta=args.ref_fasta,
        cosmic_csv=args.cosmic_csv,
        exposures_tsv=args.exposures,
        apply_subclonal_filter=False,
        detect_min_alt=args.detect_min_alt,
        signatures=signatures,
        mode=args.mode,
        subclonal_list=args.subclonal_list,
    )

    # Create genome and calculate multiplicities
    print("Creating genome and calculating multiplicities...")
    normal_cn_map = {"X": 1, "Y": 1} if args.sex == "male" else None
    g = Genome.create(
        mut_df, purity=purity, detect_min_alt=args.detect_min_alt, detect_min_vaf=0,
        normal_cn_map=normal_cn_map,
    )
    g.calculate_multiplicities(bootstrap_B=args.bootstrap_B, random_state=42)
    g.calculate_gof()

    # Run timing
    print("Running timing analysis...")
    g.time_segments(env=routes)

    # Cluster segment multiplicities
    print("Clustering segment multiplicities...")
    seg_cluster_df = cluster_segment_multiplicities(
        g, output_plot_file=args.output_cluster_plot
    )

    # Create and save times dataframe
    times_df = make_time_df(g)
    times_df.to_csv(args.output_times, sep="\t", index=False)

    # Save genome pickle
    with gzip.open(args.output_gpkl, "wb") as f:
        pickle.dump(g, f)

    # Cluster timepoints
    print("Clustering timepoints...")
    cluster_times_result, segment_cluster_ids, original_times, cluster_summary_df = (
        cluster_times_bottomup(g, cluster_output_file=args.output_time_cluster_plot)
    )

    cluster_summary_df.to_csv(args.output_time_cluster_df, sep="\t", index=False)

    cluster_definitions = {}
    for idx, row in cluster_summary_df.iterrows():
        cluster_definitions[row["cluster_id"]] = row["classification"]

    if seg_cluster_df.empty:
        # No multiplicity clusters — fall back to first-pass results for both outputs
        cluster_summary_df.to_csv(args.output_time_cluster2_df, sep="\t", index=False)
        plot_overview(
            g,
            cluster_times=cluster_times_result,
            segment_cluster_ids=segment_cluster_ids,
            original_times=original_times,
            clustered_genome=None,
            cluster_definitions=cluster_definitions,
            output_file=args.output_fig,
        )
        # Still produce improved overview files (unmerged only; no clustered genome)
        _generate_improved_plots_no_clusters(g, cluster_summary_df, times_df, args)
        print(f"Completed (no segment clusters): {args.sample}")
        return

    # Create clustered genome
    print("Creating clustered genome...")
    clustered_genome = make_clustered_genome(g, seg_cluster_df)
    clustered_genome.calculate_multiplicities(
        bootstrap_B=args.bootstrap_B, random_state=42
    )
    clustered_genome.time_segments(env=routes)

    # Second-pass timepoint clustering.
    # Propagate merged-segment posteriors back to original segments for
    # DETERMINED routes only (major+minor <= 3: e.g. (2,1), (2,0), (3,0)).
    # For determined routes the merged timing is tightly concentrated → sharper
    # KDE peaks.  Underdetermined routes (e.g. (2,2)) have free variables so
    # the merged segment's draws still spread across the feasible region;
    # propagating them creates artifical split clusters rather than improving
    # detection.  Those segments keep their original individual timing.
    #
    # IMPORTANT: build a separate g_pass2 rather than mutating g in place,
    # so that _generate_improved_plots can use the original g for the
    # unmerged overview (which should show pre-merge segment-level timings).
    print("Propagating merged-segment posteriors back to original segments...")
    merged_timing = {}  # src seg_id → timing_result
    for seg in clustered_genome.segments:
        if (
            seg.seg_id.startswith("merged_")
            and hasattr(seg, "timing_result")
            and seg.timing_result
        ):
            for src_id in seg.seg_ids:
                merged_timing[src_id] = seg.timing_result

    # Build pass-2 genome as a shallow copy with updated segments (g unchanged).
    new_segs_pass2 = []
    for seg in g.segments:
        if seg.seg_id in merged_timing:
            s2 = copy.copy(seg)
            s2.timing_result = merged_timing[seg.seg_id]
            new_segs_pass2.append(s2)
        else:
            new_segs_pass2.append(seg)
    g_pass2 = copy.copy(g)
    g_pass2.segments = new_segs_pass2

    print("Second-pass timepoint clustering on updated original genome...")
    cluster_times2_result, segment_cluster2_ids, original_times2, cluster_summary2_df = (
        cluster_times_bottomup(g_pass2, cluster_output_file=args.output_time_cluster2_plot)
    )
    cluster_summary2_df.to_csv(args.output_time_cluster2_df, sep="\t", index=False)

    cluster_definitions2 = {
        row["cluster_id"]: row["classification"]
        for _, row in cluster_summary2_df.iterrows()
    }

    # Save clustered genome pickle
    with gzip.open(args.output_cluster_gpkl, "wb") as f:
        pickle.dump(clustered_genome, f)

    # Save segment cluster assignments
    seg_cluster_df.to_csv(args.output_cluster_tsv, sep="\t", index=False)

    # Plot overview using second-pass cluster times (uses g_pass2 so merged
    # segment posteriors are reflected in the main overview.png)
    print("Generating overview plot...")
    plot_overview(
        g_pass2,
        cluster_times=cluster_times2_result,
        segment_cluster_ids=segment_cluster_ids,
        original_times=original_times,
        clustered_genome=clustered_genome,
        cluster_definitions=cluster_definitions2,
        output_file=args.output_fig,
    )

    # ── Improved overview + tree plots ────────────────────────────────────────
    # Pass original g (unmodified) so the unmerged plot reflects pre-merge timing.
    _generate_improved_plots(g, clustered_genome, cluster_summary2_df, times_df, args)

    print(f"Completed: {args.sample}")


if __name__ == "__main__":
    main()