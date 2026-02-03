#!/usr/bin/env python3
"""Command-line interface for running Tau timing analysis."""

import argparse
import gzip
import os
import pickle

import pandas as pd

from tau.core import Segment, Genome
from tau.preprocessing import preprocess_sample
from tau.utils import make_time_df
from tau.clustering import cluster_times, cluster_segment_multiplicities, make_clustered_genome
from tau.plotting import plot_overview
from tau import timing


def get_default_data_path(filename):
    """Get path to package data file."""
    try:
        from importlib import resources
        with resources.files("tau.data").joinpath(filename) as p:
            return str(p)
    except (FileNotFoundError, TypeError, AttributeError):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "tau", "data")
        return os.path.join(data_dir, filename)


# Default data file paths - can be overridden via environment variables or CLI args
DEFAULT_ROUTES_SAGE = os.environ.get(
    "TAU_ROUTES_SAGE", get_default_data_path("7_5_solutions_updated.sobj")
)
DEFAULT_MATRICES_H5 = os.environ.get(
    "TAU_MATRICES_H5", get_default_data_path("matrices_7_7.h5")
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Tau copy number timing analysis on a sample."
    )
    parser.add_argument(
        "--sample", type=str, required=True, help="Sample ID to process"
    )
    parser.add_argument(
        "--vcf", type=str, required=True, help="Path to the VCF file"
    )
    parser.add_argument(
        "--cnv", type=str, required=True, help="Path to the CNV file"
    )
    parser.add_argument(
        "--purity", type=float, default=None, help="Tumor purity (0-1)"
    )
    parser.add_argument(
        "--purity_file", type=str, default=None,
        help="Path to purity file (tab-separated with samplename and purity columns)"
    )
    parser.add_argument(
        "--ref_fasta", type=str, required=True,
        help="Path to reference genome FASTA"
    )
    parser.add_argument(
        "--cosmic_csv", type=str, required=True,
        help="Path to COSMIC signatures CSV file"
    )
    parser.add_argument(
        "--exposures", type=str, required=True,
        help="Path to signature exposures file"
    )
    parser.add_argument(
        "--routes_sage", type=str, default=DEFAULT_ROUTES_SAGE,
        help="Path to Sage solutions file"
    )
    parser.add_argument(
        "--matrices_h5", type=str, default=DEFAULT_MATRICES_H5,
        help="Path to constraint matrices HDF5 file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--output_times", type=str, default=None,
        help="Path to output segment times TSV (default: {sample}.segment_times.tsv)"
    )
    parser.add_argument(
        "--output_fig", type=str, default=None,
        help="Path to output overview figure (default: {sample}.overview.png)"
    )
    parser.add_argument(
        "--output_gpkl", type=str, default=None,
        help="Path to output genome pickle file (default: {sample}.genome.pkl.gz)"
    )
    parser.add_argument(
        "--output_cluster_tsv", type=str, default=None,
        help="Path to output segment clusters TSV (default: {sample}.segment_clusters.tsv)"
    )
    parser.add_argument(
        "--output_cluster_plot", type=str, default=None,
        help="Path to output segment clusters plot (default: {sample}.segment_clusters.png)"
    )
    parser.add_argument(
        "--output_cluster_gpkl", type=str, default=None,
        help="Path to output clustered genome pickle (default: {sample}.clustered_genome.pkl.gz)"
    )
    parser.add_argument(
        "--output_time_cluster_plot", type=str, default=None,
        help="Path to output timepoint clusters plot (default: {sample}.time_clusters.png)"
    )
    parser.add_argument(
        "--output_time_cluster_df", type=str, default=None,
        help="Path to output timepoint clusters TSV (default: {sample}.time_clusters.tsv)"
    )
    parser.add_argument(
        "--signatures", type=str, nargs="+", default=["SBS1", "SBS5"],
        help="Mutational signatures to consider (default: SBS1 SBS5)"
    )
    parser.add_argument(
        "--bootstrap_B", type=int, default=1,
        help="Number of bootstrap iterations (default: 1)"
    )
    parser.add_argument(
        "--detect_min_alt", type=int, default=3,
        help="Minimum alt reads to consider mutation detectable (default: 3)"
    )

    args = parser.parse_args()

    # Set default output paths based on sample name
    output_defaults = {
        "output_times": f"{args.sample}.segment_times.tsv",
        "output_fig": f"{args.sample}.overview.png",
        "output_gpkl": f"{args.sample}.genome.pkl.gz",
        "output_cluster_tsv": f"{args.sample}.segment_clusters.tsv",
        "output_cluster_plot": f"{args.sample}.segment_clusters.png",
        "output_cluster_gpkl": f"{args.sample}.clustered_genome.pkl.gz",
        "output_time_cluster_plot": f"{args.sample}.time_clusters.png",
        "output_time_cluster_df": f"{args.sample}.time_clusters.tsv",
    }
    for arg_name, default_filename in output_defaults.items():
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, os.path.join(args.output_dir, default_filename))

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create empty output files first (for Snakemake compatibility)
    output_args = [x for x in vars(args).keys() if x.startswith("output_")]
    for out_arg in output_args:
        out_path = getattr(args, out_arg)
        with open(out_path, "w") as f:
            pass

    # Get purity
    if args.purity is not None:
        purity = args.purity
    elif args.purity_file is not None:
        purity_df = pd.read_csv(args.purity_file, sep="\t")
        purity = float(purity_df.loc[purity_df["samplename"] == args.sample, "purity"])
    else:
        raise ValueError("Either --purity or --purity_file must be provided")

    # Load routes
    routes = timing.load_routes(args.routes_sage, args.matrices_h5)

    # Preprocess sample
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
        signatures=args.signatures,
        mode="soft",
        subclonal_list=None,
    )

    # Run Tau timing
    g = Genome.create(
        mut_df, purity=purity, detect_min_alt=args.detect_min_alt, detect_min_vaf=0
    )
    g.calculate_multiplicities(bootstrap_B=args.bootstrap_B, random_state=42)
    g.time_segments(env=routes)

    # Cluster segment multiplicities
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
    cluster_times_result, segment_cluster_ids, original_times, cluster_summary_df = (
        cluster_times(g, routes, cluster_output_file=args.output_time_cluster_plot)
    )

    cluster_summary_df.to_csv(args.output_time_cluster_df, sep="\t", index=False)

    cluster_definitions = {}
    for idx, row in cluster_summary_df.iterrows():
        cluster_definitions[row["cluster_id"]] = row["classification"]

    if seg_cluster_df.empty:
        plot_overview(
            g,
            cluster_times=cluster_times_result,
            segment_cluster_ids=segment_cluster_ids,
            original_times=original_times,
            clustered_genome=None,
            cluster_definitions=cluster_definitions,
            output_file=args.output_fig,
        )
        return

    # Create clustered genome
    clustered_genome = make_clustered_genome(g, seg_cluster_df)
    clustered_genome.calculate_multiplicities(bootstrap_B=args.bootstrap_B, random_state=42)
    clustered_genome.time_segments(env=routes)

    # Save clustered genome pickle
    with gzip.open(args.output_cluster_gpkl, "wb") as f:
        pickle.dump(clustered_genome, f)

    # Save segment cluster assignments
    seg_cluster_df.to_csv(args.output_cluster_tsv, sep="\t", index=False)

    # Plot overview
    plot_overview(
        g,
        cluster_times=cluster_times_result,
        segment_cluster_ids=segment_cluster_ids,
        original_times=original_times,
        clustered_genome=clustered_genome,
        cluster_definitions=cluster_definitions,
        output_file=args.output_fig,
    )


if __name__ == "__main__":
    main()
