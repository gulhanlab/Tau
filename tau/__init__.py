"""
Tau - Copy number amplification timing analysis for tumor genomes.

Tau infers when copy number alterations occurred during tumor evolution by analyzing
the density of clock-like mutations (SNVs) in copy number segments.

Example usage:
    from tau import Segment, Genome, preprocess_sample
    from tau import timing

    # Load routes (required for timing)
    env = timing.load_routes("7_5_solutions_updated.sobj", "matrices_7_7.h5")

    # Preprocess VCF/CNV data
    mut_df = preprocess_sample(sample="SAMPLE1", vcf_path="sample.vcf", ...)

    # Create genome and run analysis
    g = Genome.create(mut_df, purity=0.8)
    g.calculate_multiplicities(bootstrap_B=10)
    g.time_segments(env=env)
"""

__version__ = "0.1.0"

# Import core classes
from tau.core import Segment, Genome

# Import preprocessing
from tau.preprocessing import preprocess_sample

# Import timing functions and attach methods to classes
from tau import timing
from tau.timing import (
    load_routes,
    RouteEnv,
    _segment_time_segment,
    _genome_time_segments,
    _genome_times_to_df,
    _genome_summarize_discards,
    _genome_summarize_constraint_violations,
)

# Attach timing methods to Segment class
Segment.time_segment = _segment_time_segment

# Attach timing methods to Genome class
Genome.time_segments = _genome_time_segments
Genome.times_to_df = _genome_times_to_df
Genome.summarize_discards = _genome_summarize_discards
Genome.summarize_constraint_violations = _genome_summarize_constraint_violations

# Import utility functions
from tau.utils import (
    order_t,
    time_matrix,
    pick_best_key,
    extract_mat,
    make_time_df,
)

# Import clustering functions
from tau.clustering import (
    cluster_times,
    cluster_segment_multiplicities,
    make_clustered_genome,
    assign_chrom_arm,
)

# Import plotting functions
from tau.plotting import (
    plot_overview,
    plot_tau_stack,
    compute_offsets,
)

# Public API
__all__ = [
    # Core classes
    "Segment",
    "Genome",
    # Preprocessing
    "preprocess_sample",
    # Timing
    "timing",
    "load_routes",
    "RouteEnv",
    # Utils
    "order_t",
    "time_matrix",
    "pick_best_key",
    "extract_mat",
    "make_time_df",
    # Clustering
    "cluster_times",
    "cluster_segment_multiplicities",
    "make_clustered_genome",
    "assign_chrom_arm",
    # Plotting
    "plot_overview",
    "plot_tau_stack",
    "compute_offsets",
]
