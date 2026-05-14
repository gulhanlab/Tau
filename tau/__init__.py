"""
Tau - Copy number amplification timing analysis for tumor genomes.

Tau infers when copy number alterations occurred during tumor evolution by
analyzing the density of clock-like mutations (SNVs) in copy number segments.

Quick start (one-liner)
-----------------------
>>> from tau import run_sample, timing
>>> env = timing.load_routes_for_states([(3, 1), (4, 2), (2, 2)])
>>> genome, events = run_sample(
...     "MY_SAMPLE",
...     vcf_path="sample.vcf",
...     cnv_tsv="sample.cnv.tsv",
...     purity=0.75,
...     env=env,
...     ref_fasta="hg19.fa",
...     cosmic_csv="COSMIC_v3.3.tsv",
...     exposures_tsv="exposures.tsv",
...     signatures=["SBS1", "SBS5"],   # clock signatures
... )
>>> print(events[["time", "classification", "gf"]])

Step-by-step
------------
>>> from tau import Genome, preprocess_sample, timing, CLOCK_SIGNATURES
>>> env = timing.load_routes_for_states([(3, 1), (4, 2), (2, 2)])
>>> mut_df = preprocess_sample("S", vcf_path=..., cnv_tsv=..., purity=0.75,
...                            signatures=CLOCK_SIGNATURES)
>>> g = Genome.create(mut_df, purity=0.75)
>>> cluster_df = g.run(env)          # EM + timing + clustering in one call
>>> g.discard_report("S")            # see what was skipped and why
>>> df = g.times_to_df("S")         # long-form timing draws for all segments

Loading routes (fast path for large analyses)
---------------------------------------------
>>> env = timing.load_routes_for_states([(3,1), (4,2), (2,1)])  # only what you need
>>> env = timing.load_routes("path/to/solutions/", "matrices_7_7.h5")  # full load
"""

__version__ = "0.1.0"

# Core classes
from tau.core import Segment, Genome

# Preprocessing
from tau.preprocessing import preprocess_sample, CLOCK_SIGNATURES

# Timing
from tau import timing
from tau.timing import (
    load_routes,
    load_routes_for_states,
    RouteEnv,
    _segment_time_segment,
    _genome_time_segments,
    _genome_times_to_df,
    _genome_summarize_discards,
    _genome_summarize_constraint_violations,
    _genome_discard_report,
)

# Attach timing methods to Segment
Segment.time_segment = _segment_time_segment

# Attach timing methods to Genome
Genome.time_segments = _genome_time_segments
Genome.times_to_df = _genome_times_to_df
Genome.summarize_discards = _genome_summarize_discards
Genome.summarize_constraint_violations = _genome_summarize_constraint_violations
Genome.discard_report = _genome_discard_report

# Remove private implementation names from the tau namespace after attaching
del (
    _segment_time_segment,
    _genome_time_segments,
    _genome_times_to_df,
    _genome_summarize_discards,
    _genome_summarize_constraint_violations,
    _genome_discard_report,
)

# Pipeline helpers (Genome.run + top-level run_sample)
from tau.pipeline import _genome_run, run_sample
Genome.run = _genome_run
del _genome_run

# Utilities
from tau.utils import (
    order_t,
    time_matrix,
    pick_best_key,
    extract_mat,
    make_time_df,
)

# Clustering
from tau.clustering import (
    cluster_times_bottomup,
    cluster_times,
    cluster_segment_multiplicities,
    make_clustered_genome,
    assign_chrom_arm,
)

# Plotting
from tau.plotting import (
    plot_overview,
    plot_tau_stack,
    compute_offsets,
)

# Trees
from tau.trees import (
    build_allele_trees,
    plot_route,
    plot_cluster_trees,
)

__all__ = [
    # Core classes
    "Segment",
    "Genome",
    # Preprocessing
    "preprocess_sample",
    "CLOCK_SIGNATURES",
    # Pipeline
    "run_sample",
    # Timing
    "timing",
    "load_routes",
    "load_routes_for_states",
    "RouteEnv",
    # Utils
    "order_t",
    "time_matrix",
    "pick_best_key",
    "extract_mat",
    "make_time_df",
    # Clustering
    "cluster_times_bottomup",
    "cluster_times",
    "cluster_segment_multiplicities",
    "make_clustered_genome",
    "assign_chrom_arm",
    # Plotting
    "plot_overview",
    "plot_tau_stack",
    "compute_offsets",
    # Trees
    "build_allele_trees",
    "plot_route",
    "plot_cluster_trees",
]
