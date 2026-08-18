#!/usr/bin/env python3
"""High-level pipeline helpers: Genome.run() and run_sample()."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from tau.core import Genome


def _genome_run(
    self: Genome,
    env=None,
    *,
    bootstrap_B: int = 10,
    random_state: int | None = 42,
    n_draws: int = 50,
    dist_cut: float = 0.20,
    effN_cut: float = 10.0,
    **cluster_kwargs,
) -> pd.DataFrame:
    """Run the full Tau pipeline on this genome: EM → timing → clustering.

    Parameters
    ----------
    env : RouteEnv, optional
        Pre-loaded route environment.  If None, uses package-bundled routes.
    bootstrap_B : int
        Bootstrap replicates for EM uncertainty (default 10).
    random_state : int, optional
        RNG seed for reproducibility.
    n_draws : int
        Grid draws per free variable per segment (default 50).
    dist_cut : float
        Maximum normalised projection distance to accept a route (default 0.20).
    effN_cut : float
        Minimum effective N (EM counts sum) to attempt timing (default 10).
    **cluster_kwargs
        Extra keyword arguments forwarded to cluster_times_bottomup()
        (e.g. half_win, merge_tol, wgd_thresh).

    Returns
    -------
    pd.DataFrame
        Cluster summary — one row per detected event with columns:
        cluster_id, time, classification, gf, n_chroms, n_segments,
        pct_of_theoretical_genome, chrom.

        The full 4-tuple result (event_times, segment_cluster_ids,
        original_times, cluster_df) is stored on ``self._cluster_result``.

    Examples
    --------
    >>> from tau import Genome, timing
    >>> env = timing.load_routes_for_states([(3, 1), (4, 2), (2, 2)])
    >>> g = Genome.create(mut_df, purity=0.75)
    >>> cluster_df = g.run(env)
    >>> print(cluster_df[["time", "classification", "gf"]])
    """
    from tau.timing import _segment_time_segment, _genome_time_segments
    from tau.clustering import cluster_times_bottomup

    self.calculate_multiplicities(bootstrap_B=bootstrap_B, random_state=random_state)
    _genome_time_segments(self, env=env, n_draws=n_draws, dist_cut=dist_cut, effN_cut=effN_cut)
    result = cluster_times_bottomup(self, **cluster_kwargs)
    self._cluster_result = result
    return result[3]  # cluster_summary_df


def run_sample(
    sample: str,
    vcf_path: str | Path,
    cnv_tsv: str | Path,
    *,
    purity: float,
    env=None,
    ref_fasta: str | Path | None = None,
    cosmic_csv: str | Path | None = None,
    exposures_tsv: str | Path | None = None,
    signatures: list[str] | None = None,
    mode: str = "soft",
    bootstrap_B: int = 10,
    random_state: int | None = 42,
    n_draws: int = 50,
    dist_cut: float = 0.20,
    effN_cut: float = 10.0,
    normal_cn_map: dict[str, int] | None = None,
    detect_min_alt: int = 3,
    detect_min_vaf: float | None = None,
    apply_subclonal_filter: bool = False,
    subclonal_list: str | Path | None = None,
    subclonal_alpha: float = 0.01,
    **cluster_kwargs,
) -> tuple[Genome, pd.DataFrame]:
    """Run the full Tau pipeline from VCF + CNV files to cluster events.

    This is the recommended entry point for new analyses.  It chains:
    preprocess_sample → Genome.create → EM → timing → clustering.

    Parameters
    ----------
    sample : str
        Sample identifier (used for labelling output DataFrames).
    vcf_path : str or Path
        Path to the VCF file with somatic SNV calls.
    cnv_tsv : str or Path
        Path to the copy-number segment file (TSV).
    purity : float
        Tumour purity estimate (0–1).
    env : RouteEnv, optional
        Pre-loaded route environment (from load_routes or load_routes_for_states).
        If None, the package-bundled routes are used (requires Sage on PATH).
    ref_fasta : str or Path, optional
        Reference FASTA for SBS96 context extraction.  Required when
        ``signatures`` is also supplied.
    cosmic_csv : str or Path, optional
        COSMIC signature definitions CSV
        (download from https://cancer.sanger.ac.uk/signatures/).
    exposures_tsv : str or Path, optional
        Per-sample signature exposures TSV (rows = samples, cols = SBS names).
    signatures : list[str], optional
        Clock-like signatures to weight mutations by.  Typical choices:

        - ``["SBS1", "SBS5"]``    — clock (deamination + replication)
        - ``["SBS1"]``            — deamination only (stricter clock)

        Defaults to None (all mutations weighted equally).
    mode : str
        ``"soft"`` (default) — weight each mutation by P(clock | context).
        ``"hard"``  — binary: keep mutations whose most likely signature is a
        clock signature.
    bootstrap_B : int
        Number of bootstrap EM replicates (default 10).
    normal_cn_map : dict, optional
        Map chromosome name → normal copy number.
        Pass ``{"X": 1, "Y": 1}`` for male samples to correct hemizygous VAFs.
    apply_subclonal_filter : bool
        Run Tau's built-in low-CCF test; flagged mutations get zero weight in the EM.
        Defaults to False, matching the CLI and every published Tau run. Its power scales
        with purity, so enabling it applies a purity-dependent correction across a cohort.
    subclonal_list : str or Path, optional
        File of mutation IDs (``{chrom}:{pos}:{ref}/{alt}``, one per line) from an external
        caller such as PyClone-VI. ADDITIVE with ``apply_subclonal_filter`` — a mutation is
        excluded if either source flags it.
    subclonal_alpha : float
        Confidence level for the built-in test (default 0.01 = 99% upper bound).
    **cluster_kwargs
        Extra keyword arguments forwarded to cluster_times_bottomup().

    Returns
    -------
    genome : Genome
        Timed genome object with timing_result on each Segment.
    cluster_df : pd.DataFrame
        One row per detected event (time, classification, gf, n_chroms, …).

    Examples
    --------
    >>> from tau import run_sample, timing
    >>> env = timing.load_routes_for_states([(3, 1), (4, 2), (2, 2)])
    >>> g, events = run_sample(
    ...     "SAMPLE1",
    ...     vcf_path="sample.vcf",
    ...     cnv_tsv="sample.cnv.tsv",
    ...     purity=0.75,
    ...     env=env,
    ...     ref_fasta="hg19.fa",
    ...     cosmic_csv="COSMIC_v3.3.tsv",
    ...     exposures_tsv="exposures.tsv",
    ...     signatures=["SBS1", "SBS5"],
    ... )
    >>> print(events[["time", "classification", "gf"]])
    """
    from tau.preprocessing import preprocess_sample as _preprocess

    mut_df = _preprocess(
        sample=sample,
        vcf_path=vcf_path,
        cnv_tsv=cnv_tsv,
        purity=purity,
        ref_fasta=ref_fasta,
        cosmic_csv=cosmic_csv,
        exposures_tsv=exposures_tsv,
        signatures=signatures,
        mode=mode,
        detect_min_alt=detect_min_alt,
        detect_min_vaf=detect_min_vaf,
        apply_subclonal_filter=apply_subclonal_filter,
        subclonal_list=subclonal_list,
        subclonal_alpha=subclonal_alpha,
    )

    g = Genome.create(
        mut_df,
        purity=purity,
        normal_cn_map=normal_cn_map,
        detect_min_alt=detect_min_alt,
        detect_min_vaf=detect_min_vaf,
    )
    cluster_df = _genome_run(
        g,
        env=env,
        bootstrap_B=bootstrap_B,
        random_state=random_state,
        n_draws=n_draws,
        dist_cut=dist_cut,
        effN_cut=effN_cut,
        **cluster_kwargs,
    )
    return g, cluster_df
