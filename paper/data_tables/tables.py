"""Typed loaders for the Tier-1 composite tables.

Notebooks / panel scripts import from here so they never guess dtypes or hunt for paths:

    from paper.data_tables.tables import (load_samples, load_events, load_segments,
                                          load_routes, load_multiplicities, load_gains, load_sig_timing)

Each loader's docstring IS the column schema (kept in sync with SCHEMA.md and datapackage.json).
Conventions: `_frac` = molecular time 0=early→1=late; `_bp` = hg19 base pairs; `n_` = count; `is_`/`has_` = bool.

NOTE: tables are not populated yet (Phase 0/2). These loaders define the contract; they raise a clear
FileNotFoundError until `tau aggregate` has written the tables.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from paper.config import DATA, REF

# ── dtype maps (single source of truth for column types) ───────────────────────────────
SAMPLES_DTYPES = {
    "sample": "string", "donor_id": "string", "is_representative_aliquot": "boolean",
    "tumor_type": "string", "purity": "float64", "ploidy": "float64",
    # counts are nullable (Int64) so partially-populated migration tables load cleanly
    "n_snvs": "Int64", "n_segments": "Int64", "n_timed_segments": "Int64", "n_events": "Int64",
    "mean_ess": "float64", "has_wgd": "boolean", "has_pgd": "boolean",
    "wgd_time_frac": "float64", "timeable_genome_frac": "float64",
}
EVENTS_DTYPES = {
    "sample": "string", "tumor_type": "string", "event_id": "int64", "event_time_frac": "float64",
    "classification": "category", "class_strict": "category", "genome_frac": "float64",
    "n_chroms": "int64", "n_segments": "int64", "chrom": "string",
}
SEGMENTS_DTYPES = {
    "sample": "string", "seg_id": "string", "chrom": "string",
    "start_bp": "int64", "end_bp": "int64", "seg_len_bp": "int64",
    "major_cn": "int64", "minor_cn": "int64", "total_cn": "int64",
    "ess": "float64", "event_id": "Int64",
    "is_timed": "boolean", "discard_reason": "category", "merged_id": "string",
}
ROUTES_DTYPES = {
    "sample": "string", "seg_id": "string", "route_key": "string",
    "n_events": "int64", "n_draws": "int64", "free_var_count": "int64",
    "dist_rel": "float64", "inside_margin": "float64", "is_best_route": "boolean",
}
MULTIPLICITIES_DTYPES = {
    "sample": "string", "seg_id": "string", "multiplicity": "int64",
    "n_count": "float64", "pi": "float64",
}
GAINS_DTYPES = {
    "sample": "string", "seg_id": "string", "route_key": "string", "is_best_route": "boolean",
    "chrom": "string", "start_bp": "int64", "end_bp": "int64", "major_cn": "int64", "minor_cn": "int64",
    "gain_index": "int64", "draw_id": "int64", "gain_time_frac": "float64", "n_draws": "int64",
    "event_id": "Int64",
}
# ── MERGED (pooled) level — same emitter on the clustered genome ───────────────────────
# merged_segments drops physical coords (a pooled unit spans many loci) and adds n_segments.
MERGED_SEGMENTS_DTYPES = {
    "sample": "string", "seg_id": "string",
    "major_cn": "int64", "minor_cn": "int64", "total_cn": "int64",
    "ess": "float64", "event_id": "Int64", "is_timed": "boolean", "discard_reason": "category",
    "n_segments": "int64",
}
MERGED_MULTIPLICITIES_DTYPES = dict(MULTIPLICITIES_DTYPES)
MERGED_ROUTES_DTYPES = dict(ROUTES_DTYPES)
MERGED_GAINS_DTYPES = {
    "sample": "string", "seg_id": "string", "route_key": "string", "is_best_route": "boolean",
    "major_cn": "int64", "minor_cn": "int64",
    "gain_index": "int64", "draw_id": "int64", "gain_time_frac": "float64", "n_draws": "int64",
    "event_id": "Int64",
}
MERGED_EVENTS_DTYPES = {
    "sample": "string", "tumor_type": "string", "event_id": "int64", "event_time_frac": "float64",
    "classification": "category", "genome_frac": "float64",
    "n_chroms": "int64", "n_segments": "int64", "chrom": "string",
}
CYTOBAND_DTYPES = {
    "chrom": "string", "arm": "string", "start_bp": "int64", "end_bp": "int64",
    "arm_len_bp": "int64", "centromere_bp": "int64", "chrom_size_bp": "int64",
}
SIG_TIMING_DTYPES = {
    "sample": "string", "tumor_type": "string", "seg_id": "string", "signature": "string",
    "gain_index": "int64", "ess": "float64", "G_clock": "float64", "G_sig": "float64",
    "ts_clock": "float64", "ts_sig": "float64", "frac": "float64",
}


def _read_tsv(path: Path, dtypes: dict) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not built yet — run the batch layer (`tau aggregate`). Expected: {path}")
    return pd.read_csv(path, sep="\t", dtype=dtypes)


def load_samples() -> pd.DataFrame:
    """One row per **sample**. Columns: sample, donor_id, is_representative_aliquot, tumor_type, purity,
    ploidy, n_snvs, n_segments, n_timed_segments, n_events, mean_ess, has_wgd, has_pgd, wgd_time_frac,
    timeable_genome_frac. Filter is_representative_aliquot for per-sample stats (donor-dedup)."""
    return _read_tsv(DATA / "tau_samples.tsv", SAMPLES_DTYPES)


def load_events() -> pd.DataFrame:
    """One row per **clustered event**. Columns: sample, tumor_type, event_id, event_time_frac,
    classification{WGD,PGD,chrom_specific}, class_strict{WGD,PGD,WGD_with_loss}, genome_frac,
    n_chroms, n_segments, chrom."""
    return _read_tsv(DATA / "tau_events.tsv", EVENTS_DTYPES)


def load_segments() -> pd.DataFrame:
    """One row per **physical segment**. Columns: sample, seg_id, chrom, start_bp, end_bp, seg_len_bp,
    major_cn, minor_cn, total_cn, ess, event_id, is_timed, discard_reason, merged_id. Route fits are in
    tau_routes; the multiplicity distribution in tau_multiplicities. Join events on (sample, event_id);
    join the pooled unit on (sample, merged_id → tau_merged_segments.seg_id)."""
    return _read_tsv(DATA / "tau_segments.tsv", SEGMENTS_DTYPES)


def load_routes() -> pd.DataFrame:
    """One row per **(sample, seg_id, route_key)** — ALL feasible routes (not just best). Columns:
    sample, seg_id, route_key, n_events, n_draws, free_var_count, dist_rel, inside_margin, is_best_route."""
    return _read_tsv(DATA / "tau_routes.tsv", ROUTES_DTYPES)


def load_multiplicities() -> pd.DataFrame:
    """One row per **(sample, seg_id, multiplicity)** — the N_counts distribution. Columns:
    sample, seg_id, multiplicity (k=1..major_cn), n_count (weighted SNVs at k), pi (normalized)."""
    return _read_tsv(DATA / "tau_multiplicities.tsv", MULTIPLICITIES_DTYPES)


def load_route_topology() -> pd.DataFrame:
    """One row per **(route_key, event)** — the shipped tree-topology lookup (from route_topology.json.gz).
    Columns: route_key, major_cn, minor_cn, event_index, allele{major,minor}, parent_mult, split_a, split_b.
    Reconstruct an allele tree from this + a segment's gain times (tau_gains) — no Sage needed."""
    path = REF / "ref_route_topology.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"ref_route_topology not built — run build_ref_and_bench.py. Expected: {path}")
    return pd.read_csv(path, sep="\t")


def load_bench(name: str = "events") -> pd.DataFrame:
    """Benchmark tables, all DERIVED FROM THE SIMULATION run. name in
    {events, truth_events, tree_metrics, oracle_lift, pcawg_mutcount, seg_gains, route_accuracy,
     route_selection, route_agreement, route_agreement_per_gain, route_equivalence,
     route_roc, route_pair_error}.
    - bench_events       = one row per CALLED event (event_time_frac, true_time_frac, is_matched,
                           classification, true_type, timing_error_frac) → false-positives, matched timing.
    - bench_truth_events = one row per TRUE event (true_type, true_time_frac, event_size_frac, min_dist_frac,
                           is_detected, recovered_time_frac, timing_error_frac) → recall / false-negatives.
    - bench_seg_gains    = one row per simulated GAIN (SEGMENT-level benchmarking): major, minor, mut_count,
                           gain_index, true_time, time_lo/time_hi (estimated band; lo==hi for determined),
                           underdetermined (bool), truth_in_band (bool), dist_to_set (0 if truth inside the
                           band, else distance to nearest edge = the faithful error metric).
    - bench_route_accuracy = one row per simulated TREE: major, minor, mc, acc_oracle (max-gain error using
                           the TRUE route), acc_bestguess (using Tau's OWN route pick), route_correct (bool),
                           undet. Shows per-tree timing barely degrades under real route uncertainty.
    - bench_route_selection = one row per simulated TREE, scoring the SAME max-gain error four ways:
                           acc_oracle (true route), acc_chosen (Tau's pick), acc_mean (mean over all
                           feasible routes = expected accuracy with NO discrimination), acc_worst /
                           acc_bestcase (bounds over feasible routes), n_feasible, route_correct.
    - bench_route_equivalence = one row per simulated SEGMENT, scoring route recovery against the
                           EQUIVALENCE CLASS rather than the simulator's arbitrary tag. The simulator draws
                           true_route with _rng.choice from the routes compatible with the segment's history,
                           but several routes often predict an IDENTICAL multiplicity vector at the true
                           timing, so they are indistinguishable from the data and each admits the true
                           timing. Cols: major, minor, mc, n_tied (routes at min distance), class_size,
                           exact_match (== the recorded tag), class_match (in the class), harmful_ties.
    - bench_route_roc      = ROC for route selection, averaged vertically over segments on a common
                           FPR grid. Cols: scope ('overall' | 'tied' | a CN state 'M_m'), fpr, tpr,
                           n_segments, auc. Positives = the equivalence class; ranking is by
                           dist_rel with tied scores collapsed to one point on the curve.
    - bench_route_pair_error = one row per PAIR of candidate routes on a segment: max_gap /
                           mean_gap = distance between the two routes' solution intervals (0 where
                           they overlap), max/mean over the shared gains; both_tied, n_tied, n_cand.
    - bench_route_agreement_per_gain = one row per (segment, gain_index): mean_pair_diff /
                           max_pair_diff = disagreement at that gain between routes TIED at the
                           minimum distance, averaged/maxed over tied pairs; plus n_tied,
                           n_pairs, n_gains, gain_frac (index/n_gains, comparable across states).
    - bench_route_agreement = one row per pair of simultaneously-FEASIBLE routes on the same segment:
                           key_a/key_b, n_gains, max_abs_diff / mean_abs_diff between their median gain
                           times, n_feasible, spread (max disagreement across all feasible routes).
                           Rows with blank key_a/key_b are summary-only (too many routes to enumerate)."""
    path = DATA / f"bench_{name}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"bench_{name} not built — run build_ref_and_bench.py. Expected: {path}")
    return pd.read_csv(path, sep="\t")


def load_sig_timing() -> pd.DataFrame:
    """One row per **(sample, seg_id, signature, gain_index)** — cumulative signature-vs-clock activity up to
    each gain event (Fig 2 SBS17 panels). Columns: sample, tumor_type, seg_id, signature{SBS17,SBS18,SBS7,
    SBS4,SBS3,SBS2+SBS13}, gain_index, ess, G_clock, G_sig (cumulative clock/signature activity share),
    ts_clock, ts_sig (flat-time baselines for normalisation), frac (signature fractional activity on segment).
    Built from the existing per-signature caches by build_sig_timing.py; new signatures need `tau run --signatures`."""
    return pd.read_csv(DATA / "tau_sig_timing.tsv.gz", sep="\t", dtype=SIG_TIMING_DTYPES)


def load_gains(columns: list[str] | None = None, best_route_only: bool = False) -> pd.DataFrame:
    """One row per **(sample, seg_id, route_key, gain_index, draw_id)** — full polytope timing (long).
    Includes ALL feasible routes; pass best_route_only=True to filter to is_best_route (most panels).
    Columns: sample, seg_id, route_key, is_best_route, chrom, start_bp, end_bp, major_cn, minor_cn,
    gain_index, draw_id, gain_time_frac, n_draws. Parquet-backed; pass `columns=` for a subset (faster).
    Join to segments on (sample, seg_id), to routes on (sample, seg_id, route_key)."""
    # gz-TSV (pyarrow not in env; parquet is a drop-in later — then this reads tau_gains.parquet)
    path = DATA / "tau_gains.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"tau_gains.tsv.gz not built yet — run the backfill. Expected: {path}")
    use = columns
    if best_route_only and columns is not None and "is_best_route" not in columns:
        use = list(columns) + ["is_best_route"]   # ensure the filter column is read
    df = pd.read_csv(path, sep="\t", dtype=GAINS_DTYPES, usecols=use)
    if best_route_only:
        df = df[df.is_best_route]
        if columns is not None and "is_best_route" not in columns:
            df = df[list(columns)]
    return df


# ── MERGED (pooled) level loaders — the clustered-genome grain ──────────────────────────
def load_merged_segments() -> pd.DataFrame:
    """One row per **pooled unit** (merged_id). Columns: sample, seg_id (= tau_segments.merged_id),
    major_cn, minor_cn, total_cn, ess, event_id, is_timed, discard_reason, n_segments (constituent segments).
    Physical coords are dropped (a pooled unit spans many loci — recover them via tau_segments.merged_id)."""
    return _read_tsv(DATA / "tau_merged_segments.tsv", MERGED_SEGMENTS_DTYPES)


def load_merged_multiplicities() -> pd.DataFrame:
    """One row per **(sample, pooled seg_id, multiplicity)** — the pooled N_counts distribution.
    Columns: sample, seg_id, multiplicity, n_count, pi."""
    return _read_tsv(DATA / "tau_merged_multiplicities.tsv", MERGED_MULTIPLICITIES_DTYPES)


def load_merged_routes() -> pd.DataFrame:
    """One row per **(sample, pooled seg_id, route_key)** — pooled route fits. Columns:
    sample, seg_id, route_key, n_events, n_draws, free_var_count, dist_rel, inside_margin, is_best_route."""
    return _read_tsv(DATA / "tau_merged_routes.tsv", MERGED_ROUTES_DTYPES)


def load_merged_events() -> pd.DataFrame:
    """One row per **pooled clustered event**. Columns: sample, tumor_type, event_id, event_time_frac,
    classification, genome_frac, n_chroms, n_segments, chrom (no class_strict — that is a later annotation)."""
    return _read_tsv(DATA / "tau_merged_events.tsv", MERGED_EVENTS_DTYPES)


def load_merged_gains(columns: list[str] | None = None, best_route_only: bool = False) -> pd.DataFrame:
    """One row per **(sample, pooled seg_id, route_key, gain_index, draw_id)** — pooled polytope timing.
    Columns: sample, seg_id, route_key, is_best_route, major_cn, minor_cn, gain_index, draw_id,
    gain_time_frac, n_draws, event_id (physical coords dropped vs tau_gains). Join events on (sample, event_id)."""
    path = DATA / "tau_merged_gains.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"tau_merged_gains.tsv.gz not built yet — run the backfill. Expected: {path}")
    use = columns
    if best_route_only and columns is not None and "is_best_route" not in columns:
        use = list(columns) + ["is_best_route"]
    df = pd.read_csv(path, sep="\t", dtype=MERGED_GAINS_DTYPES, usecols=use)
    if best_route_only:
        df = df[df.is_best_route]
        if columns is not None and "is_best_route" not in columns:
            df = df[list(columns)]
    return df


def load_cytoband() -> pd.DataFrame:
    """One row per **chromosome arm** (hg19). Columns: chrom, arm{p,q}, start_bp, end_bp, arm_len_bp,
    centromere_bp, chrom_size_bp. Built by batch_scripts/build_ref_cytoband.py from the shipped
    centromere + chrom-size files — one source for arm-level figures instead of hardcoded dicts."""
    path = REF / "ref_hg19_cytoband.tsv"
    if not path.exists():
        raise FileNotFoundError(f"ref_hg19_cytoband not built — run build_ref_cytoband.py. Expected: {path}")
    return pd.read_csv(path, sep="\t", dtype=CYTOBAND_DTYPES)
