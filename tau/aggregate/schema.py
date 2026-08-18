"""tau.aggregate.schema — column/dtype definitions for the composite tables (source of truth for the emitter).

Kept in sync with paper/data_tables/SCHEMA.md (human) and the generated datapackage.json (machine, Frictionless
Table Schema). The paper figure layer keeps its OWN Sage-free copy in paper/data_tables/tables.py (importing
this module would pull tau/__init__ → Sage); both are validated against datapackage.json so they can't drift.

STATUS: fill in to match SCHEMA.md. Column order here defines the written column order.
"""

# {table_name: {column: pandas_dtype}}  — mirror of the maps in paper/data_tables/tables.py
COLUMNS: dict[str, dict[str, str]] = {
    "sample": {
        "sample": "string", "donor_id": "string", "is_representative_aliquot": "boolean",
        "tumor_type": "string", "purity": "float64", "ploidy": "float64",
        "n_snvs": "int64", "n_segments": "int64", "n_timed_segments": "int64", "n_events": "int64",
        "mean_ess": "float64", "has_wgd": "boolean", "has_pgd": "boolean",
        "wgd_time_frac": "float64", "timeable_genome_frac": "float64",
    },
    "events": {
        "sample": "string", "tumor_type": "string", "event_id": "int64", "event_time_frac": "float64",
        "classification": "category", "class_strict": "category", "genome_frac": "float64",
        "n_chroms": "int64", "n_segments": "int64", "chrom": "string",
    },
    "segments": {
        "sample": "string", "seg_id": "string", "chrom": "string",
        "start_bp": "int64", "end_bp": "int64", "seg_len_bp": "int64",
        "major_cn": "int64", "minor_cn": "int64", "total_cn": "int64",
        "ess": "float64", "event_id": "Int64", "is_timed": "boolean", "discard_reason": "category",
        "merged_id": "string",
    },
    "routes": {
        "sample": "string", "seg_id": "string", "route_key": "string",
        "n_events": "int64", "n_draws": "int64", "free_var_count": "int64",
        "dist_rel": "float64", "inside_margin": "float64", "is_best_route": "boolean",
    },
    "multiplicities": {
        "sample": "string", "seg_id": "string", "multiplicity": "int64", "n_count": "float64", "pi": "float64",
    },
    "gains": {
        "sample": "string", "seg_id": "string", "route_key": "string", "is_best_route": "boolean",
        "chrom": "string", "start_bp": "int64", "end_bp": "int64", "major_cn": "int64", "minor_cn": "int64",
        "gain_index": "int64", "draw_id": "int64", "gain_time_frac": "float64", "n_draws": "int64",
        "event_id": "Int64",
    },
    # ── MERGED (pooled) level: same emitter on the clustered genome ────────────────────────
    # merged_segments = segments minus physical coords (a pooled unit spans many loci) + n_segments.
    "merged_segments": {
        "sample": "string", "seg_id": "string",
        "major_cn": "int64", "minor_cn": "int64", "total_cn": "int64",
        "ess": "float64", "event_id": "Int64", "is_timed": "boolean", "discard_reason": "category",
        "n_segments": "int64",
    },
    "merged_multiplicities": {
        "sample": "string", "seg_id": "string", "multiplicity": "int64", "n_count": "float64", "pi": "float64",
    },
    "merged_routes": {
        "sample": "string", "seg_id": "string", "route_key": "string",
        "n_events": "int64", "n_draws": "int64", "free_var_count": "int64",
        "dist_rel": "float64", "inside_margin": "float64", "is_best_route": "boolean",
    },
    # merged_gains = gains minus physical coords + per-gain event_id.
    "merged_gains": {
        "sample": "string", "seg_id": "string", "route_key": "string", "is_best_route": "boolean",
        "major_cn": "int64", "minor_cn": "int64",
        "gain_index": "int64", "draw_id": "int64", "gain_time_frac": "float64", "n_draws": "int64",
        "event_id": "Int64",
    },
    # merged_events = tau_events schema as EMITTED (no class_strict, which is a later annotation).
    "merged_events": {
        "sample": "string", "tumor_type": "string", "event_id": "int64", "event_time_frac": "float64",
        "classification": "category", "genome_frac": "float64",
        "n_chroms": "int64", "n_segments": "int64", "chrom": "string",
    },
    "sig_timing": {
        "sample": "string", "tumor_type": "string", "seg_id": "string", "signature": "string",
        "gain_index": "int64", "ess": "float64", "G_clock": "float64", "G_sig": "float64",
        "ts_clock": "float64", "ts_sig": "float64", "frac": "float64",
    },
}

# partition column for the tau_gains parquet dataset
GAINS_PARTITION = "tumor_type"   # or "sample"


def columns(table: str) -> list[str]:
    """Ordered column names for a table (defines write order)."""
    return list(COLUMNS[table])
