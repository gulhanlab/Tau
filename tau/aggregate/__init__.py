"""tau.aggregate — composite table emit + cohort aggregation (SKETCH / interface).

Two grains, one schema (see paper/data_tables/SCHEMA.md):
  • per-sample tables — written by `tau run` or `tau export-tables <genome.pkl>` (backfill, no re-timing)
  • cohort tables     — `tau aggregate <run_dir>` concatenates them (partitioned parquet for tau_gains)

`schema.py` is the column/dtype source of truth (kept in sync with SCHEMA.md, paper/data_tables/tables.py,
and the generated datapackage.json). `export.py` holds the pkl→table logic (used by both `tau run` and
`tau export-tables`). `aggregate.py` does the cohort concat/partition.

The emit + aggregate layers are IMPLEMENTED (lifted from paper/batch_scripts/backfill_chunk.py +
merge_backfill.py). `benchmark.py` / `catalog.py` are the cohort evaluation + catalog tools.
"""
from tau.aggregate.export import (export_sample_tables, write_sample_tables,
                                   export_tables_from_pkl, TABLE_FILENAMES)
from tau.aggregate.aggregate import aggregate_run
from tau.aggregate.catalog import build_catalog
from tau.aggregate.benchmark import build_bench_tables

__all__ = ["export_sample_tables", "write_sample_tables", "export_tables_from_pkl",
           "aggregate_run", "build_catalog", "build_bench_tables", "TABLE_FILENAMES"]
