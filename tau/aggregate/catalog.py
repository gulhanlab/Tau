"""tau.aggregate.catalog — compile per-sample interactive viewers into a browsable HTML catalog (SKETCH).

Shippable cohort tool (`tau catalog`): after running Tau on many samples, build one searchable catalog to
click through them. Generalizes dev/pcawg/catalog/build_catalog*.py (which built the 965-sample PCAWG
catalog). STATUS: interface only.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional


def build_catalog(run_dir: Path, out_dir: Path, samples_table: Optional[Path] = None,
                  regenerate_missing: bool = True, shard_size: int = 200) -> Path:
    """Build an HTML catalog of the per-sample interactive viewers under `run_dir`.

    Steps
    -----
    1. For each sample: ensure `{sample}.overview_interactive.html` exists; if missing and
       `regenerate_missing`, render it from `{sample}.genome.pkl.gz` via `tau.viz.genome_to_html`
       (Sage-free, no re-timing). Copy/symlink viewers into `out_dir/viewers/`.
    2. Load per-sample metadata from `samples_table` (the `tau_samples.tsv` produced by `tau aggregate`;
       falls back to scanning `{sample}.tau_sample.tsv` files). Columns become the searchable/sortable
       fields (tumor_type, ploidy, has_wgd, has_pgd, n_events, wgd_time_frac, …).
    3. Write `out_dir/index.html` — a client-side searchable/filterable/sortable table, one row per sample
       linking to `viewers/{sample}.overview_interactive.html`. Shard the metadata into
       `out_dir/_meta/shard_*.json` (`shard_size` rows each) so the index stays responsive for large cohorts.

    Returns the path to `index.html`. Note: viewers dominate size (~7 MB each; the PCAWG catalog was 6.66 GB
    for 965) — the index + shards are tiny.
    """
    raise NotImplementedError
