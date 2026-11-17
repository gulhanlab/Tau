# The data tables — what each one contains

Plain-language guide to `paper/data_tables/`. These tables are the single source for every figure: they were
built once from Tau's per-sample outputs, and the figure scripts read these rather than the per-sample objects.

**Load any table** with the typed helpers (correct types + a docstring schema):
```python
from paper.data_tables.tables import load_samples, load_events, load_segments, load_routes, \
    load_multiplicities, load_gains, load_route_topology, load_bench
s = load_samples()                        # a pandas DataFrame
g = load_gains(best_route_only=True)      # the chosen route only (much smaller)
```

**Naming conventions** (so a column's meaning is obvious):
- `_frac` = molecular time, **0 = early → 1 = late** (e.g. `event_time_frac`, `gain_time_frac`)
- `_bp` = genomic coordinate in hg19 base pairs (`start_bp`, `end_bp`)
- `n_` = a count · `is_`/`has_` = a yes/no flag
- Tables **join** on `sample`, `seg_id`, and (for gains/routes) `route_key`.

---

## The core tables (one grain each — "one row per …")

### `tau_samples.tsv` — one row per **tumour sample** (2,703)
Per-sample metadata: `tumor_type`, `purity`, `ploidy`, mutation burden (`n_snvs`), segment/event counts,
and the doubling summary — `has_wgd`, `has_pgd`, `wgd_time_frac` (when the WGD happened). `donor_id` +
`is_representative_aliquot` let you de-duplicate multi-aliquot donors. *Use: cohort grouping/filtering.*

### `tau_events.tsv` — one row per **timed event** (2,351)
Each clustered copy-number event in a sample: its `classification` (WGD / PGD / chrom-specific), strict
re-class (`class_strict`), when it happened (`event_time_frac`), and how much genome it touched
(`genome_frac`, `n_chroms`). *Use: WGD/PGD timing & prevalence.*

### `tau_segments.tsv` — one row per **copy-number segment** (569,218)
The physical segments: coordinates (`chrom`, `start_bp`, `end_bp`), copy number (`major_cn`, `minor_cn`),
effective SNV support (`ess`), which event it was assigned to (`event_id`), and whether/why it was timed
(`is_timed`, `discard_reason`). *The "header" that the timing detail hangs off.*

### `tau_routes.tsv` — one row per **(segment × possible gain-ordering)** (345,536)
For each segment, every feasible "route" (order the copy-number gains could have happened in): how well it
fits the data (`dist_rel`, `inside_margin`), its degrees of freedom (`free_var_count`), and whether it's the
chosen one (`is_best_route`). *Use: route feasibility / timing identifiability.*

### `tau_multiplicities.tsv` — one row per **(segment × multiplicity)** (1,492,583)
The SNV multiplicity distribution per segment — `n_count` SNVs at each multiplicity `k`, and its normalized
form `pi`. This is Tau's raw timing input (`N_counts`). *Use: multiplicity diagnostics (e.g. is N3 ≈ 0?).*

### `tau_gains.tsv.gz` — one row per **(segment × route × gain × polytope-draw)** (14,420,187)
**The core timing output**: the molecular time (`gain_time_frac`) of each individual copy-number gain, for
every route and every polytope solution ("draw"). Because Tau timing is often under-determined, a gain has
many draws — average them, or use `load_gains(best_route_only=True)` for the chosen route (3.6M rows).
A 500-row preview sits in `tau_gains.head.tsv`. *Use: gain-density landscapes, swimmers, burst analysis —
burst analysis.* (Gzipped TSV for now; a Parquet drop-in later.)

---

## Reference tables (`ref/`)
- **`ref_oncokb_oncogenes_hg19.tsv`** (483) — OncoKB oncogene loci (`gene`, `chrom`, `pos_bp`) for labelling
  hotspots / overlap tests.
- **`ref_route_topology.tsv.gz`** (1.39M) — the **tree shape** of each route (which allele splits at each
  event). Combine with a segment's gain times (`tau_gains`) to draw its allele tree — no Sage needed.

## Benchmark tables (`bench_*`)
Tau vs known simulation truth: `bench_events` (recovered vs true event times, `is_matched`,
`timing_error_frac`), `bench_tree_metrics` (tree accuracy by copy-number state), `bench_oracle_lift`,
`bench_pcawg_mutcount`.

---

## Signature tables
- **`tau_sig_timing.tsv.gz`** — per-signature gain timing (one row per sample x segment x gain x signature),
  built by `../batch_scripts/build_sig_timing.py`.
- **`tau_sig_ratio.tsv.gz`**, **`tau_sig_ratio_SBS17_vs_APOBEC.tsv.gz`** — signature-to-clock interval ratios at
  segment and cluster level (`../batch_scripts/build_sig_ratio_tables.py`).

## Other directories in the data bundle
- **`sim/`** — the composite tables of the simulated cohort (same schema as the PCAWG tables).
- **`per_sample/<sample>/`** — per-sample bundles (re-clustered objects and per-sample tables) for the example tumours.
- **`objects/`** — the few raw Tau objects the object-based panels render (two PCAWG examples, three simulated genomes).
- **`frozen/`** — small outputs of the upstream analysis scripts (`../upstream/PROVENANCE.md`).

Full column-level schema + types: **`SCHEMA.md`**. How the tables were built: **`../batch_scripts/`**.
