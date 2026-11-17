# Composite table schemas

> Column-level schema of the composite tables. Load them with `tables.py` (typed loaders). `tau_gains` is a
> gzipped TSV (`tau_gains.tsv.gz`). `SCHEMA.md`, `tables.py` and `tau/aggregate/schema.py` describe the same tables.

## Format & conventions (interpretability contract)
1. **Tidy / long** — one row = one observation. No wide `G1…G9`; `tau_gains` is long over `(gain_index, draw_id)`.
2. **Names carry units/meaning:**
   - `_frac` = molecular-time fraction, **0 early → 1 late** (`event_time_frac`, `gain_time_frac`, `wgd_time_frac`).
   - `_bp` = hg19 base pairs (`start_bp`, `end_bp`, `seg_len_bp`).
   - `n_` = count (`n_draws`, `n_events`); `is_`/`has_` = boolean (`is_wgd`, `has_pgd`, `is_timed`).
   - established domain terms kept as-is: `ess`, `dist_rel`, `route_key`, `inside_margin`.
3. **Consistent join keys everywhere:** `sample`, `seg_id`, `event_id`, `route_key`.
4. **File format:** plain **TSV** for human-scale tables + `ref_*`/`bench_*`; **Parquet** for `tau_gains`
   (typed, ~10× faster load) with a committed `tau_gains.head.tsv` (first ~500 rows) for human inspection.
5. **Self-documenting:** this `SCHEMA.md` + a Frictionless **`datapackage.json`** + a typed loader `tables.py`.
6. **Deterministic order:** sort by `sample`, then `chrom, start_bp` (then `route_key, gain_index, draw_id`).

Design rules: Tier 1 (`tau_*`, this dir) is generic, per-draw / **lossless w.r.t. the genome object**, NO
berth-baked columns (linkage, bursts) and NO paper curation. Tier 2 (`figures/figN/cache_tables/`) applies
berths/filters/curation.

**Normalization (header → detail):** `tau_segments` (one physical segment) → `tau_routes` (each feasible
route fit) → `tau_gains` (each gain × draw). Plus `tau_multiplicities` (the N_counts distribution) and
`tau_events` (clusters). Join on `sample` + `seg_id` (+ `route_key`).

## How the tables are produced (emit → concat)
These same schemas exist at two grains, each emitted at **two levels** — the INDIVIDUAL (raw-genome)
level and the MERGED (clustered-genome / pooled) level:
- **Per-sample** (in each Tau output dir), individual: `{sample}.tau_samples.tsv`, `{sample}.tau_segments.tsv`,
  `{sample}.tau_events.tsv`, `{sample}.tau_routes.tsv`, `{sample}.tau_multiplicities.tsv`,
  `{sample}.tau_gains.tsv.gz` (+ `{sample}.tau_sig_timing.tsv` when signatures run); and merged:
  `{sample}.tau_merged_segments.tsv`, `{sample}.tau_merged_multiplicities.tsv`,
  `{sample}.tau_merged_routes.tsv`, `{sample}.tau_merged_gains.tsv.gz`, `{sample}.tau_merged_events.tsv`.
  Written by **`tau run`** (new runs) or **`tau export-tables <genome.pkl>`** (backfill an existing pkl —
  cheap reshape, no re-timing; loads the sibling `clustered_genome.pkl.gz` + `segment_clusters.tsv` and
  re-runs `cluster_times_bottomup` for events). Emit logic: `tau/aggregate/export.py::build_all_tables`.
- **Cohort** (this dir): `tau aggregate` **concatenates** the per-sample tables. Same columns, so it's a plain
  union. The heavy `tau_gains` is a **partitioned parquet dataset** (partition = `tumor_type`/`sample`) — each
  sample drops its partition, so the cohort table is just the directory: `pd.read_parquet(DATA/'tau_gains')`
  reads the union with predicate pushdown (`filters=[('is_best_route','==',True)]`), no concat step.

The pkl→table logic lives once in `tau/aggregate/export.py`; `schema.py` there is the col/dtype source of
truth (kept in sync with this file, `tables.py`, and `datapackage.json`).

---

## `tau_samples.tsv` — one row per **sample**
Collapses `samples_allpcawg.tsv` + `timeable_denom.tsv`.

| column | type | unit | description |
|---|---|---|---|
| `sample` | str | | aliquot UUID (primary key) |
| `donor_id` | str | | donor this aliquot belongs to (PCAWG has multi-aliquot donors) |
| `is_representative_aliquot` | bool | | the one aliquot to keep per donor (donor-dedup; filter for per-sample stats) |
| `tumor_type` | str | | e.g. `Breast.AdenoCA` |
| `purity` | float | | tumour purity |
| `ploidy` | float | | genome-wide average ploidy |
| `n_snvs` | int | | total SNVs (Σ N_counts across segments) — mutation burden |
| `n_segments` | int | | CN segments total |
| `n_timed_segments` | int | | segments with a timing result |
| `n_events` | int | | clustered events (rows in tau_events for this sample) |
| `mean_ess` | float | | mean effective sample size over timed segments (timing-power proxy) |
| `has_wgd` | bool | | any WGD event |
| `has_pgd` | bool | | any strict PGD event |
| `wgd_time_frac` | float | frac | earliest WGD event time; NaN if none |
| `timeable_genome_frac` | float | frac | timeable denom (genome − gained-but-untimeable); the gf denominator |

---

## `tau_events.tsv` — one row per **clustered event**
Collapses `events_allpcawg.tsv` + `event_catalogue_allpcawg_annotated.tsv`.

| column | type | unit | description |
|---|---|---|---|
| `sample` | str | | FK → tau_samples |
| `tumor_type` | str | | denormalized |
| `event_id` | int | | per-sample cluster index |
| `event_time_frac` | float | frac | 0 early → 1 late |
| `classification` | str | enum | `WGD` / `PGD` / `chrom_specific` |
| `class_strict` | str | enum | strict re-annotation: `WGD` / `PGD` / `WGD_with_loss` |
| `genome_frac` | float | frac | genome fraction affected |
| `n_chroms` | int | | chromosomes in the cluster |
| `n_segments` | int | | segments in the cluster |
| `chrom` | str | | for `chrom_specific`; else NA |

---

## `tau_segments.tsv` — one row per **physical segment** (the "header")
Physical + clustering facts only; route-specific fields moved to `tau_routes`.

| column | type | unit | description |
|---|---|---|---|
| `sample` | str | | FK → tau_samples |
| `seg_id` | str | | `{chrom}:{start_bp}-{end_bp}` (primary key with sample) |
| `chrom` | str | | 1–22, X, Y |
| `start_bp` / `end_bp` | int | bp | hg19 |
| `seg_len_bp` | int | bp | end − start |
| `major_cn` / `minor_cn` | int | | copy number |
| `total_cn` | int | | major + minor |
| `ess` | float | | Σ N_counts (effective sample size) |
| `event_id` | int | | tau_events cluster this segment's timing was assigned to (representative = unique event over the segment's gains, else NA); from `cluster_times_bottomup`'s own seg→event map |
| `is_timed` | bool | | timing succeeded |
| `discard_reason` | str | enum | NA if timed; else `no_routes_for_state` / `low_effective_N` / `dist_cut` / `nonpositive_cn` |
| `merged_id` | str | | the pooled unit this segment belongs to (`merged_{major}_{minor}_{label}`, or the seg's own id when passthrough); FK → `tau_merged_segments.seg_id`. NA if the segment is absent from `segment_clusters.tsv`. From the STORED clustering, not a re-run. |

---

## `tau_routes.tsv` — one row per **(sample, seg_id, route)** — ALL feasible routes ★NEW
Conserves the full route set from `timing_result` (not just `pick_best_key`) — feasibility, fit, identifiability.

| column | type | unit | description |
|---|---|---|---|
| `sample` | str | | FK |
| `seg_id` | str | | FK → tau_segments |
| `route_key` | str | | e.g. `4_2.3` |
| `n_events` | int | | gain events on the route (= max gain_index) |
| `n_draws` | int | | polytope solutions (1 = determined) |
| `free_var_count` | int | | timing degrees of freedom (0 = determined) |
| `dist_rel` | float | | route fit quality (lower = better) |
| `inside_margin` | float | | depth inside the feasible region (identifiability / polytope-volume signal) |
| `is_best_route` | bool | | the route `pick_best_key` selects (min dist_rel) |

---

## `tau_multiplicities.tsv` — one row per **(sample, seg_id, multiplicity)** ★NEW
Conserves `N_counts` — the per-multiplicity SNV distribution (the fundamental timing input; used by the overview panels,
N3 diagnostics). `snv_table` (per-SNV VAF) is intentionally dropped — `N_counts` is its sufficient statistic.

| column | type | unit | description |
|---|---|---|---|
| `sample` | str | | FK |
| `seg_id` | str | | FK → tau_segments |
| `multiplicity` | int | | multiplicity class k = 1..major_cn |
| `n_count` | float | | weighted SNV count at multiplicity k (`N_counts[k-1]`) |
| `pi` | float | frac | normalized (`n_count / Σ`) — the mixture proportion |

*Open:* also carry bootstrap `N_counts_boot` (a `boot_id` column) for bootstrap CIs? Current cache is
polytope-only (bootstrap collapsed) — add only if a figure needs multiplicity-bootstrap uncertainty.

---

## `tau_gains.parquet` — one row per **(sample, seg_id, route_key, gain_index, draw_id)** — full polytope
Collapses `gain_wgd_cache` + `oncogene_timing_cache` + `highcn_gain_cache` + `determined_gain_times`.
**The heavy, lossless timing table** (long). Includes ALL feasible routes (`is_best_route` to filter).
Ship `tau_gains.head.tsv` for human inspection.

| column | type | unit | description |
|---|---|---|---|
| `sample` | str | | FK |
| `seg_id` | str | | FK → tau_segments |
| `route_key` | str | | FK → tau_routes |
| `is_best_route` | bool | | denormalized from tau_routes (most panels filter to this) |
| `chrom` | str | | denormalized so gain-density is a single-table groupby |
| `start_bp` / `end_bp` | int | bp | segment extent (denormalized) |
| `major_cn` / `minor_cn` | int | | denormalized |
| `gain_index` | int | | 1..k, order of the gain event on the route |
| `draw_id` | int | | polytope solution index (0..n_draws−1) |
| `gain_time_frac` | float | frac | cumulative event time of this gain in this draw (interconvertible with raw `t`) |
| `n_draws` | int | | # solutions (for `1/n_draws` smear weight) |
| `event_id` | int | | the clustered event this specific gain was assigned to (per-gain, from the seg→event map); NA if unclustered |

Supports: **gain density** (row = gain at (extent, `gain_time_frac`), weight `1/n_draws`; filter `is_best_route`);
**swimmer/dotplot** (median `gain_time_frac` over `draw_id`); **bursts** (group `sample,seg_id,route_key,draw_id`,
sort `gain_index` → co-timed-window reduction). Linkage/oncogene added in Tier 2.
*Open (size):* including non-best routes multiplies rows on underdetermined segments — keep for strict
losslessness (recommended) or best-route-only + recompute alternatives on demand.
*Open (QC):* add per-draw `projected` / `inside_margin`? (route-level `inside_margin` already in tau_routes.)

---

## Merged (pooled) level — `tau_merged_*` ★NEW
The SAME emitter run on the **clustered genome** (multiplicity-pooled units), so every individual table has
a pooled twin at the merged grain. The pooled `seg_id` is the `merged_id` that `tau_segments` carries, so
`tau_segments.merged_id → tau_merged_segments.seg_id` joins a physical segment to its pooled unit. Pooled
`event_id` uses the POC **propagation** method: each pooled unit's (cleaner) timing is propagated onto its
constituent raw segments — which keep their real chrom — and `cluster_times_bottomup` is re-run on those
(directly clustering the merged genome fails: a unit spans many chroms). `tau_merged_events` is that re-run's
clustering; each unit's `event_id` is the modal event over its constituents; pooled gains inherit per-gain.

- **`tau_merged_segments.tsv`** — one row per **pooled unit**. = `tau_segments` **minus** `chrom/start_bp/end_bp/seg_len_bp`,
  **plus** `n_segments` (constituent raw segments) and `event_id`. Cols: `sample, seg_id, major_cn, minor_cn,
  total_cn, ess, event_id, is_timed, discard_reason, n_segments`.
- **`tau_merged_multiplicities.tsv`** — as `tau_multiplicities`, on pooled `seg_id` (`sample, seg_id, multiplicity, n_count, pi`).
- **`tau_merged_routes.tsv`** — as `tau_routes`, on pooled `seg_id`.
- **`tau_merged_gains.tsv.gz`** — as `tau_gains` **minus** `chrom/start_bp/end_bp`, **plus** per-gain `event_id`.
  Cols: `sample, seg_id, route_key, is_best_route, major_cn, minor_cn, gain_index, draw_id, gain_time_frac, n_draws, event_id`.
- **`tau_merged_events.tsv`** — as `tau_events` (as EMITTED, i.e. **no** `class_strict`): `sample, tumor_type,
  event_id, event_time_frac, classification, genome_frac, n_chroms, n_segments, chrom`.

---

## `tau_sig_timing.tsv.gz` — one row per **(sample, seg_id, signature, gain_index)** ★NEW, separate domain
Cumulative signature-vs-clock activity up to each gain event.
by `batch_scripts/build_sig_timing.py` from the existing per-signature caches
(`dev/pcawg/signatures/sig_cumulative_cache_*.tsv`) — **no re-timing**; signatures already computed (SBS17,
SBS18, SBS7, SBS4, SBS3, SBS2+SBS13) come for free. **73,528 rows, 6 signatures, 871 samples.** A *new*
signature needs a `tau run --signatures` rerun.

| column | type | unit | description |
|---|---|---|---|
| `sample` | str | | FK → tau_samples |
| `tumor_type` | str | | denormalized for convenience |
| `seg_id` | str | | FK → tau_segments |
| `signature` | str | | `SBS17`, `SBS18`, `SBS7`, `SBS4`, `SBS3`, `SBS2+SBS13` |
| `gain_index` | int | | gain event index (1-based) on the segment |
| `ess` | float | | effective sample size (weighted SNVs) on the segment |
| `G_clock` | float | frac | cumulative clock activity share up to this event |
| `G_sig` | float | frac | cumulative signature activity share up to this event |
| `ts_clock` | float | | clock flat-time baseline (expected if activity uniform over molecular time) |
| `ts_sig` | float | | signature flat-time baseline |
| `frac` | float | frac | signature fractional activity on the segment (NaN for SBS17 cache) |

Front-loading = `G_sig > G_clock` (signature acts earlier than the clock). The signature panels plot
`log2(G_sig/G_clock ÷ baseline)` over molecular time.

---

## Reference (`ref/`) — curated, paper-local
- `ref_oncokb_oncogenes_hg19.tsv` — `gene, chrom, pos_bp` (from `oncokb_cancerGeneList.json`; built).
- `ref_route_topology.tsv.gz` — one row per `(route_key, event)`: `route_key, major_cn, minor_cn, event_index,
  allele{major,minor}, parent_mult, split_a, split_b`. The shipped tree-topology lookup (from
  `tau/data/route_topology.json.gz`) — reconstruct an allele tree from this + a segment's gain times
  (`tau_gains`), NO Sage. **Built** (1.39M rows, 120,753 routes).
- `ref_hg19_cytoband.tsv` — one row per **chromosome arm** (hg19): `chrom, arm{p,q}, start_bp, end_bp,
  arm_len_bp, centromere_bp, chrom_size_bp`. Built by `batch_scripts/build_ref_cytoband.py` from the shipped
  centromere + chrom-size files (one source for arm-level figures). Load via `tables.load_cytoband()`. **Built.**

## Simulation / benchmark — symmetric with the PCAWG path
A **simulation run** (`sim_run/samples/`, e.g. `output_v8`, 2,412 sims) is Tau on synthetic samples, so it
produces the **same `tau_*` tables** via `tau export-tables` / `tau aggregate`. The only extra per sim is
`{SIM}.truth.json`. `tau benchmark` (`tau/aggregate/benchmark.py`) joins truth ⨝ the recovered clustering →
the **`bench_*`** evaluation tables (all **BUILT**, all derived from the sim outputs):
- `bench_events.tsv` — one row per **CALLED** event: `sample, event_time_frac, classification, genome_frac,
  is_matched, true_time_frac, true_type, timing_error_frac`. → false-positives (unmatched calls), matched timing. **2,841 rows.**
- `bench_truth_events.tsv` — one row per **TRUE** event: `sample, true_type{WGD,PGD}, true_time_frac,
  event_size_frac` (genome fraction the event visibly affects), `min_dist_frac` (molecular-time gap to the
  nearest other true event; NaN if only one), `is_detected`, `recovered_time_frac`, `timing_error_frac`.
  → **recall / false-negatives**. **3,294 rows, 80.8% recall.** *This is the table `bench_events`
  could not provide (calls-only tables have no missed events).*
- `bench_tree_metrics.tsv` — per-sim allele-tree accuracy (recovered vs true route/tree).
- `bench_oracle_lift.tsv`, `bench_pcawg_mutcount.tsv` — clustering-lift + mutcount panels (the latter also reads the real `tau_segments`/`tau_events`).
- `bench_seg_gains.tsv` — one row per simulated **GAIN** (SEGMENT-level benchmarking supp): `major, minor,
  mut_count, gain_index, true_time, time_lo, time_hi` (estimated timing band; `lo==hi` for determined),
  `underdetermined` (bool), `truth_in_band` (bool), `dist_to_set` (0 if truth inside the band, else distance
  to the nearest edge — the faithful error metric). **149,280 gains, ~57% underdetermined.** Cheap reshape of
  `dev/benchmarking/tree_gains_v8.tsv` in `build_ref_and_bench.py`.
- `bench_route_accuracy.tsv` — one row per simulated **TREE**: `major, minor, mc, acc_oracle` (max-gain error
  under the TRUE route), `acc_bestguess` (under Tau's OWN route pick), `route_correct` (bool), `undet`.
  **65,404 trees, 78% routes correct;** median max-gain error 0.039 (oracle) vs 0.041 (best-guess) — timing is
  robust to route uncertainty. Produced by `dev/benchmarking/bestguess_vs_oracle_treeacc.py`, reshaped in `build_ref_and_bench.py`.

---

## Conserved vs dropped (audit vs the genome Segment object)
**Conserved:** chrom/coords/CN, `ess`, timing (`gain_time_frac` ↔ raw `t`), **`N_counts`** (→ tau_multiplicities),
**all routes** + `dist_rel` + `inside_margin` + `free_var_count` (→ tau_routes), clustering (`event_id`),
discard reason, signature timing (→ tau_sig_timing).
**Dropped (intentional):** `snv_table` (per-SNV VAF — `N_counts` is the sufficient statistic), `gof`,
`loglik`/`loglik_diff`, `active_constraints`, `normal_cn`/`min_alt`/`min_vaf` (constants/thresholds).
**Deferred (opt-in):** `N_counts_boot` bootstrap replicates; per-draw `projected`.
