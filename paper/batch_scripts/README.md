# batch_scripts — per-sample Tau outputs to composite tables

These build the tables in `../data_tables/` from the per-sample Tau outputs. They reshape; they do not
re-time anything.

| script | makes |
|---|---|
| `backfill_chunk.py`, `submit_backfill.sh`, `merge_backfill.py` | `tau_segments`, `tau_routes`, `tau_multiplicities`, `tau_gains`, and the per-sample columns of `tau_samples` (run per chunk of samples, then merged) |
| `build_cheap_base_tables.py` | `tau_samples`, `tau_events` from the cohort event tables |
| `build_ref_cytoband.py`, `build_ref_and_bench.py` | `ref/` tables and the `bench_*` simulation-benchmark tables |
| `export_sim_chunk.py`, `submit_sim_export.sh` | `sim/` composite tables of the simulated cohort |
| `build_sig_timing.py`, `build_sig_ratio_tables.py` | `tau_sig_timing`, `tau_sig_ratio*` |
| `recluster_sample.py`, `emit_sample_v2.py`, `emit_signature_tables.py` | the `per_sample/<sample>/` bundles of the example tumours |

Paths are resolved from the repository root; the per-sample outputs are found through
`paper/pcawg_run/samples` (a link to the Tau output directory) or `TAU_PCAWG_OUTPUTS`.
The SLURM wrappers are examples and carry site-specific `#SBATCH` lines.
