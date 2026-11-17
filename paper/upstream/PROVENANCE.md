# upstream — where the frozen inputs come from

The panel scripts read composite tables, per-figure caches and a few small "frozen" tables. This
directory holds the analysis scripts that produced those inputs, copied from the working analysis tree.
They are kept for provenance: site-specific absolute paths were replaced by environment variables
(`TAU_REPO`, `TAU_PCAWG_INPUT_DIR`, ...), but they are otherwise unmodified, they expect the working
layout of that tree (outputs under `$TAU_REPO/dev/...`), and they have not been re-run from this
location. The cache builders in `../figures/data/` find the outputs of these scripts through
`TAU_ANALYSIS_DIR`.

| input (in the data bundle) | used by | produced by |
|---|---|---|
| per-sample Tau outputs (not distributed) | everything | `pcawg_run/run_tau_sample.py`; events by `pcawg_run/run_bottomup_final.py`; per-signature reruns set up by `pcawg_run/setup_persig_runs.py` |
| cohort event/sample tables | `batch_scripts/build_cheap_base_tables.py` | `pcawg_run/summarize_allpcawg.py`, `pcawg_run/load_events.py`; one aliquot per donor by `pcawg_run/donor_dedup.py` |
| per-solution gain times of high-copy-number segments (`genome_highcn_timing_cache*.tsv`) | burst cache builders, Supp. 11 exemplar selection | `burst/build_genome_highcn_cache.py`, `burst/build_genome_highcn_cache_v2.py` |
| burst taxonomy (`wgd_burst_taxonomy_polytope.tsv`) | `figures/data/burst_characteristics/build/` | `burst/classify_wgd_burst_all.py`, `burst/compute_max_burst_size.py` |
| matched null replicates | `figures/data/burst_characteristics/build/build_burst_null.py` (Supp. 12a) | `burst/simulate_null_perrep.py` |
| `frozen/burst_possible_necessary.tsv` | Supp. 11, Supp. 12 | `burst/burst_possible_necessary.py` |
| `frozen/doubling_blocks.tsv` | Supp. 4b | `special_case/doubling_blocks.py` |
| `frozen/supp4_dominant_34x.tsv` | Supp. 4b | `special_case/scan_special_case.py`, followed by keeping the best-powered cluster per sample and state |
| post-WGD loss tables | `figures/data/loss_landscape/build/` (Supp. 9) | `loss/compute_multi_wgd_loss.py`, `loss/build_loss_catalog.py`, `loss/single_wgd_loss_seg_compute.py`, `loss/cohort_loss_compute.py` |
| `figures/data/punctuated/pgd_arm_membership.tsv` | Fig. 3f | `ccpg_arms/pgd_arm_patterns.py` |
| signature interval tables (`sig_intervals_range_*`, `sig_clustertiming_*`; `clustered_signature_input/`) | Fig. 7b,c, Supp. 13, `tau_sig_ratio*` | `signatures/make_clustertiming_retimed.py`, `signatures/make_interval_range_segment.py`, `signatures/merge_shards.py` |
| per-signature cumulative caches | `batch_scripts/build_sig_timing.py` | `signatures/build_sig_cache.py`, `signatures/precompute_sig_timing.py`, `signatures/make_sig_timing_tables.py` |
| simulated cohort | `bench_*`, `sim/`, `objects/sim/` | `simulation/Snakefile`, `simulation/simulate_sample.py`, `simulation/config_v8.yaml` (`gen_final.py`, `build_seg_solutions_v10.py` are the batch drivers) |
| `frozen/fig3_a_example.*` | Supp. 6a | `simulation/panel_A_overview_compare.py` |
| `ref/COSMIC_v3p2_SBS_WGS_MuSiCal.csv`, `ref/MuSiCal_assignment_PCAWG.txt` | Supp. 13b | COSMIC v3.2 SBS spectra and PCAWG signature exposures from MuSiCal (Jin et al. 2024) |
| `ref/ref_oncokb_oncogenes_hg19.tsv` | oncogene annotation | OncoKB cancer gene list |
