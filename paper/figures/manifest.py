"""The paper's figure set, panel by panel, mapped to the script output that is each panel.

THIS IS THE SINGLE SOURCE OF TRUTH for "which file is which panel of which figure". `export_panels.py`
builds the export tree from it and `run_all.py --check` uses it to confirm every panel was produced.

Figures are keyed by LABEL (the directory name under main/ or supp/); `number` is the figure number in
the manuscript and `tex` its LaTeX label. Each panel is (letter, source, note): `source` is a path
relative to paper/figures/ WITHOUT extension (the script of the same name, or the one named in the note,
writes .svg/.png/.pdf next to itself). A source with no script is drawn by hand and noted as such.
"""

MAIN = [
    dict(label="overview", number=1, tex="fig:model.overview",
         title="Tau enumerates all gain routes to a copy-number state and solves their timing analytically",
         panels=[("a-e", None, "schematic drawn in Illustrator; no script")]),
    dict(label="examples", number=2, tex="fig:examples",
         title="Segment timing and route clustering reveal diverse amplification trajectories converging at shared gain times",
         panels=[
             ("a", "main/examples/a_example1", "stomach adenocarcinoma dc537fcf, two WGDs (object-based: tau.overview.overview_figure)"),
             ("b", "main/examples/b_example2", "esophageal adenocarcinoma 0250e6f7, WGD + ccPG (same renderer)"),
         ]),
    dict(label="punctuated", number=3, tex="fig:punctuated",
         title="Genome-wide timing distinguishes WGD from cross-chromosomal punctuated gains (ccPGs)",
         panels=[
             ("a", "main/punctuated/a_gain_composition", "partition of clonal gains: WGD-linked / ccPG-linked / unlinked (gain counts)"),
             ("b", "main/punctuated/b_timing_wgd", "simulated timing accuracy, WGD (bc_timing.py)"),
             ("c", "main/punctuated/c_timing_ccpg", "simulated timing accuracy, ccPG (bc_timing.py)"),
             ("d", "main/punctuated/d_event_matrix", "samples by number of WGD x ccPG events (one aliquot per donor, n=2,580)"),
             ("e", "main/punctuated/e_genome_fraction", "genome fraction per event (one aliquot per donor)"),
             ("f", "main/punctuated/f_arm_patterns", "ccPG arm programmes by tumour type"),
         ]),
    dict(label="wgd_dynamics", number=4, tex="fig:wgd_dynamics",
         title="WGD and ccPG prevalence and WGD propensity are highly heterogeneous within and across tumor types",
         panels=[
             ("a", "main/wgd_dynamics/a_timeline", "per-sample event times, eight tumour-type blocks"),
             ("b", "main/wgd_dynamics/b_prevalence_timing", "prevalence | ccPG time | WGD time by tumour type"),
             ("c", "main/wgd_dynamics/c_wgd_weibull", "median time to WGD: full cohort vs WGD-positive, first vs second "
                                                     "(model: wgd_timing_model/ e2-e10 chain)"),
         ]),
    dict(label="landscape", number=5, tex="fig:landscape",
         title="Temporal and genomic landscapes of WGD-linked, ccPG-linked and unlinked gains",
         panels=[
             ("a", "main/landscape/a_landscape", "gain density, genome x molecular time; 3 tumour types x 3 layers"),
             ("b", "main/landscape/b_myc_swimmer", "sequence of gains at MYC (_shared/_swimmer.py)"),
         ]),
    dict(label="burst", number=6, tex="fig:burst",
         title="High-copy-number amplifications occur in bursts enriched around whole-genome duplication",
         panels=[
             ("a", "main/burst/a_burst_vs_gradual", "burst vs gradual exemplars"),
             ("b", "main/burst/b_bursts_pile_at_wgd", "time to nearest WGD vs WGD-shuffle null"),
             ("c", "main/burst/c_excess_vs_explained", "excess vs WGD-explained bursts"),
             ("d", "main/burst/d_burst_density", "burst density per sample (best-route solutions)"),
             ("e", "main/burst/e_burden_by_doubling", "excess bursts per sample by class"),
             ("f", "main/burst/f_top_hits", "chromosome-arm hotspots (odds ratios) with oncogenes"),
             ("g", "main/burst/g_seglen", "length distribution of excess-burst segments"),
             ("h", "main/burst/h_wgd_enrichment", "fraction within +/-0.10 of WGD per oncogene vs permutation null"),
             ("i", "main/burst/i_wgd_swimmer", "burst-to-WGD offset per oncogene"),
         ]),
    dict(label="signatures", number=7, tex="fig:signatures",
         title="Mutational-process activity resolved across clonal history",
         panels=[
             ("a", "main/signatures/a_sbs17_overview_trees", "SBS17 vs clock in the esophageal example: overview + cluster trees (object-based)"),
             ("b", "main/signatures/b_sbs17", "per-sample activity states by tumour type, SBS17 (b_cluster_heatmaps.py; R)"),
             ("b", "main/signatures/b_apobec", "same, APOBEC (b_cluster_heatmaps.py; R)"),
             ("c", "main/signatures/c_direction_arrows", "direction-of-change arrows by signature and tumour type (R)"),
         ]),
]

SUPP = [
    dict(label="route_complexity", number=1, tex="fig:cn_route_enumeration",
         panels=[("a", "supp/route_complexity/a_routes_per_state", "routes vs segments per state"),
                 ("b", None, "schematic composed in Illustrator from the README images; no script")]),
    dict(label="multiplicity_clustering", number=2, tex="fig:multiplicity_clustering",
         panels=[("a", "supp/multiplicity_clustering/a_overview", "route_clustering.py (object-based)"),
                 ("b", "supp/multiplicity_clustering/b_profiles", "route_clustering.py"),
                 ("c", "supp/multiplicity_clustering/c_trees", "route_clustering.py")]),
    dict(label="benchmarking_segment", number=3, tex="fig:benchmarking_supplementary",
         panels=[("a", "supp/benchmarking_segment/a_mutcount_threshold", ""),
                 ("b", "supp/benchmarking_segment/b_schematic", "object-based (two simulated genomes)"),
                 ("c", "supp/benchmarking_segment/c_cluster_mutcount", ""),
                 ("d", "supp/benchmarking_segment/d_tree_accuracy", ""),
                 ("e", "supp/benchmarking_segment/e_error_by_determinacy", ""),
                 ("f", "supp/benchmarking_segment/f_error_by_cn_maxgain", "")]),
    dict(label="underdetermined", number=4, tex="fig:underdetermined",
         panels=[("a", "supp/underdetermined/a_special_case_tree", "object-based"),
                 ("b", "supp/underdetermined/b_special_case_population", "")]),
    dict(label="indeterminacy", number=5, tex="fig:indeterminacy",
         panels=[("a", "supp/indeterminacy/a_bounds_per_cn", ""),
                 ("b", "supp/indeterminacy/b_calibration_heatmaps", "")]),
    dict(label="wgd_ccpg_benchmark", number=6, tex="fig:wgd_ccpg_benchmark",
         panels=[("a", "supp/wgd_ccpg_benchmark/a_example", "frozen render of one simulated genome"),
                 ("b", "supp/wgd_ccpg_benchmark/b_recall", ""),
                 ("c", "supp/wgd_ccpg_benchmark/c_falsepos", ""),
                 ("d", "supp/wgd_ccpg_benchmark/d_allelic", ""),
                 ("e", "supp/wgd_ccpg_benchmark/e_allelic_bins", "")]),
    dict(label="wgd_swimmer_others", number=7, tex="fig:wgd_swimmer_others",
         panels=[("", "supp/wgd_swimmer_others/timeline_others", "Fig 4a drawing for the remaining tumour types")]),
    dict(label="gain_landscape", number=8, tex="fig:gain_landscape",
         panels=[("", "supp/gain_landscape/p1_other_types", "a_other_types.py: Biliary, Bone.Osteosarc, Breast, CNS.Medullo"),
                 ("", "supp/gain_landscape/p2_other_types", "ColoRect, Eso, Kidney.RCC, Liver"),
                 ("", "supp/gain_landscape/p3_other_types", "Lung.AdenoCA, Lung.SCC, Lymph.BNHL"),
                 ("", "supp/gain_landscape/p4_other_types", "Ovary, Panc.AdenoCA, Panc.Endocrine"),
                 ("", "supp/gain_landscape/p5_other_types", "Prost, Stomach, Uterus"),
                 ("", "supp/gain_landscape/s8_gain_landscape", "all five sheets merged")]),
    dict(label="loss_landscape", number=9, tex="fig:loss_landscape",
         panels=[("a", "supp/loss_landscape/a_loss_schematic", ""),
                 ("b", "supp/loss_landscape/b_loss_profiles", ""),
                 ("c", "supp/loss_landscape/c_loss_example", "")]),
    dict(label="extra_oncogenes", number=10, tex="fig:extra_oncogenes",
         panels=[("a", "supp/extra_oncogenes/a_pik3ca_swimmer", ""),
                 ("b", "supp/extra_oncogenes/b_egfr_swimmer", "")]),
    dict(label="burst_gallery", number=11, tex="fig:burst_gallery",
         panels=[("", "supp/burst_gallery/burst_gallery", "24 frozen exemplars (data/burst_gallery/exemplars.pkl)")]),
    dict(label="burst_characteristics", number=12, tex="fig:burst_characteristics",
         panels=[("a", "supp/burst_characteristics/a_burst_null", "abc_solo.py"),
                 ("b", "supp/burst_characteristics/b_excess_counts", "abc_solo.py"),
                 ("c", "supp/burst_characteristics/c_wgd_enrichment", "abc_solo.py")]),
    dict(label="signature", number=13, tex="fig:signature_method",
         panels=[("a", "supp/signature/a_multiplicity_ambiguity", ""),
                 ("b", "supp/signature/b_signature_weights", ""),
                 ("c", "supp/signature/c_expansion_contraction", ""),
                 ("d", "supp/signature/d_stepwise_bic", "caption panels d and e both come from this script")]),
]

# Scripts kept as alternatives (alt_*.py) are not listed: an `alt_` prefix means "not a manuscript panel".
UNUSED = []
