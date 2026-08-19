#!/usr/bin/env python3
"""Command-line interface for Tau.

Subcommands
-----------
``tau run``   Run the full timing pipeline on a real sample (VCF + CNV).
``tau demo``  Run end-to-end on a synthetic genome with NO external files.

Both are also reachable as ``pixi run tau ...`` / ``pixi run demo``.
``python -m scripts.run_tau`` remains a thin alias for ``tau run``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import gzip
import json
import os
import pickle
import sys

import pandas as pd


def get_default_data_path(filename):
    """Get path to a package data file."""
    try:
        from importlib import resources

        return str(resources.files("tau.data").joinpath(filename))
    except (FileNotFoundError, TypeError, AttributeError):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        return os.path.join(data_dir, filename)


# Default data file paths - can be overridden via environment variables or CLI args
def _default_routes_sage():
    """Prefer the per-state solutions/ directory, matching timing._default_paths().

    The split directory is loaded lazily one CN state at a time, so it is far faster
    than the monolithic file; the monolithic file remains the fallback.
    """
    env = os.environ.get("TAU_ROUTES_SAGE")
    if env:
        return env
    split = get_default_data_path("solutions")
    if os.path.isdir(split) and any(f.endswith(".sobj") for f in os.listdir(split)):
        return split
    return get_default_data_path("7_5_solutions_updated.sobj")


DEFAULT_ROUTES_SAGE = _default_routes_sage()
DEFAULT_MATRICES_H5 = os.environ.get(
    "TAU_MATRICES_H5", get_default_data_path("matrices_7_7.h5")
)

# COSMIC SBS definitions ship with the package, so --signatures works without pointing at one.
_packaged_cosmic = get_default_data_path("cosmic_sbs_v3.2.csv")
DEFAULT_COSMIC_CSV = os.environ.get(
    "TAU_COSMIC_CSV", _packaged_cosmic if os.path.exists(_packaged_cosmic) else None
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"tau: error: {msg}", file=sys.stderr)
    sys.exit(2)


def _require_file(path, label):
    if path is None:
        return
    if not os.path.exists(path):
        _die(f"{label} not found: {path}")
    if os.path.isdir(path):
        _die(f"{label} is a directory, expected a file: {path}")


def _check_purity(purity):
    if purity is None:
        return
    if not (0 < purity <= 1):
        _die(f"--purity must be in (0, 1], got {purity}")


# ---------------------------------------------------------------------------
# `tau run`
# ---------------------------------------------------------------------------

def _add_run_args(parser):
    parser.add_argument("--sample", type=str, required=True, help="Sample ID to process")
    parser.add_argument(
        "--snv_table", "--vcf", dest="snv_table", type=str, required=True,
        help="Somatic SNVs: either a VCF, or a plain table (tab/comma/whitespace) with "
             "chromosome, position and per-mutation read counts — column names are matched "
             "case-insensitively (e.g. Chromosome/Position/Tumor_Ref_Count/Tumor_Alt_Count). "
             "ref/alt columns are optional and only needed for signature weighting. "
             "--vcf is accepted as an alias.",
    )
    parser.add_argument(
        "--cn_table", "--cnv", dest="cn_table", type=str, required=True,
        help="Allele-specific copy-number table (tab/comma/whitespace) with chromosome, "
             "segment start/end and major/minor CN — column names matched case-insensitively "
             "(e.g. Chromosome/Segment_Start/Segment_End/Major_CN/Minor_CN).",
    )
    parser.add_argument(
        "--purity", type=float, required=True,
        help="Tumour purity in (0, 1]. If you keep purities in a table, look the value up "
             "in your submit script and pass it here.",
    )
    parser.add_argument(
        "--ref_fasta", type=str, default=None,
        help="Reference genome FASTA (required only when --cosmic_csv/--exposures are given)",
    )
    parser.add_argument(
        "--cosmic_csv", type=str, default=DEFAULT_COSMIC_CSV,
        help="COSMIC SBS signature definitions CSV. Ships with Tau — override only to use "
             "a different signature set.",
    )
    parser.add_argument(
        "--exposures", type=str, default=None,
        help="Per-sample signature exposures TSV (optional; omit for equal-weight mutations)",
    )
    parser.add_argument(
        "--routes_sage", type=str, default=DEFAULT_ROUTES_SAGE,
        help="Path to Sage solutions (.sobj file or split solutions directory)",
    )
    parser.add_argument(
        "--matrices_h5", type=str, default=DEFAULT_MATRICES_H5,
        help="Path to constraint matrices HDF5 file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument("--output_gpkl", type=str, default=None,
                        help="Path for the genome pickle (default <output_dir>/<sample>.genome.pkl.gz)")
    parser.add_argument("--output_cluster_gpkl", type=str, default=None,
                        help="Path for the pooled/clustered genome pickle")
    parser.add_argument(
        "--signatures", type=str, nargs="+", default=["SBS1", "SBS5"],
        help="Clock signatures to weight mutations by (default: SBS1 SBS5). Needs --exposures "
             "and --ref_fasta (or an sbs96/context column in --snv_table); if those are absent "
             "Tau warns and falls back to equal-weight mutations. Pass --signatures none to "
             "request equal weighting explicitly.",
    )
    parser.add_argument(
        "--bootstrap_B", type=int, default=10,
        help="Number of bootstrap EM replicates (default: 10, matching the Python API)",
    )
    parser.add_argument(
        "--detect_min_alt", type=int, default=3,
        help="Minimum alt reads to consider a mutation detectable (default: 3)",
    )
    parser.add_argument(
        "--mode", dest="mode", type=str, default="soft", choices=["soft", "hard"],
        help="Signature weighting when --signatures is given (default: soft). "
             "'soft' weights each mutation by P(clock | context, exposures); "
             "'hard' keeps a mutation whole (weight 1) only if its most-likely signature is a "
             "clock signature, else drops it. No effect without --signatures.",
    )
    parser.add_argument(
        "--sex", type=str, default=None, choices=["male", "female"],
        help="Sample sex. 'male' sets normal_cn=1 on chrX/chrY (hemizygous VAF correction)",
    )
    parser.add_argument(
        "--output_interactive", type=str, default=None,
        help="Path for the self-contained interactive HTML viewer "
             "(default <sample>.overview_interactive.html in --output_dir)",
    )
    parser.add_argument(
        "--interactive", dest="interactive", action="store_true",
        help=argparse.SUPPRESS,      # on by default; kept so existing command lines keep working
    )
    parser.add_argument(
        "--no-interactive", dest="interactive", action="store_false",
        help="Skip the self-contained interactive HTML viewer "
             "(<sample>.overview_interactive.html), which is written by default. The viewer "
             "costs a per-route tree precompute and is a few MB per sample; see --tree-top-k.",
    )
    parser.add_argument(
        "--tree-top-k", dest="tree_top_k", type=int, default=5,
        help="Interactive viewer: routes per segment to precompute tree+knobs for "
             "(0 = skip trees → faster generation, no Sage needed for the viewer step)",
    )
    parser.add_argument(
        "--signature-analysis", "--include-signatures", dest="include_signatures",
        action="store_true",
        help="Re-time the genome under every active signature (exposure share above "
             "--signature-threshold) and write the per-signature analysis: sig_summary / "
             "sig_activity / tau_sig_intervals / tau_sig_states, plus the viewer cache "
             "(<sample>.sig_timing.json.gz) "
             "so the viewer gets the signature toggle + activity-over-time curve. "
             "Adds a re-time per active signature (~minutes); needs signature exposures "
             "on the genome (falls back to hard-assignment fractions).",
    )
    parser.add_argument(
        "--signature-threshold", "--sig-thresh", dest="sig_thresh", type=float, default=0.15,
        help="Exposure-share gate for --signature-analysis (default 0.15). Signatures are "
             "grouped first (SBS17a+SBS17b -> SBS17, APOBEC family -> APOBEC), so the gate "
             "applies to the group's summed share; the clock is always included.",
    )
    parser.add_argument(
        "--subclonal-filter", dest="subclonal_filter", action="store_true",
        help="Apply Tau's built-in low-CCF subclonal test: drop mutations whose VAF upper "
             "99%% bound cannot reach CCF 1 at multiplicity 1. OFF by default (this is what "
             "every published Tau run used). Its power scales with purity, so enabling it "
             "applies a purity-dependent correction across a cohort.",
    )
    parser.add_argument(
        "--subclonal-list", dest="subclonal_list", type=str, default=None,
        help="File of mutation IDs to treat as subclonal, one {chrom}:{pos}:{ref}/{alt} per line "
             "(e.g. from PyClone-VI). Combines with --subclonal-filter by union.",
    )
    parser.add_argument(
        "--subclonal-alpha", dest="subclonal_alpha", type=float, default=0.01,
        help="Confidence level for --subclonal-filter (default 0.01 = 99%% upper bound)",
    )
    parser.add_argument(
        "--tumor_type", type=str, default=None,
        help="Tumour type label carried into tau_samples / tau_events (default: NA)",
    )
    parser.add_argument(
        "--output_tables_dir", type=str, default=None,
        help="Directory for the composite tau_* tables (default: --output_dir)",
    )
    parser.add_argument(
        "--no-tables", dest="tables", action="store_false",
        help="Skip the composite tau_* tables (segments/routes/multiplicities/gains/events, "
             "plus the merged level). They are a cheap reshape of results already computed, "
             "and are what `tau aggregate` consumes.",
    )
    parser.set_defaults(interactive=True, include_signatures=False, tables=True,
                        subclonal_filter=False)


def run_from_args(args):
    """Execute the `tau run` pipeline from parsed args."""
    # --- Input validation (before any heavy import or work) ---
    _require_file(args.snv_table, "--snv_table/--vcf")
    _require_file(args.cn_table, "--cn_table/--cnv")
    _require_file(args.ref_fasta, "--ref_fasta")
    _require_file(args.cosmic_csv, "--cosmic_csv")
    _require_file(args.exposures, "--exposures")
    _check_purity(args.purity)

    # Signatures default to equal-weight (None). If signatures are requested,
    # the COSMIC + exposures + reference inputs are all required.
    signatures = args.signatures
    if signatures and len(signatures) == 1 and str(signatures[0]).lower() in ("none", "off"):
        signatures = None
    if signatures:
        # ref_fasta is only needed when the SNV table does not already carry the
        # trinucleotide context, so it is checked later, once the table is read.
        missing = [
            name for name, val in (
                ("--cosmic_csv", args.cosmic_csv),
                ("--exposures", args.exposures),
            ) if val is None
        ]
        if missing:
            print(
                f"  [warn] signature weighting needs {', '.join(missing)} — "
                f"falling back to equal-weight mutations."
            )
            signatures = None

    # Everything lands in --output_dir: the 11 tau_* tables + the overview. One directory per sample,
    # chosen by the caller; `tau aggregate` searches recursively, so pointing it at the parent of many
    # such directories picks them all up regardless of how they are nested.
    sample_dir = getattr(args, "output_tables_dir", None) or args.output_dir
    os.makedirs(sample_dir, exist_ok=True)

    # The pickles are kept: the genome object holds everything the tables flatten, and
    # `tau export-tables` rebuilds the full table set from the pair without re-timing.
    for arg_name, default_filename in (
            ("output_gpkl", f"{args.sample}.genome.pkl.gz"),
            ("output_cluster_gpkl", f"{args.sample}.clustered_genome.pkl.gz"),
            ("output_interactive", f"{args.sample}.overview_interactive.html")):
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, os.path.join(sample_dir, default_filename))

    purity = args.purity

    # Heavy imports (Sage etc.) deferred until after validation passes.
    from tau.core import Genome
    from tau.preprocessing import preprocess_sample
    from tau.clustering import (
        cluster_times_bottomup,
        cluster_segment_multiplicities,
        make_clustered_genome,
    )
    from tau import timing

    routes = timing.load_routes(args.routes_sage, args.matrices_h5)

    mut_df = preprocess_sample(
        sample=args.sample,
        vcf_path=args.snv_table,
        cnv_tsv=args.cn_table,
        purity=purity,
        ref_fasta=args.ref_fasta,
        cosmic_csv=args.cosmic_csv,
        exposures_tsv=args.exposures,
        apply_subclonal_filter=bool(getattr(args, "subclonal_filter", False)),
        subclonal_alpha=float(getattr(args, "subclonal_alpha", 0.01)),
        detect_min_alt=args.detect_min_alt,
        signatures=signatures,
        mode=getattr(args, "mode", "soft"),
        subclonal_list=getattr(args, "subclonal_list", None),
    )

    normal_cn_map = {"X": 1, "Y": 1} if args.sex == "male" else None

    g = Genome.create(
        mut_df, purity=purity, detect_min_alt=args.detect_min_alt, detect_min_vaf=0,
        normal_cn_map=normal_cn_map,
    )
    g.calculate_multiplicities(bootstrap_B=args.bootstrap_B, random_state=42)
    g.calculate_gof()
    g.time_segments(env=routes)

    seg_cluster_df = cluster_segment_multiplicities(g, output_plot_file=None)

    with gzip.open(args.output_gpkl, "wb") as f:
        pickle.dump(g, f)

    cluster_times_result, segment_cluster_ids, original_times, cluster_summary_df = (
        cluster_times_bottomup(g, cluster_output_file=None)
    )

    # The pooled genome drives every tau_merged_* table and the overview's allele trees.
    clustered_genome = None
    if not seg_cluster_df.empty:
        clustered_genome = make_clustered_genome(g, seg_cluster_df)
        clustered_genome.calculate_multiplicities(bootstrap_B=args.bootstrap_B, random_state=42)
        clustered_genome.time_segments(env=routes)
        with gzip.open(args.output_cluster_gpkl, "wb") as f:
            pickle.dump(clustered_genome, f)

    # Optionally build the per-signature timing cache so the viewer gets the
    # signature toggle + activity curve. Written into the sample directory, where
    # the viewer step auto-finds it.
    _sig_timing = os.path.join(sample_dir, f"{args.sample}.sig_timing.json.gz")
    # The exposures FILE is authoritative for the signature step when given: `genome.exposures` only
    # holds what was baked in at preprocessing, and older genomes have none at all — in which case the
    # estimate silently degrades to hard-assignment fractions and finds fewer active signatures.
    _sig_expo = None
    if getattr(args, "include_signatures", False) and args.exposures:
        try:
            from tau.sig_timing import exposures_from_file
            _sig_expo = exposures_from_file(args.exposures, args.sample)
            print(f"  signatures: exposures from {os.path.basename(args.exposures)} "
                  f"({len(_sig_expo)} non-zero)")
        except Exception as e:
            print(f"  [warn] could not read --exposures for the signature step ({e}); "
                  f"falling back to the genome's own estimate")

    if getattr(args, "include_signatures", False):
        try:
            from tau.sig_timing import build_sig_payload, write_sig_tables
            payload = build_sig_payload(g, routes, getattr(args, "sig_thresh", 0.15),
                                        expo=_sig_expo)
            payload["sample"] = args.sample
            with gzip.open(_sig_timing, "wt") as f:
                json.dump(payload, f)
            print(f"  signatures: {os.path.basename(_sig_timing)}")
            # ...and the same analysis as flat tables, so the step has a readable
            # output and not only a viewer cache.
            for p in write_sig_tables(payload, os.path.join(sample_dir, args.sample)):
                print(f"  signatures: {os.path.basename(p)}")
        except Exception as e:  # never let the sig cache break the main run
            print(f"  [warn] signature timing cache skipped: {e}")

        # Pooled-CLUSTER activity + the stepwise-change state fit. The per-SEGMENT tables above are
        # the fine-grained view; the state model needs pooled clusters, because its intervals have to
        # carry enough mutations for a relative rate to mean anything.
        try:
            from tau.sig_timing import cluster_interval_table
            from tau.sig_states import fit_cluster_table
            iv = cluster_interval_table(
                g, routes, seg_cluster_df=seg_cluster_df, sample=args.sample,
                tumor_type=(getattr(args, "tumor_type", None) or "NA"),
                thresh=getattr(args, "sig_thresh", 0.15), expo=_sig_expo)
            if len(iv):
                p_iv = os.path.join(sample_dir, f"{args.sample}.tau_sig_intervals.tsv")
                iv.to_csv(p_iv, sep="\t", index=False)
                print(f"  signatures: {os.path.basename(p_iv)} ({len(iv)} cluster-intervals)")
                st = fit_cluster_table(iv)
                p_st = os.path.join(sample_dir, f"{args.sample}.tau_sig_states.tsv")
                st.to_csv(p_st, sep="\t", index=False)
                print(f"  signatures: {os.path.basename(p_st)} ({len(st)} fitted states)")
        except Exception as e:
            print(f"  [warn] signature state fit skipped: {e}")

    # Self-contained interactive HTML viewer (reuses the already-loaded route env).
    if getattr(args, "interactive", False):
        try:
            from tau.viz import genome_to_html
            # Auto-find a sibling <sample>.sig_timing.json.gz next to the genome
            # pickle (signature toggle + activity curve); absent → feature OFF.
            if not os.path.exists(_sig_timing):
                _sig_timing = None
            genome_to_html(
                g, cluster_df=cluster_summary_df,
                output_html=args.output_interactive, sample=args.sample,
                clustered_genome=clustered_genome, env=routes,
                tree_top_k=getattr(args, "tree_top_k", 5),
                sig_timing=_sig_timing,
            )
        except Exception as e:  # never let the viewer break the main run
            print(f"  [warn] interactive viewer skipped: {e}")
            args.output_interactive = None

    # Composite tau_* tables (the standard analysis-ready output). Built from the objects already
    # in memory, so nothing is re-timed and nothing is re-clustered that this run already did.
    #
    # MUST BE LAST: build_all_tables propagates each pooled unit's timing onto the raw segments to
    # derive merged events, which mutates g.segments[].timing_result. The signature step and the
    # viewer both read g in memory, so running this before them would hand them pooled timing.
    if getattr(args, "tables", True):
        try:
            from tau.aggregate.export import build_all_tables, write_all_tables
            tdir = sample_dir
            tables = build_all_tables(
                g, args.sample, getattr(args, "tumor_type", None) or "NA",
                seg_cluster_df=(seg_cluster_df if len(seg_cluster_df) else None),
                ctb_raw=(cluster_times_result, segment_cluster_ids, original_times,
                         cluster_summary_df),
                clustered_genome=clustered_genome,
                purity=purity,
            )
            written = write_all_tables(tables, tdir, args.sample)
            print(f"  tables:   {len(written)} tau_* tables in {os.path.abspath(tdir)}/")

            # The two-level overview: individual segment timing | pooled allele trees | timing
            # histogram. Needs the events tables above, and the RAW per-segment timing — which
            # build_all_tables restores after its pooled-clustering pass, so g is intact here.
            if clustered_genome is not None:
                from tau.overview import save_overview
                paths, _info = save_overview(
                    g, clustered_genome, tables["events"], tables["merged_events"],
                    os.path.join(tdir, f"{args.sample}.overview"), verbose=True)
                print(f"  overview: {os.path.basename(paths[0])} (+ .svg)")
        except Exception as e:   # the run's primary outputs are already on disk — don't lose them
            print(f"  [warn] composite tables skipped: {e}")

    n_ev = len(cluster_summary_df)
    evs = ", ".join(f"{r['classification']} t={r['time']:.2f}"
                    for _, r in cluster_summary_df.iterrows()) if n_ev else "none"
    print(f"\nDone -> {os.path.abspath(sample_dir)}/")
    print(f"  {len([s for s in g.segments if getattr(s, 'timing_result', None)])}"
          f"/{len(g.segments)} segments timed | {n_ev} event(s): {evs}")
    if getattr(args, "interactive", False) and args.output_interactive:
        print(f"  viewer:   {os.path.basename(args.output_interactive)}")


# ---------------------------------------------------------------------------
# `tau demo`
# ---------------------------------------------------------------------------

def demo_from_args(args):
    """Run the real `tau run` pipeline on the small simulated dataset that ships with Tau.

    The inputs are ordinary SNV/CN tables under ``tau/data/demo/`` — simulated, no patient
    data — so the demo exercises exactly the code path a user's own data takes, and its
    output is the full set of files a real run produces. Ground truth ships alongside, so
    the demo also checks itself.
    """
    import json

    demo_dir = Path(get_default_data_path("demo"))
    snv = demo_dir / "DEMO.snv_table.tsv.gz"
    cn = demo_dir / "DEMO.cn_table.tsv"
    truth_f = demo_dir / "DEMO.truth.json"
    for f in (snv, cn, truth_f):
        if not f.exists():
            _die(f"demo data missing: {f}. Reinstall Tau, or pass your own data to `tau run`.")
    meta = json.loads(truth_f.read_text())

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Tau demo — {meta['n_mutations']:,} simulated mutations over "
          f"{meta['n_segments']} segments (purity {meta['purity']:.2f}); "
          f"running the full `tau run` pipeline...")

    # Build the same args `tau run` would see, so the demo cannot drift from it.
    run_parser = argparse.ArgumentParser()
    _add_run_args(run_parser)
    run_args = run_parser.parse_args([
        "--sample", "DEMO",
        "--snv_table", str(snv),
        "--cn_table", str(cn),
        "--purity", str(meta["purity"]),
        "--output_dir", out_dir,
        "--bootstrap_B", str(args.bootstrap_B),
        "--signatures", "none",          # the demo ships no exposures
        "--tumor_type", "DEMO",
    ])
    run_from_args(run_args)

    truth = meta.get("truth", {})
    if truth:
        print("\nGround truth: " + ", ".join(f"{k} @ t={v:.2f}" for k, v in truth.items()))


# ---------------------------------------------------------------------------
# `tau export-tables` / `tau aggregate`
# ---------------------------------------------------------------------------

def export_tables_from_args(args):
    """Reshape an existing genome pkl → per-sample composite tables (no re-timing)."""
    _require_file(args.genome_pkl, "genome.pkl.gz")
    from tau.aggregate.export import export_tables_from_pkl
    written = export_tables_from_pkl(args.genome_pkl, args.out)
    print(f"Wrote per-sample tables to {os.path.abspath(args.out)}/")
    for name, path in written.items():
        print(f"  {name}: {os.path.basename(path)}")


def aggregate_from_args(args):
    """Concatenate per-sample composite tables → cohort tables."""
    from tau.aggregate.aggregate import aggregate_run
    out = aggregate_run(args.sample_tables_dir, args.out)
    print(f"Wrote cohort tables to {os.path.abspath(args.out)}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="tau",
        description="Tau — copy number amplification timing in tumor genomes.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser(
        "run", help="Run timing on a real sample (VCF + CNV).",
        description="Run the full Tau timing pipeline on a sample.",
    )
    _add_run_args(run_p)
    run_p.set_defaults(func=run_from_args)

    demo_p = sub.add_parser(
        "demo", help="Run end-to-end on a synthetic genome (no external files).",
        description="Synthesise a small tumour genome and run Tau on it end-to-end.",
    )
    demo_p.add_argument(
        "--output_dir", type=str, default="tau_demo_out",
        help="Output directory (default: tau_demo_out)",
    )
    demo_p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    demo_p.add_argument(
        "--bootstrap_B", type=int, default=10, help="Bootstrap EM replicates (default: 10)"
    )
    demo_p.set_defaults(func=demo_from_args)

    export_p = sub.add_parser(
        "export-tables", help="Reshape an existing genome pkl into per-sample composite tables.",
        description="Backfill the per-sample tau_* composite tables from a timed genome pkl (no re-timing).",
    )
    export_p.add_argument("genome_pkl", type=str, help="Path to <sample>.genome.pkl.gz")
    export_p.add_argument("--out", type=str, required=True, help="Output directory for the per-sample tables")
    export_p.set_defaults(func=export_tables_from_args)

    agg_p = sub.add_parser(
        "aggregate", help="Concatenate per-sample composite tables into cohort tables.",
        description="Union all per-sample tau_* tables under a staging dir into the cohort tau_* tables.",
    )
    agg_p.add_argument("sample_tables_dir", type=str, help="Directory of per-sample tau_* tables")
    agg_p.add_argument("--out", type=str, required=True, help="Output directory for the cohort tables")
    agg_p.set_defaults(func=aggregate_from_args)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
