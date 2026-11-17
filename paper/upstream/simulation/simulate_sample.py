#!/usr/bin/env python3
"""
Simulate a single tumor genome and write outputs for the benchmarking pipeline.

Outputs written to --outdir:
  {sample}.vcf        — plain VCF with t_alt_count / t_ref_count INFO fields
  {sample}.cna.txt    — TSV copy number segments (chromosome, start, end, major_cn, minor_cn)
  {sample}.purity.txt — single float (tumor purity)
  {sample}.truth.json — ground truth: events, purity, per-segment t_values
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import tau.simulation as sim
from tau.simulation import CHR_LEN


def write_vcf(path, mut_df):
    """Write mutation dataframe to a plain text VCF file.

    FILTER is '.' so that pysam's rec.filter.keys() returns [] (record kept).
    INFO contains t_alt_count and t_ref_count as required by read_vcf_counts().
    """
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##INFO=<ID=t_alt_count,Number=1,Type=Integer,'
                'Description="Tumor alt allele count">\n')
        f.write('##INFO=<ID=t_ref_count,Number=1,Type=Integer,'
                'Description="Tumor ref allele count">\n')
        for chrom, length in CHR_LEN.items():
            f.write(f"##contig=<ID={chrom},length={length}>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, row in mut_df.iterrows():
            info = f"t_alt_count={int(row['nalt'])};t_ref_count={int(row['nref'])}"
            f.write(
                f"{row['chrom']}\t{int(row['pos'])}\t.\t"
                f"{row['ref']}\t{row['alt']}\t.\t.\t{info}\n"
            )


def write_cna(path, segments):
    """Write copy number segments to TSV."""
    with open(path, "w") as f:
        f.write("chromosome\tstart\tend\tmajor_cn\tminor_cn\ttrue_route\n")
        for seg in segments:
            tag = getattr(seg, "tag", "")
            f.write(f"{seg.chrom}\t{seg.start}\t{seg.end}\t{seg.major}\t{seg.minor}\t{tag}\n")


def event_to_dict(ev):
    """Convert a simulation Event to a JSON-serialisable dict."""
    d = {"type": type(ev).__name__, "time": float(ev.time)}
    for attr in ("chroms", "chrom", "start", "end", "delta"):
        if hasattr(ev, attr):
            val = getattr(ev, attr)
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            elif isinstance(val, np.ndarray):
                val = val.tolist()
            d[attr] = val
    return d


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a tumor genome for the Tau benchmarking pipeline."
    )
    parser.add_argument("--sample", required=True, help="Sample ID (e.g. SIM_1)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Seed the global simulation RNG so genome structure is reproducible.
    # Note: generate_mutation_dataframe uses its own local rng (unseeded), so
    # exact read depths/VAFs vary per run; CN states and event times are fixed.
    sim._rng = np.random.default_rng(args.seed)

    # Run simulation
    g, events = sim.simulate_and_write()

    # Collect all mutation dataframes from segments
    mut_dfs = [
        seg.mut_df
        for seg in g.segments
        if seg.mut_df is not None and len(seg.mut_df) > 0
    ]
    all_mut_df = pd.concat(mut_dfs, ignore_index=True) if mut_dfs else pd.DataFrame(
        columns=["chrom", "pos", "ref", "alt", "nalt", "nref"]
    )

    sample = args.sample
    vcf_path    = os.path.join(args.outdir, f"{sample}.vcf")
    cna_path    = os.path.join(args.outdir, f"{sample}.cna.txt")
    purity_path = os.path.join(args.outdir, f"{sample}.purity.txt")
    truth_path  = os.path.join(args.outdir, f"{sample}.truth.json")

    write_vcf(vcf_path, all_mut_df)
    write_cna(cna_path, g.segments)

    with open(purity_path, "w") as f:
        f.write(f"{g.purity}\n")

    truth = {
        "purity": float(g.purity),
        "genome_events": [event_to_dict(ev) for ev in events],
        # TWO levels of route truth, and they answer different questions:
        #   route_class  what the traced lineage actually did (structural truth)
        #   equiv_class  the routes NO method could tell apart for this timing — co-temporal splits
        #                zero a t-interval, leaving the corresponding matrix column unconstrained, so
        #                several routes predict an identical N
        # Score route selection against equiv_class; scoring against true_route alone penalises a
        # method for failing to distinguish the indistinguishable.
        "segments": [
            {
                "seg_id": seg.seg_id,
                "major": int(seg.major),
                "minor": int(seg.minor),
                "true_route": getattr(seg, "tag", "") or "",
                "route_class": list(getattr(seg, "route_class", []) or []),
                "equiv_class": list(getattr(seg, "equiv_class", []) or []),
                "t_values": [float(v) for v in (seg.t_vals or [])],
            }
            for seg in g.segments
        ],
    }
    with open(truth_path, "w") as f:
        json.dump(truth, f, indent=2)

    print(f"Wrote {sample} to {args.outdir}")
    print(f"  VCF:    {vcf_path}  ({len(all_mut_df)} mutations)")
    print(f"  CNA:    {cna_path}  ({len(g.segments)} segments)")
    print(f"  Purity: {g.purity:.3f}")
    print(f"  Events: {len(events)} "
          f"({sum(1 for e in events if type(e).__name__=='WGD')} WGD, "
          f"{sum(1 for e in events if type(e).__name__=='PGD')} PGD)")


if __name__ == "__main__":
    main()
