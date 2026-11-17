"""Run the signature step of a standard Tau run for one sample, into its per_sample directory.

This is the `--include-signatures` half of `tau run`, applied to a sample whose clock run already
exists. It calls the same two functions the CLI calls (tau/cli.py:343-396) and writes the same two
files under the same names, so what the panel reads is a standard Tau output and not a
figure-specific format:

    <sample>.tau_sig_intervals.tsv     sig_timing.cluster_interval_table   per (cluster, interval)
    <sample>.tau_sig_states.tsv        sig_states.fit_cluster_table        the BIC stepwise fit

A full `tau run --include-signatures` would redo EM, timing and clustering from the VCF first, and
needs --vcf/--cnv/--purity/--ref_fasta/--cosmic_csv/--exposures. Everything downstream of the clock
timing is identical either way, so this reuses the saved genome instead. If the sample is ever re-run
properly, these files are simply overwritten and the panel keeps working.

ONE DEVIATION FROM STOCK `tau run`, and it is deliberate. DBSCAN leaves un-clustered segments at
cluster_label = -1; make_clustered_genome passes those through under their raw coordinate seg_id, and
sig_timing only re-times ids beginning "merged_", so a stock run drops them from the signature table
without warning. tau/overview.py still DRAWS them — three of this sample's sixteen trees are single
segments — so a stock signature table would be missing exactly those three. Each is promoted to its
own one-member cluster first; a one-member cluster's pooled counts are the segment's own, so nothing
is averaged away. This is arguably a package bug rather than a figure requirement.

    python paper/figures/data/signature/build/build_cluster_trees.py [uuid ...]
"""
import glob
import gzip
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
from tau import timing
from tau.sig_states import fit_cluster_table
from tau.sig_timing import cluster_interval_table

TAU = TAU_OUTPUTS   # raw genome pickles only
PER_SAMPLE = PER_SAMPLE_DIR
SAMPLES = ["0250e6f7"]      # the Fig. 2b / Fig. 7a esophageal adenocarcinoma


def load_genome(prefix):
    hits = glob.glob(str(TAU / "*" / f"{prefix}*" / f"{prefix}*.genome.pkl.gz"))
    if not hits:
        sys.exit(f"no genome pickle for {prefix}")
    p = Path(hits[0])
    with gzip.open(p, "rb") as f:
        return pickle.load(f), p.parent.name, p.parent.parent.name


def states_of(g):
    """The (major, minor) states present, so only those solutions have to be loaded."""
    return {(int(s.major_cn), int(s.minor_cn)) for s in g.segments
            if getattr(s, "major_cn", 0) and s.major_cn > 0 and s.minor_cn >= 0}


def seg_clusters(sample):
    """The run's own segment_clusters.tsv, with un-clustered segments promoted (see module docstring)."""
    d = PER_SAMPLE / sample
    if not d.is_dir():
        sys.exit(f"{sample} has no directory under {PER_SAMPLE}")
    scd = pd.read_csv(d / f"{sample}.segment_clusters.tsv", sep="\t")
    lone = scd.cluster_label == -1
    if lone.any():
        nxt = scd[~lone].groupby(["major_cn", "minor_cn"]).cluster_label.max().to_dict()
        labels = []
        for _, r in scd[lone].iterrows():
            cn = (r.major_cn, r.minor_cn)
            nxt[cn] = int(nxt.get(cn, -1)) + 1
            labels.append(nxt[cn])
        scd.loc[lone, "cluster_label"] = labels
        print(f"    promoted {int(lone.sum())} un-clustered segment(s) to single-member clusters")
    return d, scd


def one(g, sample, tt, env):
    t0 = time.time()
    d, scd = seg_clusters(sample)
    iv = cluster_interval_table(g, env, seg_cluster_df=scd, sample=sample, tumor_type=tt)
    if not len(iv):
        sys.exit(f"{sample}: cluster_interval_table returned nothing")
    p_iv = d / f"{sample}.tau_sig_intervals.tsv"
    iv.to_csv(p_iv, sep="\t", index=False)
    print(f"  {sample[:8]}: {len(iv)} cluster-intervals over {iv.cluster.nunique()} units "
          f"in {time.time()-t0:.0f}s -> {p_iv.name}")

    try:
        st = fit_cluster_table(iv)
        p_st = d / f"{sample}.tau_sig_states.tsv"
        st.to_csv(p_st, sep="\t", index=False)
        print(f"            {len(st)} fitted states -> {p_st.name}")
    except Exception as e:     # the intervals are the panel's input; the state fit is a bonus
        print(f"            [warn] state fit skipped: {e}")

    sigs = sorted(c[:-6] for c in iv.columns if c.endswith("_ratio"))
    for sig in sigs:
        ok = iv.dropna(subset=[f"{sig}_raw_interval"])
        if len(ok):
            print(f"            {sig:>8}: baseline "
                  f"{ok[f'{sig}_raw_interval'].sum()/ok.clocklike_raw_interval.sum():.4f} "
                  f"({ok.cluster.nunique()}/{iv.cluster.nunique()} units)")
    return iv


def main(argv=None):
    prefixes = (argv or sys.argv[1:]) or SAMPLES
    loaded = [load_genome(p) for p in prefixes]
    # only the states these genomes contain: load_routes() reads the whole monolithic 7_5 solutions
    # file, which is minutes of I/O for solutions no segment here needs
    states = sorted(set().union(*(states_of(x[0]) for x in loaded)))
    print(f"loading route solutions for {len(states)} CN states...", flush=True)
    env = timing.load_routes_for_states(states)
    for g, sample, tt in loaded:
        one(g, sample, tt, env)


if __name__ == "__main__":
    main()
