"""Two worked interval-comparison examples -> paper/figures/data/signature/sig_interval_examples.tsv.

Ports the two exemplar pooled clusters used by panel c out of the per-tumour-type signature tables, so
the panel reads only from paper/.

Both are chosen to be FULLY DETERMINED (clock polytope range 0.000) and well powered (>900 clock
mutations). That matters for a methods figure: an underdetermined unit would show interval boundaries
that are themselves intervals, and the expansion/contraction the panel is explaining would be confounded
with polytope width. They are illustrative examples, not a claim about typical behaviour.

  Eso.AdenoCA  9ae33a63  merged_3_0_0  SBS17   front-loaded — early intervals EXPAND under SBS17
  Lung.AdenoCA 3b20d548  merged_4_1_0  APOBEC  back-loaded  — late intervals expand under APOBEC

Different copy-number states on purpose, so the panel is not read as a property of one topology, and
the (4,1) shows three gains rather than two.

Usage:  python paper/figures/data/signature/build/build_examples.py
"""
import glob
import gzip
import pickle
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
SRC = ANALYSIS / "pcawg/signatures/timing_tables"
TAU = TAU_OUTPUTS
OUT = cache_dir_of(__file__); OUT.mkdir(parents=True, exist_ok=True)
HG19_TOTAL = 3_095_677_412        # tau/trees.py plot_cluster_trees, so the % matches the overview

# The two tumours of Figure 2, so the methods supplement explains the weighting on the same genomes
# the main figure times. Note the trade-off this makes against the earlier picks: the effect sizes are
# smaller (+0.08 / -0.11 against +0.51 / -0.46), and the stomach unit is UNDERDETERMINED (clock
# polytope range 0.36) where the earlier pair were both exactly determined. The panel therefore shows
# a subtler, more typical case rather than the clearest possible one.
# The best-powered, fully determined units in the cohort for this comparison (clock polytope range
# 0.000, >=25 pooled segments, >=950 clock mutations). Determinacy matters here: an underdetermined
# unit's gain times are themselves intervals, which would confound the expansion the panel defines.
# ROUTE is needed to draw the tree and is not in the interval tables — taken from the clustered genome.
# The APOBEC pick is the best-powered determined back-loaded unit in the cohort with three gains:
# 45 segments, 1,347 clock mutations, strongest direction (score -0.19). Its last clock interval is
# only 0.017 wide, so the panel simply does not print a width label below 0.05 — the arrow is still
# drawn and the fold change still labelled, which is what the reader needs.
#
# Searched alternatives, for the record. Requiring no interval under 0.05 forces (3,0) or (3,1):
# Lung.AdenoCA merged_3_0_0 (min 0.095) and merged_3_1_0 (min 0.054, 1,895 clock muts), plus
# Lung.SCC 34881c6b and Eso.AdenoCA 64623aaa (both (3,1), wider but weaker). Relaxing to 0.03 and
# demanding (4,1)/(4,0) leaves only very low-power units — Uterus 0c39c03a (34 clock mutations),
# Head.SCC de8ef60b (38), Panc bbdd7393 (14) — none usable for a "well powered" methods example.
EXAMPLES = [("Eso.AdenoCA", "9ae33a63", "merged_3_0_0", "SBS17", "front", "3_0.1"),
            ("Lung.AdenoCA", "3b20d548", "merged_4_1_0", "APOBEC", "back", "4_1.1")]


def cluster_identity(prefix, seg_id):
    """('(3,0) Cluster 1', 4.2) for a merged cluster — the overview plot's own naming.

    Both come from tau.trees so the supplement labels a cluster exactly as the sample's overview
    figure does: `_get_final_segment_clusters` numbers clusters 1..n within a CN state in its own
    sort order, and the percentage is the cluster's total segment length over hg19.
    """
    from tau.trees import _get_final_segment_clusters
    hits = glob.glob(str(TAU / "*" / f"{prefix}*" / f"{prefix}*.clustered_genome.pkl.gz"))
    if not hits:
        sys.exit(f"no clustered genome for {prefix}")
    with gzip.open(hits[0], "rb") as f:
        cg = pickle.load(f)
    for label, seg in _get_final_segment_clusters(cg).items():
        if seg.seg_id == seg_id:
            tl = getattr(seg, "total_length", seg.end - seg.start)
            return label, 100.0 * tl / HG19_TOTAL
    sys.exit(f"{seg_id} is not one of {prefix}'s final clusters")


def main():
    out = []
    for tt, pre, clus, sig, kind, route in EXAMPLES:
        t = pd.read_csv(SRC / f"sig_intervals_range_{tt}_cluster.tsv", sep="\t")
        d = t[t["sample"].str.startswith(pre) & (t.cluster == clus)].sort_values("time_interval")
        if d.empty:
            sys.exit(f"example not found: {tt} {pre} {clus}")
        clabel, pct = cluster_identity(pre, clus)
        keep = pd.DataFrame({
            "example": kind, "tumor_type": tt, "sample": d["sample"].values, "cluster": clus,
            "route": route, "cluster_label": clabel, "pct_genome": pct,
            "cn_state": d.cn_state.values, "n_segments": d.n_segments.values,
            "signature": sig, "time_interval": d.time_interval.values,
            "clock_norm": d.clocklike_normalized_interval.values,
            "sig_norm": d[f"{sig}_normalized_interval"].values,
            "clock_raw": d.clocklike_raw_interval.values,
            "sig_raw": d[f"{sig}_raw_interval"].values,
            "ratio": d[f"{sig}_ratio"].values,
            "clock_range": d.clocklike_range.values})
        out.append(keep)
        print(f"{tt} {pre} {clus} [{sig}]: {clabel}, {pct:.1f}% of genome, {len(d)} intervals, "
              f"clock {d.clocklike_raw_interval.sum():.0f} muts, range {d.clocklike_range.max():.3f}")
    t = pd.concat(out, ignore_index=True)
    t.to_csv(OUT / "sig_interval_examples.tsv", sep="\t", index=False)
    print(f"\n-> {OUT/'sig_interval_examples.tsv'}")


if __name__ == "__main__":
    main()
