"""Re-cluster a sample's RAW genome from scratch (now with major CN 7 enabled) and write a fresh
clustered genome + segment_clusters.tsv into the per-sample bundle, so emit/plot pick up (7,x) merges.

Reads the stored RAW genome (retains snv_table), runs cluster_segment_multiplicities (range now 2..7),
builds the merged genome, then loads Sage routes for exactly the merged units' CN states via the SPLIT
per-CN loader (tau/data/solutions/*.sobj, incl. (7,0)-(7,5)) and times them. Does NOT touch dev/ —
writes only into paper/data_tables/per_sample/{full}/.
Usage:  python paper/batch_scripts/recluster_sample.py <sample_id> [adj|noadj] [eps]

eps: DBSCAN radius for the multiplicity clustering (default 0.07 = the frozen cohort run; every bundle
in the paper is at 0.07 -- a 0.08 trial for dc537fcf on 2026-09-18 was reverted, see
paper/figures/main/examples/README.md).
"""
import sys, gzip, pickle
from pathlib import Path

R = next(p for p in Path(__file__).resolve().parents if (p / "paper").is_dir())
sys.path.insert(0, str(R))
from tau import timing
from tau.clustering import cluster_segment_multiplicities, make_clustered_genome

TAU_OUT = R / "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
OUT_ROOT = R / "paper/data_tables/per_sample"


def recluster(sample, adjacency=True, eps=0.07):
    sdir = next(iter(TAU_OUT.glob(f"*/{sample}*")))
    full = sdir.name
    with gzip.open(sdir / f"{full}.genome.pkl.gz", "rb") as f:
        g = pickle.load(f)

    # two-step (adjacency-merge → DBSCAN) is now the default in cluster_segment_multiplicities;
    # pass adjacency=False for the legacy per-segment shape clustering.
    seg_cluster_df = cluster_segment_multiplicities(g, adjacency=adjacency, eps=eps)
    clustered = make_clustered_genome(g, seg_cluster_df)
    # load routes for exactly the merged units' states via the SPLIT per-CN loader
    states = sorted({(int(s.major_cn), int(s.minor_cn)) for s in clustered.segments})
    routes = timing.load_routes_for_states(states)
    clustered.calculate_multiplicities(bootstrap_B=0, random_state=42)
    clustered.time_segments(env=routes)

    out = OUT_ROOT / full
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / f"{full}.clustered_genome.pkl.gz", "wb") as f:
        pickle.dump(clustered, f)
    seg_cluster_df.to_csv(out / f"{full}.segment_clusters.tsv", sep="\t", index=False)

    # report high-CN merged units now present
    hi = [(int(s.major_cn), int(s.minor_cn), getattr(s, "num_segs", 1))
          for s in clustered.segments if int(s.major_cn) >= 6 and getattr(s, "num_segs", 1) >= 2]
    m7 = [(M, m, n) for (M, m, n) in hi if M == 7]
    (out / "CLUSTERING_EPS").write_text(f"{eps}\n")
    print(f"{full}: eps={eps} | {len(clustered.segments)} merged units | "
          f"major>=6 merges: {sorted(hi, reverse=True)}")
    print(f"   MAJOR-7 merges: {sorted(m7, reverse=True) if m7 else 'NONE'}")
    return full


if __name__ == "__main__":
    # two-step (adjacency) is the default; pass "noadj" as arg 2 to use the legacy shape-only clustering
    adj = not (len(sys.argv) > 2 and sys.argv[2] == "noadj")
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.07
    recluster(sys.argv[1], adjacency=adj, eps=eps)
