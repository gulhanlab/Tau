#!/usr/bin/env python3
"""merge_shards.py <TT> : concat the per-shard cluster interval-range + membership outputs into the
final tables, order columns, enrich the per-segment table with cluster_label, delete the shard dir."""
import os
import sys, glob, shutil
import pandas as pd
from pathlib import Path
TABLES=Path(os.path.join(os.environ.get("TAU_REPO", "."), "dev/pcawg/signatures/timing_tables"))
TT=sys.argv[1]
sub=TABLES/f"_shards_{TT}"
rt=sorted(glob.glob(str(sub/f"sig_intervals_range_{TT}_cluster.shard*.tsv")))
mm=sorted(glob.glob(str(sub/f"sig_clustertiming_{TT}_membership.shard*.tsv")))
assert rt, f"no shards in {sub}"
dr=pd.concat([pd.read_csv(f,sep="\t") for f in rt],ignore_index=True)
# order columns: descriptors, interval+bounds, raw, normalized, ratio, range, exposure
SIGS=sorted({c[:-6] for c in dr.columns if c.endswith("_ratio")})
desc=["sample","tumor_type","cn_state","cluster_label","cluster","n_segments"]
order=desc+["time_interval","clocklike_cumulative_start","clocklike_cumulative_end","clocklike_raw_interval"]+\
      [f"{s}_raw_interval" for s in SIGS]+["clocklike_normalized_interval"]+[f"{s}_normalized_interval" for s in SIGS]+\
      [f"{s}_ratio" for s in SIGS]+["clocklike_range"]+[f"{s}_range" for s in SIGS]+[f"{s}_exposure" for s in SIGS]
dr=dr[[c for c in order if c in dr.columns]]
dr.to_csv(TABLES/f"sig_intervals_range_{TT}_cluster.tsv",sep="\t",index=False)
dm=pd.concat([pd.read_csv(f,sep="\t") for f in mm],ignore_index=True)
dm.to_csv(TABLES/f"sig_clustertiming_{TT}_membership.tsv",sep="\t",index=False)
print(f"merged {len(rt)} shards -> cluster intervals {dr.shape} ({dr['sample'].nunique()} samples), "
      f"membership {dm.shape}; signatures {SIGS}")
# enrich per-segment table with cluster_label
segf=TABLES/f"sig_timing_{TT}_segments.tsv"
if segf.exists():
    seg=pd.read_csv(segf,sep="\t")
    if "cluster_label" in seg.columns: seg=seg.drop(columns="cluster_label")
    seg=seg.merge(dm[["sample","segment_id","cluster_label"]],on=["sample","segment_id"],how="left")
    cols=list(seg.columns); cols.remove("cluster_label"); i=cols.index("cn_state")+1
    seg=seg[cols[:i]+["cluster_label"]+cols[i:]]; seg.to_csv(segf,sep="\t",index=False)
    print(f"enriched {segf.name} with cluster_label")
shutil.rmtree(sub); print("removed shard dir")
for s in SIGS:
    sub2=dr.dropna(subset=[f"{s}_ratio"])
    det=(sub2.clocklike_range<0.05)&(sub2[f"{s}_range"]<0.05)
    print(f"  {s}: {sub2['sample'].nunique()} samples, {len(sub2)} cluster-intervals, "
          f"{det.mean():.0%} well-determined (both ranges<0.05)")
