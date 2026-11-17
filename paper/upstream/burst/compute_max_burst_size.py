#!/usr/bin/env python3
"""
compute_max_burst_size.py — per-segment max_burst_size for all burst-bearing high-CN segments.

max_burst_size = largest set of CONSECUTIVE gains co-timed within a 0.05 span (the burst the
four-tier timing taxonomy finds), taken as the MAX over the segment's polytope solutions. It is
LOCATION-AGNOSTIC (no reference to WGD). excess := max_burst_size > D, D = ⌊M/2⌋+⌊m/2⌋.

Shared by the burst↔WGD panels (figure_burst_wgd_lead.py / proto_persample_bar.py) so they use one
consistent definition. Output: figures_genome_highcn_catalogue/max_burst_size.tsv  (SEG + max_burst_size)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("TAU_REPO", "."))
CAT  = BASE / "dev/pcawg/oncogenes/figures_genome_highcn_catalogue"
MISC = BASE / "dev/pcawg/misc_data_and_plots"
SEG  = ["sample", "chrom", "start", "end"]
GCOLS = [f"G{i}" for i in range(1, 10)]
BURST_SPAN = 0.05
BURST_CATS = ["complete_burst", "incomplete_burst", "burst_compatible"]

tax = pd.read_csv(CAT / "wgd_burst_taxonomy_polytope.tsv", sep="\t", dtype={"chrom": str})
seg = tax[tax["category"].isin(BURST_CATS)][SEG].drop_duplicates().reset_index(drop=True)
print(f"{len(seg):,} burst-bearing segments", flush=True)

print("Loading per-solution cache …", flush=True)
oc = pd.read_csv(MISC / "genome_highcn_timing_cache.tsv", sep="\t", dtype={"chrom": str},
                 usecols=SEG + GCOLS).merge(seg, on=SEG, how="inner")
print(f"  {len(oc):,} solution-rows; computing max_burst_size …", flush=True)

G = oc[GCOLS].values.astype(float)                       # sorted G1..Gn, NaN beyond ne
mbs = np.ones(len(G), dtype=np.int16)
with np.errstate(invalid="ignore"):
    for w in range(2, len(GCOLS) + 1):                   # window of w consecutive gains
        spans = G[:, w - 1:] - G[:, :G.shape[1] - (w - 1)]
        tight = np.nanmin(np.where(np.isnan(spans), np.inf, spans), axis=1)
        mbs[tight < BURST_SPAN] = w
oc["mbs"] = mbs
out = oc.groupby(SEG)["mbs"].max().rename("max_burst_size").reset_index()
out.to_csv(CAT / "max_burst_size.tsv", sep="\t", index=False)
print(f"Saved → {CAT/'max_burst_size.tsv'}  (median={out.max_burst_size.median():.0f})", flush=True)
print("Done.", flush=True)
