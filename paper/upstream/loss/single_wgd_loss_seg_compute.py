#!/usr/bin/env python3
"""Per-SEGMENT single-WGD loss (post-WGD band), to complement cohort_loss_segments.tsv (double-WGD) so
the loss landscape spans the SAME all-WGD cohort as the amplification landscape. One WGD time W; loss L
is ascribed to the post-WGD window [W,1]. Same tree-consistent method / dedup / PELICAN-excl.
-> dev/pcawg/FigWGDPGD/single_wgd_loss_segments.tsv  (sample,tumor_type,chrom,start,end,L,W,length)
"""
import os
import glob, gzip, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.environ.get("TAU_REPO", "."), "dev/pcawg/FigWGDPGD"))
from wgd_split_filter import corrected_wgd_multiplicity
from donor_dedup import representative_samples
from tau.trees import _pick_draw_for_tree

BASE = Path(os.environ.get("TAU_REPO", "."))
ROOT = BASE/"dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
ASSIGN_R = 0.20
CHROMS = set([str(i) for i in range(1, 23)] + ["X"])

ev = pd.read_csv(BASE/"dev/pcawg/FigWGDPGD/events_allpcawg.tsv", sep="\t")
corr = corrected_wgd_multiplicity(ev)
reps = representative_samples()
single = {s for s, k in corr.items() if k == 1} & reps
tt = ev.drop_duplicates("sample").set_index("sample")["tumor_type"]
wgd = {s: sorted(g.time.values) for s, g in ev[ev.classification == "WGD"].groupby("sample")}
gmap = {Path(f).name.replace(".genome.pkl.gz", ""): f
        for f in glob.glob(str(ROOT/"*"/"*"/"*.genome.pkl.gz"))}
todo = sorted(single & set(gmap) & set(wgd))
print(f"{len(todo)} single-WGD samples", flush=True)


def loss_single(ev_times, W):
    ev = sorted(float(x) for x in ev_times)
    tags = ["focal" if abs(t - W) > ASSIGN_R else "WGD" for t in ev]
    cb = 2 + sum(t < W for t, g in zip(ev, tags) if g == "focal")   # copies at WGD = loss OPPORTUNITY
    return max(0, cb - tags.count("WGD")), cb


rows = []
for n, s in enumerate(todo):
    W = float(wgd[s][0])
    try:
        with gzip.open(gmap[s], "rb") as fh:
            g = pickle.load(fh)
    except Exception:
        continue
    for seg in g.segments:
        if not getattr(seg, "timing_result", None) or seg.N_counts is None:
            continue
        if str(getattr(seg, "chrom", "")) not in CHROMS:
            continue
        et, _ = _pick_draw_for_tree(seg, [W])
        if et is None or len(et) == 0:
            continue
        L, cb = loss_single(et, W)
        rows.append(dict(sample=s, tumor_type=tt.get(s, "?"), chrom=str(seg.chrom),
                         start=int(seg.start), end=int(seg.end),
                         major_cn=int(seg.major_cn), minor_cn=int(seg.minor_cn), L=L, cb=cb, W=W,
                         length=int(seg.end) - int(seg.start)))
    del g
    if (n + 1) % 50 == 0:
        print(f"  …{n+1}/{len(todo)}, {len(rows)} segs", flush=True)

d = pd.DataFrame(rows)
d.to_csv(BASE/"dev/pcawg/FigWGDPGD/single_wgd_loss_segments.tsv", sep="\t", index=False)
print(f"done: {len(d)} segments, {d['sample'].nunique()} samples", flush=True)
