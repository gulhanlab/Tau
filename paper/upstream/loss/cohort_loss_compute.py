#!/usr/bin/env python3
"""Cohort-wide timed-loss table: per-segment WGD1/WGD2 loss across all double-WGD samples, for the
loss heatmaps. Uses the tree-consistent method (_pick_draw_for_tree + 0.2 focal-aware assignment loss),
each sample with its own W1<W2. Corrected double-WGD set (dedup + PELICAN-excluded).
-> dev/pcawg/FigWGDPGD/cohort_loss_segments.tsv
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
reps = representative_samples()                                   # dedup + PELICAN excluded
dbl = {s for s, k in corr.items() if k >= 2} & reps
tt = ev.drop_duplicates("sample").set_index("sample")["tumor_type"]
wgd = {s: sorted(g.time.values) for s, g in ev[ev.classification == "WGD"].groupby("sample")}
gmap = {Path(f).name.replace(".genome.pkl.gz", ""): f
        for f in glob.glob(str(ROOT/"*"/"*"/"*.genome.pkl.gz"))}
print(f"{len(dbl)} double-WGD samples (corrected/dedup/PELICAN-excl); {len(gmap)} genome pkls", flush=True)


def losses_assign(ev_times, W1, W2):
    ev = sorted(float(x) for x in ev_times)
    def a(t):
        d1, d2 = abs(t - W1), abs(t - W2)
        return "focal" if min(d1, d2) > ASSIGN_R else ("W1" if d1 <= d2 else "W2")
    tags = [a(t) for t in ev]
    cb1 = 2 + sum(t < W1 for t, g in zip(ev, tags) if g == "focal")
    d1 = tags.count("W1"); L1 = max(0, cb1 - d1)
    cb2 = cb1 + d1 + sum(W1 < t < W2 for t, g in zip(ev, tags) if g == "focal")
    d2 = tags.count("W2"); L2 = max(0, cb2 - d2)
    return L1, L2, cb1, cb2                              # cb1+cb2 = total loss OPPORTUNITY (branches)


rows = []
for n, s in enumerate(sorted(dbl)):
    if s not in gmap or s not in wgd:
        continue
    w = wgd[s]; W1, W2 = float(w[0]), float(w[-1])
    if W2 - W1 < 0.15:
        continue
    try:
        with gzip.open(gmap[s], "rb") as fh:
            g = pickle.load(fh)
    except Exception:
        continue
    for seg in g.segments:
        if not getattr(seg, "timing_result", None) or seg.N_counts is None:
            continue
        chrom = str(getattr(seg, "chrom", ""))
        if chrom not in CHROMS:
            continue
        et, _ = _pick_draw_for_tree(seg, [W1, W2])
        if et is None or len(et) == 0:
            continue
        L1, L2, cb1, cb2 = losses_assign(et, W1, W2)
        rows.append(dict(sample=s, tumor_type=tt.get(s, "?"), chrom=chrom,
                         start=int(seg.start), end=int(seg.end), L1=L1, L2=L2, cb=cb1 + cb2,
                         length=int(seg.end) - int(seg.start)))
    del g
    if (n + 1) % 25 == 0:
        print(f"  …{n+1}/{len(dbl)} samples, {len(rows)} segments", flush=True)

d = pd.DataFrame(rows)
d.to_csv(BASE/"dev/pcawg/FigWGDPGD/cohort_loss_segments.tsv", sep="\t", index=False)
print(f"done: {len(d)} segments from {d['sample'].nunique()} samples, "
      f"{d['tumor_type'].nunique()} tumour types", flush=True)
print("mean L1=%.2f L2=%.2f" % ((d.L1*d.length).sum()/d.length.sum(),
                                (d.L2*d.length).sum()/d.length.sum()))
