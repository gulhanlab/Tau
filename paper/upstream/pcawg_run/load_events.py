#!/usr/bin/env python3
"""
load_events.py — one-time consolidation of all PCAWG event clusters under the NEW WGD definition.

Walks the per-sample time_clusters.tsv files and classifies every event cluster with the updated rule:
    WGD            : gf >= 0.40  OR  n_chroms >= 8
    PGD            : n_chroms >= 2   (and not WGD)
    chrom_specific : n_chroms == 1

genome-fraction denominator (REFINED, "timeable" genome): the per-sample time_clusters.tsv
stores gf against the FULL autosomal genome (near_len / GENOME_SIZE). We rescale it here to the
refined denominator = GENOME_SIZE - gained-but-untimeable genome (major_cn>=2 segments that got
no timing estimate), keeping diploid/un-gained genome in. The per-sample denom comes from
timeable_denom.tsv (compute_timeable_denom.py). Rescale: gf_new = gf_old * GENOME_SIZE / denom,
clipped to 1.0. (tau/clustering.py now computes this natively, so future runs need no rescaling.)

Output: events_allpcawg.tsv (one row per event) — the shared input for every FigWGDPGD panel.
Columns: sample, tumor_type, cluster_id, time, classification, gf, n_chroms, n_segments, pct, chrom
"""
import os
from pathlib import Path
import glob
import pandas as pd

BASE = Path(os.environ.get("TAU_REPO", "."))
DATA = BASE / "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
OUT  = BASE / "dev/pcawg/FigWGDPGD/events_allpcawg.tsv"
DENOM_TSV = BASE / "dev/pcawg/FigWGDPGD/timeable_denom.tsv"

HG19_SIZES = {1:249250621,2:243199373,3:198022430,4:191154276,5:180915260,6:171115067,
              7:159138663,8:146364022,9:141213431,10:135534747,11:135006516,12:133851895,
              13:115169878,14:107349540,15:102531392,16:90354753,17:81195210,18:78077248,
              19:59128983,20:63025520,21:48129895,22:51304566}
GENOME_SIZE = sum(HG19_SIZES.values())

denom = pd.read_csv(DENOM_TSV, sep="\t").set_index("sample")["denom"].to_dict()
print(f"loaded refined denom for {len(denom):,} samples", flush=True)

WGD_GF_THRESH      = 0.40  # fraction of TIMEABLE genome (refined denominator)
WGD_NCHROMS_THRESH = 8     # was 10/14; lowered to 8 — CN profiles show 8-9-chrom "PGDs"
                           # are near-whole second doublings (ploidy/major>=3 match 2xWGD),
                           # not partial events. ≥8 cleans PGD to genuinely partial (2-7 chroms)
                           # and moves +34 doubled-twice tumors into the 2xWGD group.


def classify(gf, n_chroms):
    if gf >= WGD_GF_THRESH or n_chroms >= WGD_NCHROMS_THRESH:
        return "WGD"
    if n_chroms >= 2:
        return "PGD"
    return "chrom_specific"


rows = []
files = glob.glob(str(DATA / "*" / "*" / "*.time_clusters.tsv"))
print(f"reading {len(files):,} time_clusters files…", flush=True)
for i, f in enumerate(files):
    if i % 500 == 0:
        print(f"  {i}/{len(files)}", flush=True)
    p = Path(f)
    sample = p.parent.name
    tumor_type = p.parent.parent.name
    try:
        df = pd.read_csv(f, sep="\t")
    except Exception:
        continue
    if df.empty or "n_chroms" not in df.columns:
        continue
    # rescale full-genome gf -> refined (timeable) denominator
    d = denom.get(sample)
    if d is None:
        continue
    scale = GENOME_SIZE / d
    for _, r in df.iterrows():
        gf_old = float(r.get("gf", 0))
        gf_new = min(gf_old * scale, 1.0)
        rows.append({
            "sample": sample, "tumor_type": tumor_type,
            "cluster_id": r.get("cluster_id"), "time": r.get("time"),
            "classification": classify(gf_new, int(r["n_chroms"])),
            "gf": round(gf_new, 4), "n_chroms": int(r["n_chroms"]),
            "n_segments": r.get("n_segments"), "pct": round(gf_new * 100, 2),
            "chrom": r.get("chrom"),
        })

ev = pd.DataFrame(rows)
ev.to_csv(OUT, sep="\t", index=False)
print(f"\nSaved {len(ev):,} events from {ev['sample'].nunique():,} samples → {OUT}", flush=True)
print("\nevent counts (NEW classification):")
print(ev["classification"].value_counts())
print(f"\nsamples with >=1 WGD: {ev[ev.classification=='WGD']['sample'].nunique():,}")
print(f"samples with >=1 PGD: {ev[ev.classification=='PGD']['sample'].nunique():,}")
print(f"samples with >=1 WGD/PGD: {ev[ev.classification.isin(['WGD','PGD'])]['sample'].nunique():,}")
