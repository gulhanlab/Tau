#!/usr/bin/env python3
"""
Scan all segment_times.tsv files in output_sbs1sbs5_soft_allpcawg.
Keep segments with n_events >= 3 (high-CN).
Pivot time_fraction by time_point → G1..G9 per (sample, chrom, start, end, key, draw_id).
Output: dev/pcawg/misc_data_and_plots/genome_highcn_timing_cache.tsv
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

TAU_OUT = Path(os.path.join(os.environ.get("TAU_REPO", "."),
                            "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"))
OUT = Path(os.path.join(os.environ.get("TAU_REPO", "."),
                        "dev/pcawg/misc_data_and_plots/genome_highcn_timing_cache.tsv"))

MIN_EVENTS = 3

seg_cols = ["chrom", "start", "end", "key"]
chunks = []
n_samples = 0
n_skipped = 0

for tt_dir in sorted(TAU_OUT.iterdir()):
    if not tt_dir.is_dir():
        continue
    tumor_type = tt_dir.name
    for s_dir in sorted(tt_dir.iterdir()):
        if not s_dir.is_dir():
            continue
        sid = s_dir.name
        tsv = s_dir / f"{sid}.segment_times.tsv"
        if not tsv.exists():
            continue
        if tsv.stat().st_size < 200:
            n_skipped += 1
            continue
        try:
            d = pd.read_csv(tsv, sep="\t")
        except Exception as e:
            print(f"  read error {sid}: {e}")
            continue
        if d.empty:
            continue

        # Max time_point per segment (= n_events)
        ne = (d.groupby(seg_cols)["time_point"].max()
               .reset_index()
               .rename(columns={"time_point": "n_events"}))
        ne = ne[ne["n_events"] >= MIN_EVENTS]
        if ne.empty:
            n_samples += 1
            continue

        d = d.merge(ne, on=seg_cols)

        # Parse major/minor CN from key ("major_minor.variant")
        cn = d["key"].str.split(".", n=1).str[0].str.split("_")
        d["major_cn"] = cn.str[0].astype(int)
        d["minor_cn"] = cn.str[1].astype(int)

        # mutation_count: one value per (seg, draw)
        mc = (d.groupby(seg_cols + ["draw_id"])["mutation_count"]
                .first().reset_index())

        # Pivot time_fraction → G1..G9
        piv = (d.pivot_table(index=seg_cols + ["draw_id", "major_cn", "minor_cn", "n_events"],
                             columns="time_point",
                             values="time_fraction",
                             aggfunc="first")
                .reset_index())
        piv.columns.name = None
        # rename integer columns to G1..G9
        int_cols = [c for c in piv.columns if isinstance(c, (int, np.integer))]
        piv = piv.rename(columns={c: f"G{c}" for c in int_cols})

        piv = piv.merge(mc, on=seg_cols + ["draw_id"], how="left")
        piv.insert(0, "tumor_type", tumor_type)
        piv.insert(0, "sample", sid)

        chunks.append(piv)
        n_samples += 1

        if n_samples % 200 == 0:
            total = sum(len(c) for c in chunks)
            print(f"  {n_samples} samples, {total} rows", flush=True)

print(f"\nProcessed {n_samples} samples ({n_skipped} skipped as empty)")
if not chunks:
    print("No data — exiting")
    raise SystemExit(1)

cache = pd.concat(chunks, ignore_index=True)
print(f"Total rows: {len(cache)}")
print(f"Unique samples: {cache['sample'].nunique()}")
print(f"n_events range: {cache['n_events'].min()}-{cache['n_events'].max()}")

# Consistent column order
gcols = [f"G{i}" for i in range(1, 10) if f"G{i}" in cache.columns]
base = ["sample", "tumor_type", "chrom", "start", "end", "key",
        "major_cn", "minor_cn", "n_events", "draw_id"]
cache = cache[[c for c in base + gcols + ["mutation_count"] if c in cache.columns]]

cache.to_csv(OUT, sep="\t", index=False)
print(f"Saved → {OUT}")
