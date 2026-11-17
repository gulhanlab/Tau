#!/usr/bin/env python3
"""
build_genome_highcn_cache_v2.py — same as build_genome_highcn_cache.py, plus ROUTE QUALITY.

v1 kept every route Tau emitted and dropped all fit-quality columns, so every downstream consumer
(width, max_burst_size, the drawn fan) pooled a route at dist_rel 0.19 alongside one at 0.00 as if the
data could not tell them apart. It usually can: a (5,2) segment typically carries 12 routes, of which
only the handful flagged best_route are actually tied at the minimum distance.

Carried through here, per (segment, route):
    dist_rel      distance_from_route_boundary — 0 .. dist_cut(0.20); constant within (segment, route)
    best_route    True for EVERY route tied at that segment's minimum distance (not a single pick —
                  pick_best_key's random tie-break chooses among them, so the flag is the tied SET)
    route_prob    route_probability
    num_draws     draws for that route (1 = determined route)

Writes a SEPARATE file (…_v2.tsv). v1 stays put so nothing downstream shifts under its feet until we
decide the pooling rule deliberately.

Output: dev/pcawg/misc_data_and_plots/genome_highcn_timing_cache_v2.tsv
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

TAU_OUT = Path(os.path.join(os.environ.get("TAU_REPO", "."),
               "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"))
OUT = Path(os.path.join(os.environ.get("TAU_REPO", "."),
                        "dev/pcawg/misc_data_and_plots/genome_highcn_timing_cache_v2.tsv"))

MIN_EVENTS = 3
QUAL = {"distance_from_route_boundary": "dist_rel", "best_route": "best_route",
        "route_probability": "route_prob", "num_draws": "num_draws"}

seg_cols = ["chrom", "start", "end", "key"]
chunks, n_samples, n_skipped = [], 0, 0

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

        ne = (d.groupby(seg_cols)["time_point"].max().reset_index()
                .rename(columns={"time_point": "n_events"}))
        ne = ne[ne["n_events"] >= MIN_EVENTS]
        if ne.empty:
            n_samples += 1
            continue
        d = d.merge(ne, on=seg_cols)

        cn = d["key"].str.split(".", n=1).str[0].str.split("_")
        d["major_cn"] = cn.str[0].astype(int)
        d["minor_cn"] = cn.str[1].astype(int)

        mc = d.groupby(seg_cols + ["draw_id"])["mutation_count"].first().reset_index()
        # route quality is per (segment, route) — verified constant within it, so first() is exact
        have = [c for c in QUAL if c in d.columns]
        qual = d.groupby(seg_cols)[have].first().reset_index().rename(columns=QUAL)

        piv = (d.pivot_table(index=seg_cols + ["draw_id", "major_cn", "minor_cn", "n_events"],
                             columns="time_point", values="time_fraction", aggfunc="first")
                .reset_index())
        piv.columns.name = None
        int_cols = [c for c in piv.columns if isinstance(c, (int, np.integer))]
        piv = piv.rename(columns={c: f"G{c}" for c in int_cols})

        piv = piv.merge(mc, on=seg_cols + ["draw_id"], how="left")
        piv = piv.merge(qual, on=seg_cols, how="left")
        piv.insert(0, "tumor_type", tumor_type)
        piv.insert(0, "sample", sid)

        chunks.append(piv)
        n_samples += 1
        if n_samples % 200 == 0:
            print(f"  {n_samples} samples, {sum(len(c) for c in chunks):,} rows", flush=True)

print(f"\nProcessed {n_samples} samples ({n_skipped} skipped as empty)")
if not chunks:
    raise SystemExit("No data — exiting")

cache = pd.concat(chunks, ignore_index=True)
gcols = [f"G{i}" for i in range(1, 10) if f"G{i}" in cache.columns]
base = ["sample", "tumor_type", "chrom", "start", "end", "key",
        "major_cn", "minor_cn", "n_events", "draw_id"]
tail = ["mutation_count", "dist_rel", "best_route", "route_prob", "num_draws"]
cache = cache[[c for c in base + gcols + tail if c in cache.columns]]
cache.to_csv(OUT, sep="\t", index=False)

print(f"Total rows: {len(cache):,} | samples {cache['sample'].nunique()}")
if "best_route" in cache:
    seg = cache.groupby(["sample", "chrom", "start", "end"])
    print(f"  rows on a best_route: {100*cache.best_route.mean():.1f}%")
    print(f"  routes/segment: median {seg.key.nunique().median():.0f} | "
          f"best-only median {cache[cache.best_route].groupby(['sample','chrom','start','end']).key.nunique().median():.0f}")
    print(f"  dist_rel pct 50/90/99: {np.round(np.percentile(cache.dist_rel.dropna(), [50, 90, 99]), 4)}")
print(f"Saved → {OUT}")
