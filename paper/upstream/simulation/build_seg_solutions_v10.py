"""Every sampled solution for every v10 segment -> bench_seg_solutions.tsv.gz + bench_seg_truth.tsv

WHY STORE ALL SOLUTIONS
-----------------------
bench_seg_gains stores only the BAND [time_lo, time_hi] per gain, so the only error metric it can
support is dist_to_set (distance to the nearest point in the band). That metric rewards a method for
being uncertain — a wider band scores better — and it cannot express "what does a user get if they
pick a solution", which is the question a benchmark should answer.

Storing the solutions themselves makes the SELECTION RULE a post-hoc groupby instead of something
baked into the table. Rules that then become one-liners:

    random draw       an honest "pick one and see"
    medoid            the solution closest to the per-gain median
    WGD-anchored      minimise summed distance to the sample's inferred WGD
    sim_span          maximise simultaneity — the tightest window over the gains
    oracle            the solution closest to the truth (an upper bound, not a claim)

That last pair is why this exists: comparing Tau to GRITIC under a simultaneity-maximising rule needs
every solution, not a summary of them.

LAYOUT — two tables, joined on (sample, chrom, start, end)
    bench_seg_solutions   one row per (segment, route, draw); G1..G9 = that solution's gain times
                          (cumulative, sorted), NaN beyond the segment's gain count. Carries
                          is_best_route so route-selection questions can be asked too.
    bench_seg_truth       one row per segment; the simulated truth (true gain times, true route and
                          its equivalence class) plus mutation count. Kept separate so the truth is
                          not repeated across ~50 solution rows per segment.

Reads the v10 per-sample segment_times.tsv (13 GB in total), so it streams sample by sample.
"""
import os
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(os.environ.get("TAU_REPO", "."))
V10 = R / "dev/benchmarking/output_v10"
OUT = R / "paper/data_tables"
GC = [f"G{i}" for i in range(1, 10)]
SEG = ["sample", "chrom", "start", "end"]

sol_parts, truth_rows = [], []
files = sorted(glob.glob(str(V10 / "SIM_*/SIM_*.truth.json")))
print(f"{len(files):,} samples", flush=True)

for i, tf in enumerate(files):
    if i and i % 250 == 0:
        n = sum(len(p) for p in sol_parts)
        print(f"  {i:,}/{len(files):,}  ({n:,} solution rows)", flush=True)
    s = Path(tf).parent.name
    j = json.load(open(tf))

    # ---- truth, one row per segment ----
    for tr in j["segments"]:
        tv = tr.get("t_values")
        if not tv or tr["major"] + tr["minor"] <= 2:
            continue
        c, rr = tr["seg_id"].replace("chr", "").split(":")
        a, b = (int(v) for v in rr.split("-"))
        g = np.cumsum(np.asarray(tv, float))[:-1]           # true gain times
        row = dict(sample=s, chrom=c, start=a, end=b, major=tr["major"], minor=tr["minor"],
                   n_gain=len(g), true_route=tr.get("true_route") or "",
                   route_class=",".join(tr.get("route_class") or []),
                   equiv_class=",".join(tr.get("equiv_class") or []))
        for k in range(9):
            row[f"trueG{k+1}"] = float(g[k]) if k < len(g) else np.nan
        truth_rows.append(row)

    # ---- every solution, one row per (segment, route, draw) ----
    tp = V10 / s / f"{s}.segment_times.tsv"
    if not tp.exists() or tp.stat().st_size < 5:
        continue
    try:
        T = pd.read_csv(tp, sep="\t", dtype={"chrom": str},
                        usecols=["chrom", "start", "end", "key", "draw_id", "time_point",
                                 "time_fraction", "best_route", "major_cn", "minor_cn"])
    except pd.errors.EmptyDataError:
        continue
    if T.empty:
        continue
    P = T.pivot_table(index=["chrom", "start", "end", "key", "draw_id", "best_route",
                             "major_cn", "minor_cn"],
                      columns="time_point", values="time_fraction", aggfunc="first").reset_index()
    P.columns.name = None
    P = P.rename(columns={c: f"G{int(c)}" for c in P.columns if isinstance(c, (int, np.integer))})
    for c in GC:
        if c not in P.columns:
            P[c] = np.nan
    P.insert(0, "sample", s)
    sol_parts.append(P[SEG + ["major_cn", "minor_cn", "key", "best_route", "draw_id"] + GC])

sol = pd.concat(sol_parts, ignore_index=True)
sol = sol.rename(columns={"key": "route_key", "best_route": "is_best_route"})
truth = pd.DataFrame(truth_rows)

sol.to_csv(OUT / "bench_seg_solutions.tsv.gz", sep="\t", index=False, compression="gzip")
truth.to_csv(OUT / "bench_seg_truth.tsv", sep="\t", index=False)

print(f"\nbench_seg_solutions: {len(sol):,} solution rows, "
      f"{sol.groupby(SEG).ngroups:,} segments, {int(sol.is_best_route.astype(bool).sum()):,} on a best route")
print(f"bench_seg_truth    : {len(truth):,} segments")
print(f"  solutions per segment: median {sol.groupby(SEG).size().median():.0f} "
      f"(best-route only: {sol[sol.is_best_route.astype(bool)].groupby(SEG).size().median():.0f})")
print(f"-> {OUT/'bench_seg_solutions.tsv.gz'}\n-> {OUT/'bench_seg_truth.tsv'}")
