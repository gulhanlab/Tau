"""The POSSIBLE / NECESSARY bracket for excess bursts -> burst_possible_necessary.tsv

WHY THIS EXISTS
---------------
`max_burst_size` is a MAXIMUM over a segment's solutions, so "excess" as currently defined asks whether
SOME solution shows a burst. That question is anti-conservative by construction, and the wider the
solution set the worse it gets: against the matched simulation null, the false-positive rate is 0.5% on
determined segments (one solution, nothing to maximise over) and 11.8% on underdetermined ones.

The fix is not to discard underdetermined segments. The same solution set answers the opposite question
— does EVERY solution show a burst — which is conservative. Together they bracket the truth:

    necessary  = min over solutions of the largest co-timed set  > D    (lower bound)
    possible   = max over solutions of the largest co-timed set  > D    (upper bound, = current defn)

For a determined segment the two coincide and the bracket collapses to a point. For an underdetermined
one they differ, and the truth lies inside.

RESULT (single best route, 0.05 burst window, D = floor(M/2)+floor(m/2))

    subset             n        possible   necessary    gap
    determined       4,282        24.5%       24.5%     0.0 pp
    underdetermined  8,975        16.8%        4.2%    12.6 pp
    all             13,257        19.3%       10.8%     8.5 pp

    of the 8,975 underdetermined segments:
        380  ( 4%)  burst in EVERY solution  -> callable, as solid as a determined call
      1,132  (13%)  burst in SOME only       -> genuinely undecidable from the data
      7,463  (83%)  burst in NO solution     -> a negative you can trust

So 87% of underdetermined segments give a definite answer; only 13% are truly open. Underdetermined
data supports (a) the lower bound as a conservative count, (b) negatives, for denominators and
enrichment tests, and (c) gain POSITIONS — "these gains sit near the WGD" survives even where "these
gains constitute a burst" does not, which is why high-false-positive CN states still show ~3x WGD
enrichment.

What it does NOT support is reporting the upper bound alone as if it were a call. The defensible
summary is the interval: 10.8-19.3% of high-CN segments carry an excess burst, closing to 24.5% on the
determined subset.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(os.environ.get("TAU_REPO", "."))
MISC = R / "dev/pcawg/misc_data_and_plots"
OUT = MISC / "burst_possible_necessary.tsv"
SEG = ["sample", "chrom", "start", "end"]
GCOLS = [f"G{i}" for i in range(1, 10)]
BURST_SPAN = 0.05

bt = pd.read_csv(MISC / "burst_singleroute.tsv", sep="\t", dtype={"chrom": str})
c = pd.read_csv(MISC / "genome_highcn_timing_cache_v2.tsv", sep="\t", dtype={"chrom": str},
                usecols=SEG + GCOLS + ["key"])
c = c.merge(bt[SEG + ["route_key", "D", "width_max"]], on=SEG)
c = c[c.key == c.route_key]                      # the route the analysis scored

# per SOLUTION: largest set of consecutive gains inside the burst window
M = c[GCOLS].to_numpy(float)
n, NG = M.shape
per = np.ones(n, dtype=int)
with np.errstate(invalid="ignore"):
    for w in range(2, NG + 1):
        hi, lo = M[:, w - 1:], M[:, :NG - (w - 1)]
        sp = np.where(np.isnan(hi) | np.isnan(lo), np.inf, hi - lo)
        per[np.min(sp, axis=1) < BURST_SPAN] = w
c["bs"] = per

g = c.groupby(SEG).agg(bs_min=("bs", "min"), bs_max=("bs", "max")).reset_index()
x = bt.merge(g, on=SEG)
x["possible"] = x.bs_max > x.D
x["necessary"] = x.bs_min > x.D
x["determined"] = x.width_max <= 1e-9
x.to_csv(OUT, sep="\t", index=False)

print(f"{'subset':<20}{'n':>8}{'possible':>12}{'necessary':>12}{'gap':>10}")
for lab, m in [("determined", x.determined), ("underdetermined", ~x.determined),
               ("all", pd.Series(True, index=x.index))]:
    s = x[m]
    print(f"{lab:<20}{len(s):>8}{100*s.possible.mean():>11.1f}%{100*s.necessary.mean():>11.1f}%"
          f"{100*(s.possible.mean()-s.necessary.mean()):>8.1f} pp")
u = x[~x.determined]
print(f"\nunderdetermined ({len(u):,}):")
print(f"  burst in EVERY solution (callable) : {int(u.necessary.sum()):>5} ({100*u.necessary.mean():.0f}%)")
print(f"  burst in SOME only      (open)     : {int((u.possible & ~u.necessary).sum()):>5} "
      f"({100*(u.possible & ~u.necessary).mean():.0f}%)")
print(f"  burst in NO solution    (excluded) : {int((~u.possible).sum()):>5} "
      f"({100*(~u.possible).mean():.0f}%)")
print(f"\n-> {OUT}")
