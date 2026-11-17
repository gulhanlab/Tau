#!/usr/bin/env python3
"""
simulate_null_perrep.py  --major M --minor N

PER-REPLICATE null: each simulated dataset is classified on its own, so one null replicate is
comparable to one real segment.

Why this exists
---------------
simulate_null_cn_state.py pools the solutions of all N_BOOT replicates into a single solution space
before classifying. That makes the pooled span statistics meaningless as a null:

  * `max_span` for a real segment is the widest its gains sit across ONE dataset's polytope.
  * `max_span` for the pooled null is the widest across 30 DIFFERENT simulated truths unioned.

Thirty uniform-Dirichlet draws always contain a widely separated configuration, so the pooled null span
is ~1 by construction (measured: median 1.000, 5th pct 0.902, min 0.655 over 12,920 segments). The
"zero null bursts at every threshold" result follows arithmetically and is not evidence. The same
pooling makes complete_burst and incomplete_burst — which require ALL solutions to be tight —
impossible for the null to ever produce, and biases burst_compatible the other way (pooling more
solutions can only lower the minimum, so the pooled null calls 58% burst_compatible vs 53% real).

Here each replicate is simulated, solved and classified independently, and one row is written per
(segment, replicate). "What fraction of null segments are complete_burst" then has a real answer.

Matching to the real data
-------------------------
  route + CN state + mutation count   from the real segment (as before)
  all feasible routes per replicate   the real solution cache also pools across routes
                                      (median 2 keys per segment), so this is symmetric
  N_DRAWS = 50                        a real segment's polytope carries ~25-50 draws per route;
                                      the pooled null used 1, which is far too few to classify a
                                      single replicate

Usage:
    python simulate_null_perrep.py --major 4 --minor 1 [--boot 10] [--draws 50]
"""
import os
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("TAU_REPO", "."))
OUT_DIR = BASE / "dev/pcawg/oncogenes/figures_genome_highcn_catalogue"
NULL_DIR = OUT_DIR / "null_perrep"; NULL_DIR.mkdir(exist_ok=True, parents=True)
MISC = BASE / "dev/pcawg/misc_data_and_plots"

sys.path.insert(0, str(BASE))
from tau.core import Segment
from tau.timing import _segment_time_segment, _get_route_env
from tau.simulation import _load_matrices
from tau.utils import order_t

BURST_THRESH = 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--major", type=int, required=True)
ap.add_argument("--minor", type=int, required=True)
ap.add_argument("--boot", type=int, default=10)      # null replicates per real segment
ap.add_argument("--draws", type=int, default=50)     # polytope solutions per replicate per route
args = ap.parse_args()
MAJOR, MINOR, N_BOOT, N_DRAWS = args.major, args.minor, args.boot, args.draws
RNG = np.random.default_rng(4242 + 100 * MAJOR + MINOR)


def burst_size_bounds(draws, ne):
    """(necessary, possible) = (min, max) over solutions of the largest co-timed consecutive set.

    `possible` is the published statistic (does SOME solution show a burst) and is anti-conservative:
    a maximum over solutions inflates with solution-set width. `necessary` is its conservative twin
    (does EVERY solution show one). Together they bracket the truth, and for a determined replicate
    they coincide. Recording both lets the null be scored on either definition — without the min the
    conservative claim has no false-positive rate at all.
    """
    per = np.ones(len(draws), dtype=int)          # per-SOLUTION largest co-timed set
    for w in range(2, ne + 1):
        hi = draws[:, w - 1:ne]
        lo = draws[:, :ne - (w - 1)]
        per[np.min(hi - lo, axis=1) < BURST_THRESH] = w
    return int(per.min()), int(per.max())


def compute_stats(draws, ne):
    """Just the burst statistic. Everything else was tier scaffolding.

    The four-tier scheme needed percentiles ACROSS a replicate's solutions (p90_span, prefix spans and
    gaps, p5_min_window_span). Those are gone, and with them the reason a replicate needed >=3 solutions
    to be usable — a guard that silently discarded every DETERMINED replicate (one solution), leaving
    the null 0% determined while the real data is 32%. max_burst_size is a count of co-timed gains and
    is perfectly well defined on a single solution.

    max_burst_size >= 3 reproduces the old tier filter exactly (4,858 segments, 100% agreement), so
    nothing is lost by stating the definition in one line instead of a cascade.
    """
    return {"min_window_span": float(np.nanmin([
        np.min(draws[:, j + k - 1] - draws[:, j])
        for k in range(3, ne + 1) for j in range(ne - k + 1)])) if ne >= 3 else np.nan}


def one_replicate(mat, ne, mutation_count, env):
    """Simulate ONE dataset and return its own solution array (n_solutions, ne), or None."""
    t_vals = RNG.dirichlet(np.ones(mat.shape[1]))
    expected = np.maximum(mat @ t_vals * float(mutation_count), 0.0)
    N_counts = RNG.poisson(expected).astype(float)
    if N_counts.sum() < 5:
        return None
    seg = Segment(chrom="1", start=0, end=1, major_cn=MAJOR, minor_cn=MINOR, purity=1.0,
                  snv_table=pd.DataFrame(), N_counts=N_counts,
                  pi_em=N_counts / max(N_counts.sum(), 1e-9))
    try:
        _segment_time_segment(seg, env, n_draws=N_DRAWS, dist_cut=0.5)
    except Exception:
        return None
    # TIED-BEST ROUTES ONLY. Previously every route in timing_result was pooled, at dist_cut=0.5 —
    # more permissive than Tau's own 0.20, and far more permissive than the real side now is. Pooling
    # more routes can only RAISE max_burst_size, so an unrestricted null would look burstier than it
    # should and understate the real excess rate. Keep the routes at the minimum dist_rel, which is
    # what `best_route` marks on the real side.
    tr = getattr(seg, "timing_result", {}) or {}
    dist = {}
    for k, info in tr.items():
        qc = info.get("qc", [])
        if qc:
            dist[k] = float(qc[0].get("dist_rel", np.inf))
    if not dist:
        return None
    dmin = min(dist.values())
    keep = {k for k, v in dist.items() if v <= dmin + 1e-9}
    g_vecs = []
    for k in keep:
        for draw in tr[k].get("draws", []):
            g = np.cumsum(order_t(draw["t"]))[:-1]
            if len(g) == ne:
                g_vecs.append(g)
    # >=1, not >=3: a determined route yields ONE solution and must still count
    return np.array(g_vecs) if len(g_vecs) >= 1 else None


def main():
    print(f"CN state ({MAJOR},{MINOR})  boot={N_BOOT} draws={N_DRAWS}", flush=True)
    clf = pd.read_csv(OUT_DIR / "burst_classified_summary.tsv", sep="\t", dtype={"chrom": str})
    clf = clf[(clf.major_cn == MAJOR) & (clf.minor_cn == MINOR)].copy()
    oc = pd.read_csv(MISC / "genome_highcn_timing_cache.tsv", sep="\t", dtype={"chrom": str})
    oc = oc[(oc.major_cn == MAJOR) & (oc.minor_cn == MINOR)]
    SEG = ["sample", "chrom", "start", "end"]
    # the route the REAL analysis used for this segment, not an arbitrary "first" — the null has to
    # be generated through the same matrix the observation was scored on, or it is not matched
    bt = pd.read_csv(MISC / "burst_singleroute.tsv", sep="\t", dtype={"chrom": str},
                     usecols=SEG + ["route_key"]).rename(columns={"route_key": "key"})
    clf = clf.merge(bt, on=SEG, how="left")
    print(f"  {len(clf):,} segments, {clf.key.notna().sum():,} with a route key", flush=True)

    env = _get_route_env()
    MATRICES = _load_matrices()
    rows = []
    for i, row in enumerate(clf.itertuples()):
        key, mc, ne = row.key, row.mutation_count, int(row.n_events)
        if pd.isna(key) or pd.isna(mc) or mc < 5 or key not in MATRICES:
            continue
        mat = MATRICES[key].T
        for rep in range(N_BOOT):
            sol = one_replicate(mat, ne, mc, env)
            if sol is None:
                continue
            st = compute_stats(sol, ne)
            rows.append(dict(sample=row.sample, chrom=row.chrom, start=row.start, end=row.end,
                             major_cn=MAJOR, minor_cn=MINOR, n_events=ne, key=key,
                             mutation_count=mc, real_category=row.category, rep=rep,
                             null_min_window_span=st["min_window_span"],
                             n_solutions=len(sol),
                             null_min_burst_size=burst_size_bounds(sol, ne)[0],
                             null_max_burst_size=burst_size_bounds(sol, ne)[1],
                             D=MAJOR // 2 + MINOR // 2))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(clf)} segments …", flush=True)

    out = pd.DataFrame(rows)
    p = NULL_DIR / f"null_perrep_{MAJOR}_{MINOR}.tsv"
    out.to_csv(p, sep="\t", index=False)
    print(f"\n{len(out):,} null replicates from {out['sample'].nunique() if len(out) else 0} samples")
    if len(out):
        det = out.n_solutions <= 1
        print(f"  excess {100*(out.null_max_burst_size > out.D).mean():.1f}%  |  "
              f"determined replicates {int(det.sum()):,} ({100*det.mean():.0f}%)  "
              f"excess among them {100*(out[det].null_max_burst_size > out[det].D).mean():.1f}%",
              flush=True)
    print(f"Saved -> {p}\nDone.", flush=True)


if __name__ == "__main__":
    main()
