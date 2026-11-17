"""EXCESS-burst rate, real vs matched null -> paper/figures/data/burst_characteristics/burst_null_excess_{by_state,by_ngain}.tsv

Rebuilt on the definition the burst figures actually use. The previous version scored both sides with
`complete_burst` (p90_span < theta, "gains tight in ~all solutions") and swept theta. The figures now
use EXCESS — max_burst_size > D, D = floor(M/2)+floor(m/2) — and a null computed on a different
statistic from the panel it supports is not a null. The theta sweep is gone with it: max_burst_size is
defined at a fixed 0.05 window, so there is no theta to sweep without re-solving every replicate at
every theta.

The null re-simulates each real segment through its own route and mutation count with uniform Dirichlet
interval fractions, then runs Tau — so it asks what Tau OUTPUTS when gains are not clustered in time,
on exactly these segments. One replicate = one null segment (simulate_null_perrep.py).

ONE RULE, BOTH SIDES
--------------------
Both sides now take max_burst_size over TIED-BEST routes only (the routes at a segment's minimum
dist_rel). The null previously pooled every route at dist_cut = 0.5 — looser than Tau's own 0.20 and
far looser than the real side — and pooling extra routes can only RAISE a max, so the old null was
biased burst-ward and would have understated the real excess rate.

WHY BY STATE, NOT POOLED
------------------------
The null excess rate depends enormously on the CN state, because D is a weak bar when a state has many
gains relative to its doubling budget: 0.021 for (6,2) and 0.034 for (3,2), but 0.61 for (3,3) and 0.78
for (5,5). A single pooled number would average a near-impossible bar with an almost-free one and mean
nothing. The by-n_gain table makes the same point on the axis that drives it.

Usage:  python paper/figures/data/burst_characteristics/build/build_burst_null.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
CAT = ANALYSIS / "pcawg/oncogenes/figures_genome_highcn_catalogue"
MISC = ANALYSIS / "pcawg/misc_data_and_plots"
OUT = cache_dir_of(__file__)
OUT.mkdir(parents=True, exist_ok=True)


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    files = sorted(glob.glob(str(CAT / "null_perrep" / "null_perrep_*.tsv")))
    if not files:
        sys.exit("no per-replicate null — run dev/pcawg/oncogenes/simulate_null_perrep.py first")
    n = pd.concat([pd.read_csv(f, sep="\t", dtype={"chrom": str}) for f in files],
                  ignore_index=True)
    if "null_max_burst_size" not in n.columns:
        sys.exit("null files predate the excess statistic — rerun simulate_null_perrep.py")
    n = n[n.null_max_burst_size.notna()].copy()
    # POSSIBLE = some solution shows a burst (the permissive statistic, a max over solutions).
    # NECESSARY = every solution shows one. The panel plots NECESSARY: max-over-solutions is
    # anti-conservative and its false-positive rate runs to 40-90% in balanced high-CN states, which
    # is why those states read n.s. The conservative reading has a near-zero null everywhere, so every
    # state becomes interpretable instead of being dropped.
    n["excess"] = n.null_min_burst_size > n.D          # NECESSARY
    n["excess_possible"] = n.null_max_burst_size > n.D

    r = pd.read_csv(MISC / "burst_possible_necessary.tsv", sep="\t", dtype={"chrom": str})
    r["excess_possible"] = r["possible"]
    r["excess"] = r["necessary"]
    states = set(zip(n.major_cn, n.minor_cn))
    r = r[[(a, b) in states for a, b in zip(r.major_cn, r.minor_cn)]].copy()
    print(f"real {len(r):,} segments | null {len(n):,} replicates | {len(states)} CN states")

    rows = []
    for (a, b) in sorted(states):
        rr = r[(r.major_cn == a) & (r.minor_cn == b)]
        nn = n[(n.major_cn == a) & (n.minor_cn == b)]
        if len(rr) == 0 or len(nn) == 0:
            continue
        rows.append(dict(major_cn=a, minor_cn=b, n_gain=int(rr.n_events.median()),
                         real_n=int(rr.excess.sum()), real_total=len(rr),
                         null_n=int(nn.excess.sum()), null_total=len(nn),
                         real_poss=int(rr.excess_possible.sum()),
                         null_poss=int(nn.excess_possible.sum())))
    st = pd.DataFrame(rows)
    st["real"] = st.real_n / st.real_total
    st["null"] = st.null_n / st.null_total
    st["fold"] = st["real"] / st["null"].replace(0, np.nan)
    st.to_csv(OUT / "burst_null_excess_by_state.tsv", sep="\t", index=False)

    rows = []
    for g in sorted(set(r.n_events) & set(n.n_events)):
        rr, nn = r[r.n_events == g], n[n.n_events == g]
        if len(rr) < 30 or len(nn) < 30:
            continue
        rows.append(dict(n_gain=int(g), real_n=int(rr.excess.sum()), real_total=len(rr),
                         null_n=int(nn.excess.sum()), null_total=len(nn)))
    ng = pd.DataFrame(rows)
    ng["real"] = ng.real_n / ng.real_total
    ng["null"] = ng.null_n / ng.null_total
    ng.to_csv(OUT / "burst_null_excess_by_ngain.tsv", sep="\t", index=False)

    # ---- POOLED, by determinacy: the three groups panel a plots ----
    # Not by CN state. D is a weak bar in some states and a strong one in others, so 19 bars mix two
    # separate stories. Determinacy is the axis that actually governs whether the statistic can be
    # trusted, and it splits cleanly in three.
    n["det"] = n.n_solutions <= 1
    rd, ru = r[r.determined], r[~r.determined]
    nd, nu = n[n.det], n[~n.det]
    pooled = pd.DataFrame([
        dict(group="determined", detail="one solution — no choice to make",
             real_n=int(rd.possible.sum()), real_total=len(rd),
             null_n=int(nd.excess_possible.sum()), null_total=len(nd)),
        dict(group="underdetermined\nevery solution", detail="conservative reading",
             real_n=int(ru.necessary.sum()), real_total=len(ru),
             null_n=int(nu.excess.sum()), null_total=len(nu)),
        dict(group="underdetermined\nsome solution", detail="permissive reading",
             real_n=int(ru.possible.sum()), real_total=len(ru),
             null_n=int(nu.excess_possible.sum()), null_total=len(nu)),
    ])
    pooled["real"] = pooled.real_n / pooled.real_total
    pooled["null"] = pooled.null_n / pooled.null_total
    pooled["fold"] = pooled["real"] / pooled["null"].replace(0, np.nan)
    pooled.to_csv(OUT / "burst_null_pooled.tsv", sep="\t", index=False)
    print("\nPOOLED by determinacy:")
    for t in pooled.itertuples():
        print(f"  {t.group.replace(chr(10),' '):<32} real {100*t.real:>5.1f}%  null {100*getattr(t,'null'):>5.2f}%  "
              f"fold {t.fold:>5.1f}")

    tot_r, tot_n = r.excess.mean(), n.excess.mean()
    print(f"\noverall excess: real {100*tot_r:.1f}%  null {100*tot_n:.1f}%  "
          f"fold {tot_r/tot_n:.1f}  (pooled — see docstring, report by state)")
    print(f"\n{'':>7}{'':>6}{'--- NECESSARY (plotted) ---':>30}{'--- POSSIBLE ---':>26}")
    print(f"{'state':>7}{'gains':>6}{'real':>10}{'null':>10}{'fold':>10}{'real':>13}{'null':>13}")
    for t in st.sort_values("fold", ascending=False).itertuples():
        f = f"{t.fold:.0f}" if np.isfinite(t.fold) else "inf"
        rp = t.real_poss / t.real_total
        np_ = t.null_poss / t.null_total
        print(f"{f'({t.major_cn},{t.minor_cn})':>7}{t.n_gain:>6}"
              f"{100*t.real:>9.1f}%{100*t.null:>9.2f}%{f:>10}{100*rp:>12.1f}%{100*np_:>12.1f}%")
    print(f"\n-> {OUT/'burst_null_excess_by_state.tsv'}\n-> {OUT/'burst_null_excess_by_ngain.tsv'}"
          f"\n-> {OUT/'burst_null_pooled.tsv'}")


if __name__ == "__main__":
    main()
