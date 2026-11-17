"""Build the route-topology reference table (from tau/data/route_topology.json.gz — ships with Tau; lets
trees reconstruct from tau_gains WITHOUT Sage) and the bench_* tables. bench_events + bench_truth_events are
DERIVED FROM THE SIMULATION OUTPUTS via tau.aggregate.benchmark (truth.json ⨝ recovered clustering); the
tree/oracle/mutcount tables are cheap reshapes of the existing v8 result tables. No pkl/sim rerun.
Run: PATH=.pixi/envs/default/bin:$PATH python paper/batch_scripts/build_ref_and_bench.py
"""
import gzip, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R))
from tau.aggregate.benchmark import build_bench_tables
DT = R / "paper/data_tables"
BENCH = R / "dev/benchmarking"

# ── ref_route_topology: one row per (route_key, event) ─────────────────────────────────
d = json.load(gzip.open(R / "tau/data/route_topology.json.gz", "rt"))
rows = []
for rk, v in d.items():
    for ev in v.get("events", []):
        sp = ev.get("split", [None, None])
        rows.append(dict(route_key=rk, major_cn=v["major_cn"], minor_cn=v["minor_cn"],
                         event_index=ev.get("col_index"), allele=ev.get("allele"),
                         parent_mult=ev.get("parent_mult"), split_a=sp[0], split_b=sp[1]))
topo = pd.DataFrame(rows).sort_values(["route_key", "event_index"])
# gz-TSV (120k routes) — shippable lookup; trees reconstruct from this + tau_gains, no Sage
topo.to_csv(DT / "ref/ref_route_topology.tsv.gz", sep="\t", index=False, compression="gzip")
print(f"ref_route_topology: {len(topo)} event-rows, {topo.route_key.nunique()} routes")

# ── bench_events + bench_truth_events: DERIVED FROM SIM (truth.json ⨝ recovered clustering) ──
paths = build_bench_tables(sim_run_dir=BENCH / "output_v8",
                           recovered_tsv=BENCH / "bottomup_final_v8.tsv", out_dir=DT)
be = pd.read_csv(paths["bench_events"], sep="\t")
te = pd.read_csv(paths["bench_truth_events"], sep="\t")
print(f"bench_events {len(be)} calls | bench_truth_events {len(te)} true events "
      f"({int(te.is_detected.sum())} detected, recall {te.is_detected.mean():.1%})")

# ── tree / oracle / mutcount: cheap reshapes of existing v8 result tables ────────────────
pd.read_csv(BENCH / "tree_metrics_v8.tsv", sep="\t").to_csv(DT / "bench_tree_metrics.tsv", sep="\t", index=False)
pd.read_csv(BENCH / "oracle_lift_combined.tsv", sep="\t").to_csv(DT / "bench_oracle_lift.tsv", sep="\t", index=False)
pd.read_csv(BENCH / "pcawg_mutcount_clustering.tsv", sep="\t").to_csv(DT / "bench_pcawg_mutcount.tsv", sep="\t", index=False)
print("tree_metrics + oracle_lift + pcawg_mutcount written")

# ── per-GAIN segment-level calibration (SEGMENT benchmarking supp): each simulated gain's estimated
#    timing band [lo,hi] vs its true time, with determined flag + mutation count. dist_to_set = the
#    faithful error (0 if truth inside the band; distance to the nearest band edge otherwise).
_sg = pd.read_csv(BENCH / "tree_gains_v8.tsv", sep="\t").rename(columns={
    "mc": "mut_count", "gain": "gain_index", "true_t": "true_time",
    "lo": "time_lo", "hi": "time_hi", "undet": "underdetermined", "covered": "truth_in_band"})
_sg["underdetermined"] = _sg["underdetermined"].astype(bool)
_sg["truth_in_band"] = _sg["truth_in_band"].astype(bool)
_sg["dist_to_set"] = np.maximum.reduce([np.zeros(len(_sg)),
                                        _sg["time_lo"] - _sg["true_time"],
                                        _sg["true_time"] - _sg["time_hi"]])
_sg.to_csv(DT / "bench_seg_gains.tsv", sep="\t", index=False)
print(f"bench_seg_gains written ({len(_sg):,} gains, {_sg.underdetermined.mean():.0%} underdetermined, "
      f"median dist_to_set {_sg.dist_to_set.median():.3f})")

# ── per-tree accuracy under Tau's OWN route pick (best-guess) vs the oracle true route: max-gain
#    distance-to-valid-set both ways + whether the route was picked correctly. Produced by
#    bestguess_vs_oracle_treeacc.py (reads the sim segment_times.tsv); reshaped here.
pd.read_csv(BENCH / "bestguess_vs_oracle_v8.tsv", sep="\t").to_csv(DT / "bench_route_accuracy.tsv", sep="\t", index=False)
print("bench_route_accuracy written")

# ── what route discrimination BUYS, and whether the feasible routes even disagree. Both produced in one
#    pass by route_selection_agreement.py (reads the sim segment_times.tsv, which stores every route that
#    survived the distance cut, not just the chosen one).
for _src, _dst, _what in [("route_selection_v8.tsv", "bench_route_selection.tsv", "trees"),
                          ("route_agreement_v8.tsv", "bench_route_agreement.tsv", "route pairs"),
                          ("route_equivalence_v8.tsv", "bench_route_equivalence.tsv", "segments"),
                          ("route_agreement_per_gain_v8.tsv",
                           "bench_route_agreement_per_gain.tsv", "(segment, gain) rows"),
                          ("route_roc_v8.tsv", "bench_route_roc.tsv", "ROC grid points"),
                          ("route_pair_error_v8.tsv", "bench_route_pair_error.tsv", "route pairs")]:
    _p = BENCH / _src
    if _p.exists():
        _d = pd.read_csv(_p, sep="\t")
        _d.to_csv(DT / _dst, sep="\t", index=False)
        print(f"{_dst[:-4]} written ({len(_d):,} {_what})")
    else:
        print(f"{_dst[:-4]} SKIPPED — run dev/benchmarking/route_selection_agreement.py first")
