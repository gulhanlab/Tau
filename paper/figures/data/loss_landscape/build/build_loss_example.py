"""Cache builder for Supp (loss_landscape) c — the worked loss example, from pooled-segment clusters.

Panel a is a schematic: it says how implied loss is counted but reads no data. put the real
example next to it — a whole genome's solution with a cross on every copy the doublings should have
produced and the data does not show.

Sample: dc537fcf (Stomach.AdenoCA, double WGD), the canonical double-WGD example. Doubling times are
read from the run's own time_clusters.tsv, not hardcoded.

Trees come from INDIVIDUAL SEGMENTS of the raw genome — one real locus per panel, at named
coordinates. An earlier version used the pooled units
of the clustered genome (`merged_{M}_{m}_{i}`).

The cost of the switch is power, and it is not small: the best single segment carries ESS 158 against
1,613 for the top pooled unit, ~10x less. Pooled trees put the gains ON the doubling lines; single
segments scatter around them. The gain is that each panel is a locus a reader can look up rather than
a genome-wide composite. Both are defensible; this is the concrete-example reading.

SELECTION IS BY AGREEMENT WITH THE POOLED CLUSTER, NOT BY POWER. For each CN state the largest pooled
unit is timed first and used as the reference history; every individual segment of that state is then
scored by `dist_to_cluster` = mean |segment gain time - pooled gain time|, and the panel draws the
closest one. This picks the segment that best REPRESENTS its cluster.

Why not highest-ESS: loss is `copies present - gains assigned to that doubling`, and a gain only counts
within +/-0.20 of a doubling, so a segment whose gains scatter books spurious missing copies. Choosing
on power alone gave (4,4) 2 missing where the pooled cluster reads 0 -- an artefact of single-segment
noise, not a finding. Choosing on agreement removes it.

This selects for typicality and is blind to the loss outcome: `dist_to_cluster` never looks at the
crosses. It does select against noise, which is the intent -- the panel illustrates the measurement,
and the measurement is the pooled one (panel b).

Every timed segment is cached with its ESS and `dist_to_cluster`, so changing the count or the
selection rule is a re-render rather than another 1.2 GB route_metadata load.

Reads the per-sample genome pkl and the 1.2 GB route_metadata (both outside data_tables/), which is why
this lives in paper/figures/data/loss_landscape/build/. It writes a small JSON the panel script can render on its own:

    {sample, wgd: [W1, W2],
     segments: [{seg_id, chrom, start, end, len_mb, major, minor, ess, dist_to_cluster,
                 n_sol, events: [...]}]}

`events` are route-template dicts (allele / parent_mult / split / time) — exactly what
tau.trees.build_allele_trees consumes, so the panel rebuilds the same trees without touching dev/.
"""
import copy
import gzip
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
warnings.filterwarnings("ignore")
from tau.utils import pick_best_key
from tau.trees import build_allele_trees, _pick_draw_for_tree

SAMPLE = "dc537fcf-d910-4c4b-8af9-e7da429f2633"
# CANONICAL allpcawg run. There is a second, older copy of this sample under
# dev/pcawg/misc_data_and_plots/output/ with a different (unnormalised) signature weighting, no
# stored exposures and different event times (W = 0.102 / 0.809 vs 0.129 / 0.824). Panel b comes
# from multi_wgd_loss_allpcawg.tsv, which is built on the canonical run — so this must be too.
SDIR = TAU_OUTPUTS / "Stomach.AdenoCA" / SAMPLE
PKL = SDIR / f"{SAMPLE}.genome.pkl.gz"               # RAW per-segment genome (individual loci)
CPKL = SDIR / f"{SAMPLE}.clustered_genome.pkl.gz"    # POOLED units, used only as the reference
TCLS = SDIR / f"{SAMPLE}.time_clusters.tsv"
ROUTE_META = ANALYSIS / "pcawg/route_metadata.pkl"   # route metadata of the PCAWG run (set TAU_ANALYSIS_DIR)
OUT = cache_dir_of(__file__) / "loss_example.json"
CHROMS = [str(i) for i in range(1, 23)] + ["X"]



def _wgd_times():
    """The two doubling times, read from the run's own clustering rather than hardcoded."""
    import pandas as pd
    d = pd.read_csv(TCLS, sep="\t")
    t = sorted(float(x) for x in d.loc[d.classification == "WGD", "time"])
    if len(t) != 2:
        raise SystemExit(f"expected 2 WGD events in {TCLS}, found {len(t)}: {t}")
    return t


def main():
    W1, W2 = _wgd_times()
    print(f"doublings from {TCLS.name}: W1={W1:.4f} W2={W2:.4f}", flush=True)
    print("loading route_metadata (1.2 GB) …", flush=True)
    with open(ROUTE_META, "rb") as f:
        route_metadata = pickle.load(f)
    with gzip.open(PKL, "rb") as fh:
        g = pickle.load(fh)
    with gzip.open(CPKL, "rb") as fh:
        cg = pickle.load(fh)

    # reference history per CN state: the largest pooled unit, timed the same way
    ref = {}
    for seg in cg.segments:
        if not str(getattr(seg, "seg_id", "")).startswith("merged_"):
            continue
        if not getattr(seg, "timing_result", None) or seg.N_counts is None:
            continue
        M, m = int(seg.major_cn), int(seg.minor_cn)
        key = pick_best_key(seg)
        rd = route_metadata.get((M, m), {})
        if key is None or key not in rd:
            continue
        variations = rd[key]
        tpl = (variations[0] if isinstance(variations, list) else variations).get("events", [])
        if not tpl:
            continue
        et, _ = _pick_draw_for_tree(seg, [W1, W2], events_tpl=tpl)
        if et is None or len(et) != len(tpl):
            continue
        ess = float(np.sum(seg.N_counts))
        if (M, m) not in ref or ess > ref[(M, m)][1]:
            ref[(M, m)] = (np.asarray(et, float), ess)
    print(f"pooled reference histories for {len(ref)} CN states", flush=True)

    pool = []
    for seg in g.segments:
        sid = str(getattr(seg, "seg_id", ""))
        if not getattr(seg, "timing_result", None) or seg.N_counts is None:
            continue
        M, m = int(seg.major_cn), int(seg.minor_cn)
        key = pick_best_key(seg)
        rd = route_metadata.get((M, m), {})
        if key is None or key not in rd:
            continue
        variations = rd[key]
        tpl = (variations[0] if isinstance(variations, list) else variations).get("events", [])
        if not tpl:
            continue
        et, _ = _pick_draw_for_tree(seg, [W1, W2], events_tpl=tpl)
        if et is None or len(et) != len(tpl):
            continue
        events = copy.deepcopy(tpl)
        for j, e in enumerate(events):
            e["time"] = float(et[j])
        try:                                    # only keep what actually renders
            build_allele_trees(events, M, m)
        except (ValueError, IndexError):
            continue
        r = ref.get((M, m))
        dist = (float(np.abs(np.asarray(et, float) - r[0]).mean())
                if r is not None and len(r[0]) == len(et) else float("inf"))
        pool.append(dict(seg_id=sid, major=M, minor=m, ess=float(np.sum(seg.N_counts)),
                         dist_to_cluster=round(dist, 5),
                         chrom=str(seg.chrom), start=int(seg.start), end=int(seg.end),
                         len_mb=round((int(seg.end) - int(seg.start)) / 1e6, 1),
                         n_segments=1,
                         n_sol=len([d for d in seg.timing_result[key]["draws"]
                                    if d.get("boot_id", 0) == 0]),
                         events=[{k: e[k] for k in ("allele", "parent_mult", "split", "time")}
                                 for e in events]))

    # Cache the whole pool ranked by power; the panel takes the largest N from it.
    pool.sort(key=lambda d: -d["ess"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(sample=SAMPLE, wgd=[W1, W2], segments=pool), indent=1))
    states = sorted({(s["major"], s["minor"]) for s in pool})
    print(f"loss_example.json: {len(pool)} individual segments, "
          f"ESS {pool[-1]['ess']:.0f}-{pool[0]['ess']:.0f}, {len(states)} CN states -> {OUT}")
    best = {}
    for s_ in sorted(pool, key=lambda d: d["dist_to_cluster"]):
        best.setdefault((s_["major"], s_["minor"]), s_)
    print("  closest-to-cluster segment per state:")
    for st in sorted(best, key=lambda k: best[k]["dist_to_cluster"])[:12]:
        b = best[st]
        print(f"    ({st[0]},{st[1]})  dist {b['dist_to_cluster']:.4f}  ESS {b['ess']:5.0f}  "
              f"{b['len_mb']:5.1f} Mb  {b['seg_id']}")


if __name__ == "__main__":
    main()
