#!/usr/bin/env python3
"""Build bottomup_final_v7.tsv — per-detected-event table matched to truth — for the
Figure-3 panels (plot_figure2_nature.py), mirroring bottomup_final_v6.tsv.

Reads each v7 sample's time_clusters.tsv (production clustering output, already on disk)
and truth.json; greedily time-matches detected WGD/PGD events to true WGD/PGD events
(<= MATCH_TOL). No genome.pkl loading. Columns:
  sample event_time classification gf n_chroms n_cand_chroms matched true_time true_type timing_error
"""
import glob, json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
import sys
VER = sys.argv[1] if len(sys.argv) > 1 else "v7"
SIM = BASE / f"output_{VER}"
OUT = BASE / f"bottomup_final_{VER}.tsv"
MATCH_TOL = 0.15

rows = []
for sd in sorted(SIM.iterdir()):
    if not sd.is_dir():
        continue
    s = sd.name
    tc_p = sd / f"{s}.time_clusters.tsv"
    tr_p = sd / f"{s}.truth.json"
    if not tc_p.exists() or not tr_p.exists():
        continue
    tc = pd.read_csv(tc_p, sep="\t")
    truth = json.load(open(tr_p))
    true_ev = [(float(e["time"]), e["type"]) for e in truth.get("genome_events", [])
               if e["type"] in ("WGD", "PGD")]

    det = tc[tc.classification.isin(["WGD", "PGD"])].copy()
    det = det.sort_values("time").reset_index(drop=True)
    # greedy time-match detected -> true (one-to-one, nearest first, <= MATCH_TOL)
    pairs = sorted([(abs(float(r.time) - tt), di, ti)
                    for di, r in det.iterrows()
                    for ti, (tt, _) in enumerate(true_ev)])
    md, mt, match = set(), set(), {}
    for dist, di, ti in pairs:
        if dist > MATCH_TOL:
            break
        if di not in md and ti not in mt:
            md.add(di); mt.add(ti); match[di] = ti

    for di, r in det.iterrows():
        matched = di in match
        tt, ty = (true_ev[match[di]] if matched else (None, None))
        rows.append({
            "sample": s, "event_time": round(float(r.time), 4),
            "classification": r.classification, "gf": r.gf,
            "n_chroms": int(r.n_chroms), "n_cand_chroms": int(r.n_chroms),
            "matched": matched,
            "true_time": (round(tt, 4) if matched else ""),
            "true_type": (ty if matched else ""),
            "timing_error": (round(abs(float(r.time) - tt), 4) if matched else ""),
        })

out = pd.DataFrame(rows)
out.to_csv(OUT, sep="\t", index=False)
print(f"wrote {OUT}  ({len(out)} detected events, {out['sample'].nunique()} samples)")
print(out["classification"].value_counts().to_string())
print(f"matched: {out['matched'].sum()}  FP: {(~out['matched']).sum()}")
