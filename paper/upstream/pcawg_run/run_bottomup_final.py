#!/usr/bin/env python3
"""
Production bottom-up event detection (v3: ESS-weighted mean + conservative refinement).
Runs on a directory of samples and outputs per-sample event calls.

Usage:
  python run_bottomup_final.py --input_dir <dir> --output_tsv <out.tsv> [--has_truth]

Outputs TSV with columns:
  sample, event_time, classification, gf, n_chroms, n_cand_chroms
  (+ true_time, true_type if --has_truth)
"""
import argparse, gzip, json, pickle, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import poisson as sp_poisson
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parent.parents[2]
sys.path.insert(0, str(REPO))
from tau.utils import pick_best_key, order_t

HALF_WIN            = 0.06
IDENTIF_RANGE       = 0.10
REFINE_WIN          = 0.10
MIN_ESS             = 10
GENOME_SIZE         = 2_897_310_462
WGD_THRESH          = 0.40
WGD_GENOME_THRESH   = 0.50   # fraction of genome with major_cn>=2 to trigger (2,2) enforcement
MIN_CHROM_FRAC_CAND = 0.50
MERGE_TOL           = 0.15
MATCH_TOL           = 0.15

HG19_SIZES = {
    1:249250621,2:243199373,3:198022430,4:191154276,5:180915260,
    6:171115067,7:159138663,8:146364022,9:141213431,10:135534747,
    11:135006516,12:133851895,13:115169878,14:107349540,15:102531392,
    16:90354753,17:81195210,18:78077248,19:59128983,20:63025520,
    21:48129895,22:51304566,
}


def load_dfs(sample_dir, sample):
    g_path = sample_dir / f"{sample}.genome.pkl.gz"
    if not g_path.exists(): return None, None
    with gzip.open(g_path, "rb") as f: g = pickle.load(f)

    # Decide whether to enforce t2=0 for (2,2) segments (WGD genome)
    total_len = sum(s.end - s.start for s in g.segments)
    wgd_len   = sum(s.end - s.start for s in g.segments
                    if getattr(s, "major_cn", 0) >= 2)
    is_wgd_genome = (wgd_len / total_len > WGD_GENOME_THRESH) if total_len > 0 else False

    rows = []
    for seg in g.segments:
        if seg.timing_result is None: continue
        key = pick_best_key(seg)
        if key is None or key not in seg.timing_result: continue
        draws = seg.timing_result[key].get("draws", [])
        ess = float(sum(seg.N_counts)) if seg.N_counts is not None else 0
        try: chrom = int(seg.chrom)
        except: continue

        # WGD enforcement: for (2,2) segments in WGD genomes, set t2=0 → t_wgd = N2/2
        # N2/2 = max(cumsum[:,0]) across draws (upper bound of feasible range for t1)
        if is_wgd_genome and getattr(seg, "major_cn", 0) == 2 and getattr(seg, "minor_cn", 0) == 2:
            pt_draws = [d for d in draws if d.get("boot_id", 0) == 0 and d.get("t")]
            if pt_draws:
                all_cumsums = [np.cumsum(order_t(d["t"]))[:-1] for d in pt_draws]
                t_wgd = float(max(c[0] for c in all_cumsums))
                # Add a single row at t_wgd (range=0 → identifiable, one timepoint per segment)
                rows.append({"chrom":chrom,"start":seg.start,"end":seg.end,
                             "time_point":0,"time_fraction":t_wgd,
                             "ess":ess,"length":seg.end-seg.start})
            continue

        for draw in draws:
            if draw.get("boot_id", 0) != 0: continue
            t = draw.get("t", {})
            if not t: continue
            for tp_idx, tf in enumerate(np.cumsum(order_t(t))[:-1]):
                rows.append({"chrom":chrom,"start":seg.start,"end":seg.end,
                             "time_point":tp_idx,"time_fraction":float(tf),
                             "ess":ess,"length":seg.end-seg.start})
    if not rows: return None, None
    df = pd.DataFrame(rows)
    grp = ["chrom","start","end","time_point"]

    df_all_mean = (df.groupby(grp+["length"])
                     .agg(time_fraction=("time_fraction","mean"),
                          ess=("ess","first")).reset_index())
    df_all_mean = df_all_mean[df_all_mean["ess"] >= MIN_ESS].copy()

    rng = df.groupby(grp)["time_fraction"].transform(lambda x: x.max()-x.min())
    df_filt = df[rng < IDENTIF_RANGE].copy()
    if df_filt.empty: return pd.DataFrame(), df_all_mean
    df_filt = (df_filt.groupby(grp+["length"])
                      .agg(time_fraction=("time_fraction","mean"),
                           ess=("ess","first")).reset_index())
    return df_filt[df_filt["ess"] >= MIN_ESS].copy(), df_all_mean


def per_chrom_peaks(df):
    candidates = []
    for chrom, cdf in df.groupby("chrom"):
        cs = HG19_SIZES.get(chrom)
        if cs is None or cdf.empty: continue
        grid     = np.arange(0.05, 0.96, 0.025)
        total    = cdf["ess"].sum()
        expected = 2 * HALF_WIN * total
        if expected == 0: continue
        obs = np.array([cdf.loc[np.abs(cdf["time_fraction"]-c)<=HALF_WIN,"ess"].sum()
                        for c in grid])
        pv  = np.array([sp_poisson.sf(o-1, expected) for o in obs])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rej, _, _, _ = multipletests(pv, method="fdr_bh", alpha=0.05)
        sig = grid[rej]; clusters = []
        for c in sig:
            if clusters and c-clusters[-1][-1] <= 0.055: clusters[-1].append(c)
            else: clusters.append([c])
        for cl in clusters:
            center = float(np.mean(cl))
            near   = cdf[np.abs(cdf["time_fraction"]-center) <= HALF_WIN+0.02]
            near_u = near.drop_duplicates(["start","end"])
            cf     = near_u["length"].sum() / cs
            if cf < MIN_CHROM_FRAC_CAND: continue
            candidates.append({"chrom":chrom,"time":center,"chrom_frac":cf,
                                "length":near_u["length"].sum(),
                                "ess_near":float(near_u["ess"].sum())})
    return candidates


def merge_candidates(candidates, df_all_mean):
    """ESS-weighted mean + conservative refinement (only for multi-chrom events)."""
    if not candidates: return []
    cands = sorted(candidates, key=lambda x: x["time"])
    clusters = []
    for c in cands:
        merged = False
        for cl in clusters:
            cl_mean = (sum(x["time"]*x["ess_near"] for x in cl) /
                       sum(x["ess_near"] for x in cl))
            if abs(c["time"] - cl_mean) <= MERGE_TOL:
                cl.append(c); merged = True; break
        if not merged: clusters.append([c])

    events = []
    for cl in clusters:
        total_ess    = sum(x["ess_near"] for x in cl)
        center       = float(sum(x["time"]*x["ess_near"] for x in cl) / total_ess)
        cand_chroms  = list({x["chrom"] for x in cl})
        cand_n_chroms = len(cand_chroms)

        if cand_n_chroms >= 2:
            near_all   = df_all_mean[np.abs(df_all_mean["time_fraction"]-center) <= REFINE_WIN]
            near_segs  = near_all.drop_duplicates(["chrom","start","end"])
            gf         = near_segs["length"].sum() / GENOME_SIZE
            n_chroms   = len(near_segs["chrom"].unique()) if not near_segs.empty else cand_n_chroms
            cls        = "WGD" if gf >= WGD_THRESH else "PGD"
        else:
            gf         = sum(x["length"] for x in cl) / GENOME_SIZE
            n_chroms   = cand_n_chroms
            cls        = "chrom_specific"

        events.append({"time":center,"classification":cls,"gf":gf,
                       "n_chroms":n_chroms,"n_cand_chroms":cand_n_chroms})
    return events


def run_sample(sample_dir, sample):
    df_filt, df_all = load_dfs(sample_dir, sample)
    if df_filt is None or (hasattr(df_filt,"empty") and df_filt.empty):
        return []
    return merge_candidates(per_chrom_peaks(df_filt), df_all)


def load_truth(sample_dir, sample):
    tp = sample_dir / f"{sample}.truth.json"
    if not tp.exists(): return []
    with open(tp) as f: truth = json.load(f)
    return [e for e in truth.get("genome_events",[]) if e["type"] in ("WGD","PGD")]


def match_events(detected, true_events):
    """Match detected to true events within MATCH_TOL, return matched pairs."""
    det = [(e["time"], e["classification"]) for e in detected
           if e["classification"] in ("WGD","PGD")]
    tru = [(e["time"], e["type"]) for e in true_events]
    matched_det, matched_true = set(), set()
    pairs = sorted([(abs(dt-tt), di, ti)
                    for di,(dt,_) in enumerate(det)
                    for ti,(tt,_) in enumerate(tru)])
    matches = {}
    for dist, di, ti in pairs:
        if dist > MATCH_TOL: break
        if di not in matched_det and ti not in matched_true:
            matched_det.add(di); matched_true.add(ti)
            matches[di] = ti
    return det, tru, matches


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_tsv", required=True)
    ap.add_argument("--has_truth", action="store_true",
                    help="Load truth.json and add match columns (for simulations)")
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_tsv = Path(args.output_tsv)
    sample_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Processing {len(sample_dirs)} samples from {input_dir.name}...")

    rows = []
    for i, sd in enumerate(sample_dirs):
        sample = sd.name
        if i % 100 == 0: print(f"  {i}/{len(sample_dirs)}  {sample}")
        try:
            events = run_sample(sd, sample)
        except Exception as e:
            print(f"  Error {sample}: {e}")
            events = []

        if args.has_truth:
            true_events = load_truth(sd, sample)
            det, tru, matches = match_events(events, true_events)
            for di, (dt, dcls) in enumerate(det):
                ev = next(e for e in events if abs(e["time"]-dt)<1e-9)
                ti = matches.get(di)
                rows.append({
                    "sample": sample,
                    "event_time": round(dt, 4),
                    "classification": dcls,
                    "gf": round(ev["gf"], 4),
                    "n_chroms": ev["n_chroms"],
                    "n_cand_chroms": ev["n_cand_chroms"],
                    "matched": ti is not None,
                    "true_time": round(tru[ti][0], 4) if ti is not None else float("nan"),
                    "true_type": tru[ti][1] if ti is not None else "",
                    "timing_error": round(abs(dt - tru[ti][0]), 4) if ti is not None else float("nan"),
                })
            # also record undetected true events
            matched_true = {v for v in matches.values()}
            for ti, (tt, ttype) in enumerate(tru):
                if ti not in matched_true:
                    rows.append({
                        "sample": sample,
                        "event_time": float("nan"),
                        "classification": "missed",
                        "gf": float("nan"),
                        "n_chroms": 0,
                        "n_cand_chroms": 0,
                        "matched": False,
                        "true_time": round(tt, 4),
                        "true_type": ttype,
                        "timing_error": float("nan"),
                    })
            # diploid: no true events, no detections
            if not true_events and not events:
                rows.append({"sample":sample,"event_time":float("nan"),
                             "classification":"none","gf":float("nan"),
                             "n_chroms":0,"n_cand_chroms":0,
                             "matched":True,"true_time":float("nan"),
                             "true_type":"none","timing_error":float("nan")})
        else:
            if events:
                for ev in events:
                    rows.append({
                        "sample": sample,
                        "event_time": round(ev["time"], 4),
                        "classification": ev["classification"],
                        "gf": round(ev["gf"], 4),
                        "n_chroms": ev["n_chroms"],
                        "n_cand_chroms": ev["n_cand_chroms"],
                    })
            else:
                rows.append({"sample":sample,"event_time":float("nan"),
                             "classification":"none","gf":float("nan"),
                             "n_chroms":0,"n_cand_chroms":0})

    out = pd.DataFrame(rows)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_tsv, sep="\t", index=False)
    print(f"Saved {len(out)} rows → {output_tsv}")

    # Quick summary
    wgd = out[out["classification"]=="WGD"]
    pgd = out[out["classification"]=="PGD"]
    cs  = out[out["classification"]=="chrom_specific"]
    print(f"\nSummary:")
    print(f"  Samples:          {out['sample'].nunique()}")
    print(f"  WGD events:       {len(wgd)}  ({wgd['sample'].nunique()} samples)")
    print(f"  PGD events:       {len(pgd)}  ({pgd['sample'].nunique()} samples)")
    print(f"  Chrom-specific:   {len(cs)}  ({cs['sample'].nunique()} samples)")
    if args.has_truth:
        tp = out[out.get("matched", False) == True]
        miss = out[out["classification"]=="missed"]
        fp = out[(out["classification"].isin(["WGD","PGD"])) & (~out["matched"])]
        print(f"  True positives:   {len(tp[tp['classification'].isin(['WGD','PGD'])])}")
        print(f"  Missed events:    {len(miss)}")
        print(f"  False positives:  {len(fp)}")

if __name__ == "__main__":
    main()
