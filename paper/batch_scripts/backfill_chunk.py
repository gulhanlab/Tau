"""Backfill one chunk of samples: read genome pkls -> composite tables (tau_segments, tau_routes,
tau_multiplicities, tau_gains) + per-sample aggregates for tau_samples. Reshape only, NO re-timing.
usage: python backfill_chunk.py <task_id> <chunk_size>   (SLURM array)
-> paper/batch_scripts/backfill_chunks/chunk_<task>.{segments,routes,multiplicities,sampleagg}.tsv + .gains.parquet
"""
import sys, gzip, pickle
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import tau  # noqa: F401  (needed to unpickle Genome/Segment)

TAU = R / "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
OUT = R / "paper/batch_scripts/backfill_chunks"


def order_t(tdict):
    """t is {t1:.., t2:..} (sympy-keyed); return values ordered by numeric suffix."""
    def idx(k):
        d = "".join(ch for ch in str(k) if ch.isdigit())
        return int(d) if d else 0
    return [float(tdict[k]) for k in sorted(tdict, key=idx)]


def best_route(seg):
    """deterministic min-dist_rel route (first wins on ties)."""
    best, bd = None, np.inf
    for key, tr in (seg.timing_result or {}).items():
        qc = tr.get("qc", [])
        d = float(qc[0].get("dist_rel", np.inf)) if qc else np.inf
        if d < bd:
            bd, best = d, key
    return best


def sample_rows(g, sample, tumor_type, seg2evt):
    seg_rows, route_rows, mult_rows, gain_rows = [], [], [], []
    n_snvs = 0.0; ess_list = []; cn_weighted = 0.0; total_len = 0
    for seg in g.segments:
        chrom = str(getattr(seg, "chrom", "")).replace("chr", "")
        s0, s1 = int(getattr(seg, "start", -1)), int(getattr(seg, "end", -1))
        seg_id = getattr(seg, "seg_id", None) or f"{chrom}:{s0}-{s1}"
        M, m = int(getattr(seg, "major_cn", 0)), int(getattr(seg, "minor_cn", 0))
        nc = getattr(seg, "N_counts", None)
        ess = float(np.sum(nc)) if nc is not None else 0.0
        tr = getattr(seg, "timing_result", None) or {}
        is_timed = bool(tr)
        disc = getattr(seg, "discard_info", None)
        reason = (disc.get("reason") if isinstance(disc, dict) else None) if not is_timed else None
        seg_rows.append(dict(sample=sample, seg_id=seg_id, chrom=chrom, start_bp=s0, end_bp=s1,
                             seg_len_bp=max(0, s1 - s0), major_cn=M, minor_cn=m, total_cn=M + m,
                             ess=round(ess, 3), event_id=seg2evt.get(seg_id, pd.NA),
                             is_timed=is_timed, discard_reason=reason))
        # sample-level accumulators
        if nc is not None:
            n_snvs += float(np.sum(nc))
        if is_timed:
            ess_list.append(ess)
        if s1 > s0:
            cn_weighted += (M + m) * (s1 - s0); total_len += (s1 - s0)
        # multiplicities
        if nc is not None and np.sum(nc) > 0:
            tot = float(np.sum(nc))
            for k in range(len(nc)):
                mult_rows.append(dict(sample=sample, seg_id=seg_id, multiplicity=k + 1,
                                      n_count=round(float(nc[k]), 4), pi=round(float(nc[k]) / tot, 5)))
        # routes + gains
        bkey = best_route(seg)
        for key, rtr in tr.items():
            draws = rtr.get("draws", []); meta = rtr.get("meta", {}); qc = rtr.get("qc", [{}])
            n_ev = None
            for d in draws:
                tv = order_t(d.get("t", {}))
                if len(tv) < 2:
                    continue
                cum = np.cumsum(tv)[:-1]            # event (gain) times
                n_ev = len(cum)
                for gi, gt in enumerate(cum, start=1):
                    gain_rows.append(dict(sample=sample, seg_id=seg_id, route_key=key,
                                          is_best_route=(key == bkey), chrom=chrom, start_bp=s0, end_bp=s1,
                                          major_cn=M, minor_cn=m, gain_index=gi, draw_id=int(d.get("draw_id", 0)),
                                          gain_time_frac=round(float(gt), 5), n_draws=len(draws)))
            route_rows.append(dict(sample=sample, seg_id=seg_id, route_key=key,
                                   n_events=(n_ev if n_ev is not None else 0), n_draws=len(draws),
                                   free_var_count=int(meta.get("free_var_count", 0)),
                                   dist_rel=round(float(qc[0].get("dist_rel", np.nan)), 5) if qc else np.nan,
                                   inside_margin=round(float(qc[0].get("inside_margin", np.nan)), 5) if qc else np.nan,
                                   is_best_route=(key == bkey)))
    ploidy = round(cn_weighted / total_len, 4) if total_len else np.nan
    agg = dict(sample=sample, purity=float(getattr(g.segments[0], "purity", np.nan)) if g.segments else np.nan,
               n_snvs=int(round(n_snvs)), n_segments=len(g.segments),
               n_timed_segments=len(ess_list), mean_ess=round(float(np.mean(ess_list)), 3) if ess_list else np.nan,
               ploidy_seg=ploidy)
    return seg_rows, route_rows, mult_rows, gain_rows, agg


def main(task_id, chunk):
    jobs = []
    for tt in sorted(TAU.iterdir()):
        if tt.is_dir():
            for sd in sorted(tt.iterdir()):
                pkl = sd / f"{sd.name}.genome.pkl.gz"
                if sd.is_dir() and pkl.exists():
                    jobs.append((pkl, sd.name, tt.name, sd))
    start, end = (task_id - 1) * chunk, min(task_id * chunk, len(jobs))
    if start >= len(jobs):
        print(f"[task {task_id}] nothing to do"); return
    OUT.mkdir(parents=True, exist_ok=True)
    S, RT, MU, agg_rows = [], [], [], []
    GA = []
    for pkl, sid, tt, sd in jobs[start:end]:
        # seg -> event_id from the existing segment_clusters.tsv
        seg2evt = {}
        scf = sd / f"{sid}.segment_clusters.tsv"
        if scf.exists():
            try:
                sc = pd.read_csv(scf, sep="\t", usecols=lambda c: c in ("seg_id", "cluster_label"))
                seg2evt = {r.seg_id: (int(r.cluster_label) if r.cluster_label >= 0 else pd.NA)
                           for r in sc.itertuples()}
            except Exception:
                pass
        try:
            with gzip.open(pkl, "rb") as fh:
                g = pickle.load(fh)
        except Exception as e:
            print(f"  {sid}: load failed {e}", flush=True); continue
        sr, rr, mr, gr, ag = sample_rows(g, sid, tt, seg2evt)
        S += sr; RT += rr; MU += mr; GA += gr; agg_rows.append(ag)
        del g
    pd.DataFrame(S).to_csv(OUT / f"chunk_{task_id}.segments.tsv", sep="\t", index=False)
    pd.DataFrame(RT).to_csv(OUT / f"chunk_{task_id}.routes.tsv", sep="\t", index=False)
    pd.DataFrame(MU).to_csv(OUT / f"chunk_{task_id}.multiplicities.tsv", sep="\t", index=False)
    pd.DataFrame(agg_rows).to_csv(OUT / f"chunk_{task_id}.sampleagg.tsv", sep="\t", index=False)
    # gains as gzipped TSV (pyarrow/parquet not in env; gz-TSV = compact + human-readable, drop-in to parquet later)
    pd.DataFrame(GA).to_csv(OUT / f"chunk_{task_id}.gains.tsv.gz", sep="\t", index=False, compression="gzip")
    print(f"[task {task_id}] samples[{start}:{end}]  segs={len(S)} routes={len(RT)} mult={len(MU)} gains={len(GA)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
