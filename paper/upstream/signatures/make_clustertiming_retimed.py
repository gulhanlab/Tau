#!/usr/bin/env python3
"""
make_clustertiming_retimed.py  [TUMOR_TYPE] [LIMIT_SAMPLES]

ONE aggregate timing per cluster per signature, by POOLING + RE-TIMING (the supervisor's design):
  1. Cluster the CLOCK genome's segments by multiplicity (cluster_segment_multiplicities; per CN state).
  2. For each cluster, MERGE its member segments into one pseudo-segment (pool their SNVs) and RE-TIME it
     -> a single set of event times for the cluster (clock).
  3. For each signature run, merge the SAME member segments from the signature genome (signature-weighted
     SNVs) and re-time -> the cluster's event times under that signature. Same clock-defined clusters, so
     clock vs signature is a direct comparison.

Uses tau.make_clustered_genome (pools snv_tables) + Genome.time_segments (recomputes N_counts via EM and
solves timing). Best route per merged segment via pick_best_key; event times = median over draws of
cumsum(order_t(t))[:-1].

Output: sig_clustertiming_retimed_<TT>.tsv  — one row per (sample, cluster, gain): clock_time + per
signature time / shift / n_segments.
"""
import sys, gzip, pickle
import numpy as np, pandas as pd
from pathlib import Path
import tau
from tau import cluster_segment_multiplicities, make_clustered_genome
from tau.utils import pick_best_key, order_t

R = Path(os.environ.get("TAU_REPO", "."))
SIGDIR = R/"dev/pcawg/signatures"; TABLES=SIGDIR/"timing_tables"
CLOCK  = R/"dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
EXPO   = Path(os.path.join(os.environ.get("TAU_PCAWG_INPUT_DIR", "."), "MuSiCal_assignment_PCAWG.txt"))
TT = sys.argv[1] if len(sys.argv) > 1 else "Breast.AdenoCA"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None

SIGRUNS={"SBS3":SIGDIR/"output_persig_SBS3","APOBEC":SIGDIR/"output_persig_SBS2_SBS13",
         "SBS18":SIGDIR/"output_persig_SBS18","SBS17":SIGDIR/"output_persig_SBS17a_SBS17b",
         "SBS4":SIGDIR/"output_persig_SBS4","SBS7":SIGDIR/"output_persig_SBS7"}
SIGCOLS={"SBS3":["SBS3"],"APOBEC":["SBS2","SBS13"],"SBS18":["SBS18"],"SBS17":["SBS17a","SBS17b"],
         "SBS4":["SBS4"],"SBS7":["SBS7a","SBS7b","SBS7c","SBS7d"]}
ex=pd.read_csv(EXPO,sep="\t"); sbs=[c for c in ex.columns if c.startswith("SBS")]
fr={r:dict(zip(ex.aliquot_id, ex[c].sum(axis=1)/ex[sbs].sum(axis=1).replace(0,np.nan))) for r,c in SIGCOLS.items()}

def load(path):
    with gzip.open(path,"rb") as f: return pickle.load(f)

def stats(G2d):
    """point-estimate interval vector (median over polytope draws, renormalized to sum 1) + per-interval
    range (max-min of the normalized interval across draws). G2d=(n_draws,n_events) -> len n_events+1."""
    nd=G2d.shape[0]
    B=np.column_stack([np.zeros(nd),np.clip(G2d,0,1),np.ones(nd)])
    Tm=np.diff(B,axis=1)
    med=np.median(Tm,axis=0); tot=med.sum()
    if tot>0: med=med/tot
    return med, (np.nanmax(Tm,axis=0)-np.nanmin(Tm,axis=0))

def cluster_event_times(genome, seg_cluster_df, env):
    """re-time merged (clustered) genome; return {merged_seg_id: (cn_state, n_segs, G2d, ts)} where
    G2d=(n_draws,n_events) keeps ALL polytope draws (for per-interval median+range)."""
    cg=make_clustered_genome(genome, seg_cluster_df)
    cg.segments=[s for s in cg.segments if str(getattr(s,"seg_id","")).startswith("merged_")]
    if not cg.segments: return {}
    cg.calculate_multiplicities(bootstrap_B=1, random_state=0)   # EM on pooled SNVs -> N_counts (mut_w-weighted)
    cg.time_segments(env)
    out={}
    for s in cg.segments:
        if not getattr(s,"timing_result",None): continue
        best=pick_best_key(s)
        if not best or best not in s.timing_result: continue
        ev=[np.cumsum(order_t(d["t"]))[:-1] for d in s.timing_result[best]["draws"]
            if int(d.get("boot_id",0))==0 and d.get("t")]
        if not ev: continue
        L=min(len(e) for e in ev); G2d=np.array([e[:L] for e in ev])     # (n_draws, n_events)
        # ts = D/total_cn (un-normalization constant: t_unnorm = t_norm * ts)
        nc=np.asarray(s.N_counts,dtype=float); tot=float(s.major_cn+s.minor_cn)
        ts=float(np.sum((np.arange(len(nc))+1)*nc)/tot) if tot>0 and nc.sum()>0 else np.nan
        out[s.seg_id]=(f"{s.major_cn}_{s.minor_cn}", int(getattr(s,"num_segs",len(getattr(s,'seg_ids',[])))),
                       G2d, ts)
    return out

import os
env=tau.timing._get_route_env() if hasattr(tau,"timing") else None
samples=sorted(p.name for p in (CLOCK/TT).iterdir() if p.is_dir())
if LIMIT: samples=samples[:LIMIT]
SHARD_ID=os.environ.get("SHARD_ID"); SHARD_N=os.environ.get("SHARD_N")
if SHARD_ID is not None and SHARD_N is not None:                       # SLURM array shard
    sid=int(SHARD_ID); sn=int(SHARD_N); samples=samples[sid::sn]
    OUTSUB=TABLES/f"_shards_{TT}"; OUTSUB.mkdir(parents=True,exist_ok=True); SUF=f".shard{sid:04d}"
else:
    OUTSUB=TABLES; SUF=""
print(f"{TT}: {len(samples)} samples (shard {SHARD_ID}/{SHARD_N}); signatures gated at exposure>=0.15",flush=True)

rows=[]; memrows=[]
for i,s in enumerate(samples):
    cg=CLOCK/TT/s/f"{s}.genome.pkl.gz"
    if not cg.exists(): continue
    try: gclock=load(cg)
    except Exception as e: print("  skip clock",s[:8],repr(e),flush=True); continue
    scdf=cluster_segment_multiplicities(gclock)
    if scdf is None or scdf.empty or (scdf.cluster_label!=-1).sum()==0: continue
    mm=scdf[["seg_id","major_cn","minor_cn","cluster_label","x","y"]].copy(); mm["sample"]=s; memrows.append(mm)
    clock=cluster_event_times(gclock, scdf, env)
    if not clock: continue
    sig_results={}
    for sig,rd in SIGRUNS.items():
        if fr.get(sig,{}).get(s,0)<0.15: continue
        gp=rd/s/f"{s}.genome.pkl.gz"
        if not gp.exists(): continue
        try: gsig=load(gp)
        except Exception: continue
        sig_results[sig]=cluster_event_times(gsig, scdf, env)
    for mseg,(cn,nseg,G2d_c,ts_c) in clock.items():
        clabel=int(str(mseg).split("_")[-1])   # DBSCAN cluster index within the CN state (0,1,2,...)
        tcn,tcs=stats(G2d_c); cum=np.r_[0.0,np.cumsum(tcn)]
        sig_iv={}
        for sig,res in sig_results.items():
            if mseg in res and res[mseg][2].shape[1]==G2d_c.shape[1]:   # same n_events => alignable
                tsn,tss=stats(res[mseg][2]); sig_iv[sig]=(tsn,tss,res[mseg][3])
        for i_iv in range(len(tcn)):
            row=dict(sample=s,tumor_type=TT,cn_state=cn,cluster_label=clabel,cluster=mseg,n_segments=nseg,
                     time_interval=i_iv+1,clocklike_cumulative_start=cum[i_iv],clocklike_cumulative_end=cum[i_iv+1],
                     clocklike_raw_interval=tcn[i_iv]*ts_c,clocklike_normalized_interval=tcn[i_iv],clocklike_range=tcs[i_iv])
            for sig,(tsn,tss,ts_s) in sig_iv.items():
                cr=tcn[i_iv]*ts_c
                row[f"{sig}_raw_interval"]=tsn[i_iv]*ts_s; row[f"{sig}_normalized_interval"]=tsn[i_iv]
                row[f"{sig}_ratio"]=(tsn[i_iv]*ts_s)/cr if cr>0 else np.nan
                row[f"{sig}_range"]=tss[i_iv]
            for sig in sig_results: row[f"{sig}_exposure"]=fr.get(sig,{}).get(s,np.nan)
            rows.append(row)
    if (i+1)%10==0: print(f"  {i+1}/{len(samples)} samples",flush=True)

df=pd.DataFrame(rows)
out=OUTSUB/f"sig_intervals_range_{TT}_cluster{SUF}.tsv"; df.to_csv(out,sep="\t",index=False)
print("wrote",out,df.shape,flush=True)
if memrows:
    mem=pd.concat(memrows,ignore_index=True).rename(columns={"seg_id":"segment_id","x":"mult_x","y":"mult_y"})
    mem["cn_state"]=mem.major_cn.astype(int).astype(str)+"_"+mem.minor_cn.astype(int).astype(str)
    mem["clustered"]=mem.cluster_label!=-1; mem.insert(1,"tumor_type",TT)
    mf=OUTSUB/f"sig_clustertiming_{TT}_membership{SUF}.tsv"; mem.to_csv(mf,sep="\t",index=False)
    print("wrote",mf,mem.shape,flush=True)
