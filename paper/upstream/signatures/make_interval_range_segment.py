#!/usr/bin/env python3
"""
make_interval_range_segment.py  <TUMOR_TYPE>
Per-segment, per-INTERVAL table WITH polytope range, built from the per-draw segment_times.tsv (clock
+ each active signature) — so underdetermined intervals are handled honestly:

  for each (segment, interval) we take the MEDIAN interval over all polytope draws, plus the SPREAD
  (max-min of that interval across draws). range≈0 => the interval is exactly determined (trust the
  median); large range => the interval is polytope-ambiguous (filter/flag it).

Determinacy is per-interval, not per-segment: early gains of high-CN states are often range=0 even when
the segment as a whole is "underdetermined". So you keep an interval when BOTH clocklike_range and
<SIG>_range are small.

Columns: descriptors, time_interval, clock_start, clock_end,
  raw:        clocklike_raw, <SIG>_raw           (= median normalized interval * S; S=Σ_i i·N_i/total_cn)
  normalized: clocklike_normalized, <SIG>_normalized
  ratio:      <SIG>_ratio                         (= <SIG>_raw / clocklike_raw)
  range:     clocklike_range, <SIG>_range      (polytope max-min of the normalized interval)
  exposure:   <SIG>_exposure
-> sig_intervals_range_<TT>_segment.tsv
"""
import os
import sys
import numpy as np, pandas as pd
from pathlib import Path
R=Path(os.environ.get("TAU_REPO", "."))
TABLES=R/"dev/pcawg/signatures/timing_tables"
CLOCK=R/"dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
SIGDIR=R/"dev/pcawg/signatures"
EXPO=Path(os.path.join(os.environ.get("TAU_PCAWG_INPUT_DIR", "."), "MuSiCal_assignment_PCAWG.txt"))
TT=sys.argv[1] if len(sys.argv)>1 else "Eso.AdenoCA"

SIGDIRS={"SBS17":"output_persig_SBS17a_SBS17b","SBS18":"output_persig_SBS18","APOBEC":"output_persig_SBS2_SBS13",
         "SBS3":"output_persig_SBS3","SBS4":"output_persig_SBS4","SBS7":"output_persig_SBS7"}
SIGCOLS={"SBS3":["SBS3"],"SBS4":["SBS4"],"SBS17":["SBS17a","SBS17b"],"SBS18":["SBS18"],
         "SBS7":["SBS7a","SBS7b","SBS7c","SBS7d"],"APOBEC":["SBS2","SBS13"]}
ex=pd.read_csv(EXPO,sep="\t"); sbs=[c for c in ex.columns if c.startswith("SBS")]
expo={s:dict(zip(ex.aliquot_id, ex[c].sum(axis=1)/ex[sbs].sum(axis=1).replace(0,np.nan))) for s,c in SIGCOLS.items()}

seg=pd.read_csv(TABLES/f"sig_timing_{TT}_segments.tsv",sep="\t")
SIGS=[c[2:] for c in seg.columns if c.startswith("G_") and c!="G_clock"]
DCOL=["chrom","cn_state","cluster_label","route","determined","mutation_count"]
meta=seg.drop_duplicates(["sample","segment_id"]).set_index(["sample","segment_id"])
tscols=["ts_clock"]+[f"ts_{s}" for s in SIGS]
print(f"{TT}: {meta.index.get_level_values(0).nunique()} samples, signatures {SIGS}")

USE=["bootstrap_id","draw_id","segment_id","time_fraction","time_point","best_route"]
def draws_by_seg(path):
    try: d=pd.read_csv(path,sep="\t",usecols=USE,low_memory=False)
    except Exception: return {}
    d=d[(d.bootstrap_id==0)&(d.best_route==True)]
    out={}
    for sid,g in d.groupby("segment_id"):
        p=g.pivot_table(index="draw_id",columns="time_point",values="time_fraction")
        out[sid]=p.sort_index(axis=1).to_numpy()      # (n_draws, n_events)
    return out

def stats(G2d):
    """point-estimate interval vector (median over draws, renormalized to sum 1) + per-interval polytope
    range (max-min of the normalized interval across draws). len = n_events+1."""
    nd=G2d.shape[0]
    B=np.column_stack([np.zeros(nd),np.clip(G2d,0,1),np.ones(nd)])
    Tm=np.diff(B,axis=1)                                # (n_draws, n_intervals); each row sums to 1
    med=np.median(Tm,axis=0); tot=med.sum()
    if tot>0: med=med/tot                               # median isn't linear -> renormalize to sum 1
    return med, (np.nanmax(Tm,axis=0)-np.nanmin(Tm,axis=0))

rows=[]
for s,msub in meta.groupby(level=0):
    cd=draws_by_seg(CLOCK/TT/s/f"{s}.segment_times.tsv")
    if not cd: continue
    sd={sig:draws_by_seg(SIGDIR/SIGDIRS[sig]/s/f"{s}.segment_times.tsv")
        for sig in SIGS if (SIGDIR/SIGDIRS[sig]/s/f"{s}.segment_times.tsv").exists()}
    for (_,sid),m in msub.iterrows():
        if sid not in cd: continue
        tcn,tcs=stats(cd[sid]); Sc=float(m["ts_clock"])
        cum=np.r_[0.0,np.cumsum(tcn)]
        sig_iv={}
        for sig in SIGS:
            if sig in sd and sid in sd[sig] and sd[sig][sid].shape[1]==cd[sid].shape[1] and pd.notna(m.get(f"ts_{sig}")):
                tsn,tss=stats(sd[sig][sid]); sig_iv[sig]=(tsn,tss,float(m[f"ts_{sig}"]))
        for i in range(len(tcn)):
            row={"sample":s,"tumor_type":TT,"segment_id":sid}
            for k in DCOL:
                if k in m: row[k]=m[k]
            row["time_interval"]=i+1; row["clocklike_cumulative_start"]=cum[i]; row["clocklike_cumulative_end"]=cum[i+1]
            row["clocklike_raw_interval"]=tcn[i]*Sc
            for sig,(tsn,tss,Ss) in sig_iv.items(): row[f"{sig}_raw_interval"]=tsn[i]*Ss
            row["clocklike_normalized_interval"]=tcn[i]
            for sig,(tsn,tss,Ss) in sig_iv.items(): row[f"{sig}_normalized_interval"]=tsn[i]
            cr=tcn[i]*Sc
            for sig,(tsn,tss,Ss) in sig_iv.items(): row[f"{sig}_ratio"]=(tsn[i]*Ss)/cr if cr>0 else np.nan
            row["clocklike_range"]=tcs[i]
            for sig,(tsn,tss,Ss) in sig_iv.items(): row[f"{sig}_range"]=tss[i]
            for sig in SIGS: row[f"{sig}_exposure"]=expo.get(sig,{}).get(s,np.nan)
            rows.append(row)

out=pd.DataFrame(rows)
desc=["sample","tumor_type","segment_id","chrom","cn_state","cluster_label","route","determined","mutation_count"]
order=[c for c in desc if c in out.columns]+["time_interval","clocklike_cumulative_start","clocklike_cumulative_end",
    "clocklike_raw_interval"]+[f"{s}_raw_interval" for s in SIGS]+["clocklike_normalized_interval"]+\
    [f"{s}_normalized_interval" for s in SIGS]+[f"{s}_ratio" for s in SIGS]+["clocklike_range"]+\
    [f"{s}_range" for s in SIGS]+[f"{s}_exposure" for s in SIGS]
out=out[[c for c in order if c in out.columns]]
f=TABLES/f"sig_intervals_range_{TT}_segment.tsv"; out.to_csv(f,sep="\t",index=False)
print("wrote",f,out.shape)
det=(out.clocklike_range<0.05)
print(f"intervals with clocklike_range<0.05 (well-determined): {det.mean():.0%}")
