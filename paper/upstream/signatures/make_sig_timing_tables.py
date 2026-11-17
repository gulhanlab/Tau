#!/usr/bin/env python3
"""
make_sig_timing_tables.py  [TUMOR_TYPE]   (default Breast.AdenoCA)
Build tidy data tables of per-segment + state-aggregated Tau timing for each mutational signature
vs the SBS1+SBS5 clock, for one tumor type. For the supervisor to explore signature-activity ratios.

Two outputs in dev/pcawg/signatures/timing_tables/ :
  sig_timing_<TT>_segments.tsv  — one row per (sample, segment, gain event). Clock event time +
      each signature's event time, CN state / route / determinacy, and per-segment abundance scalars.
  sig_timing_<TT>_state.tsv     — aggregated per (sample, CN state): median clock time + per-signature
      median event time, timing shift, abundance ratio, n segments.

Per-signature columns (SIG in whichever are active in the tumor type):
  G_<SIG>        signature event time (cumulative fraction of that signature's mutations before the gain)
  dG_<SIG>       = G_<SIG> - G_clock   (signed timing shift; <0 = signature earlier than clock)
  ts_<SIG>       per-segment D/total_cn for the signature run (D=sum_i i*N_i)  [abundance scale]
  ratio_<SIG>    = ts_<SIG> / ts_clock  (per-segment signature:clock abundance ratio = D_sig/D_clock)
  frac_<SIG>     genome-wide signature exposure fraction for the sample
The per-INTERVAL unnormalized signature:clock ratio used in the plots can be reconstructed as
  r_interval = (dG_sig / dG_clock) * ratio_<SIG>   over consecutive gains within a segment.
"""
import os
import sys, glob
import numpy as np, pandas as pd
from pathlib import Path
R = Path(os.environ.get("TAU_REPO", "."))
EXPO = Path(os.path.join(os.environ.get("TAU_PCAWG_INPUT_DIR", "."), "MuSiCal_assignment_PCAWG.txt"))
SIGDIR = R/"dev/pcawg/signatures"
CLOCK  = R/"dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
OUTDIR = SIGDIR/"timing_tables"; OUTDIR.mkdir(exist_ok=True)
TT = sys.argv[1] if len(sys.argv) > 1 else "Breast.AdenoCA"

# signature -> (cache file, G column, exposure columns)
SIGS = {"SBS17":("sig_cumulative_cache_ts.tsv","G_sbs17",["SBS17a","SBS17b"]),
        "SBS3":("sig_cumulative_cache_SBS3.tsv","G_sig",["SBS3"]),
        "SBS4":("sig_cumulative_cache_SBS4.tsv","G_sig",["SBS4"]),
        "APOBEC":("sig_cumulative_cache_SBS2_SBS13.tsv","G_sig",["SBS2","SBS13"]),
        "SBS7":("sig_cumulative_cache_SBS7.tsv","G_sig",["SBS7a","SBS7b","SBS7c","SBS7d"]),
        "SBS18":("sig_cumulative_cache_SBS18.tsv","G_sig",["SBS18"])}
KEY=["sample","segment_id","time_point"]

ex=pd.read_csv(EXPO,sep="\t"); sbs=[c for c in ex.columns if c.startswith("SBS")]
def expo_frac(cols): return dict(zip(ex.aliquot_id, ex[cols].sum(axis=1)/ex[sbs].sum(axis=1).replace(0,np.nan)))

base=None; active=[]
for sig,(fn,gcol,ecols) in SIGS.items():
    p=SIGDIR/fn
    if not p.exists(): continue
    d=pd.read_csv(p,sep="\t")
    d=d[d.tumor_type==TT].copy()
    if len(d)==0: continue
    tscol="ts_sbs17" if sig=="SBS17" else "ts_sig"
    if "frac" in d.columns: fr=d["frac"]
    else: fr=d["sample"].map(expo_frac(ecols))
    sub=pd.DataFrame({"sample":d["sample"],"segment_id":d.segment_id,"time_point":d.time_point,
                      f"G_clock_{sig}":d.G_clock,f"ts_clock_{sig}":d.ts_clock,f"ess_{sig}":d.get("ess",np.nan),
                      f"G_{sig}":d[gcol],f"ts_{sig}":d[tscol],f"frac_{sig}":fr})
    active.append(sig)
    base = sub if base is None else base.merge(sub,on=KEY,how="outer")
if base is None:
    print(f"No signatures active in {TT}"); sys.exit(0)
# coalesce the per-signature copies of the (signature-independent) clock columns
for c in ["G_clock","ts_clock","ess"]:
    dups=[f"{c}_{sig}" for sig in active if f"{c}_{sig}" in base.columns]
    base[c]=base[dups].bfill(axis=1).iloc[:,0]; base=base.drop(columns=dups)
print(f"{TT}: signatures active = {active}; {base['sample'].nunique()} samples, {len(base)} segment-gains")

# --- attach CN state / route / determinacy from the clock run ---
meta=[]
for s in base["sample"].unique():
    g=glob.glob(str(CLOCK/TT/s/f"{s}.segment_times.tsv"))
    if not g: continue
    t=pd.read_csv(g[0],sep="\t",low_memory=False)
    t=t[(t.bootstrap_id==0)&(t.best_route==True)]
    m=t.groupby("segment_id").agg(chrom=("chrom","first"),start=("start","first"),end=("end","first"),
        major_cn=("major_cn","first"),minor_cn=("minor_cn","first"),route=("key","first"),
        mutation_count=("mutation_count","first"),num_draws=("num_draws","first")).reset_index()
    m["sample"]=s; meta.append(m)
meta=pd.concat(meta)
base=base.merge(meta,on=["sample","segment_id"],how="left")
base["cn_state"]=base.major_cn.astype("Int64").astype(str)+"_"+base.minor_cn.astype("Int64").astype(str)
base["determined"]=base.num_draws.fillna(1).le(1)   # single draw => determined timing

# per-signature derived columns
for sig in active:
    base[f"dG_{sig}"]=base[f"G_{sig}"]-base["G_clock"]
    base[f"ratio_{sig}"]=base[f"ts_{sig}"]/base["ts_clock"].replace(0,np.nan)

front=["sample","tumor_type","segment_id","chrom","start","end","cn_state","major_cn","minor_cn",
       "route","determined","num_draws","mutation_count","time_point","ess","G_clock","ts_clock"]
base["tumor_type"]=TT
percols=[]
for sig in active: percols+=[f"G_{sig}",f"dG_{sig}",f"ts_{sig}",f"ratio_{sig}",f"frac_{sig}"]
base=base[[c for c in front if c in base.columns]+percols].sort_values(["sample","chrom","start","time_point"])
segf=OUTDIR/f"sig_timing_{TT}_segments.tsv"; base.to_csv(segf,sep="\t",index=False)
print("wrote",segf,base.shape)

# --- state-based aggregation: per (sample, cn_state) ---
def agg(g):
    out={"n_segments":g.segment_id.nunique(),"n_gains":len(g),"G_clock_med":g.G_clock.median()}
    for sig in active:
        sl=g.dropna(subset=[f"G_{sig}"])
        out[f"n_{sig}"]=sl.segment_id.nunique()
        out[f"G_{sig}_med"]=sl[f"G_{sig}"].median()
        out[f"dG_{sig}_med"]=sl[f"dG_{sig}"].median()
        out[f"ratio_{sig}_med"]=sl[f"ratio_{sig}"].median()
    return pd.Series(out)
state=base.groupby(["sample","cn_state"]).apply(agg,include_groups=False).reset_index()
state.insert(1,"tumor_type",TT)
statef=OUTDIR/f"sig_timing_{TT}_state.tsv"; state.to_csv(statef,sep="\t",index=False)
print("wrote",statef,state.shape)
