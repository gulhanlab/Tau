#!/usr/bin/env python3
"""
build_sig_cache.py  SIG  THRESH
Build a per-gain cache comparing clock (SBS1+SBS5) vs signature SIG, for samples with
SIG exposure fraction >= THRESH. Per matched segment (same best route) and gain:
  G_clock, G_sig (event times), ts_clock, ts_sig (= D/total_cn per run = calc_total_sum),
  ess_clock, tumor_type. -> sig_cumulative_cache_<SIG>.tsv
SIG dir = dev/pcawg/signatures/output_persig_<SIG>/; clock = tau_outputs/.../<tt>/<sample>/.
For SBS3, exposure column is "SBS3" (single).
"""
import os
import sys, glob, gzip, pickle
import numpy as np, pandas as pd
from pathlib import Path
R = Path(os.environ.get("TAU_REPO", ".")); sys.path.insert(0, str(R))

SIG   = sys.argv[1] if len(sys.argv) > 1 else "SBS3"
THRESH= float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
SIGCOLS = {"SBS3": ["SBS3"], "SBS17a_SBS17b": ["SBS17a","SBS17b"],
           "SBS2_SBS13": ["SBS2","SBS13"], "SBS4": ["SBS4"],
           "SBS7": ["SBS7a","SBS7b","SBS7c","SBS7d"], "SBS18": ["SBS18"]}.get(SIG, [SIG])
PERSIG = R/f"dev/pcawg/signatures/output_persig_{SIG}"
CLOCKG = str(R/"dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg")+"/*/{s}/{s}.segment_times.tsv"
EXPO   = Path(os.path.join(os.environ.get("TAU_PCAWG_INPUT_DIR", "."), "MuSiCal_assignment_PCAWG.txt"))
OUT    = R/f"dev/pcawg/signatures/sig_cumulative_cache_{SIG}.tsv"
USECOLS=["bootstrap_id","draw_id","segment_id","key","time_point","time_fraction","best_route","mutation_count"]

ex=pd.read_csv(EXPO,sep="\t"); sbs=[c for c in ex.columns if c.startswith("SBS")]
ex["frac"]=ex[SIGCOLS].sum(axis=1)/ex[sbs].sum(axis=1).replace(0,np.nan)
fmap=dict(zip(ex.aliquot_id,ex.frac)); ttmap=dict(zip(ex.aliquot_id,ex.get("tumor_type",pd.Series(index=ex.index))))

def events(path):
    d=pd.read_csv(path,sep="\t",usecols=USECOLS,low_memory=False)
    d=d[(d.bootstrap_id==0)&(d.best_route==True)]
    out={}
    for (sid,key),g in d.groupby(["segment_id","key"]):
        out[sid]=(key, g.groupby("time_point")["time_fraction"].mean(), g["mutation_count"].iloc[0])
    return out
def tsmap(path):
    with gzip.open(path,"rb") as f: G=pickle.load(f)
    out={}
    for s in G.segments:
        N=getattr(s,"N_counts",None)
        if N is None or len(N)==0: continue
        N=np.asarray(N,float); D=float(sum((k+1)*N[k] for k in range(len(N)))); tc=s.major_cn+s.minor_cn
        if tc>0: out[s.seg_id]=D/tc
    return out

samps=[p.name for p in PERSIG.iterdir() if p.is_dir() and fmap.get(p.name,0)>=THRESH]
print(f"{len(samps)} {SIG} samples with frac>={THRESH}",flush=True)
rows=[]; kept=0
for i,s in enumerate(samps):
    ck=glob.glob(CLOCKG.format(s=s)); sst=PERSIG/s/f"{s}.segment_times.tsv"
    cg=glob.glob(str(R/"dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg")+f"/*/{s}/{s}.genome.pkl.gz")
    sg=PERSIG/s/f"{s}.genome.pkl.gz"
    if not ck or not sst.exists() or not cg or not sg.exists(): continue
    tt=Path(ck[0]).parents[1].name
    try:
        ec=events(ck[0]); es=events(sst); tc=tsmap(cg[0]); ts=tsmap(sg)
    except Exception as e:
        print("skip",s[:8],repr(e),flush=True); continue
    kept+=1
    for sid,(kc,Gc,ess) in ec.items():
        if sid not in es: continue
        ks,Gs,_=es[sid]
        if ks!=kc or sid not in tc or sid not in ts: continue
        for tp in Gc.index.intersection(Gs.index):
            rows.append(dict(sample=s,tumor_type=tt,segment_id=sid,time_point=int(tp),
                             G_clock=float(Gc[tp]),G_sig=float(Gs[tp]),ess=float(ess),
                             ts_clock=float(tc[sid]),ts_sig=float(ts[sid]),frac=float(fmap[s])))
    if (i+1)%30==0: print(f"  {i+1}/{len(samps)}; kept {kept}, {len(rows)} gains",flush=True)
pd.DataFrame(rows).to_csv(OUT,sep="\t",index=False)
print(f"done. {kept} samples, {len(rows)} gains -> {OUT}")
