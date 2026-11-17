"""Collect (3,x)/(4,x) underdetermined units with a STRUCTURALLY-chosen special-case solution.

The special case is picked by simultaneity alone — minimise the doubling blocks' interior intervals,
prefer the form covering most gains. No reference to the detected WGD times, so any concentration at
the WGD in the resulting figure is a finding rather than a consequence of selection.
"""
import os
import sys, gzip, pickle, glob, itertools, csv, numpy as np, pandas as pd
sys.path.insert(0,os.environ.get("TAU_REPO", "."))
from paper.data_tables.tables import load_events
from tau.utils import pick_best_key, order_t
OUT = sys.argv[1]
DB={}
for r in csv.DictReader(open("dev/pcawg/doubling_blocks.tsv"), delimiter="\t"):
    if r["blocks"]:
        DB[r["route_key"]]=[tuple(int(v) for v in b.replace("/","+").split("+")) for b in r["blocks"].split(";")]
ev=load_events(); w=ev[ev.classification=="WGD"]
wgdt={s:sorted(g.event_time_frac.tolist()) for s,g in w.groupby("sample")}
files=sorted(glob.glob("dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg/*/*/*.clustered_genome.pkl.gz"))
rows=[]; n=0
for fi,f in enumerate(files):
    s=f.split("/")[-2]
    if s not in wgdt: continue
    try: cg=pickle.load(gzip.open(f,"rb"))
    except Exception: continue
    n+=1
    if n % 100 == 0: print(f"  {n} samples, {len(rows)} units", flush=True)
    wt=wgdt[s]; mw=min(len(wt),2)
    for seg in cg.segments:
        maj,mino=int(seg.major_cn),int(seg.minor_cn)
        if maj not in (3, 4): continue   # (3,x) and (4,x); states with no doubling block
                                         # are dropped below by the `k not in DB` test
        k=pick_best_key(seg)
        if k is None or k not in DB: continue
        dr=[d for d in seg.timing_result[k].get("draws",[]) if d.get("boot_id",0)==0]
        if len(dr)<2: continue
        ess=float(np.sum(seg.N_counts)) if seg.N_counts is not None else 0
        if ess<30: continue
        ts=np.array([order_t(d["t"]) for d in dr]); cums=np.cumsum(ts,axis=1)[:,:-1]
        blocks=[b for b in DB[k] if b[1]>=2]
        best=None
        for r_ in range(1,mw+1):
            for combo in itertools.combinations(blocks,r_):
                iv=sorted((a,a+L-1) for a,L,_ in combo)
                if any(iv[j][1]>=iv[j+1][0] for j in range(len(iv)-1)): continue
                sc=sum(ts[:,a:a+L-1].sum(axis=1) for a,L,_ in combo)
                i=int(np.argmin(sc))
                if sc[i]>0.01: continue
                cand=(-sum(L for _,L,_ in combo), float(sc[i]), i, combo)
                if best is None or cand[:2]<best[:2]: best=cand
        if best is None: continue
        _,_,i,combo=best
        rows.append(dict(sample=s, seg_id=str(seg.seg_id), state=f"({maj},{mino})", route=k,
                         ess=ess, n_draws=len(dr), n_gain=cums.shape[1],
                         n_doubling=len(combo),
                         all_ev=";".join(f"{x:.4f}" for x in cums.ravel()),
                         spec_ev=";".join(f"{x:.4f}" for x in cums[i]),
                         wgd=";".join(f"{x:.4f}" for x in wt)))
d=pd.DataFrame(rows); d.to_csv(OUT, sep="\t", index=False)
print(f"DONE: {n} WGD samples scanned; {len(d)} units in {d['sample'].nunique()} samples")
print(d.state.value_counts().to_string())
