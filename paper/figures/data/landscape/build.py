"""Fig 5 cache: build the gain-density grids from the COMPOSITE TABLES (tau_gains + tau_events + ref oncokb),
NOT the genome pkls. Leaves out major_cn>7 for now. Reproduces _gain_density_grids.npz /
_unlinked_gain_density_grids.npz (+ a per-oncogene gain table for the swimmers) into figures/data/<label>/.
Run: python paper/figures/data/landscape/build.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.data_tables.tables import load_gains, load_events, load_samples
from paper.config import REF

OUT = Path(__file__).resolve().parent; OUT.mkdir(parents=True, exist_ok=True)
WGD_BERTH, PGD_BERTH = 0.20, 0.10
HG19 = {"1":249250621,"2":243199373,"3":198022430,"4":191154276,"5":180915260,"6":171115067,
"7":159138663,"8":146364022,"9":141213431,"10":135534747,"11":135006516,"12":133851895,
"13":115169878,"14":107349540,"15":102531392,"16":90354753,"17":81195210,"18":78077248,
"19":59128983,"20":63025520,"21":48129895,"22":51304566}
CHROMS = [str(i) for i in range(1, 23)]
OFF = {}; c = 0
for k in CHROMS:
    OFF[k] = c; c += HG19[k]
GEN = c; N_X, N_Y = 700, 20
x_edges = np.linspace(0, GEN, N_X + 1)

print("loading tables...", flush=True)
ev = load_events()
WGDT = {s: np.array(sorted(g.event_time_frac.values)) for s, g in ev[ev.classification == "WGD"].groupby("sample")}
PGDT = {s: np.array(sorted(g.event_time_frac.values)) for s, g in ev[ev.class_strict == "PGD"].groupby("sample")}
samp = load_samples()[["sample", "tumor_type"]]
TT = dict(zip(samp["sample"], samp.tumor_type))
g = load_gains(best_route_only=True,
               columns=["sample", "seg_id", "chrom", "start_bp", "end_bp", "major_cn", "gain_time_frac", "n_draws"])
g = g[(g.major_cn >= 2) & (g.major_cn <= 7) & g.chrom.isin(CHROMS)].copy()
g["tumor_type"] = g["sample"].map(TT)
g = g[g.tumor_type.notna()]
print(f"gains (best, 2<=major<=7, autosome): {len(g):,}", flush=True)

types = sorted(g.tumor_type.unique())
link = {t: np.zeros((N_Y, N_X), np.float32) for t in types}
nonlink = {t: np.zeros((N_Y, N_X), np.float32) for t in types}
unlk_link = {t: np.zeros((N_Y, N_X), np.float32) for t in types}     # WGD OR PGD linked
unlk_non = {t: np.zeros((N_Y, N_X), np.float32) for t in types}
# ccPG-linked = PGD-linked AND NOT WGD-linked, so the three layers (WGD / ccPG / unlinked) partition
# every gain exactly once. Derived here rather than by subtracting grids, because the segment-count
# denominators are per-segment booleans and do not subtract.
pgd_link = {t: np.zeros((N_Y, N_X), np.float32) for t in types}
pgd_non = {t: np.zeros((N_Y, N_X), np.float32) for t in types}
seg_link = {t: np.zeros(N_X, np.float32) for t in types}; seg_non = {t: np.zeros(N_X, np.float32) for t in types}
seg_ulink = {t: np.zeros(N_X, np.float32) for t in types}; seg_unon = {t: np.zeros(N_X, np.float32) for t in types}
seg_plink = {t: np.zeros(N_X, np.float32) for t in types}; seg_pnon = {t: np.zeros(N_X, np.float32) for t in types}
n_pgd = {t: 0 for t in types}
n_all = {t: 0 for t in types}; n_wgd = {t: 0 for t in types}

for (s, tt), gs in g.groupby(["sample", "tumor_type"]):
    n_all[tt] += 1
    isw = s in WGDT; isp = s in PGDT
    if isw:
        n_wgd[tt] += 1
    if isp:
        n_pgd[tt] += 1
    tf = gs.gain_time_frac.values
    w = 1.0 / np.clip(gs.n_draws.values, 1, None)
    gx0 = gs.chrom.map(OFF).values + gs.start_bp.values
    gx1 = gs.chrom.map(OFF).values + gs.end_bp.values
    wl = (np.min(np.abs(tf[:, None] - WGDT[s][None, :]), axis=1) <= WGD_BERTH) if isw else np.zeros(len(gs), bool)
    pl = (np.min(np.abs(tf[:, None] - PGDT[s][None, :]), axis=1) <= PGD_BERTH) if isp else np.zeros(len(gs), bool)
    ul = wl | pl
    po = pl & ~wl                     # ccPG-linked only
    yb = np.clip((tf * N_Y).astype(int), 0, N_Y - 1)
    i0 = np.clip(np.searchsorted(x_edges, gx0) - 1, 0, N_X)
    i1 = np.clip(np.searchsorted(x_edges, gx1), 0, N_X)
    segk = gs.seg_id.values
    for k in range(len(gs)):
        if i0[k] >= i1[k]:
            continue
        (link[tt] if wl[k] else nonlink[tt])[yb[k], i0[k]:i1[k]] += w[k]
        (unlk_link[tt] if ul[k] else unlk_non[tt])[yb[k], i0[k]:i1[k]] += w[k]
        (pgd_link[tt] if po[k] else pgd_non[tt])[yb[k], i0[k]:i1[k]] += w[k]
    for key in np.unique(segk):
        m = segk == key; a, b = i0[m][0], i1[m][0]
        if a >= b:
            continue
        if wl[m].any(): seg_link[tt][a:b] += 1
        if (~wl[m]).any(): seg_non[tt][a:b] += 1
        if ul[m].any(): seg_ulink[tt][a:b] += 1
        if (~ul[m]).any(): seg_unon[tt][a:b] += 1
        if po[m].any(): seg_plink[tt][a:b] += 1
        if (~po[m]).any(): seg_pnon[tt][a:b] += 1

def _save(name, L, N, SL, SN, evtkey, evtval=None):
    np.savez(OUT / name, **{f"link__{t}": L[t] for t in types}, **{f"nonlink__{t}": N[t] for t in types},
             **{f"seglink__{t}": SL[t] for t in types}, **{f"segnonlink__{t}": SN[t] for t in types},
             types=np.array(types), n_all=np.array([n_all[t] for t in types]),
             **{evtkey: np.array([(evtval or n_wgd)[t] for t in types])}, N_X=N_X, N_Y=N_Y)
_save("_gain_density_grids.npz", link, nonlink, seg_link, seg_non, "n_wgd")
_save("_unlinked_gain_density_grids.npz", unlk_link, unlk_non, seg_ulink, seg_unon, "n_evt")
_save("_ccpg_gain_density_grids.npz", pgd_link, pgd_non, seg_plink, seg_pnon, "n_wgd")
np.savez(OUT / "_ccpg_counts.npz", types=np.array(types),
         n_pgd=np.array([n_pgd[t] for t in types]))
print(f"saved density grids for {len(types)} tumour types -> figures/data/landscape/")

# per-oncogene gains for the swimmers (tau_gains at oncogene loci)
onc = pd.read_csv(REF / "ref_oncokb_oncogenes_hg19.tsv", sep="\t")
onc["chrom"] = onc.chrom.astype(str)
gg = load_gains(best_route_only=True,
                columns=["sample", "seg_id", "chrom", "start_bp", "end_bp", "major_cn", "minor_cn",
                         "gain_index", "gain_time_frac", "n_draws"])
gg = gg[(gg.major_cn <= 7) & gg.chrom.isin(CHROMS)].copy(); gg["tumor_type"] = gg["sample"].map(TT)
rows = []
for _, o in onc.iterrows():
    hit = gg[(gg.chrom == o.chrom) & (gg.start_bp <= o.pos_bp) & (gg.end_bp >= o.pos_bp)].copy()
    if len(hit):
        hit["oncogene"] = o.gene; rows.append(hit)
if rows:
    pd.concat(rows, ignore_index=True).to_csv(OUT / "oncogene_gains.tsv.gz", sep="\t", index=False, compression="gzip")
    print(f"oncogene_gains.tsv.gz: gains at {onc.gene.nunique()} oncogene loci")
