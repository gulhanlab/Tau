"""Post-WGD loss profiles for three contrasting tumour types -> paper/figures/data/loss_landscape/loss_profiles.tsv.

Ports the binned profile and its bootstrap band out of dev/pcawg/loss_catalog so the panel reads only
from paper/. This is the cache_build step and the only place allowed to reach into dev/.

The quantity
------------
    loss per opportunity in a bin  =  sum over (sample, segment, doubling) of L * len
                                      -----------------------------------------------
                                      sum over the same of cb * len

cb is the copy budget: the copies present when a doubling fired, all of which it should have
duplicated. L is how many never appeared. Doublings are POOLED rather than split, because L and cb at
successive doublings are tied by the identity cb_2 = cb_1 + d_1 and a per-doubling ratio is therefore
not identifiable; the pooled total never has to decide which doubling lost what.

The band is a 95% percentile bootstrap over SAMPLES within each tumour type, 1000 replicates, seeded.
The tumour is the unit that repeats in the cohort — resampling segments would treat one genome's ~90
segments as independent observations and give a band several times too tight.

The three types are chosen to contrast in both level and shape, not to be representative: Liver.HCC
loses least (0.34) and is the best powered (n=83); Ovary.AdenoCA is intermediate (0.52) and close to
flat; Lung.SCC is lowest-but-most-structured (0.40), with broad multi-chromosome plateaus.

Input: dev/pcawg/loss_catalog/multi_wgd_loss_allpcawg.tsv (compute_multi_wgd_loss.py), which
generalises the L1/L2 recursion to k doublings so 3- and 4-WGD samples keep their middle doublings.

Usage:  python paper/figures/data/loss_landscape/build/build_loss_profiles.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "supp/loss_landscape"))   # the figure dir (_geom helper)
import _geom as G

SRC = ANALYSIS / "pcawg/loss_catalog/multi_wgd_loss_allpcawg.tsv"
OUT = cache_dir_of(__file__); OUT.mkdir(parents=True, exist_ok=True)
TYPES = ["Liver.HCC", "Ovary.AdenoCA", "Lung.SCC"]
BIN_MB = 10
MIN_SAMP = 5
N_BOOT = 1000
SEED = 7


def main():
    d = pd.read_csv(SRC, sep="\t", dtype={"chrom": str})
    d = d[d.chrom.isin(G.CHROMS)]
    nb = int(np.ceil(G.GEN / (BIN_MB * 1e6)))
    edges = np.linspace(0, nb * BIN_MB * 1e6, nb + 1)
    rng = np.random.default_rng(SEED)

    rows = []
    for t in TYPES:
        g = d[d.tumor_type == t]
        samples = sorted(g["sample"].unique())
        idx = {s: i for i, s in enumerate(samples)}
        num = np.zeros((len(samples), nb)); den = np.zeros((len(samples), nb))
        for r in g.itertuples():
            x0, x1 = G.OFF[r.chrom] + r.start, G.OFF[r.chrom] + r.end
            i0 = max(0, np.searchsorted(edges, x0) - 1)
            i1 = min(nb, np.searchsorted(edges, x1))
            si = idx[r.sample]
            for i in range(i0, i1):
                ov = min(x1, edges[i + 1]) - max(x0, edges[i])
                if ov > 0:
                    num[si, i] += r.L * ov; den[si, i] += r.cb * ov
        ok = (den > 0).sum(0) >= MIN_SAMP
        tot = den.sum(0)
        prof = np.divide(num.sum(0), tot, out=np.full(nb, np.nan), where=tot > 0)
        boot = np.empty((N_BOOT, nb))
        for b in range(N_BOOT):
            p = rng.integers(0, len(samples), len(samples))
            n_, d_ = num[p].sum(0), den[p].sum(0)
            boot[b] = np.divide(n_, d_, out=np.full(nb, np.nan), where=d_ > 0)
        with np.errstate(invalid="ignore"):
            lo, hi = (np.nanpercentile(boot, q, axis=0) for q in (2.5, 97.5))
        gw = float(num.sum() / max(den.sum(), 1))
        for i in range(nb):
            if not ok[i]:
                continue
            rows.append(dict(tumor_type=t, n_samples=len(samples), bin_index=i,
                             x_mb=float((edges[i] + edges[i + 1]) / 2e6),
                             loss_per_opp=float(prof[i]), lo=float(lo[i]), hi=float(hi[i]),
                             n_bin=int((den[:, i] > 0).sum()), genome_wide=gw))
        print(f"{t:<16} n={len(samples):>3}  genome-wide {gw:.3f}  "
              f"{int(ok.sum())} bins  median band {np.nanmedian(hi - lo):.3f}")

    t = pd.DataFrame(rows)
    p = OUT / "loss_profiles.tsv"
    t.to_csv(p, sep="\t", index=False)
    print(f"\n{len(t):,} rows -> {p}")


if __name__ == "__main__":
    main()
