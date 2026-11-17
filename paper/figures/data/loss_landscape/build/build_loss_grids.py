"""Post-WGD loss landscape -> paper/figures/data/loss_landscape/loss_grids.npz + loss_by_chrom.tsv.

Ports the inputs of dev/pcawg/FigWGDPGD/loss_per_opportunity.py into the figure's own cache, so the
panel scripts read only from paper/. This is the cache_build step, and is the only place allowed to
reach into dev/.

What is being measured
----------------------
A WGD doubles every allele present. Any copy that should be there afterwards and is not was lost. So for
each segment, walking its route back through the doubling gives the copies the doubling produced, and
the shortfall against the observed copy number is the implied loss (dev/pcawg/FigWGDPGD/
single_wgd_loss_segments.tsv for single-WGD samples, cohort_loss_segments.tsv for double, where L1/L2
split the loss between the two doublings).

Loss alone is not comparable across the genome: a locus that was amplified more has more copies
available to lose. The measure is therefore loss per OPPORTUNITY —

    loss density (copies lost per WGD sample, spread over [t_WGD, 1] and divided by 1 - t_WGD)
    ÷ WGD-linked gain density at the same locus (the doublings available to lose)

which is a per-branch loss rate, decoupled from how much a region amplified. Loci with essentially no
opportunity are masked rather than divided by a near-zero denominator (25th-percentile floor).

Emits both the genome x time grids (for the heatmap) and a per-(tumour type, chromosome) collapse (for
the homogeneity comparison, where the question is whether any chromosome stands out).

Usage:  python paper/figures/data/loss_landscape/build/build_loss_grids.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
DEV = ANALYSIS / "pcawg/FigWGDPGD"
GAINS = Path(__file__).resolve().parents[2] / "landscape/_gain_density_grids.npz"   # the landscape figure cache
OUT = cache_dir_of(__file__); OUT.mkdir(parents=True, exist_ok=True)

HG19 = {"1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260, "6": 171115067,
        "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
        "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753, "17": 81195210, "18": 78077248,
        "19": 59128983, "20": 63025520, "21": 48129895, "22": 51304566}
CHROMS = [str(i) for i in range(1, 23)]
OFF = {}; _c = 0
for _k in CHROMS:
    OFF[_k] = _c; _c += HG19[_k]
GEN = _c
MIN_WGD, MIN_LOSS = 8, 8


def main():
    g = np.load(GAINS, allow_pickle=True)
    GT = list(g["types"]); N_X, N_Y = int(g["N_X"]), int(g["N_Y"])
    n_wgd = {t: int(n) for t, n in zip(GT, g["n_wgd"])}
    x_edges = np.linspace(0, GEN, N_X + 1)

    ev = pd.read_csv(DEV / "events_allpcawg.tsv", sep="\t")
    wgd = {s: sorted(v.time.values) for s, v in ev[ev.classification == "WGD"].groupby("sample")}

    sg = (pd.read_csv(DEV / "single_wgd_loss_segments.tsv", sep="\t", dtype={"chrom": str})
            .rename(columns={"L": "loss"}))[["sample", "tumor_type", "chrom", "start", "end", "loss", "W"]]
    db = pd.read_csv(DEV / "cohort_loss_segments.tsv", sep="\t", dtype={"chrom": str})
    db["loss"] = db.L1 + db.L2
    db["W"] = db["sample"].map(lambda s: float(wgd[s][0]) if s in wgd else np.nan)
    d = pd.concat([sg, db[["sample", "tumor_type", "chrom", "start", "end", "loss", "W"]]],
                  ignore_index=True)
    # W < 0.98: a WGD at the very end of molecular time leaves no post-WGD window for loss to occur in,
    # so 1 - W is a near-zero divisor rather than a real rate
    d = d[d.chrom.isin(CHROMS) & (d.loss > 0) & d.W.notna() & (d.W < 0.98)]
    nloss = d.groupby("tumor_type")["sample"].nunique()

    types = [t for t in GT if t != "Unknown" and n_wgd.get(t, 0) >= MIN_WGD
             and nloss.get(t, 0) >= MIN_LOSS]
    print(f"{len(types)} tumour types ({d['sample'].nunique():,} samples with implied loss)")

    grids, opps, rows = {}, {}, []
    for t in types:
        H = np.zeros((N_Y, N_X), np.float32)
        for r in d[d.tumor_type == t].itertuples():
            yw = min(max(int(r.W * N_Y), 0), N_Y - 1)
            i0 = max(0, np.searchsorted(x_edges, OFF[r.chrom] + r.start) - 1)
            i1 = min(N_X, np.searchsorted(x_edges, OFF[r.chrom] + r.end))
            if i0 < i1:
                H[yw:N_Y, i0:i1] += r.loss / max(1e-3, 1 - r.W)
        L = gaussian_filter(H / nloss[t], (1.0, 1.5))
        opp = gaussian_filter(g[f"link__{t}"].astype(float).sum(0) / max(1, n_wgd[t]), 1.5)
        grids[t], opps[t] = L, opp

        # per-chromosome collapse: the same ratio, aggregated over each chromosome's columns
        flo = np.percentile(opp[opp > 0], 25) if (opp > 0).any() else 1e-3
        for ch in CHROMS:
            c0 = max(0, np.searchsorted(x_edges, OFF[ch]) - 1)
            c1 = min(N_X, np.searchsorted(x_edges, OFF[ch] + HG19[ch]))
            sel = np.arange(c0, c1)
            sel = sel[opp[sel] >= flo]
            if sel.size == 0:
                continue
            rows.append(dict(tumor_type=t, chrom=ch, n_samples=int(nloss[t]),
                             loss=float(L[:, sel].sum(0).mean()),
                             opportunity=float(opp[sel].mean()),
                             loss_per_opportunity=float((L[:, sel].sum(0) / opp[sel]).mean())))

    np.savez_compressed(OUT / "loss_grids.npz", types=np.array(types), N_X=N_X, N_Y=N_Y,
                        n_loss=np.array([int(nloss[t]) for t in types]),
                        **{f"loss__{t}": grids[t] for t in types},
                        **{f"opp__{t}": opps[t] for t in types})
    tb = pd.DataFrame(rows)
    tb.to_csv(OUT / "loss_by_chrom.tsv", sep="\t", index=False)
    print(f"-> {OUT/'loss_grids.npz'}\n-> {OUT/'loss_by_chrom.tsv'}  ({len(tb)} rows)")
    s = tb.groupby("chrom").loss_per_opportunity.median().reindex(CHROMS)
    print(f"\nper-chromosome median loss/opportunity: {s.min():.3f}–{s.max():.3f} "
          f"(ratio {s.max()/s.min():.2f}x)")


if __name__ == "__main__":
    main()
