#!/usr/bin/env python3
"""Per-sample post-WGD loss catalogue: one plot per WGD sample, raw loss and loss per opportunity.

Why per sample
--------------
The cohort loss landscapes average tens of genomes per tumour type, and that average is a bad estimator
here: at a given locus, samples of the same type disagree 2.2-5.4x more than the type-level mean profile
varies along the genome. A flat mean row is therefore uninformative — it is equally consistent with
diffuse loss and with strong loss at DIFFERENT loci in different samples. This catalogue keeps every
genome separate so the question can be looked at directly.

What is drawn, per sample
-------------------------
  top     Opportunity and loss on the same axis. Grey is `cb`, the copies present at the doubling — what
          the WGD had available to lose. Blue is `L`, how many of them are missing. The blue is drawn
          inside the grey, so the reading is "of the copies the doubling produced, this many went".
          Double-WGD samples split the blue into the first and second doubling.
  bottom  L / cb, the fraction of the available copies lost, on a fixed 0-1 axis so samples are directly
          comparable to each other.

Both are step functions over the sample's OWN segments — no smoothing, no binning, no cohort
normalisation, and gaps where the sample has no timed segment are left empty rather than interpolated.

Simplifications, deliberate
---------------------------
Loss is attributed to the WGD itself rather than spread over the post-WGD window [W,1] and divided by
1-W as the cohort grids do. That rate normalisation exists to make tumour types with different WGD
timing comparable; within one sample there is nothing to make comparable, and the raw copy count is the
more direct quantity.

The denominator here is the segment's OWN cb from the reconstruction, not the cohort's WGD-linked gain
density. That is the honest per-sample denominator and it avoids the confound that made the cohort map
hard to read, where the ratio tracked its denominator (rho -0.75) more tightly than it tracked loss.

Caveat that applies to every panel: L = max(0, cb - d) is floored at zero, so a segment whose
reconstruction assigns more doublings than the budget allows contributes 0 rather than a negative. Loss
is one-sided and cannot cancel.

Usage:
    python dev/pcawg/loss_catalog/build_loss_catalog.py                  # all 762 samples
    python dev/pcawg/loss_catalog/build_loss_catalog.py --limit 6        # a quick look
    python dev/pcawg/loss_catalog/build_loss_catalog.py --types Ovary.AdenoCA Liver.HCC
"""
import os
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

R = Path(os.environ.get("TAU_REPO", "."))
DEV = R / "dev/pcawg/FigWGDPGD"
OUT = R / "dev/pcawg/loss_catalog"

HG19 = {"1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260, "6": 171115067,
        "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747, "11": 135006516,
        "12": 133851895, "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753,
        "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520, "21": 48129895,
        "22": 51304566}
CEN = {"1": 125e6, "2": 93.3e6, "3": 91e6, "4": 50.4e6, "5": 48.4e6, "6": 61e6, "7": 59.9e6,
       "8": 45.6e6, "9": 49e6, "10": 40.2e6, "11": 53.7e6, "12": 35.8e6, "13": 17.9e6, "14": 17.6e6,
       "15": 19e6, "16": 36.6e6, "17": 24e6, "18": 17.2e6, "19": 26.5e6, "20": 27.5e6, "21": 13.2e6,
       "22": 14.7e6}
CHROMS = [str(i) for i in range(1, 23)]
OFF = {}
_c = 0
for _k in CHROMS:
    OFF[_k] = _c
    _c += HG19[_k]
GEN = _c

C_OPP = "#dcdcdc"       # copies available at the doubling
C_L1 = "#08306b"        # lost at the first doubling
C_L2 = "#4292c6"        # ...at the second
C_RATIO = "#2a5db0"


def load():
    """Both loss tables, in one frame with a uniform (L1, L2, cb) shape and the WGD times attached."""
    ev = pd.read_csv(DEV / "events_allpcawg.tsv", sep="\t")
    wgd = ev[ev.classification == "WGD"].groupby("sample").time.apply(lambda s: sorted(s.values))

    sg = pd.read_csv(DEV / "single_wgd_loss_segments.tsv", sep="\t", dtype={"chrom": str})
    sg = sg.rename(columns={"L": "L1"}); sg["L2"] = 0.0; sg["n_wgd"] = 1

    db = pd.read_csv(DEV / "cohort_loss_segments.tsv", sep="\t", dtype={"chrom": str})
    db["n_wgd"] = 2

    keep = ["sample", "tumor_type", "chrom", "start", "end", "L1", "L2", "cb", "length", "n_wgd"]
    d = pd.concat([sg[keep], db[keep]], ignore_index=True)
    d = d[d.chrom.isin(CHROMS)].copy()
    d["L"] = d.L1 + d.L2
    d["x0"] = [OFF[c] + s for c, s in zip(d.chrom, d.start)]
    d["w"] = d.end - d.start
    return d, wgd


def _chrom_axis(ax, top=False):
    for c in CHROMS[1:]:
        ax.axvline(OFF[c] / 1e6, color="0.75", lw=0.6)
    for c in CHROMS:
        ax.axvline((OFF[c] + CEN[c]) / 1e6, color="0.88", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(0, GEN / 1e6)
    if top:
        ax.set_xticks([])
    else:
        ax.set_xticks([(OFF[c] + HG19[c] / 2) / 1e6 for c in CHROMS])
        ax.set_xticklabels(CHROMS, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_sample(d, wgd, sample, outdir):
    g = d[d["sample"] == sample].sort_values("x0")
    tt = g.tumor_type.iloc[0]
    W = [float(x) for x in wgd.get(sample, [])]
    x, w = g.x0.to_numpy(float) / 1e6, g.w.to_numpy(float) / 1e6
    cb, L1, L2 = g.cb.to_numpy(float), g.L1.to_numpy(float), g.L2.to_numpy(float)
    L = L1 + L2
    ratio = np.divide(L, cb, out=np.zeros_like(L, dtype=float), where=cb > 0)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.12})

    # opportunity behind, loss in front, so the bar is read as a fraction consumed
    ax.bar(x, cb, width=w, align="edge", color=C_OPP, linewidth=0, label="copies at WGD (opportunity)")
    ax.bar(x, L1, width=w, align="edge", color=C_L1, linewidth=0, label="lost")
    if L2.any():
        ax.bar(x, L2, width=w, align="edge", bottom=L1, color=C_L2, linewidth=0,
               label="lost at 2nd WGD")
    ax.set_ylabel("copies", fontsize=8)
    ax.set_ylim(0, max(1.0, cb.max()) * 1.08)
    # above the axes, not inside it: the bars reach the top of the panel wherever a segment kept its
    # full copy budget, so any in-axes corner is occupied for some sample
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
              handlelength=1.2, columnspacing=1.0, borderpad=0.2)
    _chrom_axis(ax, top=True)

    bx.bar(x, ratio, width=w, align="edge", color=C_RATIO, linewidth=0)
    lw_mean = float((ratio * g.w).sum() / max(g.w.sum(), 1))
    bx.axhline(lw_mean, color="#b0402f", lw=1.0, ls=(0, (4, 2)))
    # boxed: on a heavily-lost genome the bars run the width of the panel and bare red text on blue
    # is unreadable
    bx.text(GEN / 1e6 * 0.004, lw_mean, f"genome mean {lw_mean:.2f}", fontsize=7, color="#b0402f",
            ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
    bx.set_ylim(0, 1); bx.set_ylabel("fraction of\navailable copies lost", fontsize=8)
    bx.set_xlabel("chromosome", fontsize=8)
    _chrom_axis(bx)
    for a in (ax, bx):
        a.tick_params(labelsize=7, length=2, pad=1.5)

    wtxt = " · ".join(f"WGD {i+1} at t={t:.2f}" for i, t in enumerate(W)) or "WGD time unavailable"
    cov = float(g.w.sum()) / GEN
    mL = float((L * g.w).sum() / max(g.w.sum(), 1))
    mcb = float((cb * g.w).sum() / max(g.w.sum(), 1))
    ax.set_title(f"{tt}   {sample}\n{wtxt}   ·   {len(g)} timed segments covering {100*cov:.0f}% of the "
                 f"genome   ·   {mL:.2f} of {mcb:.2f} copies lost per locus (length-weighted)",
                 fontsize=8.5, fontweight="bold", pad=16, linespacing=1.35)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.845, bottom=0.095)

    sub = outdir / tt; sub.mkdir(parents=True, exist_ok=True)
    fig.savefig(sub / f"{sample}.png", dpi=130)
    plt.close(fig)
    return dict(sample=sample, tumor_type=tt, n_wgd=int(g.n_wgd.iloc[0]),
                W1=W[0] if W else np.nan, W2=W[-1] if len(W) > 1 else np.nan,
                n_segments=len(g), genome_frac=cov,
                mean_loss_frac=lw_mean,
                mean_L=float((L * g.w).sum() / max(g.w.sum(), 1)),
                mean_cb=float((cb * g.w).sum() / max(g.w.sum(), 1)),
                frac_genome_any_loss=float(g.w[L > 0].sum() / max(g.w.sum(), 1)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--outdir", default=str(OUT / "plots"))
    a = ap.parse_args(argv)

    d, wgd = load()
    if a.types:
        d = d[d.tumor_type.isin(a.types)]
    samples = sorted(d["sample"].unique())
    if a.limit:
        # spread the sample over tumour types rather than taking one type's worth alphabetically
        step = max(1, len(samples) // a.limit)
        samples = samples[::step][:a.limit]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(samples)} samples -> {outdir}", flush=True)

    rows = []
    for i, s in enumerate(samples, 1):
        rows.append(plot_sample(d, wgd, s, outdir))
        if i % 50 == 0:
            print(f"  …{i}/{len(samples)}", flush=True)
    t = pd.DataFrame(rows).sort_values("mean_loss_frac", ascending=False)
    t.to_csv(OUT / "loss_catalog_summary.tsv", sep="\t", index=False)
    print(f"\n{len(t)} plots written")
    print(f"mean fraction of available copies lost: median {t.mean_loss_frac.median():.3f}, "
          f"range {t.mean_loss_frac.min():.3f}-{t.mean_loss_frac.max():.3f}")
    print(f"-> {OUT/'loss_catalog_summary.tsv'}")


if __name__ == "__main__":
    main()
