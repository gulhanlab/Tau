"""punctuated figure (main), panel a — LENGTH-WEIGHTED variant of a_gain_composition.

Identical partition (WGD berth 0.20, ccPG berth 0.10, WGD precedence, containment over solutions,
autosomes with 2 <= major_cn <= 7) but each gain contributes its SEGMENT LENGTH rather than 1, so the
column reads "% of gained DNA" instead of "% of gains". Kept as an alternative: the shared fraction is
83.7% weighted vs 77.9% unweighted (ccPG 4.2% vs 2.5%, unlinked 16.3% vs 22.1%), because ccPGs are
arm-level (median 29.6 Mb) while unlinked gains are more focal (10.8 Mb).

Run: python paper/figures/main/punctuated/alt_gain_composition_weighted.py -> PNG+SVG+PDF next to this file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
import a_gain_composition as A             # geometry, colours, berths, style
from paper.data_tables.tables import load_gains, load_events
from paper.config import cache_dir_of

PANELS = A.PANELS
S = A.S
CACHE = cache_dir_of(__file__) / "gain_composition_weighted.tsv"


def compute(force=False):
    """-> (n_gains, {category: summed segment length in bp}). Cached; the join is over ~3.5M gain rows."""
    if CACHE.exists() and not force:
        t = pd.read_csv(CACHE, sep="\t")
        return int(t.n_gains.iloc[0]), t.set_index("category").weight.to_dict()

    ev = load_events()
    WGDT = {s: np.sort(g.event_time_frac.values)
            for s, g in ev[ev.classification == "WGD"].groupby("sample")}
    PGDT = {s: np.sort(g.event_time_frac.values)
            for s, g in ev[ev.class_strict == "PGD"].groupby("sample")}
    g = load_gains(best_route_only=True,
                   columns=["sample", "seg_id", "chrom", "start_bp", "end_bp", "major_cn",
                            "gain_time_frac", "gain_index"])
    g = g[(g.major_cn >= 2) & (g.major_cn <= 7) & g.chrom.isin(A.CHROMS)]

    parts = []
    for s, gs in g.groupby("sample", sort=False):
        tf = gs.gain_time_frac.to_numpy(float)
        dw = (np.min(np.abs(tf[:, None] - WGDT[s][None, :]), axis=1)
              if s in WGDT else np.full(len(gs), np.inf))
        dp = (np.min(np.abs(tf[:, None] - PGDT[s][None, :]), axis=1)
              if s in PGDT else np.full(len(gs), np.inf))
        parts.append(pd.DataFrame({"sample": s, "seg_id": gs.seg_id.values,
                                   "gi": gs.gain_index.values, "dw": dw, "dp": dp,
                                   "seglen": (gs.end_bp.values - gs.start_bp.values)}))
    a = pd.concat(parts).groupby(["sample", "seg_id", "gi"]).agg(
        dw=("dw", "min"), dp=("dp", "min"), seglen=("seglen", "first"))
    n_gains = len(a)
    L = a["seglen"].to_numpy(float)
    wl = (a.dw <= A.WGD_BERTH).to_numpy()
    po = ((a.dp <= A.PGD_BERTH).to_numpy() & ~wl)
    tot = {"WGD-linked": float(L[wl].sum()), "ccPG-linked": float(L[po].sum()),
           "unlinked": float(L[~(wl | po)].sum())}
    print(f"  {n_gains:,} gains, {L.sum()/1e9:.1f} Gb gained")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"category": list(tot), "weight": list(tot.values()),
                  "n_gains": n_gains}).to_csv(CACHE, sep="\t", index=False)
    return n_gains, tot


def make(save=True):
    n_gains, tot = compute()
    T = sum(tot.values())
    parts = [("WGD-linked", A.WGD_C), ("ccPG-linked", A.CCPG_C), ("unlinked", A.UNL_C)]

    fig, ax = plt.subplots(figsize=(A.W, A.H))
    top = 100.0
    for lab, col in parts:
        v = 100 * tot[lab] / T
        ax.bar(A.BAR_X, v, bottom=top - v, width=A.BAR_W, color=col, edgecolor="white",
               linewidth=0.8, zorder=3)
        mid = top - v / 2
        ax.plot([A.BAR_X + A.BAR_W / 2, A.LAB_X - 0.04], [mid, mid], color=col, lw=0.5, zorder=2,
                clip_on=False)
        ax.text(A.LAB_X, mid, f"{lab}\n{v:.0f}%", ha="left", va="center",   # Gb totals overflow the 32 mm canvas
                fontsize=S.annot, color=col, linespacing=1.25, zorder=4)
        top -= v

    shared = 100 * (tot["WGD-linked"] + tot["ccPG-linked"]) / T

    ax.set_ylim(0, 100); ax.set_xlim(*A.XLIM); ax.set_xticks([])
    ax.set_ylabel("% gained DNA", fontsize=S.label, labelpad=2)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=S.tick)
    ax.spines["left"].set_position(("data", A.SPINE_X))
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=2.0, pad=1.5)
    fig.subplots_adjust(left=0.40, right=0.99, top=1 - 3.0 / 44.0, bottom=3.1 / 44.0)
    if save:
        PANELS.mkdir(parents=True, exist_ok=True)
        stem = "alt_gain_composition_weighted"
        fig.savefig(PANELS / f"{stem}.png", dpi=300)
        fig.savefig(PANELS / f"{stem}.svg")
        fig.savefig(PANELS / f"{stem}.pdf")
        sys.path.insert(0, str(R / "paper/figures"))
        from _svgfix import longhand_fonts
        longhand_fonts(PANELS / f"{stem}.svg")
    print(f"gains={n_gains:,}  " + "  ".join(
        f"{l}={100*tot[l]/T:.1f}%" for l, _ in parts) + f"  shared={shared:.1f}%")
    return fig


if __name__ == "__main__":
    make()
