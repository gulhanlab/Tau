"""punctuated figure (main), panel a — how every clonal gain in PCAWG partitions into WGD-linked, ccPG-linked and unlinked.

This panel DEFINES the three categories the rest of the figure and the landscape panels (Fig. 5a,
Supp. Fig. 8) are drawn from, so it uses exactly their partition — same berths, same weighting, same gain
set — rather than a separate calculation that would nearly agree:

  WGD-linked    a solution places the gain within 0.20 of a detected WGD
  ccPG-linked   within 0.10 of a detected ccPG, and NOT WGD-linked
  unlinked      neither

The two berths differ because the events do: a WGD is a genome-wide landmark whose per-segment gains
scatter around it, while a ccPG is a tighter, smaller event. Precedence goes to WGD so the three
categories partition the gains exactly once each.

Classification is by CONTAINMENT: a gain is linked if ANY of its solutions reaches the event. That is
the project convention for polytope output — all solutions are equally valid, so the question is whether
the solution space contains the pattern, not what fraction of it does. Each gain is then counted once.

Both choices move the headline, so the sensitivity is worth stating rather than burying (shared % =
WGD + ccPG):

    berths (WGD/ccPG)   WGD    ccPG   unlinked   shared
    0.05 / 0.05        43.9%   1.9%     54.2%     45.8%
    0.10 / 0.05        60.4%   1.8%     37.8%     62.2%
    0.10 / 0.10        60.4%   2.6%     37.0%     63.0%
    0.20 / 0.10        75.4%   2.5%     22.1%     77.9%   <- used here, = the landscape berths

The default matches the landscape so Fig 3a and the landscape panels describe the same partition. Note
the draft's "57%" predates this: it came from a 0.05 berth with mutation-count weighting, and no
combination of the two conventions reproduces it. Pick a row and make the text match.

ccPG-linked is small under every setting (~2%). The "shared across chromosomes" figure is overwhelmingly
WGD; ccPGs matter for WHERE and WHEN they land, not for how much of the genome they move.

Restricted, like the landscape, to autosomal gains with 2 <= major_cn <= 7.

Table-driven: tau_gains + tau_events. `make()` returns the figure and writes PNG+SVG next to this script.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
matplotlib.rcParams.update({"svg.fonttype": "none", "font.size": 7, "font.family": "sans-serif",
                            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
                            "figure.facecolor": "none", "axes.facecolor": "none",
                            "savefig.facecolor": "none", "pdf.fonttype": 42})
from paper.data_tables.tables import load_gains, load_events
from paper.config import cache_dir_of, panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

PANELS = panels_dir_of(__file__)
WGD_BERTH, PGD_BERTH = 0.20, 0.10      # = the landscape berths; see the table in the docstring
CHROMS = [str(i) for i in range(1, 23)]
WGD_C, CCPG_C, UNL_C = "#0b6979", "#cd9f2c", "#b8b8b8"
CACHE = cache_dir_of(__file__) / "gain_composition.tsv"
MM = 1 / 25.4


def compute(force=False):
    """-> (n_gains, weighted counts per category). Cached; the join is over ~3.5M gain rows."""
    if CACHE.exists() and not force:
        t = pd.read_csv(CACHE, sep="\t")
        return int(t.n_gains.iloc[0]), t.set_index("category").weight.to_dict()

    ev = load_events()
    WGDT = {s: np.sort(g.event_time_frac.values)
            for s, g in ev[ev.classification == "WGD"].groupby("sample")}
    PGDT = {s: np.sort(g.event_time_frac.values)
            for s, g in ev[ev.class_strict == "PGD"].groupby("sample")}
    g = load_gains(best_route_only=True,
                   columns=["sample", "seg_id", "chrom", "major_cn", "gain_time_frac", "gain_index"])
    g = g[(g.major_cn >= 2) & (g.major_cn <= 7) & g.chrom.isin(CHROMS)]

    parts = []
    for s, gs in g.groupby("sample", sort=False):
        tf = gs.gain_time_frac.to_numpy(float)
        dw = (np.min(np.abs(tf[:, None] - WGDT[s][None, :]), axis=1)
              if s in WGDT else np.full(len(gs), np.inf))
        dp = (np.min(np.abs(tf[:, None] - PGDT[s][None, :]), axis=1)
              if s in PGDT else np.full(len(gs), np.inf))
        parts.append(pd.DataFrame({"sample": s, "seg_id": gs.seg_id.values,
                                   "gi": gs.gain_index.values, "dw": dw, "dp": dp}))
    # min over solutions = does ANY solution reach the event
    a = pd.concat(parts).groupby(["sample", "seg_id", "gi"]).agg(dw=("dw", "min"), dp=("dp", "min"))
    n_gains = len(a)
    wl = a.dw <= WGD_BERTH
    po = (a.dp <= PGD_BERTH) & ~wl
    tot = {"WGD-linked": float(wl.sum()), "ccPG-linked": float(po.sum()),
           "unlinked": float((~(wl | po)).sum())}
    print(f"  {n_gains:,} gains")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"category": list(tot), "weight": list(tot.values()),
                  "n_gains": n_gains}).to_csv(CACHE, sep="\t", index=False)
    return n_gains, tot


# PORTRAIT. One vertical 100% column, 32 x 44 mm, as placed in the assembled figure: y axis at the
# left ("% gains"), the column, and the three category labels (with a leader each) to its right. The
# shared-across-chromosomes bracket and the "n = ... gains" title of the earlier version are gone; the
# total is quoted in the caption instead.
W, H = 32 * MM, 44 * MM
# x units are arbitrary but the SPACING matters, so they are set together: with XLIM below the plot
# box is ~7 mm per unit, which makes the column ~5.5 mm wide and leaves the label block just enough
# room not to run off the canvas.
XLIM = (-0.55, 2.95)
BAR_X, BAR_W = 0.0, 0.80      # the column
LAB_X = 0.55                  # left edge of the text block beside it
SPINE_X = -0.45               # the y axis


def make(save=True):
    n_gains, tot = compute()
    T = sum(tot.values())
    parts = [("WGD-linked", WGD_C), ("ccPG-linked", CCPG_C), ("unlinked", UNL_C)]

    fig, ax = plt.subplots(figsize=(W, H))
    # drawn top-down so WGD-linked, the majority and the subject, sits at the top of the column
    # rather than being read last
    top = 100.0
    for lab, col in parts:
        v = 100 * tot[lab] / T
        ax.bar(BAR_X, v, bottom=top - v, width=BAR_W, color=col, edgecolor="white",
               linewidth=0.8, zorder=3)
        mid = top - v / 2
        # a 2.5% band is ~1.5 mm tall and cannot hold a label, so every category is named OUTSIDE
        # the column and tied to its band by a leader. Doing it for all three keeps one convention.
        ax.plot([BAR_X + BAR_W / 2, LAB_X - 0.04], [mid, mid], color=col, lw=0.5, zorder=2,
                clip_on=False)
        ax.text(LAB_X, mid, f"{lab}\n{v:.0f}% ({tot[lab]/1000:,.0f}k)", ha="left", va="center",
                fontsize=S.annot, color=col, linespacing=1.25, zorder=4)
        top -= v

    shared = 100 * (tot["WGD-linked"] + tot["ccPG-linked"]) / T   # WGD + ccPG = everything NOT unlinked

    ax.set_ylim(0, 100); ax.set_xlim(*XLIM); ax.set_xticks([])
    ax.set_ylabel("% gains", fontsize=S.label, labelpad=2)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=S.tick)
    ax.spines["left"].set_position(("data", SPINE_X))
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=2.0, pad=1.5)
    # no legend: each band is named at its own leader, so a key would repeat all three
    fig.subplots_adjust(left=0.40, right=0.99, top=1 - 3.0 / 44.0, bottom=3.1 / 44.0)   # axis 37.9 mm tall (her figure)
    if save:
        PANELS.mkdir(parents=True, exist_ok=True)
        fig.savefig(PANELS / "a_gain_composition.png", dpi=300)
        fig.savefig(PANELS / "a_gain_composition.svg")
        sys.path.insert(0, str(R / "paper/figures"))
        from _svgfix import longhand_fonts
        longhand_fonts(PANELS / "a_gain_composition.svg")
    print(f"gains={n_gains:,}  " + "  ".join(
        f"{l}={100*tot[l]/T:.1f}% ({tot[l]:,.0f})" for l, _ in parts) + f"  shared={shared:.1f}%")
    return fig


if __name__ == "__main__":
    make()
