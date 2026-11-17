"""benchmarking_segment figure (supp) — standalone timing accuracy by copy-number state, at >200 mutations.

Per-GAIN error against a single random solution from the band, NOT the max-gain-per-segment error the
earlier version used and NOT distance-to-set. Both of those measure something other than accuracy here:

  max-gain per segment   correlates with gains-per-segment at rho = +0.95, and gains-per-segment is
                         major + minor - 1 — so its state ranking is close to a restatement of total
                         copy number. It put (2,2) best (0.017) and (4,4) worst (0.220).
  distance-to-set        scores 0 whenever the truth falls inside the solution band, so the widest,
                         least informative polytopes score best. It puts (2,2) and (4,4) both at 0.000.

A per-gain single solution removes both: it does not scale with gain count, and a wide band earns no
credit for containing the truth. Under it the ordering inverts — (2,2) becomes the WORST state (0.128)
and the determined LOH states the best ((2,0) 0.017). What the ranking tracks is polytope WIDTH
(rho = +0.83), i.e. identifiability, with a weaker residual dependence on gain count (rho = +0.67).

WHY THE (2,2) RESULT IS NOT A PIPELINE WEAKNESS — read this before quoting the number
------------------------------------------------------------------------------------
This panel scores each segment ALONE, from its own multiplicity spectrum with no genome-wide
constraint. That is the raw signal, and it is the right thing for a segment-level benchmark, but it is
not what the pipeline delivers for the wide-polytope states. A (2,2) segment in a WGD sample has both
gains AT the doubling, so anchoring to the detected WGD collapses its polytope to essentially a point.
In PCAWG that is nearly all of them:

    (2,2)  17,097 timed segments (10.0% of all)   95.6% in WGD samples
    (3,2)   4,399                ( 2.6%)          95.3%
    (4,2)   1,899                ( 1.1%)          95.3%
    (4,4)     150                ( 0.1%)          99.3%
    all minor_cn >= 2                (15.0%)      95.5%

    (2,2) with NO anchor available (non-WGD sample): 750 segments = 0.44% of the timed cohort,
    median ESS 17 — low-powered in any case.

So this panel is best read as the motivation for the anchoring step rather than as a limitation:
states with wide standalone polytopes are precisely the ones the WGD constraint resolves. Legend
entries are daggered where >90% of that state's real timed segments sit in WGD samples and are
therefore anchored in practice.

Table-driven from bench_seg_gains (per-gain) + tau_segments/tau_events (anchoring fractions).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, S
_apply_style(compact=True)
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
MC_CUT, MIN_N = 200, 50
XMAX, XMARK = 0.5, 0.10          # shared with panels d and e
ANCHOR_CUT = 0.90                # dagger a state when this share of its real segments are in WGD samples


def _ecdf(a):
    a = np.sort(np.asarray(a, float)); a = a[~np.isnan(a)]
    return a, np.arange(1, len(a) + 1) / max(len(a), 1)


def _anchored_fraction():
    """Share of each state's real timed segments that sit in a WGD sample (so are anchored)."""
    sg = pd.read_csv(R / "paper/data_tables/tau_segments.tsv", sep="\t",
                     usecols=["sample", "major_cn", "minor_cn", "is_timed"]).dropna(
                     subset=["major_cn", "minor_cn"])
    sm = pd.read_csv(R / "paper/data_tables/tau_samples.tsv", sep="\t")
    ev = pd.read_csv(R / "paper/data_tables/tau_events.tsv", sep="\t")
    keep = set(sm[sm.is_representative_aliquot.astype(bool)]["sample"])
    wgd = set(ev.loc[ev.classification == "WGD", "sample"])
    t = sg[(sg.is_timed.astype(bool)) & (sg["sample"].isin(keep))].copy()
    t["cn"] = t.major_cn.astype(int).astype(str) + "_" + t.minor_cn.astype(int).astype(str)
    return t.assign(w=t["sample"].isin(wgd)).groupby("cn").w.mean()


def make(save=True):
    g = load_bench("seg_gains")
    g = g[g.mut_count > MC_CUT].copy()
    g["cn"] = g.major.astype(int).astype(str) + "_" + g.minor.astype(int).astype(str)
    # point solution: a RANDOM time inside the segment's solution band, not the midpoint — the
    # midpoint is a central summary the method never computes, and it understates how much a wide
    # polytope actually costs. Determined gains have lo == hi so the draw is exact for them.
    # CAVEAT: bench_seg_gains stores only [lo, hi], not the draws, so this is uniform-in-interval
    # rather than a true draw from the polytope density. Midpoint gives nearly the same answer
    # (2,2) 0.148 vs 0.128, (4,4) 0.076 vs 0.080, (3,2) 0.070 vs 0.063 — so nothing here turns on it.
    rng = np.random.default_rng(0)
    g["err"] = (g.time_lo + (g.time_hi - g.time_lo) * rng.random(len(g)) - g.true_time).abs()
    g["width"] = g.time_hi - g.time_lo

    med = g.groupby("cn").agg(size=("err", "size"), median=("err", "median"), width=("width", "median"))
    med = med[med["size"] >= MIN_N].sort_values("median", ascending=False)
    anch = _anchored_fraction()

    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=-0.15, vmax=1.05)          # keep both ends off the extremes of the ramp
    fig, ax = plt.subplots(figsize=(89.0 * MM, 58.0 * MM))
    for i, (cn, row) in enumerate(med.iterrows()):
        x, y = _ecdf(g.loc[g.cn == cn, "err"].values)
        maj, mino = cn.split("_")
        dag = "†" if anch.get(cn, 0.0) >= ANCHOR_CUT else " "
        ax.plot(x, y, color=cmap(norm(i / max(len(med) - 1, 1))), lw=1.2, zorder=4,
                label=f"({maj},{mino}){dag} {row['median']:.3f}")

    ax.axvline(XMARK, ls=":", c="#999", lw=0.8)
    ax.text(XMARK + 0.008, 0.03, f"{XMARK:.2f}", fontsize=S.tiny, color="#999")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.0)
    ax.set_xlabel("per-gain timing error, single solution")
    ax.set_ylabel("fraction of gains below error")
    ax.set_title("Standalone per-gain timing error\nby copy-number state (>200 mut)",
                 fontweight="bold", pad=3, fontsize=S.annot)
    # the dagger key goes in the legend TITLE, not a fig.text footnote: with 22 states the legend runs
    # the full height of the canvas and a bottom-anchored note lands on top of it
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
              handlelength=1.0, labelspacing=0.16, handletextpad=0.3, fontsize=S.tiny,
              title="state   median\n† ≥90% anchored to a WGD", title_fontsize=S.legend)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(lw=0.3, color="0.93"); ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.1236, right=0.5843, top=0.8707, bottom=0.1638)
    if save:
        _save_panel(fig, PANELS, "alt_error_by_cn_pergain")
    print(med.assign(anchored=[anch.get(c, np.nan) for c in med.index]).round(3).to_string())
    return fig


if __name__ == "__main__":
    make(); print(f"benchmarking_segment/f: written -> {PANELS}")
