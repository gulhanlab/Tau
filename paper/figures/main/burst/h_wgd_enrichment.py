"""burst figure (main), panel h — WGD enrichment (flipped dot plot): fraction of EXCESS bursts whose centroid sits within
+-0.10 of ANY of their sample's WGD events, observed vs a sample-level WGD-shuffle null, for the pooled
set plus 8 high-powered oncogenes. Table-driven: fig6_bursts.tsv + tau_events (via _wgd.py). Mirrors the
flipped variant in dev figure_burst_wgd_lead.py. make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.labelsize": 7,
                     "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "svg.fonttype": "none"})
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import _wgd
from _wgd import WGD_COL, NULL_COL, NULL_TICK, TXT, MM, HIGHPOWER, OG_MID

PANELS = panels_dir_of(__file__)


def make(save=True):
    bursts = _wgd.load_bursts(); events = _wgd.wgd_event_sets()
    b = _wgd.excess_wgd_bursts(bursts, events)
    near, sidx, P, _, _ = _wgd.near_matrix(b, events)
    own = near[np.arange(len(b)), sidx]
    chrom, start, end = b["chrom"].values, b["start_bp"].values, b["end_bp"].values

    rows = [{"label": "Excess bursts", **_wgd.enrich(np.ones(len(b), bool), near, own, sidx, P), "all": True}]
    for g in HIGHPOWER:
        c, m = OG_MID[g]
        r = _wgd.enrich((chrom == c) & (start <= m) & (end >= m), near, own, sidx, P)
        if r:
            rows.append({"label": g, **r, "all": False})
    res = pd.DataFrame(rows)
    genes = res[~res["all"]].sort_values("obs")
    o2 = pd.concat([genes, res[res["all"]]], ignore_index=True)
    xs = np.arange(len(o2))

    fig, ax = plt.subplots(figsize=(44 * MM, 65 * MM))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.94, bottom=0.30)
    ax.errorbar(xs, o2["mu"], yerr=2 * o2["sd"], fmt="none", ecolor=NULL_COL, elinewidth=3, alpha=0.9, zorder=1)
    ax.scatter(xs, o2["mu"], color=NULL_TICK, s=16, marker="_", linewidths=1.2, zorder=2)
    ax.scatter(xs, o2["obs"], s=28, color=WGD_COL, zorder=3, edgecolors="white", linewidths=0.5)
    for i, r in o2.iterrows():
        ax.text(i, r["obs"] + 0.03, _wgd.stars(r["p"]), ha="center", va="bottom", fontsize=S.annot, color=TXT)
    ax.set_xticks(xs); ax.set_xticklabels(o2["label"], rotation=40, ha="right", fontsize=S.tick)
    for lab in ax.get_xticklabels():
        if lab.get_text() == "Excess bursts":
            lab.set_fontweight("bold")
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("frac. within ±0.10 WGD", fontsize=S.label, color=TXT)
    ax.set_title("WGD enrichment", fontsize=S.title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    if save:
        fig.savefig(PANELS / "h_wgd_enrichment.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "h_wgd_enrichment.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make(); print("e_wgd_enrichment: PNG+SVG written")
