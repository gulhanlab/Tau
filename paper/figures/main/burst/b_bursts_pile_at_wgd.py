"""burst figure (main), panel b — bursts pile at WGD (PER-SOLUTION, weighted) vs a WGD-shuffle null. For every bursting polytope
solution of every high-CN segment in a WGD sample, δ = burst centroid − nearest WGD event; each
burst-bearing segment contributes total weight 1 split across ITS bursting solutions (containment: a
segment that CAN burst counts once, its location following the actual bursting solutions). Observed vs a
null that permutes which sample's WGD-event set each segment is scored against. Table-driven:
fig6_perdraw.tsv.gz + tau_events (via _wgd.py). Mirrors dev panel_b_perdraw_wgd_profile.py. make() -> fig.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "svg.fonttype": "none"})
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
from paper.config import cache_dir_of, panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import _wgd
from _wgd import WGD_WIN, MM

CACHE = cache_dir_of(__file__); PANELS = panels_dir_of(__file__)
N_PERM = 1000
TEAL, TEALD, GREY, GREYL, TXT = "#0d8a93", "#0b6979", "#717c91", "#cacdd9", "#273246"
EDGES = np.arange(-1.0, 1.0001, 0.04); CENT = (EDGES[:-1] + EDGES[1:]) / 2


def make(save=True):
    events = _wgd.wgd_event_sets()
    bu = pd.read_csv(CACHE / "fig6_perdraw.tsv.gz", sep="\t", dtype={"chrom": str})
    bu = bu[bu["sample"].isin(events.keys())].copy()
    # each burst-bearing SEGMENT weight 1, split across its bursting solutions
    nb = bu.groupby(["sample", "seg_id"])["draw_id"].transform("size")
    bu["w"] = 1.0 / nb
    nseg = bu.groupby(["sample", "seg_id"]).ngroups

    pool = np.array(list(events.keys())); pool_ev = [events[s] for s in pool]
    P = len(pool); spos = {s: i for i, s in enumerate(pool)}
    cen = bu["centroid"].values; wt = bu["w"].values; sidx = bu["sample"].map(spos).values
    n = len(bu)
    delta = np.empty((n, P))
    for p in range(P):
        diffs = cen[:, None] - pool_ev[p][None, :]
        delta[:, p] = diffs[np.arange(n), np.argmin(np.abs(diffs), axis=1)]

    def whist(assign):
        return np.histogram(delta[np.arange(n), assign], bins=EDGES, weights=wt)[0]

    rng = np.random.default_rng(7)
    obs = whist(sidx)
    null = np.array([whist(rng.permutation(P)[sidx]) for _ in range(N_PERM)])
    nm, ns = null.mean(0), null.std(0)

    fig, ax = plt.subplots(figsize=(43 * MM, 40 * MM))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.88, bottom=0.18)
    ax.axvspan(-WGD_WIN, WGD_WIN, color="#cee7ed", alpha=0.55, zorder=0)
    ax.axvline(0, color=TEALD, lw=1.0, ls="--", zorder=2)
    ax.fill_between(CENT, np.clip(nm - 2 * ns, 0, None), nm + 2 * ns, color=GREYL, alpha=0.8,
                    step="mid", zorder=1, label="WGD-shuffle null")
    ax.plot(CENT, nm, color=GREY, lw=0.8, drawstyle="steps-mid", zorder=3)
    ax.fill_between(CENT, 0, obs, step="mid", color=TEAL, alpha=0.85, zorder=4,
                    edgecolor=TEALD, linewidth=0.4, label="Observed bursts")
    ax.text(0, ax.get_ylim()[1] * 0.98, " WGD", color=TEALD, fontsize=S.label, va="top", ha="left", fontweight="bold")
    ax.set_xlim(-1.0, 1.0); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("centroid − WGD time", fontsize=S.label, color=TXT)
    ax.set_ylabel("Bursts (weighted)", fontsize=S.label, color=TXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=S.legend, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
              handlelength=1.2, handletextpad=0.4, labelspacing=0.25)
    inwin = np.abs(CENT) < WGD_WIN
    print(f"c: {nseg:,} burst segments; |δ|<0.10 obs={obs[inwin].sum():.0f} vs null={nm[inwin].sum():.0f} "
          f"({obs[inwin].sum()/max(nm[inwin].sum(),1):.1f}x)", flush=True)
    if save:
        fig.savefig(PANELS / "b_bursts_pile_at_wgd.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "b_bursts_pile_at_wgd.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make(); print("c_bursts_pile_at_wgd: PNG+SVG written")
