"""route_complexity figure (supp), panel a — # routes versus # PCAWG segments per copy-number state.

Tau enumerates every distinct order in which the gains of an allele-specific state (major, minor)
could have occurred. That count is a property of the state alone: (2,1) admits a single history,
(7,7) admits 77,775. Each is a different temporal hypothesis, so the inference has to carry all of
them and then discriminate between them.

Each cell is split on the diagonal and the two halves are coloured on independent log scales:

    upper-left   number of distinct amplification routes  (blue)
    lower-right  number of PCAWG segments in that state   (orange)

The two ramps run opposite to one another — the states with the most histories are the rarest — which
is the situation a method has to handle: difficulty concentrated in states that are individually
uncommon but collectively substantial. Exact counts are printed in each half.

Table-driven from ref_route_topology (distinct route_key per state) + tau_segments (segment counts).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_route_topology, load_segments

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
# ONE colormap and ONE normalisation for both halves: routes top out at 77,775 and segments at 84,604,
# so a shared log scale is honest and makes the halves directly comparable — a cell whose upper-left is
# brighter than its lower-right has more possible histories than observed instances. Dark -> bright, so
# brighter = more.
CMAP = "magma"
EMPTY = "#f4f4f4"


def _grids():
    rt = load_route_topology()
    n_rt = rt.groupby(["major_cn", "minor_cn"]).route_key.nunique().reset_index(name="n_routes")
    majors = sorted(n_rt.major_cn.unique())
    minors = sorted(n_rt.minor_cn.unique())

    seg = load_segments()
    n_sg = seg.groupby(["major_cn", "minor_cn"]).size().reset_index(name="n_seg")

    routes = np.full((len(minors), len(majors)), np.nan)
    segs = np.full((len(minors), len(majors)), np.nan)
    for _, r in n_rt.iterrows():
        routes[minors.index(r.minor_cn), majors.index(r.major_cn)] = r.n_routes
    for _, r in n_sg.iterrows():
        if r.major_cn in majors and r.minor_cn in minors:
            segs[minors.index(r.minor_cn), majors.index(r.major_cn)] = r.n_seg
    segs[np.isnan(routes)] = np.nan            # states with no route are not part of the grid
    return routes, segs, majors, minors, seg


# triangle vertices for a unit cell centred at (x, y); split on the leading diagonal
def _upper_left(x, y):
    return [(x - .5, y - .5), (x - .5, y + .5), (x + .5, y + .5)]


def _lower_right(x, y):
    return [(x - .5, y - .5), (x + .5, y - .5), (x + .5, y + .5)]


def _txt_colour(cmap, norm, v):
    """White text on the dark (low) end, dark text once the fill brightens."""
    return "white" if norm(max(v, 1)) < 0.58 else "#111111"


def make(save=True):
    routes, segs, majors, minors, seg = _grids()
    vmax = max(np.nanmax(routes), np.nanmax(segs))
    nR = nS = LogNorm(vmin=1, vmax=vmax)         # shared scale -> the two halves are comparable
    cR = cS = plt.get_cmap(CMAP)

    fig, ax = plt.subplots(figsize=(SINGLE, 70 * MM))
    for i in range(len(minors)):
        for j in range(len(majors)):
            if np.isnan(routes[i, j]):
                ax.add_patch(Polygon([(j - .5, i - .5), (j + .5, i - .5),
                                      (j + .5, i + .5), (j - .5, i + .5)],
                                     facecolor=EMPTY, edgecolor="white", lw=0.5))
                continue
            r = routes[i, j]
            s = 0 if np.isnan(segs[i, j]) else segs[i, j]
            ax.add_patch(Polygon(_upper_left(j, i), facecolor=cR(nR(max(r, 1))),
                                 edgecolor="white", lw=0.5))
            ax.add_patch(Polygon(_lower_right(j, i), facecolor=cS(nS(max(s, 1))),
                                 edgecolor="white", lw=0.5))
            # centroid of each triangle, nudged apart so the two numbers never touch
            ax.text(j - 0.17, i + 0.20, f"{int(r):,}", ha="center", va="center",
                    fontsize=S.tiny, color=_txt_colour(cR, nR, r), fontweight="bold")
            ax.text(j + 0.17, i - 0.20, f"{int(s):,}", ha="center", va="center",
                    fontsize=S.tiny, color=_txt_colour(cS, nS, s))

    ax.set_xlim(-0.5, len(majors) - 0.5); ax.set_ylim(-0.5, len(minors) - 0.5)
    ax.set_xticks(range(len(majors))); ax.set_xticklabels(majors)
    ax.set_yticks(range(len(minors))); ax.set_yticklabels(minors)
    ax.set_xlabel("major copy number"); ax.set_ylabel("minor copy number")
    ax.set_title("# routes versus # PCAWG segments per copy-number state",
                 fontweight="bold", pad=4)
    ax.tick_params(length=0); ax.set_aspect("auto")
    for s_ in ("top", "right", "left", "bottom"):
        ax.spines[s_].set_visible(False)

    # key: one split cell drawn in the empty upper-left of the grid
    kx, ky = 0.35, len(minors) - 1.25
    ax.add_patch(Polygon(_upper_left(kx, ky), facecolor=cR(0.55), edgecolor="white", lw=0.5))
    ax.add_patch(Polygon(_lower_right(kx, ky), facecolor=cS(0.80), edgecolor="white", lw=0.5))
    ax.text(kx + 0.62, ky + 0.22, "routes", fontsize=S.tiny, color="0.30", va="center", ha="left")
    ax.text(kx + 0.62, ky - 0.22, "segments", fontsize=S.tiny, color="0.30", va="center", ha="left")

    sm = ScalarMappable(norm=nR, cmap=cR); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes([0.845, 0.16, 0.030, 0.70]))
    cb.set_label("count (routes or segments)", fontsize=S.tick); cb.ax.tick_params(labelsize=S.tick)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.yaxis.set_minor_formatter(plt.NullFormatter())

    fig.subplots_adjust(left=0.105, right=0.815, top=0.92, bottom=0.115)
    if save:
        _save_panel(fig, PANELS, "a_routes_per_state")
    return fig


if __name__ == "__main__":
    make(); print(f"supp/route_complexity/a: written -> {PANELS}")
