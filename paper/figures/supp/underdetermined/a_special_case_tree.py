"""underdetermined figure (supp), panel a — one underdetermined route drawn three ways.

An underdetermined route admits a whole polytope of equally valid timings. This panel shows what that
means for a single well-powered unit — merged_4_4_0 from the double-WGD stomach tumour (route 4_4.4,
ESS 425, 51 solutions, 6 gains) — and what a SPECIAL-CASE solution is:

    all solutions   every polytope point overlaid; gain 3 alone spans 0.287-0.822
    random draw     one arbitrary point: gains scattered, nothing co-timed
    special case    the point where gains coincide on multiple branches — here 2 gains at 0.055 and
                    4 at 0.822, i.e. two doublings, with every gain belonging to one of them

The special case is selected by SIMULTANEITY alone (the intervening t-intervals collapse to zero); no
event time enters the choice. The detected WGDs are drawn afterwards for reference, so their agreement
with the special case is an observation, not a constraint.

Object-based: reads the sample's clustered genome pickle.
"""
import copy
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
from paper.config import panels_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE, DOUBLE
_apply_style(compact=True)
from tau.utils import pick_best_key, order_t
from tau.trees import build_allele_trees, plot_route
from tau.tree_polytope import _events_for_route

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
SAMPLE = "dc537fcf-d910-4c4b-8af9-e7da429f2633"
UNIT = "merged_4_4_0"
WGD = [0.13, 0.83]
BLOCKS = [(1, 2), (3, 4)]        # the two doublings of route 4_4.4, in event-index terms
EST, WGD_C = "#1a1a1a", "#0b6979"
RNG_DRAW = 3                      # index of the illustrative random solution


def _load():
    # cohort run, NOT the per-sample bundle: panel b can only be built from the cohort (the bundles
    # cover 13 samples), and the two vintages differ for this unit — the bundle's newer multiplicity
    # clustering gives ESS 236 / 57 solutions, the cohort ESS 425 / 51. Using one source keeps the two
    # panels talking about the same object, at the cost of not matching the tree drawn in Fig 2a.
    import glob
    from paper.config import OBJECTS_PCAWG
    p = glob.glob(str(OBJECTS_PCAWG / "*" / SAMPLE / f"{SAMPLE}.clustered_genome.pkl.gz"))[0]
    with gzip.open(p, "rb") as f:
        cg = pickle.load(f)
    seg = next(s for s in cg.segments if str(s.seg_id) == UNIT)
    key = pick_best_key(seg)
    draws = [d for d in seg.timing_result[key].get("draws", []) if d.get("boot_id", 0) == 0]
    ts = np.array([order_t(d["t"]) for d in draws])
    return seg, key, ts, np.cumsum(ts, axis=1)[:, :-1]


def _draw_tree(ax, key, cn, et, colour, lw, alpha=1.0, z=4):
    evs = copy.deepcopy(_events_for_route(key))
    for j, ev in enumerate(evs):
        ev["time"] = float(np.clip(et[j], 0, 1))
    mj, mn = build_allele_trees(evs, cn[0], cn[1])
    plot_route(ax, mj, mn, major_color=colour, minor_color=colour, linewidth_scale=lw,
               spacing=0.32, alpha=alpha, zorder=z, minor_dashes=False)


def make(save=True):
    seg, key, ts, ev = _load()
    cn = (int(seg.major_cn), int(seg.minor_cn))
    # special case: interior intervals of both blocks collapse
    score = sum(ts[:, s:s + L - 1].sum(axis=1) for s, L in BLOCKS)
    i_spec = int(np.argmin(score))

    fig, axes = plt.subplots(1, 3, figsize=(SINGLE, 52 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    panels = [("all solutions", None), ("random solution", RNG_DRAW), ("special case", i_spec)]
    for ax, (title, idx) in zip(axes, panels):
        if idx is None:
            a = max(0.04, min(0.22, 5.0 / len(ev)))
            for row in ev:
                _draw_tree(ax, key, cn, row, EST, 0.8, alpha=a, z=3)
        else:
            _draw_tree(ax, key, cn, ev[idx], EST, 1.4)
        for w in WGD:
            ax.axhline(w, color=WGD_C, ls=":", lw=0.7, zorder=8)
        ax.set_ylim(0, 1); ax.set_xticks([])
        ax.set_title(title, fontsize=S.title, pad=3, fontweight="bold" if idx == i_spec else "normal")
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        if ax is not axes[0]:
            ax.spines["left"].set_visible(False); ax.tick_params(left=False, labelleft=False)
        xd = [x for ln in ax.get_lines() for x in ln.get_xdata()]
        if xd:
            ax.set_xlim(min(xd) - 0.4, max(xd) + 0.4)
    axes[0].set_yticks([0, 0.5, 1]); axes[0].set_ylabel("molecular time")

    # annotate the special case with the times its gains collapse onto
    g = ev[i_spec]
    for t in (g[0], g[-1]):        # inside the axes, or the label runs off the canvas
        axes[2].text(0.04, t + 0.035, f"{t:.2f}", transform=axes[2].get_yaxis_transform(),
                     ha="left", va="bottom", fontsize=S.tiny, color=EST, zorder=12)

    h = [mlines.Line2D([], [], color=EST, lw=1.4, label="one solution"),
         mlines.Line2D([], [], color=EST, lw=2.4, alpha=0.25, label="all solutions"),
         mlines.Line2D([], [], color=WGD_C, lw=0.7, ls=":", label="detected WGD")]
    fig.legend(handles=h, fontsize=S.legend, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               ncol=3, frameon=False, handlelength=1.6, columnspacing=1.2)
    fig.suptitle(f"{cn[0]},{cn[1]} unit, {len(ev)} equally valid solutions", fontsize=S.title, y=0.985)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.845, bottom=0.135)
    if save:
        _save_panel(fig, PANELS, "a_special_case_tree")
    return fig


if __name__ == "__main__":
    make(); print(f"supp/underdetermined/a: written -> {PANELS}")
