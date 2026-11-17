"""benchmarking_segment figure (supp), panel b — estimated allele-trees (determined (3,1) vs underdetermined (3,2)) on real
simulated segments, with the TRUE gain time marked as a short blue bar sitting directly on the matching
branch: the vertical offset between the black branch and its blue bar IS that gain's timing error, read
locally without guide lines across the panel. Underdetermined trees show the polytope solution set only
(no median line -> no implied point estimate). Object-based (reads sim genome pkls + truth).
"""
import sys, gzip, pickle, json, copy
from pathlib import Path
import numpy as np
from paper.config import panels_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE, DOUBLE, HALF
_apply_style(compact=True)
from tau.utils import pick_best_key, order_t
from tau.trees import build_allele_trees, plot_route
from tau.tree_polytope import _events_for_route

from paper.config import OBJECTS_SIM as SIM   # the two simulated genomes drawn here
PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
EST, TRUTH, ERR = "#1a1a1a", "#0072B2", "#b0402f"   # estimate=black, truth=Okabe–Ito blue, error=red
SPACING = 0.32                                      # plot_route node spacing (major tree then minor)
CASES = [("SIM_1003", "4:50000000-191154276", "Determined (3,1)", False),
         ("SIM_1132", "9:49000000-141213431", "Underdetermined (3,2)", True)]



def _norm(sid):
    c, rng = sid.split(":"); a, b = rng.split("-"); return (c.replace("chr", ""), int(a), int(b))


def _coord_label(sid):
    c, rng = sid.split(":"); a, b = rng.split("-")
    return f"chr{c}:{int(a)//10**6}–{int(b)//10**6} Mb"


def _load(sample, seg_id):
    with gzip.open(SIM / sample / f"{sample}.genome.pkl.gz", "rb") as f:
        g = pickle.load(f)
    seg = next(s for s in g.segments if s.seg_id == seg_id)
    truth = json.load(open(SIM / sample / f"{sample}.truth.json"))
    tm = {}
    for ts in truth.get("segments", []):
        tv = ts.get("t_values", [])
        if len(tv) >= 2:
            tm[_norm(ts["seg_id"])] = np.sort(np.cumsum(tv)[:-1])
    return seg, tm.get(_norm(seg_id))


def _tree(ax, cn, ev_tpl, et, color, alpha, lw, z):
    evs = copy.deepcopy(ev_tpl)
    for j, e in enumerate(evs):
        e["time"] = float(np.clip(et[j], 0, 1))
    mj, mn = build_allele_trees(evs, cn[0], cn[1])
    plot_route(ax, mj, mn, major_color=color, minor_color=color, linewidth_scale=lw,
               spacing=SPACING, alpha=alpha, zorder=z, minor_dashes=False)


def _branch_xranges(ax, cn, ev_tpl, times):
    """x-extent of each gain's horizontal branch (drawn invisibly, measured, then removed)."""
    n0 = len(ax.lines)
    _tree(ax, cn, ev_tpl, times, EST, 0.0, 1.0, 0)
    segs = []
    for ln in ax.lines[n0:]:
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(x) == 2 and abs(y[0] - y[1]) < 1e-9:
            segs.append((float(y[0]), float(min(x)), float(max(x))))
    for ln in ax.lines[n0:][::-1]:
        ln.remove()
    return sorted(segs)


def _draw_case(ax, samp, sid, blur):
    seg, true_t = _load(samp, sid)
    k = pick_best_key(seg); cn = (int(seg.major_cn), int(seg.minor_cn))
    ev_tpl = _events_for_route(k)
    draws = [d for d in seg.timing_result[k].get("draws", []) if d.get("boot_id", 0) == 0]
    est_ev = [np.sort(np.cumsum(order_t(d["t"]))[:-1]) for d in draws]
    med = np.sort(np.median(np.array(est_ev), 0))
    if blur:                                    # polytope solution set only
        a = max(0.04, min(0.24, 5.0 / len(est_ev)))
        for et in est_ev:
            _tree(ax, cn, ev_tpl, et, EST, a, 0.9, 4)
    else:
        _tree(ax, cn, ev_tpl, med, EST, 1.0, 1.5, 5)
    tt = np.sort(true_t)
    xr = _branch_xranges(ax, cn, ev_tpl, med)
    xd = [x for ln in ax.get_lines() for x in ln.get_xdata()]
    xmin, xmax = min(xd), max(xd)
    ax.set_xlim(xmin - 0.52, xmax + 0.52)       # tight margins -> wide trees
    # per-gain error: determined = |estimate − truth|; underdetermined = distance to the solution set
    n = min(len(tt), len(med))
    if blur:
        arr = np.array(est_ev); lo, hi = arr.min(0), arr.max(0)
        err = np.maximum.reduce([np.zeros(n), lo[:n] - tt[:n], tt[:n] - hi[:n]])
    else:
        err = np.abs(med[:n] - tt[:n])

    # vertical tree elements, used to keep each label out of the lines it would otherwise cross
    verts = []
    for ln in ax.get_lines():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(x) == 2 and abs(x[0] - x[1]) < 1e-9:
            verts.append((float(x[0]), float(min(y)), float(max(y))))

    # the minor tree is laid out to the RIGHT of the major tree (node index (i+n_major)*spacing),
    # so a gain's allele follows from where its branch sits; label each on its own outer side,
    # tucked into the free gap NEXT TO that branch rather than flush at a shared margin
    boundary = (cn[0] + 0.5) * SPACING
    labels = []                                 # (artist, y, side) for the "error" callout
    for i, t in enumerate(tt):                  # true gain = short bar on the matching branch
        if i < len(xr):
            _, x0, x1 = xr[i]
            pad = 0.12 * max(x1 - x0, 0.4)
            ax.plot([x0 - pad, x1 + pad], [t, t], color=TRUTH, lw=1.5, solid_capstyle="butt",
                    zorder=10)
            if i < n:
                e = float(err[i])
                txt = "0 (truth\nwithin bounds)" if e < 0.005 else f"{e:.3f}"
                cross = [vx for vx, y0, y1 in verts if y0 - 0.01 <= t <= y1 + 0.01]
                if 0.5 * (x0 + x1) > boundary:                      # minor allele -> right side
                    blk = [vx for vx in cross if vx > x1 + 1e-6]
                    lx = (min(blk) - 0.08 if blk and min(blk) - x1 < 0.42 else x1 + pad + 0.08)
                    ha = "left" if not (blk and min(blk) - x1 < 0.42) else "right"
                else:                                               # major allele -> left side
                    blk = [vx for vx in cross if vx < x0 - 1e-6]
                    room = x0 - (max(blk) if blk else xmin - 0.52)
                    lx = x0 - pad - 0.08 if room >= 0.42 else (max(blk) - 0.08 if blk else x0 - 0.1)
                    ha = "right"
                art = ax.text(lx, t, txt, ha=ha, va="center", fontsize=S.tiny, color=ERR,
                              clip_on=False, zorder=11)
                labels.append((art, t, ha))
    return dict(maxerr=float(err[:n].max()) if n else 0.0, labels=labels)


def make(save=True):
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE, 50 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.42})
    res = []
    for ax, (samp, sid, title, blur) in zip(axes, CASES):
        r = _draw_case(ax, samp, sid, blur); res.append(r)
        ax.text(0.5, -0.045, f"max gain error = {r['maxerr']:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=S.annot, fontweight="bold", color="0.15")
        ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([0, 0.5, 1])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        if ax is not axes[0]:                   # only the left panel carries the time axis
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title(f"{title}\n{samp} · {_coord_label(sid)}", fontsize=S.annot, color=EST,
                     fontweight="bold", pad=4, linespacing=1.5)
    axes[0].set_ylabel("molecular time (0 early, 1 late)", fontsize=S.label)
    # name the quantity once (in place of a legend key): point at the MIDPOINT OF THE LEFT EDGE of
    # the topmost left-side error value in the underdetermined panel, filling the space beside it
    ax1 = axes[1]
    left = [(art, y) for art, y, ha in res[1]["labels"] if ha == "right"]
    if left:
        art = max(left, key=lambda p: p[1])[0]
        fig.canvas.draw()                      # need a renderer for the text extent
        bb = art.get_window_extent(fig.canvas.get_renderer())
        inv = ax1.transData.inverted()
        x_l, y_c = inv.transform((bb.x0, 0.5 * (bb.y0 + bb.y1)))
        ax1.annotate("error", xy=(x_l, y_c), xytext=(-16, 11), textcoords="offset points",
                     ha="right", va="bottom", fontsize=S.tiny, color=ERR, zorder=12,
                     annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", color=ERR, lw=0.7, shrinkA=1, shrinkB=2))
    h = [mlines.Line2D([], [], color=EST, lw=1.5, label="determined gain"),
         mlines.Line2D([], [], color=EST, lw=2.6, alpha=0.28, label="underdetermined solutions"),
         mlines.Line2D([], [], color=TRUTH, lw=1.5, label="true gain time")]
    fig.legend(handles=h, fontsize=S.legend, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, frameon=False, handlelength=1.8, columnspacing=1.2)
    # reserve room at the bottom for the "max gain error" captions and the legend, both of which sit
    # outside the axes (the panel is saved at its exact canvas size, so nothing may overflow it)
    # the right margin also has to hold the "0 (truth within bounds)" annotation, which sits outside
    # the right-hand axes
    fig.subplots_adjust(left=0.12, right=0.74, top=0.86, bottom=0.16)
    if save:
        _save_panel(fig, PANELS, 'b_schematic')
    return fig


if __name__ == "__main__":
    make(); print(f"benchmarking_segment/b: written -> {PANELS}")
