"""loss_landscape figure (supp), panel c — the schematic of panel a, run on a real genome.

One double-WGD sample (dc537fcf, Stomach.AdenoCA), ONE REAL SEGMENT per copy-number state, drawn as
the allele tree Tau actually solved. Every copy the two doublings should have produced and the data does not
show is marked with a cross at the doubling that should have made it. Panel a says how the count is
defined; this says what it looks like on real data.

Each panel is one locus at named coordinates, not a genome-wide composite (design note:
"segments might be better"). An earlier version used pooled clusters, which carry ~10x the mutations
(N = 1,613 against 158 here) and put the gains ON the doubling lines rather than scattering around
them. Single segments are noisier and that is visible; what they buy is a locus a reader can look up.
The header gives the state, the coordinates and N.

Reading a tree: a lineage that BRANCHES at a doubling line was doubled; one that passes STRAIGHT THROUGH
it was doubled and the copy is gone, so it carries a cross. The header gives the observed state and the
TOTAL number of missing copies.

The total is the only quantity quoted. WHICH doubling a given cross is charged to is imposed by the
recursion rather than measured — the second doubling's copy budget already contains the first's loss
(cb2 = cb1 + d1), so the per-epoch split L1/L2 is tied by an identity and is not identifiable. The
crosses are drawn at a doubling because they have to be drawn somewhere; read the count, not the epoch.

Gains far from either doubling (> ASSIGN_R in molecular time) are non-WGD gains, not doublings. They are
excluded from the accounting and marked in amber — a non-WGD gain MASKS loss (it restores copy number
without a doubling), so leaving it in would undercount.

SELECTION — one panel per copy-number state, the HIGHEST-N segment of each, taken from the 275 timed
segments in the cache. Selection is on power alone and never looks at the loss outcome, so a state
coming out at 0 missing is a genuine null case rather than a chosen contrast. The row is capped at
NCOL states, ordered by N, because a single row of 183 mm cannot carry more and stay legible. The
quantitative claim lives in panel b; this panel shows what the measurement looks like.

SELECTION IS BY AGREEMENT WITH THE POOLED CLUSTER. Loss is `copies present - gains assigned to that
doubling`, and a gain counts only within +/-ASSIGN_R of a doubling. At single-segment power gain times
scatter, fewer land inside the window, and spurious missing copies are booked -- picking the
highest-N segment of each state gave (4,4) 2 missing where the pooled cluster reads 0. The cache
scores every segment by `dist_to_cluster` (mean |segment gain time - pooled gain time| against the
largest pooled unit of its state) and the panel draws the closest, which recovers the pooled reading.
The rule never looks at the crosses. The quantitative claim remains panel b, on the pooled accounting
in `multi_wgd_loss_allpcawg.tsv`.

Two things a reader should know about the counts:

  * The drawn solution is not a median. For an underdetermined segment `_pick_draw_for_tree` scores the
    polytope solutions by proximity of their events to the doubling times and draws the best-fitting
    one. Since loss = max(0, copies present - gains assigned to that doubling), the most WGD-aligned
    solution credits the most gains to doublings and so reports the FEWEST missing copies. **The
    crosses are a lower bound — the most WGD-parsimonious reading.** Determined segments have one
    solution and no choice is made.
  * The +/-0.20 assignment window can push the other way. If two gains on one allele both fall inside a
    doubling's window, only one can be that doubling; the extra inflates the copy budget going into the
    next doubling, and the shortfall there is charged as loss. The two effects do not cancel in any
    principled way.

Table-driven: paper/figures/data/loss_landscape/loss_example.json, written by paper/figures/data/loss_landscape/build/build_loss_example.py (that script
is the only one that touches the genome pkl and route_metadata). make() -> fig + PNG/SVG/PDF.
"""
import json
import sys
from pathlib import Path

from paper.config import panels_dir_of, cache_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from tau.trees import build_allele_trees, plot_route, sort_nodes
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, DOUBLE, S
_apply_style(compact=True)

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
SRC = cache_dir_of(__file__) / "loss_example.json"

MAJ, MIN_ = "#2f4858", "#7a8b99"          # same allele colours as panel a
LOST, FOCAL = "#b0402f", "#cd9f2c"
W1_C, W2_C = "#0b6979", "#7a3b8f"         # doubling 1 / doubling 2
ASSIGN_R = 0.20                           # a gain counts as a doubling only within this of a WGD
NCOL, NROW = 8, 1                         # single row
DASH_MIN, DASH_LOST = (2.6, 1.4), (0, (1.6, 1.2))


def _pick(pool, n):
    """The segment of each CN state that best AGREES with its pooled cluster's history.

    `dist_to_cluster` = mean |segment gain time - pooled gain time|, computed in the cache builder
    against the largest pooled unit of the same state. Picking on power alone instead gave (4,4)
    2 missing copies where the pooled cluster reads 0 -- single-segment scatter pushes gains outside
    the +/-ASSIGN_R window and books them as loss. Selection is blind to the loss outcome; states
    with no pooled reference (dist inf) are skipped.
    """
    out, seen = [], set()
    for s in sorted(pool, key=lambda s: s.get("dist_to_cluster", float("inf"))):
        if not np.isfinite(s.get("dist_to_cluster", float("inf"))):
            continue
        st = (s["major"], s["minor"])
        if st in seen:
            continue
        seen.add(st); out.append(s)
        if len(out) >= n:
            break
    return out


def _assign(t, W1, W2):
    d1, d2 = abs(t - W1), abs(t - W2)
    if min(d1, d2) > ASSIGN_R:
        return "focal"
    return "W1" if d1 <= d2 else "W2"


def _crosses(ax, root, sign, W1, W2):
    """Per allele: at each doubling, every copy present should split. Copies present minus splits
    actually taken = missing copies; a cross goes on the pass-through lineages first.

    PARSIMONIOUS: a lineage charged as lost at one doubling is gone by the next, so it is removed from
    the copies present and cannot be charged again. Without this a (3,0) segment books its absent minor
    allele twice — once for failing to double at W1, again for still being absent at W2 — when a single
    loss before W1 explains the whole observation. The surviving copies entering doubling k are then
    exactly 2 x (the copies that successfully doubled at k-1).

    Returns (L1, L2) for this allele.
    """
    nodes = sort_nodes(root)                   # .index was set by plot_route
    for n in nodes:
        n._tag = "root" if n.parent is None else _assign(n.time, W1, W2)
    out = {}
    gone = set()          # PARSIMONY: a lineage charged as lost is gone, so it cannot be lost again
    for W, wtag in ((W1, "W1"), (W2, "W2")):
        if wtag == "W1":
            below = [n for n in nodes if n._tag == "root" or (n._tag == "focal" and n.time < W1)]
        else:
            below = [n for n in nodes if n._tag in ("root", "W1")
                     or (n._tag == "focal" and n.time < W2)]
        below = [n for n in below if n not in gone]
        d = sum(1 for n in nodes if n._tag == wtag)
        loss = max(0, len(below) - d)
        passth = sorted([n for n in below if not any(c._tag == wtag for c in n.branches)],
                        key=lambda n: n.index)
        targets = passth + [n for n in sorted(below, key=lambda n: n.index) if n not in passth]
        for n in targets[:loss]:
            gone.add(n)
            bx = n.index; gx = bx + sign * 0.30
            ax.plot([bx, gx], [W, W], color=LOST, lw=0.6, ls=DASH_LOST, zorder=5)
            ax.plot([gx, gx], [W, W + 0.05], color=LOST, lw=0.6, ls=DASH_LOST, zorder=5)
            ax.plot(gx, W + 0.065, marker="x", color=LOST, ms=2.6, mew=0.8, zorder=6)
        out[wtag] = loss
    return out["W1"], out["W2"]


def _gain_dots(ax, root, W1, W2):
    for n in sort_nodes(root):
        if n.parent is None:
            continue
        ax.plot(n.index, n.time, marker="o", ms=1.9, zorder=8, mec="white", mew=0.3,
                color={"W1": W1_C, "W2": W2_C, "focal": FOCAL}[_assign(n.time, W1, W2)])


def make(save=True):
    d = json.loads(SRC.read_text())
    W1, W2 = d["wgd"]
    segs = _pick(d["segments"], NCOL * NROW)

    fig, axes = plt.subplots(NROW, NCOL, figsize=(DOUBLE, 62 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.30})
    flat = np.atleast_1d(axes).ravel()
    for ax, s in zip(flat, segs):
        maj, mn = build_allele_trees(s["events"], s["major"], s["minor"])
        plot_route(ax, maj, mn, major_color=MAJ, minor_color=MIN_,
                   linewidth_scale=0.45, spacing=0.5, minor_dashes=DASH_MIN)
        for w, c in ((W1, W1_C), (W2, W2_C)):
            ax.axhline(w, color=c, lw=0.5, ls=":", zorder=1)
        a1, a2 = _crosses(ax, maj, -1, W1, W2)
        b1, b2 = _crosses(ax, mn, +1, W1, W2)
        _gain_dots(ax, maj, W1, W2); _gain_dots(ax, mn, W1, W2)
        L = a1 + b1 + a2 + b2                  # pooled — the per-epoch split is not identifiable
        xs = [x for ln in ax.get_lines() for x in ln.get_xdata()]
        if xs:
            ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
        ax.set_ylim(-0.03, 1.06); ax.set_xticks([])
        loc = s.get("seg_id", "")
        if ":" in loc:                       # chr1:179.5-196.1 Mb, short enough for a narrow column
            ch, rr = loc.split(":"); a_, b_ = (int(v) / 1e6 for v in rr.split("-"))
            loc = f"chr{ch}:{a_:.0f}\u2013{b_:.0f} Mb"
        ax.set_title(f"({s['major']},{s['minor']})\n{loc}\nN={s['ess']:,.0f}   {L} missing",
                     fontsize=S.tiny, pad=3, linespacing=1.35,
                     color=(LOST if L else "#55606d"))
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
    for ax in flat[len(segs):]:
        ax.set_visible(False)
    flat[0].set_yticks([0, 0.5, 1]); flat[0].set_ylabel("molecular time", fontsize=S.label)

    h = [mlines.Line2D([], [], color=MAJ, lw=1.0, label="major allele"),
         mlines.Line2D([], [], color=MIN_, lw=1.0, ls=(0, DASH_MIN), label="minor allele"),
         mlines.Line2D([], [], color=LOST, lw=0.7, ls=DASH_LOST, marker="x", ms=2.6,
                       label="missing copy"),
         mlines.Line2D([], [], color=W1_C, lw=0.5, ls=":", label=f"doubling 1 (t={W1:.2f})"),
         mlines.Line2D([], [], color=W2_C, lw=0.5, ls=":", label=f"doubling 2 (t={W2:.2f})"),
         mlines.Line2D([], [], color=FOCAL, marker="o", ms=2.4, ls="none", label="non-WGD gain")]
    fig.legend(handles=h, fontsize=S.legend, loc="lower center", bbox_to_anchor=(0.5, 0.002),
               ncol=6, frameon=False, handlelength=1.5, columnspacing=1.1)
    fig.suptitle(f"{d['sample']}   \u00b7   Stomach.AdenoCA   \u00b7   double WGD "
                 f"(t = {W1:.2f}, {W2:.2f})", fontsize=S.annot, y=0.985, color="#273246")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.755, bottom=0.135)
    if save:
        _save_panel(fig, PANELS, "c_loss_example")
    print(f"c_loss_example: one segment per CN state, {len(segs)} states, {d['sample'][:8]} "
          f"(doublings {W1:.3f}, {W2:.3f}); {len(d['segments'])} segments available")
    for s in segs:
        print(f"    ({s['major']},{s['minor']})  N={s['ess']:5.0f}  {s['seg_id']}")
    return fig


if __name__ == "__main__":
    make()
