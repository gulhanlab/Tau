"""examples figure (main) — shared renderer for the two per-sample overview panels (a: Stomach, b: Eso).

The panel IS `tau.overview.overview_figure` (the per-sample two-level overview the `tau run` pipeline
emits: individual segment timing | pooled allele trees | pooled timing histogram), drawn on the same
objects the panels are drawn on:

    raw genome        <data>/objects/pcawg/<tt>/<id>/<id>.genome.pkl.gz   (paper.config.OBJECTS_PCAWG)
    pooled genome     paper/data_tables/per_sample/<id>/<id>.clustered_genome.pkl.gz   (recluster_sample.py)
    events            paper/data_tables/per_sample/<id>/<id>.tau_events.tsv / .tau_merged_events.tsv

`dress()` then adds the titles and labels of the assembled figure:
the sample title, "Segment-level" and "Sample-level route
clustering" section titles, "Genomic loci" / "Routes" axis titles, a "Per CN state clusters" key title,
event-time labels reduced to the bare number (right of the histogram), and the bottom line-style
legend and left-panel event labels removed. Nothing numeric is touched.

OBJECT-BASED (not table-driven): needs the pixi env (tau importable) and the pickles above.
"""
import gzip
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"]})
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(R))   # examples/main/figures/paper/repo
from paper.config import panels_dir_of, OBJECTS_PCAWG, PER_SAMPLE  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from tau.overview import overview_figure, MIN_ESS  # noqa: E402
import numpy as np  # noqa: E402

CLOCK = OBJECTS_PCAWG          # raw clock-timed genomes of the example samples
PER = PER_SAMPLE
PANELS = panels_dir_of(__file__)
# How many trees to draw: `overview_figure` keeps the top-n_trees pooled units by member-segment count
# plus one unit per high-CN state (major 5-7) not already among them, so the drawn total can exceed
# n_trees. MAX_TREES is the TOTAL we want (decision 2026-09-18: 16, the number kept in the
# assembled Figure 2); `_n_trees_for()` finds the largest n_trees whose total stays within it.
# 0250e6f7: all 16 qualifying units; dc537fcf: the 16 largest by genome span incl. the high-CN rescues.
MAX_TREES = 16
# Canvas: the assembled figure is 180 mm wide with a and b stacked (~55 mm each). Fixed here so the panel
# drops in at 100% (fonts at their nominal 6.5-8 pt); the overview renderer's default would grow the
# canvas with the number of trees (265 mm for 16 trees) and then crop it.
W_MM = 180.0
TITLE_MM, BOTTOM_MM = 4.0, 9.0      # above the tracks (sample title) / below the axes (chromosome numbers + "Genomic loci")

FS_TITLE, FS_SECTION, FS_AXIS, FS_TIME = S.title, S.label, S.label, S.tick


def _load(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def resolve(short):
    """-> (full sample id, tumour-type dir) for a uuid prefix."""
    sdir = next(iter(CLOCK.glob(f"*/{short}*")))
    return sdir.name, sdir.parent.name


def _n_trees_for(cg, max_total, min_ess=MIN_ESS):
    """Largest n_trees for which overview_figure's selection (top-n by genome span + one rescued unit per
    high-CN state) draws <= max_total trees. Mirrors the selection in tau.overview.overview_figure."""
    from tau.utils import pick_best_key
    ess = lambda s: float(np.sum(s.N_counts))
    units = [s for s in cg.segments if pick_best_key(s) is not None and ess(s) >= min_ess]
    span = lambda s: float(getattr(s, "total_length", s.end - s.start))
    units.sort(key=lambda s: -span(s))                     # overview_figure ranks by genome span
    for n in range(min(max_total, len(units)), 0, -1):
        top = units[:n]; ids = {id(s) for s in top}
        states = {(int(s.major_cn), int(s.minor_cn)) for s in top}; hi = {}
        for s in sorted(units, key=lambda x: -ess(x)):
            cn = (int(s.major_cn), int(s.minor_cn))
            if 5 <= cn[0] <= 7 and id(s) not in ids and cn not in hi and cn not in states:
                hi[cn] = s
        if n + len(hi) <= max_total:
            return n
    return 1


def _n_track_rows(cg, max_per_state, min_ess=MIN_ESS):
    """Number of CN-state rows the track will draw (states with a qualifying unit, major <= 7)."""
    from tau.utils import pick_best_key
    ess = lambda s: float(np.sum(s.N_counts))
    return len({(int(s.major_cn), int(s.minor_cn)) for s in cg.segments
                if pick_best_key(s) is not None and ess(s) >= min_ess and int(s.major_cn) <= 7})


def render(short, **kw):
    full, tt = resolve(short)
    g = _load(CLOCK / tt / full / f"{full}.genome.pkl.gz")
    cg = _load(PER / full / f"{full}.clustered_genome.pkl.gz")
    ev = pd.read_csv(PER / full / f"{full}.tau_events.tsv", sep="\t")
    mev = pd.read_csv(PER / full / f"{full}.tau_merged_events.tsv", sep="\t")
    kw.setdefault("n_trees", _n_trees_for(cg, MAX_TREES))
    # canvas height from the track: the overview axes get a fixed 30 mm, the CN track above it needs
    # 0.06 x that per CN-state row (+2% gap), plus the title above and the x-axis text below
    n_rows = _n_track_rows(cg, kw.get("track_max_per_state", 2))
    ax_mm = 30.0
    track_mm = ax_mm * (0.02 + 0.06 * n_rows)
    h_mm = BOTTOM_MM + ax_mm + max(track_mm + TITLE_MM + 3.0, 31.0)   # right side needs 31 mm above the trees
    kw.setdefault("figsize", (W_MM / 25.4, h_mm / 25.4))
    kw.setdefault("tree_lw", (0.18, 0.55))       # 0.36-1.1 pt strokes: 16 trees on 180 mm, branches must not touch
    # block widths in mm as measured on the assembled figure: overview 8.6-94, trees 97-171, then the
    # histogram to the right edge (the axes block spans margins[0]..margins[1] of the 180 mm canvas)
    kw.setdefault("width_ratios", (83.0, 3.5, 70.0, 6.0))
    kw.setdefault("track_label_fontsize", 5.5)   # CN-state labels on the cluster track
    kw.setdefault("track_min_span", 10e6)        # extra (non-tree) track units must be >= 10 Mb to show
    kw.setdefault("margins", (8.6 / W_MM, 1 - 9.0 / W_MM, BOTTOM_MM / h_mm, (BOTTOM_MM + ax_mm) / h_mm))   # 9 mm right: histogram + boxed event times
    fig, info = overview_figure(g, cg, ev, mev, verbose=True, **kw)
    return fig, info, full, tt


def dress(fig, title):
    """The titles and labels of the assembled figure, in matplotlib. Works on the axes `overview_figure` built:
    axes[0] = overview, the sharey tree axes, the last sharey axis = pooled histogram."""
    axes = fig.get_axes()
    ax_ov = axes[0]
    shared = [a for a in axes[1:] if a.get_shared_y_axes().joined(a, ax_ov)]
    trees, ax_h = shared[:-1], shared[-1]

    for lg in list(fig.legends):                       # bottom line-style legend
        lg.remove()
    for t in list(ax_ov.texts):                        # "WGD 0.83" labels right of the overview
        if re.match(r"^(WGD|ccPG|chr\w+)\s", t.get_text()):
            t.remove()
    for t in ax_h.texts:                               # "merged WGD 0.80" -> "0.80" in a box
        m = re.search(r"([01]\.\d+)\s*$", t.get_text())
        if m:
            t.set_text(m.group(1)); t.set_fontsize(FS_TIME)

    p_ov = ax_ov.get_position()
    p_t0, p_t1 = trees[0].get_position(), trees[-1].get_position()
    # the CN-state track is an inset axes of the overview (child, not in fig.axes): title goes above it
    fig.canvas.draw()
    insets = [c for c in ax_ov.get_children() if isinstance(c, plt.Axes)]
    y_top = max([c.get_window_extent().transformed(fig.transFigure.inverted()).y1 for c in insets]
                + [p_ov.y1])
    fh_mm = fig.get_size_inches()[1] * 25.4
    fig.text(p_ov.x0 - 0.05, 1 - 0.5 / fh_mm, title, fontsize=FS_TITLE, ha="left", va="top")
    fig.text((p_ov.x0 + p_ov.x1) / 2, y_top + 1.0 / fh_mm, "Segment-level", fontsize=FS_SECTION, fontweight="bold",
             ha="center", va="bottom")
    # 8 mm above the tree axes (clear of the tree titles), right-aligned to the tree block so it never
    # runs off the canvas; the CN key sits above it (tau.overview places it 18 mm above the trees)
    fig.text(p_t1.x1, p_t0.y1 + 8.0 / fh_mm, "Sample-level route clustering", fontsize=FS_SECTION,
             fontweight="bold", ha="right", va="bottom")
    # chromosome numbers are text artists just below the axes, so the axis title is placed under them
    fig.text((p_ov.x0 + p_ov.x1) / 2, p_ov.y0 - 5.0 / fh_mm, "Genomic loci", fontsize=FS_AXIS, ha="center", va="top")
    ax_ov.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax_ov.tick_params(axis="y", labelsize=S.tick, length=2, width=0.5)
    for a in trees + [ax_h]:                           # sharey: keep the trees/histogram tick-free
        a.tick_params(axis="y", left=False, labelleft=False)
    fig.text((p_t0.x0 + p_t1.x1) / 2, p_t0.y0 - 1.5 / fh_mm, "Routes", fontsize=FS_AXIS, ha="center", va="top")
    ax_ov.set_ylabel("Molecular time", fontsize=FS_AXIS)
    # the cluster fill key ("Cluster 1/2/3") gets its title; it is the small axes right of the CN key
    for a in axes:
        labs = [t.get_text() for t in a.texts]
        if labs and all(l.startswith("Cluster ") for l in labs):
            p = a.get_position()
            fig.text(p.x0, p.y1 + 0.5 / fh_mm, "Per CN state clusters", fontsize=FS_SECTION, ha="left", va="bottom")
            for t in a.texts:
                t.set_text(t.get_text().replace("Cluster ", "C")); t.set_fontsize(FS_SECTION - 1)
    return fig


def make(short, stem, title, save=True, **kw):
    fig, info, full, tt = render(short, **kw)
    dress(fig, f"{title}: {full}")
    if save:
        fig.savefig(PANELS / f"{stem}.png", dpi=300)          # exact canvas, no crop: 180 mm wide
        fig.savefig(PANELS / f"{stem}.svg")
        print(f"  saved {stem} -> {PANELS}")
    return fig
