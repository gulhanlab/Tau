"""Shared figure style for every panel script (main and supplementary).

ONE type ladder, named by ROLE, defined here and nowhere else:

    S.title   8.0   panel/axes titles
    S.label   7.0   axis labels
    S.tick    6.5   tick labels
    S.legend  6.5   legend entries
    S.annot   6.5   in-plot annotations (n=, %, gene names at normal size)
    S.tiny    5.0   deliberately small marks (chromosome numbers in genome tracks, dense gene labels)

Scripts use `fontsize=S.tick` etc. -- never a number -- so the whole figure set is re-sized from this file
(`SIZES`, or `apply(scale=...)` for one figure). The ladder matches the assembled figures
(6.5 / 7 / 8 pt). Import and call at the top of every panel script:

    from paper.figures._style import apply, S, MM, SINGLE, DOUBLE
    apply()

Widths follow the journal column grid -- SINGLE 89 mm, DOUBLE 183 mm -- and a panel exports at its final
size so that placing it in the Illustrator artboard needs no rescaling (rescaling after export is what
makes fonts disagree across a figure). Text is kept as text in the SVG (svg.fonttype "none") and PDFs
embed TrueType (fonttype 42), so labels stay editable and "select same font size" works in Illustrator.
"""
from types import SimpleNamespace
from pathlib import Path
MM = 1 / 25.4

SIZES = dict(title=8.0, label=7.0, tick=6.5, legend=6.5, annot=6.5, tiny=5.0)
S = SimpleNamespace(**SIZES)

SINGLE = 89 * MM          # single column
DOUBLE = 183 * MM         # double column / full width
HALF = 43 * MM            # two panels within a single column

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    # transparent surround AND transparent plot area. Set as rcParams rather than
    # savefig(transparent=True) on purpose: transparent=True forces EVERY axes patch clear, which
    # would wipe the deliberate dark background of the calibration heatmaps. As a default, an
    # explicit ax.set_facecolor(...) in a panel still wins.
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "font.size": SIZES["annot"],
    "axes.titlesize": SIZES["title"],
    "axes.labelsize": SIZES["label"],
    "xtick.labelsize": SIZES["tick"],
    "ytick.labelsize": SIZES["tick"],
    "legend.fontsize": SIZES["legend"],
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply(compact=False, scale=1.0):
    """Push the ladder into matplotlib's rcParams and (re)build `S` at the requested scale.

    compact=True is kept for callers that used the old dense variant; it now just scales the ladder by
    0.92 (the +1 pt bump everywhere else still applies). scale multiplies every role.
    """
    import matplotlib
    f = scale * (0.92 if compact else 1.0)
    for k, v in SIZES.items():
        setattr(S, k, round(v * f, 2))
    rc = {**RC,
          "font.size": S.annot, "axes.titlesize": S.title, "axes.labelsize": S.label,
          "xtick.labelsize": S.tick, "ytick.labelsize": S.tick, "legend.fontsize": S.legend}
    if compact:
        rc.update({"axes.linewidth": 0.4, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                   "xtick.major.size": 1.6, "ytick.major.size": 1.6, "lines.linewidth": 1.0})
    matplotlib.rcParams.update(rc)
    _install_lint_hook()
    return rc


def _svg_longhand_fonts(path):
    """Illustrator ignores matplotlib's `font:` shorthand and defaults to 12 pt — see paper/figures/_svgfix.py."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))   # _svgfix.py is a sibling
    from _svgfix import longhand_fonts
    return longhand_fonts(path)


def save(fig, panels_dir, stem, dpi=300, pdf=True):
    """Write a panel as PNG (preview), SVG (Illustrator, editable text) and PDF (vector fallback).

    Never pass bbox_inches="tight" here — it crops to content and discards the authored canvas size,
    which is what keeps panels on the column grid.
    """
    from pathlib import Path
    d = Path(panels_dir)
    fig.savefig(d / f"{stem}.png", dpi=dpi)
    fig.savefig(d / f"{stem}.svg")
    _svg_longhand_fonts(d / f"{stem}.svg")
    if pdf:
        fig.savefig(d / f"{stem}.pdf")
    return d / f"{stem}.svg"


# ───────────────────────── text-overlap lint and minimal de-collision ─────────────────────────
def text_overlaps(fig, tol_pt=0.4):
    """All pairs of visible text artists whose rendered boxes intersect (each shrunk by tol_pt).

    Returns [(text_a, text_b, overlap_pt2)], worst first. Tick labels, axis labels, titles, legend
    entries and annotations are all Text artists, so all are covered. Ignores empty strings.
    """
    import matplotlib.text as mtext
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    px = fig.dpi / 72.0 * tol_pt
    items = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            b = t.get_window_extent(ren)
        except Exception:
            continue
        if b.width <= 0 or b.height <= 0:
            continue
        items.append((t, b.shrunk(1, 1).expanded(1, 1).translated(0, 0)))
    out = []
    for i in range(len(items)):
        ta, a = items[i]
        for j in range(i + 1, len(items)):
            tb, b = items[j]
            x0, y0 = max(a.x0, b.x0) + px, max(a.y0, b.y0) + px
            x1, y1 = min(a.x1, b.x1) - px, min(a.y1, b.y1) - px
            if x1 > x0 and y1 > y0:
                out.append((ta, tb, (x1 - x0) * (y1 - y0) / (fig.dpi / 72.0) ** 2))
    out.sort(key=lambda r: -r[2])
    return out


def lint(fig, name="", tol_pt=0.4, limit=12):
    """Print the overlaps of `fig` (if any) in a one-line-per-pair form; returns their number."""
    pairs = text_overlaps(fig, tol_pt)
    for ta, tb, area in pairs[:limit]:
        sa, sb = ta.get_text().replace("\n", "⏎")[:40], tb.get_text().replace("\n", "⏎")[:40]
        print(f"  [lint] {name}: '{sa}' overlaps '{sb}' ({area:.0f} pt²)")
    if len(pairs) > limit:
        print(f"  [lint] {name}: … {len(pairs) - limit} more")
    return len(pairs)


def decollide(texts, axis="x", gap_mm=1.0):
    """Push colliding NEIGHBOURS apart by the minimum needed (half each way) along one axis, in place.

    `texts` in reading order (left→right for axis='x', bottom→top for 'y'). Non-colliding labels are
    not moved, so this never spreads text out; it only resolves actual overlaps. Positions are
    adjusted in the artist's own transform coordinates.
    """
    import numpy as np
    if len(texts) < 2:
        return
    fig = texts[0].figure
    fig.canvas.draw(); ren = fig.canvas.get_renderer()
    gap = gap_mm * MM * fig.dpi
    for a, b in zip(texts[:-1], texts[1:]):
        ea, eb = a.get_window_extent(ren), b.get_window_extent(ren)
        over = (ea.x1 + gap - eb.x0) if axis == "x" else (ea.y1 + gap - eb.y0)
        if over <= 0:
            continue
        for t, sign in ((a, -0.5), (b, +0.5)):
            tr = t.get_transform()
            px = tr.transform(t.get_position())
            px = px + np.array([sign * over, 0.0]) if axis == "x" else px + np.array([0.0, sign * over])
            t.set_position(tr.inverted().transform(px))
        fig.canvas.draw()


def _install_lint_hook():
    """When TAU_FIG_LINT is set, every Figure.savefig first prints the figure's text overlaps."""
    import os
    if not os.environ.get("TAU_FIG_LINT"):
        return
    import matplotlib.figure as mfig
    if getattr(mfig.Figure, "_tau_lint_hooked", False):
        return
    orig = mfig.Figure.savefig

    def savefig(self, fname, *a, **kw):
        try:
            lint(self, name=str(fname).rsplit("/", 1)[-1])
        except Exception as e:      # never let the lint break a render
            print(f"  [lint] skipped ({e})")
        return orig(self, fname, *a, **kw)
    mfig.Figure.savefig = savefig
    mfig.Figure._tau_lint_hooked = True
