"""Dimension arrows whose tips sit exactly on the interval they measure.

Shared by panels a and c so the two draw the same object. Both mark intervals of molecular time
against a tree, and a reader moving between them should not have to work out whether two differently
built arrows mean the same thing.

Why not `annotate(..., arrowstyle="<->")`
----------------------------------------
A FancyArrowPatch shortens its own shaft to make room for its heads, so once the span drops below
roughly twice the head length the arrow visibly stops short of what it is measuring — at a 0.017-wide
interval it rendered as a stub. Setting `shrinkA/B=0` does not help; that controls padding, not head
allowance. Drawing the shaft at full span and letting annotate decorate it fixes the span but then
leaves the shaft poking out past the shrunken heads.

Here the heads are explicit triangles with their tips ON the endpoints, the head length is capped at
40% of the span so two heads can never exceed it, and the shaft is inset behind them. Exact at any
length, by construction rather than by tuning.

Proportions are specified in POINTS and converted through transData, so an arrow keeps its shape
between axes that span different amounts of data. That means the axes limits must already be set when
`dim_arrow` is called.
"""
HEAD_PT = 2.6          # head length
HALF_W_PT = 1.15       # ...and half-width, at full size


def dims(ax):
    """(x, y) data units per typographic point, for this axes' current limits."""
    fig = ax.get_figure()
    d = fig.dpi / 72.0
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(0, 0), (d, d)])
    return x1 - x0, y1 - y0


def dim_arrow(ax, x, lo, hi, colour, lw=0.55, zorder=3):
    """A vertical double-headed arrow spanning exactly [lo, hi] at x."""
    ux, uy = dims(ax)
    hl = min(HEAD_PT * uy, 0.40 * (hi - lo))          # head length, in data units
    hw = HALF_W_PT * ux * (hl / (HEAD_PT * uy))       # ...width shrinks with it, keeping the shape
    ax.plot([x, x], [lo + 0.8 * hl, hi - 0.8 * hl], color=colour, lw=lw, zorder=zorder,
            solid_capstyle="butt")
    for y, s in ((hi, 1.0), (lo, -1.0)):
        ax.fill([x, x - hw, x + hw], [y, y - s * hl, y - s * hl], color=colour, lw=0, zorder=zorder)
