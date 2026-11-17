"""loss_landscape figure (supp), panel a — how implied post-WGD loss is counted.

A whole-genome doubling splits every lineage that is present. So the copy number a segment SHOULD carry
after the doubling is fixed by the copy number it carried before, and any shortfall against what is
actually observed is a copy that was lost afterwards. That is the whole measurement — it needs no loss
model, only the doubling and the observed state.

Left: one doubling. A diploid segment doubles to (2,2). The observed state is (2,1), so one copy of the
minor allele is missing: implied loss 1.

Right: two doublings. The expected state is (4,4). The observed (3,2) is three copies short.

Only inferred loss is counted. A minor allele the data show is absent BEFORE the doubling was never
doubled, so nothing is assumed about it and nothing is charged.

RENDERING — these are real Tau allele trees, not hand-drawn shapes. The two histories are written as
route event lists, handed to `tau.trees.build_allele_trees`, and drawn with `tau.trees.plot_route`, the
same renderer panel c uses on real data. So the schematic and the worked example are visually the same
object, and a reader who learns to read one can read the other. The crosses are placed by the same rule
panel c applies: at each doubling, every lineage already present that does NOT branch there is a copy
the doubling should have produced and the data does not show.

Pure schematic — no data is read. The states and times are illustrative.

NOTE the total is the only count quoted. An earlier version of this panel attributed the right-hand
example's three missing copies to specific epochs ("2 in epoch 1, 1 in epoch 2"). That split is not
identifiable: the second doubling's copy budget already contains the first doubling's loss
(cb2 = cb1 + d1), so L1 and L2 are tied by an identity. Same convention as panel c.
"""
import sys
from pathlib import Path

import numpy as np
from paper.config import panels_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from tau.trees import build_allele_trees, plot_route, sort_nodes
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, DOUBLE, S
_apply_style(compact=True)

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
# identical palette to panel c, so the schematic and the worked example read as one figure
MAJ, MIN_ = "#2f4858", "#7a8b99"
LOST = "#b0402f"
W1_C, W2_C = "#0b6979", "#7a3b8f"
DASH_MIN, DASH_LOST = (2.6, 1.4), (0, (1.6, 1.2))
EPS = 1e-6

# --- the two illustrative histories, as Tau route events -----------------------------------------
# An event splits the branch of multiplicity `parent_mult` into min(split) (continuing) and max(split)
# (new), at `time`. Root multiplicities are the OBSERVED copy numbers.
ONE_W = 0.45
TWO_W1, TWO_W2 = 0.28, 0.68

CASES = [
    dict(title="one doubling", exp="(2,2)", obs="(2,1)", major=2, minor=1, wgd=[(ONE_W, W1_C, "WGD")],
         events=[dict(allele="major", parent_mult=2, split=(1, 1), time=ONE_W)]),
    dict(title="two doublings", exp="(4,4)", obs="(3,2)", major=3, minor=2,
         wgd=[(TWO_W1, W1_C, "WGD 1"), (TWO_W2, W2_C, "WGD 2")],
         events=[dict(allele="major", parent_mult=3, split=(1, 2), time=TWO_W1),
                 dict(allele="minor", parent_mult=2, split=(1, 1), time=TWO_W1),
                 dict(allele="major", parent_mult=2, split=(1, 1), time=TWO_W2)]),
]


def _ghosts(ax, root, W, sign):
    """Cross every lineage alive before W that does not branch at W. Returns how many.

    This is panel c's rule stated directly: a doubling splits everything present, so a lineage that
    passes straight through the line was doubled and the copy is gone.
    """
    n = 0
    for node in sort_nodes(root):
        if node.time >= W - EPS:                                   # not yet present at the doubling
            continue
        if any(abs(b.time - W) < EPS for b in node.branches):      # it did double
            continue
        bx = node.index; gx = bx + sign * 0.34
        ax.plot([bx, gx], [W, W], color=LOST, lw=0.8, ls=DASH_LOST, zorder=5)
        ax.plot([gx, gx], [W, W + 0.055], color=LOST, lw=0.8, ls=DASH_LOST, zorder=5)
        ax.plot(gx, W + 0.075, marker="x", color=LOST, ms=3.4, mew=1.0, zorder=6)
        n += 1
    return n


def make(save=True):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 62 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.14, "width_ratios": [1, 1.5]})

    for ax, cs in zip(axes, CASES):
        maj, mn = build_allele_trees(cs["events"], cs["major"], cs["minor"])
        plot_route(ax, maj, mn, major_color=MAJ, minor_color=MIN_,
                   linewidth_scale=0.8, spacing=0.5, minor_dashes=DASH_MIN)
        for w, col, lab in cs["wgd"]:
            ax.axhline(w, color=col, lw=0.7, ls=":", zorder=1)
        lost = sum(_ghosts(ax, root, w, sign)
                   for w, _, _ in cs["wgd"] for root, sign in ((maj, -1), (mn, +1)))

        xs = [x for ln in ax.get_lines() for x in ln.get_xdata()]
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.7)
        ax.set_ylim(-0.04, 1.06); ax.set_xticks([])
        # label ABOVE the rule at the right edge — on the rule the dots strike through the text
        for w, col, lab in cs["wgd"]:
            ax.text(ax.get_xlim()[1], w + 0.02, lab, color=col, fontsize=S.annot,
                    va="bottom", ha="right")
        ax.set_title(f"{cs['title']}\nexpected {cs['exp']}, observed {cs['obs']}",
                     fontsize=S.annot, pad=3, linespacing=1.5)
        ax.text(0.5, -0.045, f"implied loss {lost}", transform=ax.transAxes, ha="center", va="top",
                fontsize=S.annot, color=LOST, fontweight="bold")
        for s_ in ("top", "right", "bottom"):
            ax.spines[s_].set_visible(False)

    axes[0].set_yticks([0, 0.5, 1]); axes[0].set_ylabel("molecular time", fontsize=S.label)

    h = [mlines.Line2D([], [], color=MAJ, lw=1.2, label="major allele"),
         mlines.Line2D([], [], color=MIN_, lw=1.2, ls=(0, DASH_MIN), label="minor allele"),
         mlines.Line2D([], [], color=LOST, lw=0.8, ls=DASH_LOST, marker="x", ms=3.4,
                       label="missing copy"),
         mlines.Line2D([], [], color=W1_C, lw=0.7, ls=":", label="doubling")]
    fig.legend(handles=h, fontsize=S.legend, loc="lower center", bbox_to_anchor=(0.5, -0.012),
               ncol=4, frameon=False, handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.845, bottom=0.165)
    if save:
        _save_panel(fig, PANELS, "a_loss_schematic")
    print("a_loss_schematic: " + " | ".join(
        f"{c['title']} expected {c['exp']} observed {c['obs']}" for c in CASES))
    return fig


if __name__ == "__main__":
    make()
