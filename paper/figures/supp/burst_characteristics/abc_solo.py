"""Supp (burst_characteristics) a, b, c as THREE SEPARATE files, sized for manual placement.

One file per panel, sized for manual placement. A composed row used to live alongside this
(`burst_row_abc.py`, since removed); it was the wrong artefact for placing panels by hand, because
you cannot move them independently and any nudge means re-rendering all three.

Geometry is specified in MILLIMETRES, not figure fractions, so the axes stay put if a margin
changes:

    axes width   a 37.0   b 37.0   c 43.0 mm      axes height 26.0 mm
    file  width  a 52.0   b 52.0   c 58.0 mm      file height 43.5 mm

The three total 162 mm, so they drop into a 183 mm double column with 21 mm left for gaps, at their
authored size — no rescaling, which is what would otherwise throw the type off the 5 pt floor.

Panels are drawn by their own modules (`a_burst_null`, `b_excess_determinacy`), never redrawn here,
so these cannot drift from the numbers those modules compute.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of  # noqa: E402
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper.figures._style import save as _save_panel, MM, S
from paper.figures._style import apply as _style_apply  # noqa: E402
_style_apply()

PANELS = panels_dir_of(__file__)
# manuscript panel names for Supp. Fig. 12 a/b/c (script produces all three)
STEMS = {"a": "a_burst_null", "b": "b_excess_counts", "c": "c_wgd_enrichment"}

# Margins in mm — left carries the y-axis label + ticks, bottom the category labels.
#
# FS=1.0, NOT the row's 0.82. The builders' base sizes are 5.0-7.0 pt, so fs=0.82 renders most type at
# 4.1-4.9 pt — under Nature's 5 pt floor. The panels are made smaller by shrinking the AXES instead,
# which costs plot area rather than legibility.
LEFT, RIGHT, TOP, BOTTOM = 14.0, 1.0, 2.5, 15.0   # LEFT fits a 2-line rotated label + 3-digit ticks;
                                                  # BOTTOM fits panel a's 4-line category label
AX_H = 26.0
AX_W = {"a": 37.0, "b": 37.0, "c": 43.0}
FS = 1.0


def _fig(letter):
    """A figure whose axes is exactly AX_W[letter] x AX_H mm, with the margins above."""
    w = LEFT + AX_W[letter] + RIGHT
    h = TOP + AX_H + BOTTOM
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([LEFT / w, BOTTOM / h, AX_W[letter] / w, AX_H / h])
    return fig, ax, w, h


def make(save=True):
    import _burst_null as A
    import _excess_determinacy as B

    x = pd.read_csv(B.SRC, sep="\t", dtype={"chrom": str})
    x["bin"] = np.where(x.determined & x.possible, "determined",
                        np.where(~x.determined & x.necessary, "every solution",
                                 np.where(~x.determined & x.possible, "compatible", "excluded")))
    x["excess"] = x["bin"] != "excluded"
    ev = pd.read_csv(__import__("paper.config", fromlist=["DATA"]).DATA / "tau_events.tsv", sep="\t")
    wgd = ev[ev.classification == "WGD"].groupby("sample").event_time_frac.apply(list).to_dict()
    pool = np.array([t for v in wgd.values() for t in v], float)
    st = B._stats(x, wgd, pool, np.random.default_rng(0))

    out = []
    for letter, draw in (("a", lambda ax: A.make(save=False, ax=ax, fs=FS)),
                         ("b", lambda ax: B.panel_counts(st, ax=ax, fs=FS)),
                         ("c", lambda ax: B.panel_enrichment(st, bx=ax, fs=FS))):
        fig, ax, w, h = _fig(letter)
        draw(ax)
        if save:
            _save_panel(fig, PANELS, STEMS[letter])
        out.append((letter, w, h))
        plt.close(fig)
    for letter, w, h in out:
        print(f"{STEMS[letter]}: {w:.1f} x {h:.1f} mm  (axes {AX_W[letter]:.1f} x {AX_H} mm)")
    print(f"three files total {sum(w for _, w, _ in out):.1f} mm wide — fits 183 mm with "
          f"{183 - sum(w for _, w, _ in out):.1f} mm for gaps")
    return out


if __name__ == "__main__":
    make()
