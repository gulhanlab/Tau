"""loss_landscape figure (supp), panel b — post-WGD loss along the genome in three tumour types.

Support for a single point: the WGD gain landscapes are a net picture, and loss is a large enough term
to move them. Between a third and a half of every copy the doublings produced is already gone —
genome-wide 0.34 in Liver.HCC, 0.52 in Ovary.AdenoCA — and it is not spread evenly, so a locus that
loses more will read as having gained less. The panel is there to make that term visible and to give
its size; it is NOT making a claim about where loss concentrates.

What is plotted
---------------
    loss per opportunity in a bin  =  sum over (sample, segment, doubling) of L * len
                                      -----------------------------------------------
                                      sum over the same of cb * len

cb is the copy budget: the copies present when a doubling fired, all of which it should have
duplicated. L is how many never appeared. Doublings are pooled, not split — L and cb at successive
doublings are tied by the identity cb_2 = cb_1 + d_1, so a per-doubling ratio is not identifiable,
whereas the pooled total never has to decide which doubling lost what.

Three types chosen to contrast in level and in shape, not to be representative, and shown in
ascending order of genome-wide loss: Liver.HCC loses least (0.34) and is best powered (n=83);
Lung.SCC is next (0.40) but by far the most structured, with broad multi-chromosome plateaus
running from ~0.55 on 1p and chr4 down to ~0.20 on 19-22; Ovary.AdenoCA loses most (0.52) and is
close to flat.

READING THE BAND. It is a 95% bootstrap over samples, and it is wide — median 0.16 to 0.22 on a 0-1
quantity. Most bin-to-bin wiggle is inside it. It is drawn precisely so the eye does not read texture
as structure; only the broad plateaus in Lung.SCC clear it convincingly. The band is pointwise over
~283 bins, so it cannot be used to call an individual locus without a multiplicity correction, and no
such call is made here.

Cache-driven from paper/figures/data/loss_landscape/loss_profiles.tsv (paper/figures/data/loss_landscape/build/build_loss_profiles.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of, cache_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _geom as G
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, DOUBLE, S
_apply_style(compact=True)

CACHE = cache_dir_of(__file__) / "loss_profiles.tsv"
PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
# rows are ordered by genome-wide loss, read from the cache rather than hardcoded — the previous
# hand-written order claimed to be ascending and was not
# band and line must differ in VALUE, not just alpha: at 283 bins the fill is contiguous and
# a line of the same hue disappears inside it however thin the band is made
BAND, LINE = "#a8cbe8", "#08306b"
MEAN_C = "#b0402f"
ROW_H = 27 * MM


def make(save=True):
    t = pd.read_csv(CACHE, sep="\t")
    order = (t.drop_duplicates("tumor_type").sort_values("genome_wide").tumor_type.tolist())
    fig, axes = plt.subplots(len(order), 1, figsize=(DOUBLE, ROW_H * len(order) + 16 * MM),
                             sharex=True, gridspec_kw={"hspace": 0.30})

    for ax, tt in zip(axes, order):
        d = t[t.tumor_type == tt].sort_values("x_mb")
        x, gw = d.x_mb.to_numpy(float), float(d.genome_wide.iloc[0])
        ax.fill_between(x, d.lo, d.hi, step="mid", color=BAND, lw=0, zorder=2)
        ax.step(x, d.loss_per_opp, where="mid", color=LINE, lw=0.8, zorder=3)
        ax.axhline(gw, color=MEAN_C, lw=0.7, ls=(0, (3, 2)), zorder=4)
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, G.GEN / 1e6)
        for c in G.CHROMS[1:]:
            ax.axvline(G.OFF[c] / 1e6, color="0.85", lw=0.4, zorder=1)
        ax.tick_params(labelsize=S.tick, length=1.4, pad=1.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{tt}   (n = {int(d.n_samples.iloc[0])} WGD samples,  "
                     f"genome-wide {gw:.2f})", fontsize=S.annot, fontweight="bold", pad=2.5)

    axes[1].set_ylabel("loss per opportunity", fontsize=S.label)
    axes[-1].set_xticks([G.CMID[c] for c in G.CHROMS])
    axes[-1].set_xticklabels(G.CHROMS, fontsize=S.tick)
    axes[-1].set_xlabel("chromosome", fontsize=S.label)
    # no in-figure methods note: it lives in the caption (see caption.md)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.955, bottom=0.10)
    if save:
        _save_panel(fig, PANELS, "b_loss_profiles")
    return fig


if __name__ == "__main__":
    make(); print("b_loss_profiles: PNG+SVG written (cache-driven from loss_profiles.tsv)")
