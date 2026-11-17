"""wgd_ccpg_benchmark figure (supp), panel d — WGD events gain both alleles; ccPG events gain one.

Shown alongside e_allelic_bins.py, which bins this same quantity. Was panel e until the two
timing-accuracy heatmaps moved to main Fig. 3b,c and the letters shifted up.

One measure, computed the obvious way. Take a window of +/-0.10 molecular time around each detected
event, keep the segments with a gain in that window, and ask how much of the sample's TIMED GENOME that
event made biallelic — gained on both alleles rather than one:

  a segment is biallelic at the event if one of its polytope solutions places a major-allele gain and a
  minor-allele gain both inside the window

Allele comes from the route topology (tau/data/route_topology.json.gz), so it is read off the gain
itself, not inferred from the final copy number. Membership is by time proximity rather than
`tau_segments.event_id`, which carries the multiplicity cluster label and does not join tau_events.

Result: ccPG events sit on zero — they gain a single allele — and reach 7.3% of the timed genome at the
very most. WGD events run to a median of 8.0% and a maximum of 66%.

DENOMINATOR. The biallelic length is divided by the sample's whole timed genome, not by the length the
event itself gained. That is what puts a small ccPG and a genome-wide WGD on one absolute scale: on the
gained-length denominator an event covering three segments can score 100% biallelic and outrank a real
doubling, which is exactly the comparison this panel exists to make. The cost is that the x axis is no
longer independent of the y: a bigger event necessarily makes more of the genome biallelic, so the
correlation with genome fraction rises from r = 0.54 to r = 0.64 and is partly definitional. Read the
SEPARATION between the two clouds, not the slope.

Why the WGD median is only 8%: it is bounded by how much WGD-linked length still HAS a minor allele to
gain. Post-WGD loss erodes (2,2) to (2,1), which then reads monoallelic by observation rather than by
history, and 54% of the length under a median WGD event is in that state. So this is a FLOOR on the
doubling's allelic reach, not an estimate of it, and the shortfall is the same post-WGD loss the paper
reports independently. The WGD-vs-ccPG contrast does not rest on this: ccPG is at zero because its
gains are monoallelic, not because its minor alleles were lost afterwards.

Per-solution containment is used rather than a representative solution because determined routes are
exactly those with minor_cn <= 1 — no minor allele to gain — so biallelic gains are only ever visible in
underdetermined segments. Collapsing those to a median solution places the two gains at a point the
polytope does not contain and pulls them apart even where a coincident solution exists; measured that
way 95% of WGD events read as monoallelic, which is an artefact of the summary rather than a result.

Cache-driven from paper/figures/data/wgd_ccpg_benchmark/event_alleles.tsv (paper/figures/data/wgd_ccpg_benchmark/build/build_event_alleles.py). Events need >=3
contributing segments; contributing segments need ESS >= 100, since the biallelic test is
power-sensitive (the rate climbs 40% -> 76% from ESS <25 to >400).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of, cache_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style()

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
SRC = cache_dir_of(__file__) / "event_alleles.tsv"
WGD_COL, CCPG_COL = "#0b6979", "#cd9f2c"
MIN_SEG = 3

# Geometry shared with e_allelic_bins.py, which is shown BESIDE this panel rather than instead of it:
# the pair tiles to 88 mm, inside a SINGLE column. 42 mm tall to match b_recall and c_falsepos, so all
# four small panels share a row height. Near-square rather than the original 89x46, which at nearly 2:1
# pressed both clouds flat; this gives the biallelic axis room to separate them.
W, H = 44 * MM, 42 * MM
# margins in mm, not figure fractions: a fraction is relative to W or H, so changing the panel size
# would silently rescale every margin with it
#              right margin holds the "1.0" x tick label, which the axis runs right up to
#                                   top matched to b_recall/c_falsepos so the plot boxes line up
MAR_L, MAR_R, MAR_T, MAR_B = 10.0 * MM, 2.4 * MM, 3.4 * MM, 8.2 * MM


def make(save=True):
    t = pd.read_csv(SRC, sep="\t")
    t = t[t.n_seg >= MIN_SEG].dropna(subset=["bi_frac_timed"]).copy()

    fig, ax = plt.subplots(figsize=(W, H))
    for c, col in (("WGD", WGD_COL), ("ccPG", CCPG_COL)):
        s = t[t.classification == c]
        ax.scatter(s.genome_frac, s.bi_frac_timed, s=3.4, color=col, alpha=0.45,
                   linewidths=0, zorder=3,
                   label=f"{c} (n={len(s)})")
        ax.axhline(s.bi_frac_timed.median(), color=col, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.set_xlabel("event genome fraction")
    # shortened, not wrapped: at 44 mm "biallelic fraction of timed genome" is longer than the plot
    # box it labels, and a two-line version is wider than the left margin can hold. The denominator
    # is stated in the caption instead.
    ax.set_ylabel("biallelic genome fraction")
    # y stops at 0.70, just above the largest WGD event: on this denominator nothing approaches
    # 1, and a 0-1 axis would press both clouds into the bottom third
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 0.70)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", handlelength=0.8, handletextpad=0.4, labelspacing=0.3,
              borderpad=0.2, markerscale=2.2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    # the "segments with a gain within +/-0.10 of the event" note that used to sit here is caption
    # material now: at 44 mm it needed 4.6 pt, below the 5 pt floor the guidelines allow
    fig.subplots_adjust(left=MAR_L / W, right=1 - MAR_R / W, top=1 - MAR_T / H, bottom=MAR_B / H)
    if save:
        _save_panel(fig, PANELS, "d_allelic")
    return fig


if __name__ == "__main__":
    make(); print("d_allelic: PNG+SVG written (cache-driven from event_alleles.tsv)")
