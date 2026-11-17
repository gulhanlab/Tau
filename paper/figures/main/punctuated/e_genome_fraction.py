"""punctuated figure (main), panel e — genome fraction per WGD/ccPG event (two-class histogram).
Table-driven: reads tau_events via the `_wgdpgd_common` shim.
`make()` returns the figure and writes PNG+SVG next to this script.
Ported from dev/pcawg/FigWGDPGD/panel_C_genome_fraction.py.
"""
import numpy as np

from _wgdpgd_common import load_events, save_panel, plt, MM, WGD, PGD, TXT
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()


def make(save=True):
    ev = load_events()
    fig, ax = plt.subplots(figsize=(58 * MM, 38 * MM))   # landscape (x>y); stacked over d on the left
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.20)
    BINS = np.linspace(0, 1, 26); W = BINS[1] - BINS[0]   # 0.04-wide bins
    for cls, col in [("WGD", WGD), ("PGD", PGD)]:
        v = ev[ev.classification == cls]["gf"].values.astype(float)
        c, _ = np.histogram(v, bins=BINS)
        ax.bar(BINS[:-1], c, width=W, align="edge", color=col, alpha=0.5,
               edgecolor="white", lw=0.3, label=f"{'ccPG' if cls == 'PGD' else cls} (n={len(v):,})")
    ax.set_xlabel("Genome fraction", color=TXT, fontsize=S.label)
    ax.set_ylabel("Number of events", color=TXT, fontsize=S.label)
    ax.tick_params(labelsize=S.tick)
    ax.set_xlim(0, 1); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=S.legend)
    return save_panel(fig, "e_genome_fraction", save=save, exact=True)


if __name__ == "__main__":
    make(); print("e_genome_fraction: PNG+SVG written")
