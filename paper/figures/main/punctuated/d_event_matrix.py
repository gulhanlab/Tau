"""punctuated figure (main), panel d — joint #WGD x #ccPG events per sample (3x3 count matrix).
Rows = number of WGD events (0/1/2+), cols = number of PGD events (0/1/2+); log-scaled colour with
counts annotated. Table-driven: reads tau_events / tau_samples via the `_wgdpgd_common` shim.
`make()` returns the figure and writes PNG+SVG next to this script.
Ported from dev/pcawg/FigWGDPGD/panel_D_event_matrix.py.
"""
import numpy as np

from _wgdpgd_common import load_events, load_roster, save_panel, plt, MM, TXT, GREY, WGD, PGD
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()


def make(save=True):
    ev = load_events(); roster = load_roster()
    nw = ev[ev.classification == "WGD"].groupby("sample").size()
    npg = ev[ev.classification == "PGD"].groupby("sample").size()
    per = roster.set_index("sample")
    per["nw"] = nw.reindex(per.index, fill_value=0).clip(upper=2)
    per["np"] = npg.reindex(per.index, fill_value=0).clip(upper=2)

    M = np.zeros((3, 3), int)
    for w in range(3):
        for p in range(3):
            M[w, p] = int(((per.nw == w) & (per.np == p)).sum())
    row_tot = M.sum(1); col_tot = M.sum(0); N = M.sum()

    # 8.5 mm cells; canvas leaves room for the axis titles on the left/bottom
    fig, ax = plt.subplots(figsize=(38 * MM, 38 * MM))
    fig.subplots_adjust(left=8.5 / 38, right=34.0 / 38, top=34.0 / 38, bottom=8.5 / 38)

    # bivariate palette from the assembled figure: teal deepens down the WGD axis, gold along the ccPG
    # axis, lightness with the column (0 -> 2+); count magnitude is carried by the printed number.
    CELL = [["#4c4928", "#89722c", "#d4a43e"],
            ["#40655c", "#949c6f", "#d9c885"],
            ["#4b8897", "#9ec6b4", "#d7e7c0"]]
    LIGHT = {(0, 2), (1, 2), (2, 1), (2, 2)}          # dark text on the light cells
    for w in range(3):
        for p in range(3):
            ax.add_patch(plt.Rectangle((p - 0.5, w - 0.5), 1, 1, facecolor=CELL[w][p], edgecolor="white", lw=1.0))
            ax.text(p, w, f"{M[w, p]:,}", ha="center", va="center", fontsize=S.annot,
                    color=TXT if (w, p) in LIGHT else "white")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(2.5, -0.5); ax.set_aspect("equal")

    PL = ["0", "1", "2+"]
    # colour-code the axes by category: ccPG (cols) gold, WGD (rows) teal (marginals -> caption)
    ax.set_xticks(range(3)); ax.set_xticklabels(PL, color=PGD, fontsize=S.tick)
    ax.set_yticks(range(3)); ax.set_yticklabels(PL, color=WGD, fontsize=S.tick)
    ax.set_xlabel("ccPG events", fontsize=S.label, color=PGD)
    ax.set_ylabel("WGD events", fontsize=S.label, color=WGD)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)

    out = save_panel(fig, "d_event_matrix", save=save, exact=True)
    print(f"  matrix totals: WGD {row_tot.tolist()}  PGD {col_tot.tolist()}  N={N}")
    return out


if __name__ == "__main__":
    make(); print("d_event_matrix: PNG+SVG written")
