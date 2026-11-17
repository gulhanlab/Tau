"""wgd_dynamics figure (main), panel a — per-sample WGD/ccPG event timeline, LANDSCAPE, eight tumour types.

Layout follows the assembled figure: the eight
blocks run left-to-right along the bottom of the figure at full page width, one COLUMN per sample swept by
its earliest event, event molecular time on y (0 at bottom). Above each block a two-line label
("<type>\n(n=<event-bearing>/<cohort>)"), then two full-width-per-sample strips: event COUNT (white=1 ...
near-black=4+) and COMPOSITION (ccPG only / ccPG+WGD / WGD only). Both strip legends sit BELOW the axes
next to the "-> Sample" cue; the WGD/ccPG dot legend sits inside the axes at the lower right, where the
last block is empty.

Data step, block order and marks are from `wgd_timing_model/e6_timeline_all_types.py`
(`_prepare()`, `make_horizontal()`): Lung.AdenoCA + Lung.SCC pooled into one "Lung" block, blocks ordered
by mean first-event time, dot size ~ sqrt(genome fraction). Only the layout is redone here.

Also used by supp/wgd_swimmer_others (every other eligible type, same layout).
`make()` returns the figure and writes PNG+SVG next to this script.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "wgd_timing_model"))
from e6_timeline_all_types import (  # noqa: E402
    _prepare, _punct_color, PUNCT_EDGE, PUNCT_SHADES, DEFAULT_PUNCT_SHADE, GORDER, GCOL, BLOCK_GAP,
    SIZE_SC, MIN_SAMPLES, LAB, WGD, PGD, TXT, MM, plt, _all_types,
)
from _wgdpgd_common import save_panel  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

SELECTED_TYPES = ["Eso.AdenoCA", "Lung.AdenoCA", "Lung.SCC", "Skin.Melanoma", "Liver.HCC",
                  "Kidney.RCC", "Ovary.AdenoCA", "CNS.Medullo", "CNS.GBM"]
LUNG_MERGE = {"Lung.AdenoCA": "Lung", "Lung.SCC": "Lung"}
GLAB = {"PGD only": "ccPG only", "PGD + WGD": "ccPG + WGD", "WGD only": "WGD only"}   # paper term

FS_LABEL, FS_TICK, FS_LEG = S.label, S.tick, S.legend


def other_types():
    """Every eligible type not in the main panel (the supp 'others' figure)."""
    return [t for t in _all_types() if t not in SELECTED_TYPES]


def make(save=True, types=None, merge_map=None, stem="a_timeline", panels_dir=None,
         width_mm=210.0, col_mm=None, order_by="first_time", rotate_labels=False, n_rows=1):
    """types=None -> the eight main-figure blocks. col_mm fixes the per-sample column width; by default the
    blocks are scaled to fill `width_mm`. rotate_labels=True: one-line block labels rotated 90 deg BELOW
    each row and the WGD/ccPG legend outside the axes -- for the supp figure, whose many narrow blocks
    cannot carry horizontal labels. n_rows>1 splits the blocks (in order) over that many rows of roughly
    equal sample count, as in the supplementary figure."""
    if types is None:
        types, merge_map = SELECTED_TYPES, LUNG_MERGE
    d = _prepare(types, merge_map, order_by)
    cohort, ev, samp, kept = d["cohort"], d["ev"], d["samp"], d["kept"]
    n_event_by_type, pos_of, block_mid, block_end, N = (
        d["n_event_by_type"], d["pos_of"], d["block_mid"], d["block_end"], d["N"])

    # ── split blocks over rows (greedy, by cumulative width) ──
    rows, cur, acc = [], [], 0.0
    target = (N + BLOCK_GAP * len(kept)) / n_rows
    for t in kept:
        w = block_end[t] - min(pos_of[s] for s in samp[samp.tumor_type == t]["sample"]) + 1 + BLOCK_GAP
        if cur and acc + w / 2 > target and len(rows) < n_rows - 1:
            rows.append(cur); cur, acc = [], 0.0
        cur.append(t); acc += w
    rows.append(cur)
    row_start = {}; row_N = {}
    for ri, rk in enumerate(rows):
        first = min(pos_of[s] for s in samp[samp.tumor_type.isin(rk)]["sample"])
        last = max(block_end[t] for t in rk)
        row_start[ri] = first; row_N[ri] = last - first + 1
    N_row = max(row_N.values())

    # ── geometry (mm): axes 26 tall per row; labels + two strips above (or labels below), legends below ──
    left_mm, right_mm = 9.0, (16.0 if rotate_labels else 1.0)
    top_mm = 8.0 if rotate_labels else 16.0             # above the first row's axes: strips (+ labels)
    lab_mm = 34.0 if rotate_labels else 0.0             # rotated labels sit below each row
    ax_h_mm, bottom_mm, row_gap_mm = 27.0, (13.0 if rotate_labels else 9.0), 8.0
    row_pitch = ax_h_mm + lab_mm + row_gap_mm + (0.0 if rotate_labels else 16.0 - 8.0)
    height_mm = top_mm + ax_h_mm + (n_rows - 1) * row_pitch + lab_mm + bottom_mm
    if col_mm is None:
        col_mm = (width_mm - left_mm - right_mm) / (N_row + BLOCK_GAP)
    else:
        width_mm = left_mm + right_mm + col_mm * (N_row + BLOCK_GAP)
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    axes = []
    for ri in range(n_rows):
        y1 = 1 - (top_mm + ri * row_pitch) / height_mm
        w_frac = col_mm * (row_N[ri] + BLOCK_GAP) / width_mm
        axes.append(fig.add_axes([left_mm / width_mm, y1 - ax_h_mm / height_mm, w_frac, ax_h_mm / height_mm]))

    def _above(mm):            # axes-fraction y for `mm` above the axes top edge
        return 1 + mm / ax_h_mm

    def _below(mm):
        return -mm / ax_h_mm

    STRIP_H, COMP_Y, PUNCT_Y = 2.0, 1.0, 4.0
    for ri, (ax, rk) in enumerate(zip(axes, rows)):
        off = row_start[ri]
        ev_r = ev[ev.tumor_type.isin(rk)]; samp_r = samp[samp.tumor_type.isin(rk)]
        # ── marks ──
        for _, g in ev_r.groupby("sample"):
            if len(g) > 1:
                gg = g.sort_values("time")
                ax.plot(gg["pos"] - off, gg["time"], color="grey", alpha=0.25, lw=0.4, zorder=1)
        for cls, col in [("WGD", WGD), ("PGD", PGD)]:
            sub = ev_r[ev_r.classification == cls]
            ax.scatter(sub["pos"] - off, sub["time"], s=np.sqrt(sub["gf"].clip(0.01)) * SIZE_SC,
                       c=col, alpha=0.85, edgecolors="white", linewidth=0.15, zorder=2)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        labels = []
        for t in rk:
            if t != rk[-1]:
                ax.axvline(block_end[t] - off + BLOCK_GAP / 2, color="#ddd", lw=0.5, zorder=0)
            n_ev = int(n_event_by_type.get(t, 0)); n_tot = int(cohort.get(t, n_ev))
            if rotate_labels:
                ax.text(block_mid[t] - off, _below(2.0), f"{LAB.get(t, t)} (n={n_ev}/{n_tot})", transform=trans,
                        ha="center", va="top", rotation=90, fontsize=FS_LABEL, color=TXT, clip_on=False)
                continue
            labels.append(ax.text(block_mid[t] - off, _above(8.5), f"{LAB.get(t, t)}\n(n={n_ev}/{n_tot})",
                                  transform=trans, ha="center", va="bottom", fontsize=FS_LABEL, color=TXT,
                                  linespacing=1.15, clip_on=False))
        # narrow neighbouring blocks (CNS GBM | Medulloblastoma) get labels wider than the blocks: push
        # each colliding pair apart symmetrically by the overlap plus a 1 mm gap, in data units.
        fig.canvas.draw(); ren = fig.canvas.get_renderer()
        px_per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
        gap_px = 1.0 * MM * fig.dpi
        for a, b in zip(labels[:-1], labels[1:]):
            ea, eb = a.get_window_extent(ren), b.get_window_extent(ren)
            over = ea.x1 + gap_px - eb.x0
            if over > 0:
                xa, ya = a.get_position(); xb, yb = b.get_position()
                a.set_position((xa - over / 2 / px_per_unit, ya)); b.set_position((xb + over / 2 / px_per_unit, yb))

        # ── strips: event count (upper), composition (lower, touching the axes) ──
        for _, r in samp_r.iterrows():
            x = pos_of[r["sample"]] - off - 0.5
            ax.add_patch(mpatches.Rectangle((x, _above(COMP_Y)), 1.0, STRIP_H / ax_h_mm, transform=trans,
                                            clip_on=False, facecolor=GCOL[r.group], lw=0))
            ax.add_patch(mpatches.Rectangle((x, _above(PUNCT_Y)), 1.0, STRIP_H / ax_h_mm, transform=trans,
                                            clip_on=False, facecolor=_punct_color(r["n_events"]),
                                            edgecolor=PUNCT_EDGE, lw=0.15))

        # ── axes ──
        ax.set_ylim(-0.02, 1.02); ax.set_xlim(-0.8 - BLOCK_GAP / 2, row_N[ri] - 0.2)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.set_ylabel("Event molecular time", fontsize=FS_LABEL, color=TXT)
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)

    ax = axes[-1]                                            # legends hang off the last row
    ax.text(0.0, _below(2.2 + lab_mm), "→ Sample", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_TICK, color=TXT, clip_on=False)      # the arrow glyph is converted to a path at export
    leg = [mlines.Line2D([], [], color=WGD, marker="o", ls="None", ms=4, label="WGD"),
           mlines.Line2D([], [], color=PGD, marker="o", ls="None", ms=4, label="ccPG")]
    if rotate_labels:
        # right of the SHORTEST row (the empty space there), at the full-width x of the longest row
        ri_short = min(row_N, key=row_N.get)
        x_right = (left_mm + col_mm * (N_row + BLOCK_GAP) + 1.0) / width_mm
        y_mid = axes[ri_short].get_position().y0 + axes[ri_short].get_position().height / 2
        fig.legend(handles=leg, fontsize=FS_LEG, frameon=False, loc="center left", bbox_to_anchor=(x_right, y_mid),
                   ncol=1, handletextpad=0.3, labelspacing=0.3)
    else:
        ax.legend(handles=leg, fontsize=FS_LEG, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0),
                  ncol=1, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.2)

    # ── strip legends below the last row: two figure legends side by side, centred on the full width ──
    gc = samp["group"].value_counts()
    h1 = [mpatches.Patch(facecolor=GCOL[g], label=f"{GLAB[g]} (n={int(gc.get(g, 0))})") for g in GORDER]
    counts = sorted(int(v) for v in samp["n_events"].unique())
    h2 = [mpatches.Patch(facecolor=_punct_color(n), edgecolor=PUNCT_EDGE, lw=0.5,
                         label=f"{n} event" + ("s" if n != 1 else "")) for n in counts]
    y = ax.get_position().y0 - (6.5 + lab_mm) / height_mm
    x_mid = (left_mm + col_mm * (N_row + BLOCK_GAP) / 2) / width_mm
    kw = dict(fontsize=FS_LEG, frameon=False, handlelength=1.1, handleheight=1.0, handletextpad=0.4,
              columnspacing=1.2, borderaxespad=0.0)
    lg1 = fig.legend(handles=h1, loc="center right", bbox_to_anchor=(x_mid - 0.02, y), ncol=len(h1), **kw)
    lg2 = fig.legend(handles=h2, loc="center left", bbox_to_anchor=(x_mid + 0.02, y), ncol=len(h2), **kw)
    fig.canvas.draw()                                       # if the pair overflows the canvas, left-align it
    x0 = lg1.get_window_extent().transformed(fig.transFigure.inverted()).x0
    x1 = lg2.get_window_extent().transformed(fig.transFigure.inverted()).x1
    if x0 < 0.005 or x1 > 0.995:
        lg1.remove(); lg2.remove()
        fig.legend(handles=h1, loc="center left", bbox_to_anchor=(left_mm / width_mm, y), ncol=len(h1), **kw)
        fig.legend(handles=h2, loc="center left", bbox_to_anchor=(left_mm / width_mm, y - 4.0 / height_mm),
                   ncol=len(h2), **kw)

    out = save_panel(fig, stem, save=save, exact=True, panels_dir=panels_dir)
    print(f"  [a_timeline] {len(kept)} blocks ({len(samp)} samples) in {n_rows} row(s): "
          + ", ".join(f"{LAB.get(t, t)} {int(n_event_by_type.get(t, 0))}/{int(cohort.get(t, 0))}" for t in kept))
    return out


if __name__ == "__main__":
    make(); print("a_timeline: PNG+SVG written")
