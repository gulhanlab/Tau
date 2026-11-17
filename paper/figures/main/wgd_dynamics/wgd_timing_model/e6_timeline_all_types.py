"""Fig 4e6 — per-sample WGD/PGD event timeline, ALL tumor types (not just Joseph's 4 selected ones in
`paper/figures/fig4/panel_scripts/a_timeline.py`). Same core idea, ported near-verbatim: one row per
sample, samples swept by their EARLIEST qualifying event time within each tumor-type block, dot per
WGD/PGD event (size ~ sqrt(genome fraction)), thin grey connector for samples with both, a composition
BAR (PGD only / PGD+WGD / WGD only) per sample, and a "punctuated" BAR right next to it (same full-width-
per-sample style, not a dot, so adjacent samples never visually overlap) shaded by exact event count:
white=1, light grey=2, dark grey=3, near-black=4+.

Differences from the original:
  - ALL tumor types, not 4 hand-picked ones — using `_all_types()` (excludes the same EXCLUDE_TYPES
    used everywhere else in this folder — mostly-empty ultra-rare/benign subtypes).
  - A tumor type is additionally DROPPED if fewer than 5 of its samples have a qualifying WGD or PGD
    event at all (too little data to show a meaningful sweep).
  - Blocks get a visible GAP (empty band) in addition to the separator line, since with ~25 blocks
    instead of 4 a thin line alone is easy to miss.
  - `make()` (vertical: time on x, one row per sample) and `make_horizontal()` (time on y, 0 at bottom /
    1 at top, one COLUMN per sample, blocks arranged left-to-right) share the same `_prepare()` data step
    and just lay it out transposed — useful when a subset is small enough that a wide, short figure reads
    better than an extremely tall, narrow one.

Reuses `e2_arm_patterns_segment._events_deduped()` / `_all_types()` (same Tier-1 tables, same donor-dedup,
same excluded-type list as every other figure in this folder) rather than importing Joseph's
`_wgdpgd_common` shim, so this script has no dependency on `paper/figures/fig4/panel_scripts` at all.

Run:
    PATH=".pixi/envs/default/bin:$PATH" .pixi/envs/default/bin/python e6_timeline_all_types.py
Writes e6_timeline_all_types.{png,svg} and e6_timeline_selected_types.{png,svg} to ./output/.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_arm_patterns_segment import (   # noqa: E402
    LAB, MM, TXT, OUT, _events_deduped, _all_types, _sample_counts, plt,
)

WGD, PGD, BOTH = "#0b6979", "#cd9f2c", "#7b3f9e"   # same NatRev palette as Joseph's `_wgdpgd_common.py`
GORDER = ["PGD only", "PGD + WGD", "WGD only"]
GCOL = {"PGD only": PGD, "PGD + WGD": BOTH, "WGD only": WGD}
SIZE_SC = 11.0
MIN_SAMPLES = 5   # drop a tumor type if fewer than this many samples have a qualifying WGD/PGD event
BLOCK_GAP = 3.5   # empty row/column-units left between tumor-type blocks
# "Punctuated" marker shade by EXACT qualifying-event count (1 = a single one-off duplication, 2+ = a
# genome that took multiple discrete WGD/PGD hits) — grey scale, darker = more events. Needs a visible
# edge colour since 1-event markers are white.
PUNCT_EDGE = "#777777"
PUNCT_SHADES = {1: "#ffffff", 2: "#bbbbbb", 3: "#555555"}
DEFAULT_PUNCT_SHADE = "#111111"   # 4+


def _punct_color(n):
    return PUNCT_SHADES.get(int(n), DEFAULT_PUNCT_SHADE)


def _punct_items(samp):
    vals = sorted(int(v) for v in samp["n_events"].unique())
    return [(_punct_color(n), PUNCT_EDGE, f"{n} event" + ("s" if n != 1 else "")) for n in vals]


def _legend_row(ax, x0, y0, items, axes_width_in, fontsize=6):
    """Draws a horizontal row of small square swatches + labels in PURE axes-fraction coordinates
    (`ax.transAxes` for both x and y) — deliberately NOT `ax.legend()`: a real Legend positioned outside
    the axes box via `bbox_to_anchor` + `add_artist` gets silently clipped by `bbox_inches="tight"` in
    this matplotlib version once the figure has multiple such legends (reproduced empirically — the
    composition/punctuation legends vanished or were cut off mid-word). Plain patches+text drawn directly
    are exactly what already works reliably for the block labels/strips elsewhere in this script.

    items: list of (facecolor, edgecolor_or_None, label). Character width is estimated from `fontsize`
    (points) and converted to AXES-fraction via `axes_width_in` (the axes box's own width in inches —
    NOT the full figure width, which is wider due to margins; using figure width here previously caused
    severe overlap since ax.transAxes(1.0) is the axes' right edge, not the figure's).
    Returns the X position right after the last item, so a second row can start there instead of guessing
    a fixed offset (which is exactly what under-shot and caused this row to overlap the previous one)."""
    char_w = (0.68 * fontsize / 72.0) / axes_width_in   # ~inches/char at `fontsize`, as an axes-fraction
    gap = char_w * 1.5   # swatch-to-text gap, ~1.5 characters wide
    x = x0
    for fc, ec, label in items:
        ax.scatter([x], [y0], transform=ax.transAxes, s=16, marker="s", c=[fc],
                  edgecolors=ec or "none", linewidth=0.5, clip_on=False, zorder=5)
        ax.text(x + gap, y0, label, transform=ax.transAxes, ha="left", va="center",
               fontsize=fontsize, color=TXT, clip_on=False, zorder=5)
        x += gap + char_w * (len(label) + 3)
    return x


def _prepare(types_subset=None, merge_map=None, order_by="alpha"):
    """Shared data step for both `make()` and `make_horizontal()`: which types/samples qualify, their
    "sweep" position within each block, and per-block bookkeeping. See `make()`'s docstring for what
    types_subset/merge_map/order_by mean."""
    cohort = _sample_counts()
    ev = _events_deduped()
    ev = ev[ev.classification.isin(["WGD", "PGD"])].copy()
    ev["time"] = ev["time"].astype(float); ev["gf"] = ev["gf"].astype(float)

    all_types = set(_all_types())
    if types_subset is not None:
        all_types &= set(types_subset)
    ev = ev[ev.tumor_type.isin(all_types)].copy()
    cohort = cohort.loc[cohort.index.isin(all_types)]
    if merge_map:
        ev["tumor_type"] = ev["tumor_type"].map(lambda t: merge_map.get(t, t))
        cohort = cohort.groupby(lambda t: merge_map.get(t, t)).sum()
        all_types = {merge_map.get(t, t) for t in all_types}

    n_event_by_type = ev.groupby("tumor_type")["sample"].nunique()
    kept = [t for t in all_types if n_event_by_type.get(t, 0) >= MIN_SAMPLES]
    dropped = sorted(all_types - set(kept))
    ev = ev[ev.tumor_type.isin(kept)].copy()

    has_w = set(ev[ev.classification == "WGD"]["sample"]); has_p = set(ev[ev.classification == "PGD"]["sample"])

    def grp(s):
        w, p = s in has_w, s in has_p
        return "PGD only" if (p and not w) else "PGD + WGD" if (p and w) else "WGD only"

    samp = (ev.groupby("sample").agg(tumor_type=("tumor_type", "first"),
            first_time=("time", "min")).reset_index())
    samp["group"] = samp["sample"].map(grp)
    n_events_by_sample = ev.groupby("sample").size()   # "punctuated" = >1 qualifying WGD/PGD event
    samp["n_events"] = samp["sample"].map(n_events_by_sample)

    if order_by == "first_time":
        mean_first = samp.groupby("tumor_type")["first_time"].mean()
        kept = sorted(kept, key=lambda t: mean_first[t])
    else:
        kept = sorted(kept, key=lambda t: LAB.get(t, t).lower())

    # position: samples swept by earliest event WITHIN each block, blocks separated by BLOCK_GAP empty
    # units (in addition to the separator line drawn later) so many blocks stay visually distinct.
    pos_of = {}; block_mid = {}; block_end = {}
    pos = 0.0
    for t in kept:
        sub = samp[samp.tumor_type == t].sort_values("first_time")
        ps = []
        for s in sub["sample"]:
            pos_of[s] = pos; ps.append(pos); pos += 1.0
        block_mid[t] = float(np.mean(ps)); block_end[t] = pos - 1.0
        pos += BLOCK_GAP
    N = pos - BLOCK_GAP   # total data extent (excluding the trailing gap)
    ev["pos"] = ev["sample"].map(pos_of)

    return dict(cohort=cohort, ev=ev, samp=samp, kept=kept, dropped=dropped,
                n_event_by_type=n_event_by_type, pos_of=pos_of, block_mid=block_mid,
                block_end=block_end, N=N)


def make(title=None, stem="e6_timeline_all_types", save=True, types_subset=None, merge_map=None,
         order_by="alpha"):
    """types_subset: optional iterable of raw tumor_type keys to restrict the figure to (still subject
    to the same MIN_SAMPLES floor) — e.g. only the cancer types called out in a specific finding, rather
    than every cohort type.

    merge_map: optional {raw_tumor_type: display_group} dict — every raw type mapped to the same
    display_group is pooled into ONE block (e.g. {"Lung.AdenoCA": "Lung", "Lung.SCC": "Lung"}), applied
    AFTER `types_subset` filtering. Types not present in the dict keep their own name/block.

    order_by: "alpha" (default) sorts blocks by display label. "first_time" instead sorts blocks by the
    MEAN of each block's samples' first (earliest) qualifying WGD/PGD event time — "average timing of
    the first punctuated event" — ascending, earliest-onset types first."""
    d = _prepare(types_subset, merge_map, order_by)
    cohort, ev, samp, kept, dropped = d["cohort"], d["ev"], d["samp"], d["kept"], d["dropped"]
    n_event_by_type, pos_of, block_mid, block_end, N = (
        d["n_event_by_type"], d["pos_of"], d["block_mid"], d["block_end"], d["N"])

    if title is None:
        title = "WGD/PGD event timeline — all tumour types"

    top_mm, bottom_mm, row_mm = 26.0, 16.0, 0.62
    axes_height_mm = row_mm * (N + BLOCK_GAP)
    height_mm = top_mm + bottom_mm + axes_height_mm
    fig_w_mm = 118.0; left_frac, right_frac = 0.22, 0.85
    axes_width_in = (right_frac - left_frac) * fig_w_mm * MM
    fig, ax = plt.subplots(figsize=(fig_w_mm * MM, height_mm * MM))
    fig.subplots_adjust(left=left_frac, right=right_frac, top=1 - top_mm / height_mm, bottom=bottom_mm / height_mm)
    # bbox_to_anchor y-fractions below are ABSOLUTE-mm-above-axes-top, converted via axes_height_mm — a
    # fixed axes-fraction offset (e.g. 1.003) would be a tiny/huge absolute gap depending on how many rows
    # this particular figure has, which is exactly what caused the title/legend to visually collide when
    # this function is called with very different row counts (e.g. 11 selected types vs all 25).
    def _mm_above(mm):
        return 1 + mm / axes_height_mm

    for _, g in ev.groupby("sample"):
        if len(g) > 1:
            gg = g.sort_values("time"); ax.plot(gg["time"], gg["pos"], color="grey", alpha=0.2, lw=0.4, zorder=1)
    for cls, col in [("WGD", WGD), ("PGD", PGD)]:
        sub = ev[ev.classification == cls]
        ax.scatter(sub["time"], sub["pos"], s=np.sqrt(sub["gf"].clip(0.01)) * SIZE_SC,
                  c=col, alpha=0.85, edgecolors="white", linewidth=0.15, zorder=2)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for t in kept:
        ax.axhline(block_end[t] + BLOCK_GAP / 2, color="#ddd", lw=0.5, zorder=0)
        n_ev = int(n_event_by_type.get(t, 0)); n_tot = int(cohort.get(t, n_ev))
        ax.text(-0.01, block_mid[t], f"{LAB.get(t, t)} (n={n_ev}/{n_tot})", transform=trans,
               ha="right", va="center", fontsize=5.2, fontweight="bold", color=TXT, clip_on=False)

    gc = {g: 0 for g in GORDER}
    for _, r in samp.iterrows():
        gc[r.group] += 1
        ax.add_patch(mpatches.Rectangle((1.01, pos_of[r["sample"]] - 0.5), 0.02, 1.0,
                    transform=trans, clip_on=False, facecolor=GCOL[r.group], lw=0))
    x_end = _legend_row(ax, 0.14, _mm_above(10.0), [(GCOL[g], None, f"{g} (n={gc[g]})") for g in GORDER],
                        axes_width_in)

    # "Punctuated" annotation column: every sample gets a full-width BAR (same style as the composition
    # strip just to its left, not a dot) shaded by its EXACT qualifying WGD/PGD event count (white=1 ...
    # near-black=4+) — bars span each sample's own row so adjacent samples never visually overlap, unlike
    # small dots which can blend together when many samples are packed tightly.
    for _, r in samp.iterrows():
        ax.add_patch(mpatches.Rectangle((1.04, pos_of[r["sample"]] - 0.5), 0.02, 1.0, transform=trans,
                    clip_on=False, facecolor=_punct_color(r["n_events"]), edgecolor=PUNCT_EDGE, lw=0.15))
    _legend_row(ax, x_end + 0.03, _mm_above(10.0), _punct_items(samp), axes_width_in, fontsize=5.5)

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(N - 0.2, -0.8 - BLOCK_GAP / 2)
    ax.set_xlabel("Event molecular time", fontsize=7, color=TXT)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    leg = [mlines.Line2D([], [], color=WGD, marker="o", ls="None", ms=4, label="WGD"),
           mlines.Line2D([], [], color=PGD, marker="o", ls="None", ms=4, label="PGD")]
    ax.legend(handles=leg, fontsize=6, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              ncol=1, handletextpad=0.3, labelspacing=0.3)
    ax.set_title(title, fontsize=7.5, color=TXT, loc="left", pad=6)

    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    print(f"  [timeline] {len(kept)} types shown ({len(samp)} samples), "
          f"{len(dropped)} types dropped (<{MIN_SAMPLES} event-bearing samples): {dropped}")
    return fig


def make_horizontal(title=None, stem="e6_timeline_horizontal", save=True, types_subset=None,
                    merge_map=None, order_by="alpha"):
    """Transposed layout: time on the Y axis (0 at bottom, 1 at top), one COLUMN per sample, blocks
    arranged left-to-right — a wide-short figure instead of narrow-tall, better suited to a subset small
    enough that "tall and thin" wastes space. Same data/marks/colours as `make()`, just rotated; see that
    function's docstring for what types_subset/merge_map/order_by mean."""
    d = _prepare(types_subset, merge_map, order_by)
    cohort, ev, samp, kept, dropped = d["cohort"], d["ev"], d["samp"], d["kept"], d["dropped"]
    n_event_by_type, pos_of, block_mid, block_end, N = (
        d["n_event_by_type"], d["pos_of"], d["block_mid"], d["block_end"], d["N"])

    if title is None:
        title = "WGD/PGD event timeline"

    left_mm, right_mm, col_mm = 12.0, 30.0, 0.62
    top_mm, bottom_mm = 34.0, 34.0   # bottom needs room for rotated (90deg) single-line block labels
    axes_width_mm = col_mm * (N + BLOCK_GAP)
    axes_width_in = axes_width_mm * MM
    width_mm = left_mm + right_mm + axes_width_mm
    height_mm = 95.0
    fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM))
    fig.subplots_adjust(left=left_mm / width_mm, right=1 - right_mm / width_mm,
                        top=1 - top_mm / height_mm, bottom=bottom_mm / height_mm)

    def _mm_above(mm):
        return 1 + mm / (height_mm - top_mm - bottom_mm)

    # Absolute-mm stacking order above the axes top edge (closest to farthest): composition strip (1mm),
    # punctuation-marker strip (5mm), title (9mm, via pad in points), then the two swatch legends (16mm) —
    # same "fixed axes-fraction offsets collide once row/column count changes" lesson as `make()`.
    COMP_STRIP_MM, PUNCT_STRIP_MM, LEGEND_MM, TITLE_PAD_PT = 1.0, 5.0, 16.0, 26

    for _, g in ev.groupby("sample"):
        if len(g) > 1:
            gg = g.sort_values("time"); ax.plot(gg["pos"], gg["time"], color="grey", alpha=0.2, lw=0.4, zorder=1)
    for cls, col in [("WGD", WGD), ("PGD", PGD)]:
        sub = ev[ev.classification == cls]
        ax.scatter(sub["pos"], sub["time"], s=np.sqrt(sub["gf"].clip(0.01)) * SIZE_SC,
                  c=col, alpha=0.85, edgecolors="white", linewidth=0.15, zorder=2)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for t in kept:
        ax.axvline(block_end[t] + BLOCK_GAP / 2, color="#ddd", lw=0.5, zorder=0)
        n_ev = int(n_event_by_type.get(t, 0)); n_tot = int(cohort.get(t, n_ev))
        # Rotated, single-line label: a horizontal two-line label only fits blocks wider than its own
        # text — with many narrow blocks (e.g. the "remaining types" figure) neighbouring labels bled
        # into each other. Vertical text needs only ~1 column-width of horizontal room regardless.
        ax.text(block_mid[t], -0.02, f"{LAB.get(t, t)} (n={n_ev}/{n_tot})", transform=trans,
               ha="right", va="top", rotation=90, fontsize=5.2, fontweight="bold", color=TXT, clip_on=False)

    comp_h = _mm_above(COMP_STRIP_MM + 2.0) - _mm_above(COMP_STRIP_MM)   # ~2mm-tall strip, in axes-fraction
    gc = {g: 0 for g in GORDER}
    for _, r in samp.iterrows():
        gc[r.group] += 1
        ax.add_patch(mpatches.Rectangle((pos_of[r["sample"]] - 0.5, _mm_above(COMP_STRIP_MM)), 1.0, comp_h,
                    transform=trans, clip_on=False, facecolor=GCOL[r.group], lw=0))
    x_end = _legend_row(ax, 0.0, _mm_above(LEGEND_MM), [(GCOL[g], None, f"{g} (n={gc[g]})") for g in GORDER],
                        axes_width_in)

    # Full-width BAR per sample (same style as the composition strip below it, not a dot) so adjacent
    # samples' punctuation marks never visually overlap.
    punct_h = _mm_above(PUNCT_STRIP_MM + 2.0) - _mm_above(PUNCT_STRIP_MM)
    for _, r in samp.iterrows():
        ax.add_patch(mpatches.Rectangle((pos_of[r["sample"]] - 0.5, _mm_above(PUNCT_STRIP_MM)), 1.0, punct_h,
                    transform=trans, clip_on=False, facecolor=_punct_color(r["n_events"]),
                    edgecolor=PUNCT_EDGE, lw=0.15))
    _legend_row(ax, x_end + 0.03, _mm_above(LEGEND_MM), _punct_items(samp), axes_width_in, fontsize=5.5)

    ax.set_ylim(-0.02, 1.02); ax.set_xlim(-0.8 - BLOCK_GAP / 2, N - 0.2)
    ax.set_ylabel("Event molecular time", fontsize=7, color=TXT)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    leg = [mlines.Line2D([], [], color=WGD, marker="o", ls="None", ms=4, label="WGD"),
           mlines.Line2D([], [], color=PGD, marker="o", ls="None", ms=4, label="PGD")]
    ax.legend(handles=leg, fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.005, 0.5),
              ncol=1, handletextpad=0.3, labelspacing=0.3)
    ax.set_title(title, fontsize=7.5, color=TXT, loc="left", pad=TITLE_PAD_PT)

    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    print(f"  [timeline, horizontal] {len(kept)} types shown ({len(samp)} samples), "
          f"{len(dropped)} types dropped (<{MIN_SAMPLES} event-bearing samples): {dropped}")
    return fig


# Cancer types called out in the WGD-prevalence/timing summary: highest WGD prevalence (Esophageal,
# Bladder, Lung), latest WGD timing (Melanoma, Lung SCC, Liver, Kidney), earliest WGD timing (Ovarian,
# Medulloblastoma), plus GBM added on top. Bladder.TCC and Kidney.ChRCC later dropped (too few samples to
# be worth their own block here); Lung.AdenoCA + Lung.SCC pooled into one "Lung" block.
SELECTED_TYPES = ["Eso.AdenoCA", "Lung.AdenoCA", "Lung.SCC", "Skin.Melanoma", "Liver.HCC",
                 "Kidney.RCC", "Ovary.AdenoCA", "CNS.Medullo", "CNS.GBM"]
LUNG_MERGE = {"Lung.AdenoCA": "Lung", "Lung.SCC": "Lung"}

if __name__ == "__main__":
    make()
    make_horizontal(title="WGD/PGD timeline — selected types", stem="e6_timeline_selected_types",
                    types_subset=SELECTED_TYPES, merge_map=LUNG_MERGE, order_by="first_time")
    # Same horizontal format, same alphabet of eligible types, but everything NOT already shown in the
    # "selected types" figure above (e.g. Bladder.TCC/Kidney.ChRCC, dropped from that one for being too
    # sparse there, still get a block here since MIN_SAMPLES only filters per-figure, not globally).
    remaining = [t for t in _all_types() if t not in SELECTED_TYPES]
    make_horizontal(title="WGD/PGD timeline — remaining types", stem="e6_timeline_remaining_types",
                    types_subset=remaining, order_by="first_time")
    print("e6_timeline_all_types: PNG+SVG written")
