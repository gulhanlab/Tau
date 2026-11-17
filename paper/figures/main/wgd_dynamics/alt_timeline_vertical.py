"""wgd_dynamics figure (main), panel a ALTERNATIVE (vertical layout) — per-sample WGD/PGD event timeline for selected tumour types.
Table-driven: reads tau_events / tau_samples via the `_wgdpgd_common` shim (dev logic near-verbatim).
`make()` returns the figure and writes PNG+SVG next to this script.
Ported from dev/pcawg/FigWGDPGD/panel_A_timeline.py.
"""
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory

from _wgdpgd_common import load_events, load_roster, save_panel, plt, MM, WGD, PGD, BOTH, TXT
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

SELECTED = ["CNS.GBM", "Ovary.AdenoCA", "Lung.SCC", "Stomach.AdenoCA"]
TTL = {"CNS.GBM": "CNS GBM", "Ovary.AdenoCA": "Ovary\nAdenoCA", "Lung.SCC": "Lung SCC",
       "Stomach.AdenoCA": "Stomach\nAdenoCA"}
GORDER = ["PGD only", "PGD + WGD", "WGD only"]
GCOL = {"PGD only": PGD, "PGD + WGD": BOTH, "WGD only": WGD}
SIZE_SC = 11.0


def make(save=True, types=None, stem="alt_timeline_vertical", panels_dir=None, height_mm=80, compact_labels=False):
    """types: tumour types to show (default SELECTED). The supplementary 'other types' panel
    (supp/wgd_swimmer_others) calls this with the remaining types and its own stem/panels_dir."""
    types = list(types) if types is not None else SELECTED
    TOTALS = load_roster().tumor_type.value_counts().to_dict()   # total samples per type
    ev = load_events()
    ev = ev[ev.tumor_type.isin(types) & ev.classification.isin(["WGD", "PGD"])].copy()
    ev["time"] = ev["time"].astype(float); ev["gf"] = ev["gf"].astype(float)
    has_w = set(ev[ev.classification == "WGD"]["sample"]); has_p = set(ev[ev.classification == "PGD"]["sample"])

    def grp(s):
        w, p = s in has_w, s in has_p
        return "PGD only" if (p and not w) else "PGD + WGD" if (p and w) else "WGD only"

    samp = (ev.groupby("sample").agg(tumor_type=("tumor_type", "first"),
            first_time=("time", "min")).reset_index())   # sort by EARLIEST event (onset sweep)
    samp["group"] = samp["sample"].map(grp)
    samp["gr"] = samp["group"].map({g: i for i, g in enumerate(GORDER)})
    samp["tr"] = samp["tumor_type"].map({t: i for i, t in enumerate(types)})
    samp = samp.sort_values(["tr", "first_time"]).reset_index(drop=True)   # earliest-event sweep within type
    y_of = {s: i for i, s in enumerate(samp["sample"])}; N = len(samp)
    ev["y"] = ev["sample"].map(y_of)

    fig, ax = plt.subplots(figsize=(92 * MM, height_mm * MM))   # wider top-left; same height as Panel B
    # exact=True save means no bbox crop, so the left margin must fit the labels; the one-line
    # compact labels (e.g. 'ColoRect.AdenoCA (35/47)') need more room than the stacked defaults
    fig.subplots_adjust(left=0.32 if compact_labels else 0.18, right=0.94, top=0.89, bottom=0.14)
    for _, g in ev.groupby("sample"):
        if len(g) > 1:
            gg = g.sort_values("time"); ax.plot(gg["time"], gg["y"], color="grey", alpha=0.2, lw=0.4, zorder=1)
    for cls, col in [("WGD", WGD), ("PGD", PGD)]:
        sub = ev[ev.classification == cls]
        ax.scatter(sub["time"], sub["y"], s=np.sqrt(sub["gf"].clip(0.01)) * SIZE_SC,
                   c=col, alpha=0.85, edgecolors="white", linewidth=0.2, zorder=2)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, tt in enumerate(types):
        ys = [y_of[s] for s in samp[samp.tumor_type == tt]["sample"]]
        if not ys:
            continue
        if i < len(types) - 1:
            ax.axhline(max(ys) + 0.5, color="#ddd", lw=0.4, zorder=0)
        # compact: one line per type, for the many-type supplementary version where bands are short
        lab = (f"{tt} ({len(ys)}/{TOTALS.get(tt, len(ys))})" if compact_labels
               else f"{TTL.get(tt, tt.replace('.', chr(10)))}\n(n={len(ys)}/{TOTALS.get(tt, len(ys))})")   # event-bearing / total samples
        ax.text(-0.01, np.mean(ys), lab, transform=trans, ha="right", va="center",
                fontsize=S.annot if compact_labels else 6, fontweight="bold", color=TXT, clip_on=False)
    # right per-row composition tick (groups now interleave under the time-sweep)
    gc = {g: 0 for g in GORDER}
    for _, r in samp.iterrows():
        gc[r.group] += 1
        ax.add_patch(mpatches.Rectangle((1.01, y_of[r["sample"]] - 0.5), 0.02, 1.0,
                     transform=trans, clip_on=False, facecolor=GCOL[r.group], lw=0))
    comp_leg = [mpatches.Patch(facecolor=GCOL[g], label=f"{g} (n={gc[g]})") for g in GORDER]
    # composition legend: top, one line. In compact mode the axes are narrower than the legend, so
    # centre it on the FIGURE instead of the axes or it runs off the right edge.
    cleg = ax.legend(handles=comp_leg, fontsize=S.legend, frameon=False, loc="lower center",
                     bbox_to_anchor=((0.5, 0.905) if compact_labels else (0.5, 1.005)),
                     bbox_transform=(fig.transFigure if compact_labels else ax.transAxes),
                     ncol=3, columnspacing=0.8, handlelength=0.9, handletextpad=0.4)
    ax.add_artist(cleg)

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(N - 0.2, -0.8)   # inverted: early type (top) -> late (bottom)
    ax.set_xlabel("Event molecular time", fontsize=S.label, color=TXT)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    leg = [mlines.Line2D([], [], color=WGD, marker="o", ls="None", ms=4, label="WGD"),
           mlines.Line2D([], [], color=PGD, marker="o", ls="None", ms=4, label="PGD")]
    ax.legend(handles=leg, fontsize=S.legend, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              ncol=1, handletextpad=0.3, labelspacing=0.3)   # inside plot (top-right empty) -> clear of xlabel
    return save_panel(fig, stem, save=save, exact=True, panels_dir=panels_dir)


if __name__ == "__main__":
    make(); print("alt_timeline_vertical: PNG+SVG written")
