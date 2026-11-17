"""wgd_dynamics figure (main), panel b — sample fraction (no event / ccPG only / 1 WGD / >=2 WGD / WGD+ccPG) |
ccPG event time | WGD event time, by tumour type. Rows sorted by WGD prevalence (highest at the top).

This is the WGD-split version (single vs multiple WGD in the stacked bar) — the one in the paper.
The unsplit original is kept as alt_prevalence_nosplit.py. `wgd` is split into `1 WGD` and
`>=2 WGD` by the number of WGD events the sample carries in tau_events.

Three sections: (1) sample-fraction stacked bars, (2) ccPG event-time horizontal violins,
(3) WGD event-time horizontal violins (jittered points where a row has <MIN_VIO events).
Table-driven: reads tau_events / tau_samples via the `_wgdpgd_common` shim.
`make()` returns the figure and writes PNG+SVG next to this script.
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from _wgdpgd_common import load_events, load_roster, save_panel, plt, MM, WGD, PGD, BOTH, NONE, TXT, GREY
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

MIN_N = 15           # tumour types with >= this many samples
MIN_VIO = 4          # >= this many events -> violin, else jittered points
VW = 0.85            # violin full width in row units (<1 = zero overlap)
WGD_MULTI = "#04313a"        # darker step of the WGD teal; see the docstring
LAB = {"CNS.GBM": "CNS GBM", "Lung.SCC": "Lung SCC", "Eso.AdenoCA": "Esophageal",
       "Prost.AdenoCA": "Prostate", "Ovary.AdenoCA": "Ovarian", "CNS.Medullo": "Medulloblastoma",
       "ColoRect.AdenoCA": "Colorectal", "Stomach.AdenoCA": "Stomach", "Breast.AdenoCA": "Breast",
       "Head.SCC": "Head/neck SCC", "Lung.AdenoCA": "Lung adeno", "Liver.HCC": "Liver",
       "Panc.AdenoCA": "Pancreatic", "Kidney.RCC": "Kidney", "Bladder.TCC": "Bladder",
       "Skin.Melanoma": "Melanoma", "Uterus.AdenoCA": "Uterine", "Biliary.AdenoCA": "Biliary",
       "Lymph.BNHL": "Lymphoma B-NHL", "Bone.Osteosarc": "Osteosarcoma"}


def rows():
    """Per-tumour-type table behind the panel, sorted by WGD prevalence ascending (= bottom-to-top row
    order). Shared with panel c (c_wgd_weibull.py) so its rows line up with this panel's."""
    ev = load_events(); roster = load_roster()
    roster = roster[roster.tumor_type != "Unknown"].copy()
    n_wgd = ev[ev.classification == "WGD"].groupby("sample").size()
    has_p = set(ev[ev.classification == "PGD"]["sample"])
    nw = roster["sample"].map(n_wgd).fillna(0).astype(int)
    pg = roster["sample"].isin(has_p)
    # the only change from b_prevalence_timing: wgd -> wgd1 / wgd2
    roster["cat"] = np.where(nw == 0, np.where(pg, "pgd", "none"),
                             np.where(pg, "both", np.where(nw == 1, "wgd1", "wgd2")))

    out = []
    for tt, g in roster.groupby("tumor_type"):
        n = len(g)
        if n < MIN_N:
            continue
        fr = g["cat"].value_counts(normalize=True)
        wt = ev[(ev.tumor_type == tt) & (ev.classification == "WGD")]["time"].astype(float).values
        pt = ev[(ev.tumor_type == tt) & (ev.classification == "PGD")]["time"].astype(float).values
        out.append(dict(tt=tt, n=n, both=fr.get("both", 0), wgd1=fr.get("wgd1", 0),
                         wgd2=fr.get("wgd2", 0), pgd=fr.get("pgd", 0), none=fr.get("none", 0),
                         wgd_prev=fr.get("wgd1", 0) + fr.get("wgd2", 0) + fr.get("both", 0),
                         wt=wt, pt=pt))
    return pd.DataFrame(out).sort_values("wgd_prev").reset_index(drop=True)


def row_order():
    """Tumour types top-to-bottom as drawn in this panel."""
    return list(rows().tt)[::-1]


# axes box shared with panel c so the two panels' rows align when placed side by side
FIG_H_MM = 82.0
TOP, BOTTOM = 1 - 9.5 / FIG_H_MM, 11.3 / FIG_H_MM     # axes 61.2 mm tall, as in the assembled figure


def make(save=True):
    d = rows()
    y = np.arange(len(d))
    labels = [LAB.get(t, t) for t in d.tt]

    fig, (axL, axP, axW) = plt.subplots(
        1, 3, figsize=(88 * MM, FIG_H_MM * MM),
        gridspec_kw=dict(width_ratios=[17.4, 20.4, 20.4], wspace=0.125))   # column widths in mm, as in her figure
    # top margin pulled in slightly: the legend is five entries over three rows rather than four over two
    fig.subplots_adjust(left=24.0 / 88.0, right=(24.0 + 63.2) / 88.0, top=TOP, bottom=BOTTOM)

    # ── col 1: stacked prevalence ──
    left = np.zeros(len(d))
    for key, col in [("none", "#ebebeb"), ("pgd", PGD), ("wgd1", WGD), ("wgd2", WGD_MULTI),
                     ("both", BOTH)]:
        axL.barh(y, d[key], left=left, color=col, height=1.0, edgecolor="white", lw=0.3)
        left = left + d[key]
    axL.set_yticks(y); axL.set_yticklabels([f"{l} ({n})" for l, n in zip(labels, d.n)], fontsize=S.tick)
    axL.set_xlim(0, 1); axL.set_xticks([0, 0.5, 1.0]); axL.set_xticklabels(["0", "0.5", "1"])
    axL.set_xlabel("Sample fraction", fontsize=S.label, color=TXT)
    axL.spines[["top", "right"]].set_visible(False)
    axL.legend(handles=[Patch(color=c, label=l) for c, l in
                        [("#ebebeb", "no event"), (PGD, "ccPG only"), (WGD, "1 WGD"),
                         (WGD_MULTI, "≥2 WGD"), (BOTH, "WGD+ccPG")]],
               fontsize=S.annot, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.05, 1.005),
               handlelength=1.0, columnspacing=0.8, handletextpad=0.4, labelspacing=0.3)

    def violins(ax, key, col, title):
        """One horizontal violin of event time per tumour-type row (jittered points if sparse)."""
        for i in range(0, len(d), 2):
            ax.axhspan(i - 0.5, i + 0.5, color="#f5f5f7", lw=0, zorder=0)
        data, pos = [], []
        for yi in y:
            v = np.asarray(d[key].iloc[yi], float)
            if len(v) >= MIN_VIO and np.ptp(v) > 1e-3:
                data.append(v); pos.append(yi)
            elif len(v) > 0:
                jit = (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.3
                ax.scatter(v, yi + jit, s=2.0, color=col, alpha=0.7, lw=0, zorder=4)
        if data:
            vp = ax.violinplot(data, positions=pos, vert=False, widths=VW,
                               showmeans=False, showmedians=True, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.55); b.set_lw(0.4)
            vp["cmedians"].set_color(col); vp["cmedians"].set_lw(0.9)
        ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(title, fontsize=S.title, color=col, loc="center", pad=2)
        ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)

    violins(axP, "pt", PGD, "ccPG")
    violins(axW, "wt", WGD, "WGD")
    axL.set_ylim(-0.6, len(d) - 0.4)
    fig.text((axP.get_position().x0 + axW.get_position().x1) / 2, 0.035,
             "Event molecular time", ha="center", va="center", fontsize=S.label, color=TXT)

    n_tot = int(d.n.sum())
    print(f"  {n_tot:,} samples in {len(d)} types (>= {MIN_N}) | 1 WGD {int((d.wgd1*d.n).sum()):,} | "
          f">=2 WGD {int((d.wgd2*d.n).sum()):,} | both {int((d.both*d.n).sum()):,} | "
          f"ccPG only {int((d.pgd*d.n).sum()):,} | none {int((d.none*d.n).sum()):,}")
    top = d.sort_values("wgd2", ascending=False).head(5)
    print("  highest multi-WGD fraction: " +
          ", ".join(f"{LAB.get(t,t)} {100*v:.0f}%" for t, v in zip(top.tt, top.wgd2)))
    return save_panel(fig, "b_prevalence_timing", save=save, exact=True)


if __name__ == "__main__":
    make(); print("b_prevalence_timing: PNG+SVG written")
