"""wgd_dynamics figure (main), panel b ALTERNATIVE (unsplit WGD bar; superseded by b_prevalence_timing.py) — sample fraction | ccPG event time | WGD event time, by tumour type.
Three sections: (1) sample-fraction stacked bars, (2) PGD event-time horizontal violins,
(3) WGD event-time horizontal violins (jittered points where a row has <MIN_VIO events).
Table-driven: reads tau_events / tau_samples via the `_wgdpgd_common` shim.
`make()` returns the figure and writes PNG+SVG next to this script.
Ported from dev/pcawg/FigWGDPGD/panel_B_prevalence_timing.py.
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
LAB = {"CNS.GBM": "CNS GBM", "Lung.SCC": "Lung SCC", "Eso.AdenoCA": "Esophageal",
       "Prost.AdenoCA": "Prostate", "Ovary.AdenoCA": "Ovarian", "CNS.Medullo": "Medulloblastoma",
       "ColoRect.AdenoCA": "Colorectal", "Stomach.AdenoCA": "Stomach", "Breast.AdenoCA": "Breast",
       "Head.SCC": "Head/neck SCC", "Lung.AdenoCA": "Lung adeno", "Liver.HCC": "Liver",
       "Panc.AdenoCA": "Pancreatic", "Kidney.RCC": "Kidney", "Bladder.TCC": "Bladder",
       "Skin.Melanoma": "Melanoma", "Uterus.AdenoCA": "Uterine", "Biliary.AdenoCA": "Biliary",
       "Lymph.BNHL": "Lymphoma B-NHL", "Bone.Osteosarc": "Osteosarcoma"}


def make(save=True):
    ev = load_events(); roster = load_roster()
    roster = roster[roster.tumor_type != "Unknown"].copy()
    has_w = set(ev[ev.classification == "WGD"]["sample"]); has_p = set(ev[ev.classification == "PGD"]["sample"])
    roster["cat"] = ["both" if (s in has_w and s in has_p) else "wgd" if s in has_w
                     else "pgd" if s in has_p else "none" for s in roster["sample"]]

    rows = []
    for tt, g in roster.groupby("tumor_type"):
        n = len(g)
        if n < MIN_N:
            continue
        fr = g["cat"].value_counts(normalize=True)
        wt = ev[(ev.tumor_type == tt) & (ev.classification == "WGD")]["time"].astype(float).values
        pt = ev[(ev.tumor_type == tt) & (ev.classification == "PGD")]["time"].astype(float).values
        rows.append(dict(tt=tt, n=n, both=fr.get("both", 0), wgd=fr.get("wgd", 0), pgd=fr.get("pgd", 0),
                         none=fr.get("none", 0), wgd_prev=fr.get("wgd", 0) + fr.get("both", 0),
                         wt=wt, pt=pt))
    d = pd.DataFrame(rows).sort_values("wgd_prev").reset_index(drop=True)
    y = np.arange(len(d))
    labels = [LAB.get(t, t) for t in d.tt]

    fig, (axL, axP, axW) = plt.subplots(
        1, 3, figsize=(88 * MM, 80 * MM),   # same height as Panel A (side-by-side top row)
        gridspec_kw=dict(width_ratios=[0.85, 1.0, 1.0], wspace=0.13))   # gap so column 0/1 ticks don't overlap
    fig.subplots_adjust(left=0.30, right=0.98, top=0.89, bottom=0.14)   # axes box matched to Panel A

    # ── col 1: stacked prevalence ──
    left = np.zeros(len(d))
    for key, col in [("none", "#ebebeb"), ("pgd", PGD), ("wgd", WGD), ("both", BOTH)]:
        axL.barh(y, d[key], left=left, color=col, height=1.0, edgecolor="white", lw=0.3)
        left = left + d[key]
    axL.set_yticks(y); axL.set_yticklabels([f"{l} ({n})" for l, n in zip(labels, d.n)], fontsize=S.tick)
    axL.set_xlim(0, 1); axL.set_xticks([0, 0.5, 1.0]); axL.set_xticklabels(["0", "0.5", "1"])
    axL.set_xlabel("Sample fraction", fontsize=S.label, color=TXT)
    axL.spines[["top", "right"]].set_visible(False)
    axL.legend(handles=[Patch(color=c, label=l) for c, l in
                        [("#ebebeb", "no event"), (PGD, "PGD only"), (WGD, "WGD only"), (BOTH, "WGD+PGD")]],
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
            elif len(v) > 0:                              # too few events -> jittered points
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

    violins(axP, "pt", PGD, "PGD")
    violins(axW, "wt", WGD, "WGD")
    axL.set_ylim(-0.6, len(d) - 0.4)
    # single shared x-label centred under the two distribution columns
    fig.text((axP.get_position().x0 + axW.get_position().x1) / 2, 0.035,
             "Event molecular time", ha="center", va="center", fontsize=S.label, color=TXT)
    return save_panel(fig, "alt_prevalence_nosplit", save=save, exact=True)


if __name__ == "__main__":
    make(); print("alt_prevalence_nosplit: PNG+SVG written")
