"""landscape figure (main) — per-OncoKB-oncogene gain frequency (dot size) × median timing (colour), WGD-linked vs unlinked,
all tumour types (n>=15). Table-driven: reads the density grids in figures/data/landscape/ + OncoKB loci.
make() -> fig + PNG/SVG.
"""
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "font.size": 7, "font.family": "sans-serif"})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.config import REF, cache_dir_of, panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

CACHE = cache_dir_of(__file__); PANELS = panels_dir_of(__file__)
HG19 = {"1":249250621,"2":243199373,"3":198022430,"4":191154276,"5":180915260,"6":171115067,
"7":159138663,"8":146364022,"9":141213431,"10":135534747,"11":135006516,"12":133851895,
"13":115169878,"14":107349540,"15":102531392,"16":90354753,"17":81195210,"18":78077248,
"19":59128983,"20":63025520,"21":48129895,"22":51304566}
CHROMS = [str(i) for i in range(1, 23)]
OFF = {}; c = 0
for k in CHROMS:
    OFF[k] = c; c += HG19[k]
GEN = c
WINc, HOT_FRAC, MERGE_MB, MAXCOL, MIN_N = 2, 0.5, 22, 44, 15
CANON = {"MYC","MYCN","MYCL","EGFR","ERBB2","CCND1","CCND2","CCND3","CCNE1","CDK4","CDK6","MDM2","MDM4","KRAS",
         "NRAS","HRAS","TERT","SOX2","PIK3CA","MET","FGFR1","FGFR2","KIT","PDGFRA","VEGFA","AKT1","AKT2","MCL1",
         "BCL2","AR","IGF1R","AURKA","MYB","NKX2-1","ZNF217","GATA6","KLF5","YAP1","JUN"}
LAB = {"CNS.GBM":"CNS GBM","CNS.Medullo":"Medulloblastoma","CNS.PiloAstro":"Pilocytic astro",
       "CNS.Oligo":"Oligodendroglioma","Eso.AdenoCA":"Esophageal","Prost.AdenoCA":"Prostate",
       "Ovary.AdenoCA":"Ovarian","ColoRect.AdenoCA":"Colorectal","Stomach.AdenoCA":"Stomach",
       "Breast.AdenoCA":"Breast","Head.SCC":"Head/neck SCC","Lung.AdenoCA":"Lung adeno","Lung.SCC":"Lung SCC",
       "Liver.HCC":"Liver","Panc.AdenoCA":"Pancreatic","Panc.Endocrine":"Panc. endocrine","Kidney.RCC":"Kidney",
       "Kidney.ChRCC":"Kidney chromophobe","Bladder.TCC":"Bladder","Skin.Melanoma":"Melanoma",
       "Uterus.AdenoCA":"Uterine","Biliary.AdenoCA":"Biliary","Lymph.BNHL":"Lymphoma B-NHL","Lymph.CLL":"CLL",
       "Bone.Osteosarc":"Osteosarcoma","Thy.AdenoCA":"Thyroid","Myeloid.MPN":"Myeloid MPN",
       "SoftTissue.Leiomyo":"Leiomyosarcoma","SoftTissue.Liposarc":"Liposarcoma","Cervix.SCC":"Cervical"}


def make(save=True):
    gW = np.load(CACHE / "_gain_density_grids.npz", allow_pickle=True)
    gU = np.load(CACHE / "_unlinked_gain_density_grids.npz", allow_pickle=True)
    GT = list(gW["types"]); N_X = int(gW["N_X"]); N_Y = int(gW["N_Y"])
    x_edges = np.linspace(0, GEN, N_X + 1); yc = (np.arange(N_Y) + 0.5) / N_Y
    ONC = {}
    with open(REF / "ref_oncokb_oncogenes_hg19.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["chrom"] in OFF:
                ONC[row["gene"]] = OFF[row["chrom"]] + float(row["pos_bp"])
    COL = {g: int(np.clip(np.searchsorted(x_edges, gx) - 1, 0, N_X - 1)) for g, gx in ONC.items()}

    def wmedian(w):
        return yc[np.searchsorted(np.cumsum(w), 0.5 * w.sum())] if w.sum() > 0 else np.nan

    def freq_time(grid, gk, t, gene):
        G = grid[f"{gk}__{t}"].astype(float); col = COL[gene]; win = G[:, max(0, col-WINc):col+WINc+1]
        return float(win.sum()), wmedian(win.sum(axis=1))

    n_all = {t: int(n) for t, n in zip(GT, gW["n_all"])}
    cat = sorted([t for t in GT if n_all[t] >= MIN_N],
                 key=lambda t: wmedian(gW[f"link__{t}"].astype(float).sum(axis=1)))
    PAN = [("WGD-linked gains", gW, "link"), ("unlinked gains", gU, "nonlink")]
    VMAX = {title: float(np.percentile(np.concatenate([grid[f"{gk}__{t}"][grid[f"{gk}__{t}"] > 0].ravel()
            for t in cat]), 99)) for title, grid, gk in PAN}
    score = {}
    for g in ONC:
        best, hot = 0.0, False
        for title, grid, gk in PAN:
            for t in cat:
                G = grid[f"{gk}__{t}"].astype(float); win = G[:, max(0, COL[g]-WINc):COL[g]+WINc+1]
                best = max(best, win.sum())
                if float(win.max()) >= HOT_FRAC * VMAX[title]:
                    hot = True
        if hot:
            score[g] = best
    ranked = sorted(score, key=lambda g: -(score[g] * (1.3 if g in CANON else 1.0)))
    kept = []
    for g in ranked:
        if all(abs(ONC[g] - ONC[k]) >= MERGE_MB * 1e6 for k in kept):
            kept.append(g)
        if len(kept) >= MAXCOL:
            break
    genes = sorted(kept, key=lambda g: ONC[g])
    data = {title: {t: [freq_time(grid, gk, t, g) for g in genes] for t in cat} for title, grid, gk in PAN}
    fmax = max(f for title in data for t in cat for f, _ in data[title][t])

    fig, axes = plt.subplots(1, 2, figsize=(250/25.4, (26 + 6.6 * len(cat))/25.4), gridspec_kw=dict(wspace=0.34))
    sc = None
    for ax, (title, grid, gk) in zip(axes, PAN):
        for yi, t in enumerate(cat):
            for xi, (fr, tm) in enumerate(data[title][t]):
                if fr <= 0:
                    continue
                s = (fr / fmax) * 150 + 3
                if np.isnan(tm):
                    ax.scatter(xi, yi, s=s, c="#bbbbbb", edgecolor="#3a3a3a", lw=0.5, zorder=2)
                else:
                    sc = ax.scatter(xi, yi, s=s, c=[tm], cmap="viridis", vmin=0, vmax=1, edgecolor="#3a3a3a", lw=0.5, zorder=3)
        ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, fontsize=S.tick, rotation=90)
        ax.set_yticks(range(len(cat))); ax.set_yticklabels([LAB.get(t, t) for t in cat], fontsize=S.tick)
        ax.set_xlim(-0.6, len(genes) - 0.4); ax.set_ylim(-0.6, len(cat) - 0.4); ax.invert_yaxis()
        ax.set_title(title, fontsize=S.tick, pad=4); ax.tick_params(length=0)
        ax.set_axisbelow(True); ax.grid(True, color="#eeeeee", lw=0.3)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(sc, ax=axes, fraction=0.018, pad=0.02)
    cb.set_label("median timing", fontsize=S.tick); cb.set_ticks([0, 0.5, 1.0]); cb.set_ticklabels(["0 early", "0.5", "1 late"])
    cb.ax.tick_params(labelsize=S.tick, length=0); cb.outline.set_visible(False)
    hands = [Line2D([0], [0], marker="o", ls="", mfc="#bbbbbb", mec="#3a3a3a", mew=0.5,
                    ms=np.sqrt(fr * 150 + 3), label=f"{fr:.1f}") for fr in (0.25, 0.5, 1.0)]
    axes[1].legend(handles=hands, title="gains/sample (rel.)", fontsize=S.title, title_fontsize=S.legend, frameon=False,
                   loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, handletextpad=0.2, columnspacing=0.7)
    fig.suptitle("Per-OncoKB-oncogene gain frequency (dot size) × median timing (colour): WGD-linked vs "
                 "unlinked — all tumour types (n≥15), rows sorted by WGD-linked timing", fontsize=S.title, y=1.005)
    if save:
        fig.savefig(PANELS / "alt_oncogene_dotplot.png", dpi=185, bbox_inches="tight")
        fig.savefig(PANELS / "alt_oncogene_dotplot.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make(); print("d_oncogene_dotplot: PNG+SVG written")
