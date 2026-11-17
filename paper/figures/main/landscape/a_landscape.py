"""landscape figure (main), panel a — amplification-timing landscape, three tumour types x three layers.

Layout as in the assembled figure: tumour types as COLUMNS
(Head/neck SCC | Skin.Melanoma | CNS GBM), the three gain layers as ROWS (WGD-linked / ccPG-linked /
unlinked -- the partition of Fig 3a), oncogene labels above the top row, a layer tag at the right of each
row, and one horizontal "Gain density" colourbar per layer underneath. Each layer is divided by its own
per-locus segment denominator (gains per segment that could have contributed to that layer); the colour
scale is shared across the three types WITHIN a layer (99th percentile), not between layers.

Table-driven: reads the density grids in paper/figures/data/landscape/ (built from tau_gains by paper/figures/data/landscape/build.py) + the
OncoKB oncogene loci. `make()` returns the figure and writes PNG+SVG next to this script. The supplementary
gain_landscape sheets reuse the geometry/helpers here (types stacked instead of side by side).
"""
import csv
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "font.size": 7, "font.family": "sans-serif",
                            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"]})
import matplotlib.pyplot as plt

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
GEN = c; BOUND = [OFF[k] for k in CHROMS] + [GEN]; CMID = {k: OFF[k] + HG19[k]/2 for k in CHROMS}
CEN = {"1":125e6,"2":93.3e6,"3":91e6,"4":50.4e6,"5":48.4e6,"6":61e6,"7":59.9e6,"8":45.6e6,"9":49e6,"10":40.2e6,
"11":53.7e6,"12":35.8e6,"13":17.9e6,"14":17.6e6,"15":19e6,"16":36.6e6,"17":24e6,"18":17.2e6,"19":26.5e6,
"20":27.5e6,"21":13.2e6,"22":14.7e6}
CENPOS = {k: (OFF[k] + CEN[k]) / 1e6 for k in CHROMS}
CURATED = ["Head.SCC", "Skin.Melanoma", "CNS.GBM"]      # the three columns of the main panel, in order
LAB = {"Head.SCC": "Head/neck SCC", "CNS.GBM": "CNS GBM"}
LAYERS = [("wgd", "WGD", "_gain_density_grids.npz", "link", "seglink", "#0b6979"),
          ("ccpg", "ccPG", "_ccpg_gain_density_grids.npz", "link", "seglink", "#cd9f2c"),
          ("unlk", "Unlinked", "_unlinked_gain_density_grids.npz", "nonlink", "segnonlink", "#8a8a8a")]
GENE_COL = "#aa4e94"
CANON = {"MYC","MYCN","MYCL","EGFR","ERBB2","CCND1","CCND2","CCND3","CCNE1","CDK4","CDK6","MDM2","MDM4","KRAS",
         "NRAS","HRAS","TERT","SOX2","PIK3CA","MET","FGFR1","FGFR2","KIT","PDGFRA","VEGFA","AKT1","AKT2","MCL1",
         "BCL2","AR","IGF1R","AURKA","MYB","NKX2-1","ZNF217","GATA6","KLF5","YAP1","JUN"}


def _onc():
    d = {}
    with open(REF / "ref_oncokb_oncogenes_hg19.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["chrom"] in OFF:
                d[row["gene"]] = (row["chrom"], float(row["pos_bp"]))
    return d


def _declutter(items, gap):
    lab = [x for x, _ in items]
    for i in range(1, len(lab)):
        if lab[i] - lab[i-1] < gap:
            lab[i] = lab[i-1] + gap
    if lab and lab[-1] > GEN/1e6:
        lab[-1] = GEN/1e6
        for i in range(len(lab)-2, -1, -1):
            if lab[i+1] - lab[i] < gap:
                lab[i] = lab[i+1] - gap
    return [(xt, lx, nm) for (xt, nm), lx in zip(items, lab)]


def _gnorm(grid, num, den, t):
    G = gaussian_filter(grid[f"{num}__{t}"].astype(float), (0.8, 1.5))
    seg = gaussian_filter(grid[f"{den}__{t}"].astype(float), 1.5)
    flo = np.percentile(seg[seg > 0], 10) if (seg > 0).any() else 1.0
    Rr = G / np.clip(seg, flo, None)[None, :]; Rr[:, seg < flo] = 0
    return Rr


def hot_genes(Rr, vthr, x_edges, N_X, ONC, max_n=11):
    """Oncogenes whose locus sits on a hot column of one density grid (>= vthr), decluttered."""
    hits = []
    for name, (ch, p) in ONC.items():
        col = int(np.clip(np.searchsorted(x_edges, OFF[ch] + p) - 1, 0, N_X - 1))
        val = float(Rr[:, max(0, col - 2):col + 3].max())
        if val >= vthr:
            hits.append((val * (1.6 if name in CANON else 1.0), (OFF[ch] + p) / 1e6, name))
    hits.sort(reverse=True); kept = []
    for eff, gx, name in hits:
        if all(abs(gx - kx) >= 22 for _, kx, _ in kept):
            kept.append((eff, gx, name))
        if len(kept) >= max_n:
            break
    return sorted((gx, nm) for _, gx, nm in kept)


def make(save=True, types=None, stem="a_landscape", out_dir=None, vmax=None, pdf=False):
    """types: tumour types shown as columns (default CURATED). vmax: {layer: float} colour-scale ceiling to
    share across several calls (the supplementary sheets); default is the 99th percentile of the shown types."""
    G = {k: np.load(CACHE / f, allow_pickle=True) for k, _, f, _, _, _ in LAYERS}
    gW = G["wgd"]; GT = list(gW["types"]); N_X = int(gW["N_X"]); N_Y = int(gW["N_Y"])
    n_all = {t: int(n) for t, n in zip(GT, gW["n_all"])}; n_wgd = {t: int(n) for t, n in zip(GT, gW["n_wgd"])}
    cc = np.load(CACHE / "_ccpg_counts.npz", allow_pickle=True)
    n_pgd = {t: int(n) for t, n in zip(list(cc["types"]), cc["n_pgd"])}
    xe = np.linspace(0, GEN, N_X + 1); xc = 0.5 * (xe[:-1] + xe[1:]) / 1e6; yc = (np.arange(N_Y) + 0.5) / N_Y
    ONC = _onc()
    cat = [t for t in (types if types is not None else CURATED) if t in GT]
    dens = {t: {k: _gnorm(G[k], num, den, t) for k, _, _, num, den, _ in LAYERS} for t in cat}
    VMAX = dict(vmax) if vmax is not None else {}
    for k, *_ in LAYERS:
        if k in VMAX:
            continue
        v = np.concatenate([dens[t][k][dens[t][k] > 0].ravel() for t in cat if (dens[t][k] > 0).any()])
        VMAX[k] = float(np.percentile(v, 99)) if v.size else 1.0

    # ── geometry (mm): 210 wide; three 11 mm rows per type; gene labels above; colourbars below ──
    W_MM, H_MM = 210.0, 64.0
    L_MM, R_MM, TOP_MM, ROW_MM, VGAP_MM, HGAP_MM = 8.0, 9.0, 13.0, 11.0, 1.6, 4.0
    col_w = (W_MM - L_MM - R_MM - HGAP_MM * (len(cat) - 1)) / len(cat)
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    IMS = {}
    for ci, t in enumerate(cat):
        x0 = (L_MM + ci * (col_w + HGAP_MM)) / W_MM
        for li, (k, lab, _, _, _, lcol) in enumerate(LAYERS):
            y1 = 1 - (TOP_MM + li * (ROW_MM + VGAP_MM)) / H_MM
            ax = fig.add_axes([x0, y1 - ROW_MM / H_MM, col_w / W_MM, ROW_MM / H_MM])
            IMS[k] = ax.pcolormesh(xc, yc, dens[t][k], cmap="inferno", shading="auto", vmin=0, vmax=VMAX[k],
                                   rasterized=True)
            for b in BOUND[1:-1]:
                ax.axvline(b / 1e6, color="white", lw=0.6, alpha=0.9)
            ax.set_ylim(0, 1); ax.set_xlim(0, GEN / 1e6)
            ax.set_yticks([0, 1]); ax.tick_params(labelsize=S.tick, length=1.5, pad=1.2, width=0.4)
            for lab_, va in zip(ax.get_yticklabels(), ("bottom", "top")):   # keep 0/1 inside the row
                lab_.set_va(va)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if li == len(LAYERS) - 1:
                ax.set_xticks([CMID[c] / 1e6 for c in CHROMS])
                ax.set_xticklabels([c if int(c) < 18 else "" for c in CHROMS], fontsize=S.tick)
                for c in ("18", "19", "20", "21", "22"):                  # crowded tail: smaller
                    ax.text(CMID[c] / 1e6, -0.06, c, transform=ax.get_xaxis_transform(), ha="center",
                            va="top", fontsize=S.tiny)
            else:
                ax.set_xticks([])
            if ci == 0 and li == 1:
                ax.set_ylabel("Molecular time", fontsize=S.label, labelpad=4)
            if ci == len(cat) - 1:                                        # layer tag, boxed, right of the row
                ax.text(1.012, 0.5, lab, transform=ax.transAxes, rotation=-90, ha="left", va="center",
                        fontsize=S.label, color=lcol, bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=lcol, lw=0.6))
            if li == 0:
                ax.set_title(f"{LAB.get(t, t)} (WGD {n_wgd[t]} / ccPG {n_pgd.get(t, 0)} / total {n_all[t]})",
                             fontsize=S.annot, pad=22)
                # oncogene labels above the top row: loci hot in the WGD-linked layer (as in the PDF)
                for xt, lx, nm in _declutter(hot_genes(dens[t]["wgd"], 0.35 * VMAX["wgd"], xe, N_X, ONC), 70.0):
                    ax.annotate(nm, xy=(xt, 1.0), xytext=(lx, 1.06), textcoords="data", rotation=45, ha="left",
                                va="bottom", fontsize=S.tiny, color=GENE_COL, annotation_clip=False,
                                arrowprops=dict(arrowstyle="-", lw=0.35, color=GENE_COL, ls=(0, (1.5, 1.5)),
                                                shrinkA=0, shrinkB=0))
    # horizontal colourbars, one per layer, in a row below the panels
    y_cb = 1 - (TOP_MM + 3 * ROW_MM + 2 * VGAP_MM + 8.5) / H_MM
    cb_w = 28.0 / W_MM
    for i, (k, lab, *_r) in enumerate(LAYERS):
        cx = (52.0 + i * 41.0) / W_MM
        cax = fig.add_axes([cx, y_cb, cb_w, 2.2 / H_MM])
        cb = fig.colorbar(IMS[k], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=S.tick, length=1.5, pad=1.5, rotation=90); cb.outline.set_linewidth(0.4)
        cax.set_title(f"Gain density ({lab})", fontsize=S.title, pad=2)
    if save:
        d = Path(out_dir) if out_dir is not None else PANELS
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.png", dpi=300)
        fig.savefig(d / f"{stem}.svg")
        if pdf:
            fig.savefig(d / f"{stem}.pdf")
    return fig


if __name__ == "__main__":
    make(); print("a_landscape: PNG+SVG written")
