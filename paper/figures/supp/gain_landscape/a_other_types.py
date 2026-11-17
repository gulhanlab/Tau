"""gain_landscape figure (supp), panel a — the gain landscape split three ways, for every other tumour type.

Fig. 5a shows three curated types. This is the full partition — WGD-linked, ccPG-linked and unlinked — for
the rest of the cohort, three stacked rows per type, so the tissue-specific structure highlighted in the
main figure can be checked against the whole cohort.

The three layers are the same partition as Fig. 3a: a gain is WGD-linked if a solution places it within
0.20 of a detected WGD, ccPG-linked if within 0.10 of a ccPG and not already WGD-linked, unlinked
otherwise. Each layer is divided by its OWN per-locus segment denominator, so a row is "gains per
segment that could have contributed to this layer" and the three are on comparable footings rather than
all being scaled by total coverage.

Colour scale is shared across tumour types WITHIN a layer (99th percentile), so rows of the same colour
mean the same rate in different tissues; it is NOT shared BETWEEN layers, because unlinked gains are an
order of magnitude sparser than WGD-linked ones and a common scale would render them blank.

Types are those with >=25 timed samples and >=5 WGD samples, excluding the three already in Fig. 5a and
the `Unknown` bucket — 17 types. Emitted as SHEETS of TYPES_PER_SHEET types, each sized to a journal
page, because at full-cohort height the rows have to be squeezed to the point of illegibility. The split
is presentational and the colour scale is shared across ALL sheets, so panels remain comparable between
them. Sheets are merged into s8_gain_landscape.pdf for `\\includegraphics[page=N]`.
"""
import sys
from pathlib import Path

import numpy as np
from paper.config import panels_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "paper/figures/main/landscape"))
import a_landscape as L                     # geometry, oncogene loci, declutter, _gnorm
# a_landscape sets its own rcParams at import (font.size 7, for the main-figure canvas), so the
# supplementary style has to be applied AFTER it or the sheets inherit main-figure type sizes.
from paper.figures._style import apply as _apply_style, save as _save_panel, MM, DOUBLE, S
_apply_style(compact=True)

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
MIN_ALL, MIN_WGD = 25, 5
TYPES_PER_SHEET = 4
# Nature double-column. ROW_H keeps the per-row aspect the 300 mm draft had (11.81/0.72 = 16.4), so
# narrowing the canvas does not squash the 20 time bins: 7.20/16.4 = 0.44.
WIDTH = DOUBLE
ROW_H = 0.44
EXCLUDE = set(L.CURATED) | {"Unknown"}
LAYERS = [("wgd", "WGD-linked", "_gain_density_grids.npz", "link", "seglink"),
          ("ccpg", "ccPG-linked", "_ccpg_gain_density_grids.npz", "link", "seglink"),
          ("unlk", "unlinked", "_unlinked_gain_density_grids.npz", "nonlink", "segnonlink")]


def _grids():
    return {k: np.load(L.CACHE / f, allow_pickle=True) for k, _, f, _, _ in LAYERS}


def other_types(g):
    t = list(g["types"]); n_all = dict(zip(t, g["n_all"])); n_wgd = dict(zip(t, g["n_wgd"]))
    return sorted(x for x in t if x not in EXCLUDE
                  and int(n_all[x]) >= MIN_ALL and int(n_wgd[x]) >= MIN_WGD)


def make(save=True):
    G = _grids()
    gW = G["wgd"]
    types = other_types(gW)
    n_all = {t: int(n) for t, n in zip(list(gW["types"]), gW["n_all"])}
    n_wgd = {t: int(n) for t, n in zip(list(gW["types"]), gW["n_wgd"])}
    cc = np.load(L.CACHE / "_ccpg_counts.npz", allow_pickle=True)
    n_pgd = {t: int(n) for t, n in zip(list(cc["types"]), cc["n_pgd"])}

    N_X, N_Y = int(gW["N_X"]), int(gW["N_Y"])
    xe = np.linspace(0, L.GEN, N_X + 1)
    xc = 0.5 * (xe[:-1] + xe[1:]) / 1e6
    yc = (np.arange(N_Y) + 0.5) / N_Y

    dens = {t: {k: L._gnorm(G[k], num, den, t) for k, _, _, num, den in LAYERS} for t in types}
    VMAX = {}
    for k, _, _, _, _ in LAYERS:
        v = np.concatenate([dens[t][k][dens[t][k] > 0].ravel() for t in types
                            if (dens[t][k] > 0).any()])
        VMAX[k] = float(np.percentile(v, 99)) if v.size else 1.0

    figs = []
    # even split, so a sheet is never left with one lonely type
    n_sheets = max(1, -(-len(types) // TYPES_PER_SHEET))
    sheets = [list(c) for c in np.array_split(np.array(types, dtype=object), n_sheets)]
    for si, chunk in enumerate(sheets, start=1):
        nrow = len(chunk) * len(LAYERS)
        fig = plt.figure(figsize=(WIDTH, ROW_H * nrow + 0.95))
        # nested: the three layers of one type sit tight together, types are separated — so a reader
        # groups by tumour type rather than by row position
        outer = fig.add_gridspec(len(chunk), 1, hspace=0.62)
        IMS = {}
        for ti, t in enumerate(chunk):
            inner = outer[ti].subgridspec(len(LAYERS), 1, hspace=0.16)
            for li, (k, lab, _, _, _) in enumerate(LAYERS):
                ax = fig.add_subplot(inner[li])
                Rr = dens[t][k]
                IMS[k] = ax.pcolormesh(xc, yc, Rr, cmap="inferno", shading="auto",
                                       vmin=0, vmax=VMAX[k], rasterized=True)
                for b in L.BOUND[1:-1]:
                    ax.axvline(b / 1e6, color="white", lw=0.8, alpha=0.9)
                for c in L.CHROMS:
                    ax.axvline(L.CENPOS[c], color="white", lw=0.25, ls=(0, (2, 2)), alpha=0.45)
                ax.set_ylim(0, 1); ax.set_xlim(0, L.GEN / 1e6)
                ax.set_yticks([0, 1]); ax.tick_params(labelsize=S.tick, length=1.4, pad=1.2)
                # rows are stacked tightly: pull the 0 up and the 1 down so neighbouring rows' labels
                # don't collide
                for lab_, va in zip(ax.get_yticklabels(), ("bottom", "top")):
                    lab_.set_va(va)
                ax.set_ylabel(lab, fontsize=S.label, rotation=0, ha="right", va="center", labelpad=5)
                if li == len(LAYERS) - 1:
                    ax.set_xticks([L.CMID[c] / 1e6 for c in L.CHROMS])
                    ax.set_xticklabels(L.CHROMS, fontsize=S.tick)
                else:
                    ax.set_xticks([])
                if li == 0:
                    ax.set_title(f"{t}    (WGD {n_wgd[t]} · ccPG {n_pgd.get(t, 0)} · "
                                 f"non-WGD {n_all[t]-n_wgd[t]})",
                                 fontsize=S.annot, fontweight="bold", pad=3)
        fig.subplots_adjust(top=0.935, bottom=0.075, left=0.095, right=0.885)
        for (k, lab, _, _, _), y0 in zip(LAYERS, (0.68, 0.40, 0.12)):
            cax = fig.add_axes([0.900, y0, 0.008, 0.22])
            cb = fig.colorbar(IMS[k], cax=cax); cb.set_label(f"Gain density ({lab})", fontsize=S.tick)
            cb.ax.tick_params(labelsize=S.tick, length=1.4, pad=1.0)
        if save:
            # no bbox_inches="tight": it discards the authored canvas and takes the sheets off the
            # column grid. _save_panel also applies the longhand-font SVG fix for Illustrator.
            _save_panel(fig, PANELS, f"p{si}_other_types", dpi=300, pdf=True)
        print(f"  sheet {si} ({nrow} rows): {', '.join(chunk)}")
        figs.append(fig)
        plt.close(fig)
    if save:
        _merge_pdf(n_sheets)
    return figs


def _merge_pdf(n_sheets):
    """Stitch the per-sheet PDFs into one multi-page s8 PDF, for includegraphics[page=N]."""
    from pypdf import PdfWriter
    w = PdfWriter()
    for si in range(1, n_sheets + 1):
        w.append(str(PANELS / f"p{si}_other_types.pdf"))
    out = PANELS / "s8_gain_landscape.pdf"
    with open(out, "wb") as f:
        w.write(f)
    print(f"  merged {n_sheets} pages -> {out}")


if __name__ == "__main__":
    make(); print(f"a_other_types: {len(list(PANELS.glob('p*_other_types.pdf')))} sheets -> {PANELS}")
