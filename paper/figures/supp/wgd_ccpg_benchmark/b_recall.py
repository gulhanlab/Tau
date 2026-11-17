"""wgd_ccpg_benchmark figure (supp), panel b — WGD/PGD detection recall vs event size (B1) and vs event separation (B2).
TABLE-DRIVEN from bench_truth_events (one row per TRUE event, with event_size_frac, min_dist_frac,
is_detected — derived from the simulation truth by tau.aggregate.benchmark). An event counts as detected
if any called event fell within ±0.15 molecular-time units of it. Mirrors dev Panel B1/B2.
make() -> fig + PNG/SVG.

Letter is d (not c) only because the timing heatmap was split into panels b and c; the plot itself is
unchanged.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE, DOUBLE, HALF
_apply_style()
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
NAVY = "#26456e"

ES_BINS, ES_LAB = [0.0, 0.10, 0.30, 1.01], ["<10%", "10–30%", ">30%"]
MD_BINS, MD_LAB = [0.0, 0.20, 0.30, 1.01], ["<0.20", "0.20–0.30", ">0.30"]


def _bar(ax, binned, labels, xlabel):
    x = np.arange(len(labels))
    ax.bar(x, binned["sens"].fillna(0), 0.66, color=NAVY, alpha=0.85, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0); ax.set_xlabel(xlabel)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def make(save=True):
    te = load_bench("truth_events")
    te["es_bin"] = pd.cut(te.event_size_frac, bins=ES_BINS, labels=ES_LAB, include_lowest=True)
    b_es = (te.groupby("es_bin", observed=True).agg(n=("is_detected", "count"), sens=("is_detected", "mean"))
            .reindex(ES_LAB).reset_index())
    md = te.dropna(subset=["min_dist_frac"]).copy()   # separation only defined for genomes with >=2 events
    md["md_bin"] = pd.cut(md.min_dist_frac, bins=MD_BINS, labels=MD_LAB, include_lowest=True)
    b_md = (md.groupby("md_bin", observed=True).agg(n=("is_detected", "count"), sens=("is_detected", "mean"))
            .reindex(MD_LAB).reset_index())

    # 48 x 42 mm — the original figure-3 geometry: each subplot's plot-width matches panel d, so the bars
    # are equally thin, and top/bottom are set to d's so the two y-axis boxes align in a shared bar row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(48 * MM, 42 * MM), sharey=True)
    _bar(ax1, b_es, ES_LAB, "WGD/ccPG event\nsize (gen. frac.)")
    _bar(ax2, b_md, MD_LAB, "Distance to\nnearest event")
    ax1.set_ylabel("WGD/ccPG recall", fontsize=S.label)
    fig.subplots_adjust(left=0.26, right=0.99, top=0.92, bottom=0.42, wspace=0.18)
    if save:
        _save_panel(fig, PANELS, 'b_recall')
    return fig


if __name__ == "__main__":
    make(); print("b_recall: PNG+SVG written (table-driven from bench_truth_events)")
