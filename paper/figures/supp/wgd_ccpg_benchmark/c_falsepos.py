"""wgd_ccpg_benchmark figure (supp), panel c — false-positive rate: fraction of samples with 0, 1, 2, ... spurious WGD/ccPG calls.
Table-driven: reads bench_events (a false positive = a WGD/PGD call with is_matched == False).
`make()` returns the figure and writes PNG+SVG to the supp panels dir. Reproduces the FP histogram of
dev/benchmarking/plot_figure3_benchmarking.py "Panel d".

Denominator caveat: bench_events lists CALLED events only, so the denominator here is the set of
samples that produced at least one call. The dev script normalises over the full valid-sample roster
(including samples that produced zero calls, all of which are 0-FP), which is truth-derived and NOT
recoverable from the bench tables. Those extra 0-FP samples only raise the 0-FP bar, so the honest
bench-table figure reads ~91% zero-FP vs the dev snapshot's 92%; the distribution shape is identical.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE, DOUBLE, HALF
_apply_style()
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
NAVY_COL = "#273246"



def make(save=True):
    df = load_bench("events")
    is_fp = (~df.is_matched.astype(bool)) & df.classification.isin(["WGD", "PGD"])
    fp_per_sample = df.assign(_fp=is_fp).groupby("sample")["_fp"].sum()
    n_total = fp_per_sample.shape[0]

    fp_binned = {}
    for k in fp_per_sample:
        key = int(k) if int(k) < 3 else 3
        fp_binned[key] = fp_binned.get(key, 0) + 1
    x_vals = sorted(fp_binned.keys())
    x_labels = [str(k) if k < 3 else "≥3" for k in x_vals]
    pct_arr = np.array([fp_binned[k] for k in x_vals]) / n_total

    # 30 x 42 mm — the original figure-3 geometry: same plot-area width as ONE panel-c subplot, so the
    # bars are the same thickness, and the same top/bottom margins so the two y-axis boxes line up
    fig, ax = plt.subplots(figsize=(30 * MM, 42 * MM))
    ax.bar(range(len(x_vals)), pct_arr, color=NAVY_COL, alpha=0.88, width=0.66, zorder=3)
    ax.set_xticks(range(len(x_vals))); ax.set_xticklabels(x_labels)
    ax.set_xlabel("# false positives", fontsize=S.label)
    ax.set_ylabel("Fraction of samples", fontsize=S.label)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.0)
    ax.grid(False)
    fig.subplots_adjust(left=0.42, right=0.96, top=0.92, bottom=0.42)
    if save:
        _save_panel(fig, PANELS, 'c_falsepos')
    return fig


if __name__ == "__main__":
    make(); print("c_falsepos: PNG+SVG written")
