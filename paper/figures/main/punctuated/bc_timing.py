"""punctuated figure (main), panels b and c — timing-accuracy density heatmaps (estimated vs true event
time) on the benchmark simulations: b = WGD, c = ccPG. One script, two single-panel SVGs
(b_timing_wgd, c_timing_ccpg), each with its own horizontal "events" colourbar underneath, as laid out in
the assembled figure.
Table-driven: reads bench_events (matched WGD/PGD calls). Reproduces dev/benchmarking/plot_figure3_benchmarking.py "Panel b".

Note: the dev script additionally drops samples with >3 true events (a truth-derived filter that is
NOT recoverable from bench_events), so the per-class n is marginally higher here (all matched calls);
the heatmap shape and MAE are unchanged.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__)
MM = 1 / 25.4
WGD_COL, PGD_COL = "#0b6979", "#cd9f2c"

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 6.0, "axes.titlesize": 7.0, "axes.labelsize": 6.0,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "figure.dpi": 300,
})
_style_apply()   # the shared ladder wins over the sizes in the block above


def _matched():
    df = load_bench("events")
    m = df[df.is_matched.astype(bool) & df.classification.isin(["WGD", "PGD"])].copy()
    m["true_time"] = pd.to_numeric(m["true_time_frac"], errors="coerce")
    m["event_time"] = pd.to_numeric(m["event_time_frac"], errors="coerce")
    m = m.dropna(subset=["true_time", "event_time"])
    m["signed_error"] = m["event_time"] - m["true_time"]
    return m


PANEL = {"b": ("WGD", "WGD", WGD_COL, "magma", "b_timing_wgd"),
         "c": ("PGD", "ccPG", PGD_COL, "cividis", "c_timing_ccpg")}   # letter -> (class, title, colour, cmap, stem)


def make(save=True, letter="b"):
    cls, title, col, cmname, stem = PANEL[letter]
    df_matched = _matched()
    NB = 25
    edges = np.linspace(0, 1, NB + 1)
    sub = df_matched[df_matched["true_type"] == cls]
    mae = sub["signed_error"].abs().mean()
    H, _, _ = np.histogram2d(sub["true_time"], sub["event_time"], bins=[edges, edges])
    cm = plt.get_cmap(cmname).copy(); cm.set_bad(cm(0)); cm.set_under(cm(0))

    # 36 x 46 mm: 25 mm square axes, title above, colourbar strip below the x-label
    W, Hm = 36.0, 46.0
    fig = plt.figure(figsize=(W * MM, Hm * MM))
    ax = fig.add_axes([9.0 / W, 1 - 30.2 / Hm, 25.4 / W, 25.4 / Hm])
    im = ax.imshow(H.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal",
                   cmap=cm, norm=LogNorm(vmin=1, vmax=max(H.max(), 2)))
    ax.plot([0, 1], [0, 1], ls="--", color="white", lw=0.8, alpha=0.8)
    ax.set_title(title, color=col, loc="center", fontsize=S.title, pad=3)
    ax.text(0.95, 0.06, f"n={len(sub)}\nMAE {mae:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=S.annot, color="white")
    ax.set_xlabel("True time", fontsize=S.label, labelpad=1.5)
    ax.set_ylabel("Estimated time", fontsize=S.label if letter == "b" else 6.5, labelpad=1.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=S.tick, length=1.5, pad=1.5)
    ax.grid(True, color="white", lw=0.4, alpha=0.6)
    cax = fig.add_axes([14.0 / W, 5.0 / Hm, 20.0 / W, 1.8 / Hm])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=S.tick, length=1.5, pad=1)
    cb.outline.set_linewidth(0.5)
    fig.text(3.0 / W, 5.9 / Hm, "events", fontsize=S.label, ha="left", va="center")
    if save:
        fig.savefig(PANELS / f"{stem}.png", dpi=300)
        fig.savefig(PANELS / f"{stem}.svg")
        print(f"  saved {stem} -> {PANELS}")
    return fig


if __name__ == "__main__":
    make(letter="b"); make(letter="c"); print("b_timing_wgd + c_timing_ccpg: PNG+SVG written")
