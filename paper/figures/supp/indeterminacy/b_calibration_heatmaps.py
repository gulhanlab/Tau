"""indeterminacy figure (supp), panel b — calibration heatmaps: estimated vs true cumulative gain time,
stratified by DETERMINED / UNDETERMINED (rows) × mutation-count tier (cols) = 2×4 = 8 heatmaps.
Faithful scoring (no band midpoint): determined = point estimate (time_lo==time_hi) vs truth;
underdetermined = the WHOLE polytope band [time_lo,time_hi] smeared (equal mass per gain) vs truth,
error = distance-to-set (0 if the truth is inside the band); shown as |err| for symmetry with
the determined row, where it is the plain absolute error. Both are defined in Supp. Fig. 3b. Table-driven from bench_seg_gains.
Ported from dev/benchmarking/panel_accuracy_stratified.py.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE, DOUBLE
_apply_style()          # full double-column figure: standard 6-7 pt scale, not compact
from paper.config import panels_dir_of  # noqa: E402
from paper.data_tables.tables import load_bench

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
TIERS = [("<50", 0, 50), ("50–100", 50, 100), ("100–200", 100, 200), ("≥200", 200, 1e12)]
BINS = np.linspace(0, 1, 41)



def make(save=True):
    df = load_bench("seg_gains")
    # split per tier, then subsample each row to a COMMON n so colour density is comparable across columns
    splits = []
    for lab, lo, hi in TIERS:
        t = df[(df.mut_count >= lo) & (df.mut_count < hi)]
        splits.append((lab, t[~t.underdetermined], t[t.underdetermined]))
    cap_det = min(len(d) for _, d, _ in splits)
    cap_und = min(len(u) for _, _, u in splits)

    fig, axes = plt.subplots(2, len(TIERS), figsize=(DOUBLE, 96 * MM), sharex=True, sharey=True)
    last = None
    for c, (lab, det, und) in enumerate(splits):
        det = det.sample(cap_det, random_state=0)
        und = und.sample(cap_und, random_state=0)
        # row 0 — determined: point estimate (time_lo == time_hi)
        a = axes[0, c]; a.set_facecolor("#000004")
        a.hist2d(det.true_time, det.time_lo, bins=BINS, cmap="magma", norm=LogNorm(vmin=1), cmin=1)
        a.plot([0, 1], [0, 1], "--", c="white", lw=0.8, alpha=0.8)
        e = (det.time_lo - det.true_time).abs()
        a.set_title(f"{lab} mut   (n={len(det):,})", fontsize=S.title, pad=3)
        a.text(0.95, 0.06, f"|err| {e.median():.3f}\n{(e < 0.05).mean():.0%} <0.05",
               transform=a.transAxes, ha="right", va="bottom", fontsize=S.annot, color="white")
        # row 1 — underdetermined: valid-set band [lo,hi] smeared, equal mass per gain
        b = axes[1, c]; b.set_facecolor("#000004")
        K = 25; frac = np.linspace(0, 1, K)
        xs = np.repeat(und.true_time.values, K)
        ys = (und.time_lo.values[:, None] + (und.time_hi - und.time_lo).values[:, None] * frac[None, :]).ravel()
        last = b.hist2d(xs, ys, bins=BINS, weights=np.full(xs.shape, 1.0 / K),
                        cmap="magma", norm=LogNorm(vmin=0.5), cmin=0.04)[3]
        b.plot([0, 1], [0, 1], "--", c="white", lw=0.8, alpha=0.8)
        d = und.dist_to_set
        b.set_title(f"{lab} mut   (n={len(und):,})", fontsize=S.title, pad=3)
        b.text(0.95, 0.06, f"|err| {d.median():.3f}\n{(d < 0.05).mean():.0%} <0.05",
               transform=b.transAxes, ha="right", va="bottom", fontsize=S.annot, color="white")

    for a in axes.ravel():
        a.set_aspect("equal"); a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
    axes[0, 0].set_ylabel("DETERMINED\nestimated gain time", fontsize=S.label)
    axes[1, 0].set_ylabel("UNDETERMINED\nestimated gain times", fontsize=S.label)
    fig.supxlabel("true cumulative gain time", fontsize=S.label, y=0.02)
    cb = fig.colorbar(last, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label("gain density per cell", fontsize=S.tick); cb.ax.tick_params(labelsize=S.tick)
    # plain decade labels — the default log formatter draws 10^n, whose superscript renders at ~0.7x
    # the tick size and so falls under the 5 pt floor
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.yaxis.set_minor_formatter(plt.NullFormatter())
    if save:
        _save_panel(fig, PANELS, 'b_calibration_heatmaps')
    return fig


if __name__ == "__main__":
    make(); print(f"identifiability/b: written -> {PANELS}")
