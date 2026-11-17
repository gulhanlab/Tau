"""burst figure (main), panel i — per-oncogene δ_wgd swimmer (2 rows × 4 oncogenes). For each high-powered oncogene, every
excess-burst segment overlapping the locus is a dot at δ = burst centroid − nearest WGD event; sorted by
δ, teal within ±0.10 of WGD. Table-driven: fig6_bursts.tsv + tau_events (via _wgd.py). Mirrors the
panel_e_split variant of dev figure_burst_wgd_lead.py. make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl; mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.linewidth": 0.6,
                     "svg.fonttype": "none"})
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import _wgd
from _wgd import NATREV, WGD_COL, TXT, MM, HIGHPOWER, WGD_WIN

PANELS = panels_dir_of(__file__)


def make(save=True):
    bursts = _wgd.load_bursts(); events = _wgd.wgd_event_sets()
    b = _wgd.excess_wgd_bursts(bursts, events)
    near, sidx, P, _, _ = _wgd.near_matrix(b, events)
    own = near[np.arange(len(b)), sidx]

    # order oncogenes by observed enrichment (descending), like the dev split swimmer
    chrom, start, end = b["chrom"].values, b["start_bp"].values, b["end_bp"].values
    obs = {}
    for g in HIGHPOWER:
        c, m = _wgd.OG_MID[g]
        r = _wgd.enrich((chrom == c) & (start <= m) & (end >= m), near, own, sidx, P, n_perm=2000)
        if r:
            obs[g] = r["obs"]
    swim = sorted(obs, key=lambda g: -obs[g])[:8]
    alld = np.concatenate([_wgd.og_deltas(b, g, events) for g in swim])
    XL = min(1.0, float(np.abs(alld).max()) * 1.05 + 0.04)

    fig, axes = plt.subplots(2, 4, figsize=(82 * MM, 56 * MM))
    for idx, g in enumerate(swim):
        ax = axes[idx // 4][idx % 4]
        d = np.sort(_wgd.og_deltas(b, g, events)); n = len(d); yy = np.arange(n); nb = np.abs(d) < WGD_WIN
        ax.axvspan(-WGD_WIN, WGD_WIN, color=NATREV["teal"][0], alpha=0.5, zorder=0)
        ax.axvline(0, color=WGD_COL, lw=0.8, ls="--", zorder=2)
        ax.scatter(d, yy, c=[WGD_COL if x else NATREV["grey"][2] for x in nb], s=4, zorder=3, linewidths=0)
        med = float(np.median(d)); rk = (n - 1) / 2.0
        ax.scatter([med], [rk], color=WGD_COL, s=20, zorder=6, edgecolors="white", linewidths=0.7)
        ax.text(med + 0.16, rk, f"{med:+.2f}", fontsize=S.tiny, color=WGD_COL, va="center", ha="left", fontweight="bold")
        ax.set_xlim(-XL, XL); ax.set_ylim(-1, n + n * 0.08); ax.set_yticks([])
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xticklabels(["−.5", "0", ".5"] if idx >= 4 else [], fontsize=S.tick)
        ax.tick_params(axis="x", length=2, pad=1)
        ax.set_title(f"{g} (n={n})", fontsize=S.title, fontweight="bold", color=WGD_COL, pad=2)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.supxlabel("Time from burst centre to nearest WGD", fontsize=S.label, color=TXT, y=0.0)
    fig.tight_layout(h_pad=1.3, w_pad=1.1)
    if save:
        fig.savefig(PANELS / "i_wgd_swimmer.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "i_wgd_swimmer.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make(); print("f_wgd_swimmer: PNG+SVG written")
