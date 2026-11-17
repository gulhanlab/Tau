"""burst figure (main), panel d — EXCESS-burst density per sample by WGD relationship (stacked, window-normalized). One bar per
group; excess (dark teal) stacked over explained/≤D (light teal). Groups: wgd-linked (centroid within
±0.10 of a WGD event, ÷1 per 0.2), wgd-separate (÷4), non-WGD samples (÷5). Table-driven: fig6_bursts.tsv
+ tau_events (via _wgd.py). Mirrors dev proto_persample_bar.py (main stacked panel). make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.labelsize": 7,
                     "axes.linewidth": 0.6, "xtick.labelsize": 6, "ytick.labelsize": 6, "svg.fonttype": "none"})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import _wgd
from _wgd import NATREV, WGD_WIN, BURST_CATS, MM

PANELS = panels_dir_of(__file__)
GREY = NATREV["grey"]


def _per_sample(sel, samples, samp_idx, fac):
    ps = (pd.Series(np.asarray(sel, float), index=samp_idx).groupby(level=0).sum()
          .reindex(samples, fill_value=0).values) / fac
    return ps.mean(), ps.std(ddof=1) / np.sqrt(len(ps))


def make(save=True):
    bursts = _wgd.load_bursts(); events = _wgd.wgd_event_sets()
    burst = bursts[bursts["classification"].isin(BURST_CATS)].copy()
    burst["is_wgd"] = burst["sample"].isin(events.keys())
    wgd_samples = burst[burst.is_wgd]["sample"].unique()
    non_samples = burst[~burst.is_wgd]["sample"].unique()

    wb = burst[burst.is_wgd].reset_index(drop=True)
    linked = (wb["dist_to_wgd"].values < WGD_WIN)            # centroid within ±0.10 of nearest WGD event
    excess = wb["is_excess"].values
    samp = wb["sample"].values
    exc_o = linked & excess                                  # wgd-linked excess
    exp_o = linked & ~excess                                 # wgd-linked explained (≤D)
    sep_exc = ~linked & excess                               # wgd-separate excess
    sep_exp = ~linked & ~excess

    nb = burst[~burst.is_wgd]
    non_exc = nb["is_excess"].values

    exc_m = _per_sample(exc_o, wgd_samples, samp, 1.0)[0]
    exp_m = _per_sample(exp_o, wgd_samples, samp, 1.0)[0]
    sep_exc_m = _per_sample(sep_exc, wgd_samples, samp, 4.0)[0]
    sep_exp_m = _per_sample(sep_exp, wgd_samples, samp, 4.0)[0]
    non_exc_m = _per_sample(non_exc, non_samples, nb["sample"].values, 5.0)[0]
    non_exp_m = _per_sample(~non_exc, non_samples, nb["sample"].values, 5.0)[0]
    _, wl_s = _per_sample(linked, wgd_samples, samp, 1.0)
    _, sep_s = _per_sample(~linked, wgd_samples, samp, 4.0)
    _, non_s = _per_sample(np.ones(len(nb), bool), non_samples, nb["sample"].values, 5.0)

    STACKS = [(0.0, "wgd-linked", exc_m, exp_m, wl_s),
              (1.0, "wgd-separate", sep_exc_m, sep_exp_m, sep_s),
              (2.0, "non-WGD", non_exc_m, non_exp_m, non_s)]

    fig, ax = plt.subplots(figsize=(50 * MM, 42.5 * MM))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.91, bottom=0.22)
    for x, g, exc, expl, tot_s in STACKS:
        ax.bar(x, exc, width=0.62, color=NATREV["teal"][5], edgecolor="white", lw=0.5, zorder=3)
        ax.bar(x, expl, bottom=exc, width=0.62, color=NATREV["teal"][2], edgecolor="white", lw=0.5, zorder=3)
        tot = exc + expl
        ax.errorbar(x, tot, yerr=tot_s, fmt="none", ecolor=GREY[4], elinewidth=0.8, capsize=2.5, zorder=6)
        ax.text(x, -0.03, g, ha="right", va="top", fontsize=S.annot, fontweight="bold", rotation=30,
                rotation_mode="anchor", color=GREY[5], transform=ax.get_xaxis_transform())
    ax.set_xticks([]); ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, max(exc_m + exp_m, sep_exc_m + sep_exp_m, non_exc_m + non_exp_m) * 1.25 + 0.3)   # (no value labels above the bars any more)
    ax.set_ylabel("Mean burst density per sample\n(window-normalized, per 0.2)", fontsize=S.label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color=NATREV["teal"][5], label="excess"),
                       mpatches.Patch(color=NATREV["teal"][2], label="explained")],
              fontsize=S.annot, frameon=False, loc="upper right", handlelength=1.1, labelspacing=0.3)
    if save:
        fig.savefig(PANELS / "d_burst_density.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "d_burst_density.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make(); print("d_burst_density: PNG+SVG written")
