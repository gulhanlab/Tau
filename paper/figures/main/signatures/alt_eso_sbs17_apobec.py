"""signatures figure (main), ALTERNATIVE (superseded by b_cluster_heatmaps.py, which is the version in the paper) — esophageal per-sample activity surface: SBS17 and APOBEC over molecular time. TABLE-DRIVEN.

Reads ONLY `tau_sig_timing` (load_sig_timing), tumor_type=="Eso.AdenoCA". Each heatmap column is one sample,
each row a molecular-time bin (y); colour = log2( activity ratio ÷ that sample's own flat-time baseline ),
so red = signature fires earlier than the clock in that interval, blue = later. Same interval math as
panel d: r = (ws/wc)·(ts_sig/ts_clock); per (sample, time-bin) pooled ratio = Σ ws·ts_sig / Σ wc·ts_clock;
per-sample baseline = Σ ts_sig / Σ ts_clock. Columns sorted by front-loading (early−late), so the
front-loaded majority / late minority split reads left→right. APOBEC (SBS2+SBS13) is the counterpart
(few esophageal samples carry it — small n, shown honestly). make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "font.size": 7, "font.family": "sans-serif"})
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.data_tables.tables import load_sig_timing

PANELS = panels_dir_of(__file__)
NB = 12; EDGES = np.linspace(0, 1, NB + 1)
CLIP = 1.5


def _intervals(sub):
    out = []
    for _, g in sub.groupby("seg_id"):
        g = g.sort_values("gain_index")
        Gc = np.r_[0.0, g.G_clock.values, 1.0]; Gs = np.r_[0.0, g.G_sig.values, 1.0]
        wc = np.diff(Gc); ws = np.diff(Gs); mid = 0.5 * (Gc[:-1] + Gc[1:])
        tc = g.ts_clock.iloc[0]; ts = g.ts_sig.iloc[0]
        for a, b, m in zip(ws * ts, wc * tc, mid):
            out.append((m, a, b))
    return np.array(out) if out else np.empty((0, 3))


def _sample_columns(sub):
    """-> (matrix [NB x n_samples] of log2(ratio/baseline), front-score per sample)."""
    cols, score = [], []
    for _, d in sub.groupby("sample"):
        iv = _intervals(d)
        if len(iv) == 0 or iv[:, 2].sum() <= 0:
            continue
        base = iv[:, 1].sum() / iv[:, 2].sum()
        col = np.full(NB, np.nan)
        for b in range(NB):
            hi = iv[:, 0] <= EDGES[b + 1] if b == NB - 1 else iv[:, 0] < EDGES[b + 1]
            sel = (iv[:, 0] >= EDGES[b]) & hi
            den = iv[:, 2][sel].sum()
            if den > 0 and iv[:, 1][sel].sum() > 0:
                col[b] = np.log2((iv[:, 1][sel].sum() / den) / base)
        cols.append(col)
        e = iv[iv[:, 0] < 0.5]; l = iv[iv[:, 0] >= 0.5]
        re = e[:, 1].sum() / max(e[:, 2].sum(), 1e-9); rl = l[:, 1].sum() / max(l[:, 2].sum(), 1e-9)
        score.append(np.log2(max(re, 1e-9) / max(rl, 1e-9)))
    if not cols:
        return np.empty((NB, 0)), np.array([])
    order = np.argsort(score)[::-1]                 # most front-loaded first
    return np.array(cols)[order].T, np.array(score)[order]


def _panel(ax, M, title):
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-CLIP, vmax=CLIP, origin="lower",
                   extent=[0, M.shape[1], 0, 1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "0.5", "1"], fontsize=S.tick)
    ax.set_xlabel(f"{M.shape[1]} samples  (front-loaded to late)", fontsize=S.label)
    ax.set_title(title, fontsize=S.title, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def make(save=True):
    df = load_sig_timing()
    eso = df[df.tumor_type == "Eso.AdenoCA"]
    M17, sc17 = _sample_columns(eso[eso.signature == "SBS17"])
    Map, scap = _sample_columns(eso[eso.signature == "SBS2+SBS13"])
    f17 = int((sc17 > 0).sum())
    fap = int((scap > 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 60 / 25.4),
                             gridspec_kw=dict(width_ratios=[M17.shape[1] + 4, Map.shape[1] + 4], wspace=0.18))
    im = _panel(axes[0], M17, f"SBS17  (n={M17.shape[1]}; {f17} front-loaded, {f17/max(M17.shape[1],1):.0%})")
    _panel(axes[1], Map, f"APOBEC (SBS2+SBS13)  (n={Map.shape[1]}; {fap} front-loaded)")
    axes[0].set_ylabel("molecular time  (0 early, 1 late)", fontsize=S.label)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("log$_2$( activity ÷ baseline )", fontsize=S.label)
    cb.set_ticks([-CLIP, 0, CLIP]); cb.set_ticklabels(["under", "0", "over"])
    cb.ax.tick_params(labelsize=S.tick, length=0); cb.outline.set_visible(False)
    fig.suptitle("Esophageal cohort: per-sample signature activity over molecular time", fontsize=S.title, y=1.02)
    if save:
        fig.savefig(PANELS / "alt_eso_sbs17_apobec.png", dpi=200, bbox_inches="tight")
        fig.savefig(PANELS / "alt_eso_sbs17_apobec.svg", bbox_inches="tight")
        try:      # display inline in a notebook (module forces Agg → show the saved PNG, like panels a/b)
            from IPython.display import Image, display
            display(Image(str(PANELS / "alt_eso_sbs17_apobec.png")))
        except Exception:
            pass
    return fig


if __name__ == "__main__":
    make(); print("alt_eso_sbs17_apobec: PNG+SVG written")
