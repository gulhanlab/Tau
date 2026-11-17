"""signatures figure (main) — SBS17-vs-clock activity over molecular time (gastric/eso cohort). TABLE-DRIVEN.

Reads ONLY `tau_sig_timing` (load_sig_timing). Per timed segment we rebuild the gain intervals from the
cumulative activity shares: Gc = [0, G_clock…, 1], Gs = [0, G_sig…, 1]; interval widths wc=diff(Gc),
ws=diff(Gs). The UNNORMALIZED interval ratio is r = (ws/wc)·(ts_sig/ts_clock) (use unnormalized,
ts_*/ its baseline = D_sig/D_clock abundance). Pooled per molecular-time bin (clock midpoint):
    ratio(bin) = Σ ws·ts_sig / Σ wc·ts_clock
    baseline   = Σ ts_sig / Σ ts_clock            # flat-time expectation (activity uniform over time)
    y          = log2( ratio(bin) / baseline )
Front-loaded ⇒ early bins > 0, late bins < 0. Mirrors dev sig_absolute_ratio_v2.py LEFT panel, re-centred
on the ts baseline. Eso ≈75% front-loaded / 25% late; gastric+eso pooled ≈59%.
NOTE: SBS17 rows carry no per-segment `frac` in the table, so the dev exposure stratification
(minimal/moderate/high) is not reproducible here — the SBS17-bearing tumour types are inherently high-SBS17.
make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "font.size": 7, "font.family": "sans-serif"})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.data_tables.tables import load_sig_timing

PANELS = panels_dir_of(__file__)
GASTRIC_ESO = ["Eso.AdenoCA", "Stomach.AdenoCA"]
NB = 10; EDGES = np.linspace(0, 1, NB + 1); CTR = 0.5 * (EDGES[:-1] + EDGES[1:])
COL = {"Eso.AdenoCA": "#b23a48", "Stomach.AdenoCA": "#e0995e"}
LAB = {"Eso.AdenoCA": "Esophageal", "Stomach.AdenoCA": "Stomach"}


def _per_sample_intervals(sub):
    """sample -> array [clock_mid, abs_sig, abs_clock] per gain interval (built once, reused in bootstrap)."""
    per = {}
    for samp, sd in sub.groupby("sample"):
        out = []
        for _, g in sd.groupby("seg_id"):
            g = g.sort_values("gain_index")
            Gc = np.r_[0.0, g.G_clock.values, 1.0]; Gs = np.r_[0.0, g.G_sig.values, 1.0]
            wc = np.diff(Gc); ws = np.diff(Gs); mid = 0.5 * (Gc[:-1] + Gc[1:])
            tc = g.ts_clock.iloc[0]; ts = g.ts_sig.iloc[0]
            for a, b, m in zip(ws * ts, wc * tc, mid):
                out.append((m, a, b))
        per[samp] = np.array(out) if out else np.empty((0, 3))
    return per


def _binned_log2(iv, baseline):
    y = np.full(NB, np.nan)
    for b in range(NB):
        sel = (iv[:, 0] >= EDGES[b]) & (iv[:, 0] < EDGES[b + 1] if b < NB - 1 else iv[:, 0] <= EDGES[b + 1])
        d = iv[:, 2][sel].sum()
        if d > 0 and iv[:, 1][sel].sum() > 0:
            y[b] = np.log2((iv[:, 1][sel].sum() / d) / baseline)
    return y


def _front_frac(per):
    """Per sample: front-loaded if pooled early(<0.5) ratio > late(>=0.5) ratio."""
    f = t = 0
    for iv in per.values():
        if len(iv) == 0:
            continue
        e = iv[iv[:, 0] < 0.5]; l = iv[iv[:, 0] >= 0.5]
        re = e[:, 1].sum() / max(e[:, 2].sum(), 1e-9); rl = l[:, 1].sum() / max(l[:, 2].sum(), 1e-9)
        f += re > rl; t += 1
    return f, t


def make(save=True):
    df = load_sig_timing()
    s17 = df[(df.signature == "SBS17") & (df.tumor_type.isin(GASTRIC_ESO))].copy()

    fig, ax = plt.subplots(figsize=(88 / 25.4, 66 / 25.4))
    ax.axhline(0, color="#888888", ls="--", lw=0.7, zorder=1)
    notes = []
    for tt in GASTRIC_ESO:
        per = _per_sample_intervals(s17[s17.tumor_type == tt])
        samps = np.array(list(per))
        iv = np.vstack([per[s] for s in samps if len(per[s])])
        base = iv[:, 1].sum() / iv[:, 2].sum()
        y = _binned_log2(iv, base)
        # bootstrap 95% CI over samples (resample whole samples from precomputed intervals)
        rng = np.random.default_rng(0); B = 400
        boot = np.full((B, NB), np.nan)
        for bi in range(B):
            pick = [per[s] for s in rng.choice(samps, len(samps), replace=True) if len(per[s])]
            boot[bi] = _binned_log2(np.vstack(pick), base)
        lo = np.nanpercentile(boot, 2.5, axis=0); hi = np.nanpercentile(boot, 97.5, axis=0)
        m = np.isfinite(y)
        ax.fill_between(CTR[m], lo[m], hi[m], color=COL[tt], alpha=0.15, lw=0, zorder=2)
        ax.plot(CTR[m], y[m], color=COL[tt], lw=2.0, marker="o", ms=3.5, zorder=3)
        f, t = _front_frac(per); notes.append(f"{LAB[tt]} {f}/{t} front-loaded ({f/t:.0%}); base {base:.1f}×")

    ax.set_xlim(0, 1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("molecular time  (0 early, 1 late)", fontsize=S.label)
    ax.set_ylabel("log$_2$( SBS17:clock activity ÷ baseline )", fontsize=S.label)
    ax.text(0.03, 1.02, "SBS17 over-active (front-loaded)", fontsize=S.annot, color="#b23a48", va="bottom")
    ax.text(0.03, -1.02, "SBS17 under-active (late)", fontsize=S.annot, color="#4064a0", va="top")
    ax.legend(handles=[Line2D([0], [0], color=COL[t], lw=2, marker="o", ms=3.5, label=LAB[t])
                       for t in GASTRIC_ESO], frameon=False, fontsize=S.annot, loc="upper right")
    ax.set_title("SBS17 activity is front-loaded relative to the clock\n" + "; ".join(notes),
                 fontsize=S.annot, pad=4)
    ax.tick_params(length=2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(PANELS / "alt_sbs17_ratio.png", dpi=200, bbox_inches="tight")
        fig.savefig(PANELS / "alt_sbs17_ratio.svg", bbox_inches="tight")
        try:      # display inline in a notebook (module forces Agg → show the saved PNG, like panels a/b)
            from IPython.display import Image, display
            display(Image(str(PANELS / "alt_sbs17_ratio.png")))
        except Exception:
            pass
    return fig


if __name__ == "__main__":
    make(); print("d_sbs17_ratio: PNG+SVG written")
