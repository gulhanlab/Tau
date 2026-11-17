"""burst figure (main), panel e — excess-burst BURDEN: mean excess bursts per sample by doubling class (non-WGD / PGD / 1×WGD /
2×WGD). Table-driven: fig6_bursts.tsv + tau_events (WGD-event count and PGD membership per sample). Mirrors
dev panel_burden.py. make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6.5, "svg.fonttype": "none"})
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).parent))
from paper.config import panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.data_tables.tables import load_events
import _wgd
from _wgd import NATREV, BURST_CATS, MM

PANELS = panels_dir_of(__file__)
TEAL, GREY, TXT = NATREV["teal"], NATREV["grey"], "#273246"
CLS = ["non-WGD", "PGD", "1×WGD", "2×WGD"]
CCOL = {"non-WGD": GREY[3], "PGD": TEAL[2], "1×WGD": TEAL[4], "2×WGD": TEAL[5]}


def make(save=True):
    ev = load_events()
    nw = ev[ev.classification == "WGD"].groupby("sample").size()
    pgd = set(ev[ev.classification == "PGD"]["sample"])

    def cl(s):
        k = int(nw.get(s, 0))
        return "2×WGD" if k >= 2 else "1×WGD" if k == 1 else "PGD" if s in pgd else "non-WGD"

    df = _wgd.load_bursts()
    df["excess"] = df["classification"].isin(BURST_CATS) & df["is_excess"]
    df["class"] = df["sample"].map(cl)
    per = df.groupby(["sample", "class"])["excess"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(45 * MM, 43.5 * MM))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.91, bottom=0.22)
    xs = np.arange(len(CLS)); ms = []; se = []
    for c in CLS:
        v = per[per["class"] == c]["excess"].values
        ms.append(v.mean()); se.append(v.std(ddof=1) / np.sqrt(len(v)))
    ax.bar(xs, ms, yerr=se, width=0.7, color=[CCOL[c] for c in CLS], edgecolor="white", lw=0.5,
           error_kw=dict(elinewidth=0.8, capsize=2.5, ecolor=GREY[4]))
    ax.set_xticks(xs); ax.set_xticklabels([c.replace("PGD", "ccPG") for c in CLS], rotation=25, ha="right", fontsize=S.tick)
    ax.set_ylabel("excess bursts / sample", fontsize=S.label, color=TXT)
    ax.set_title("Excess-burst burden", fontsize=S.title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    if save:
        fig.savefig(PANELS / "e_burden_by_doubling.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "e_burden_by_doubling.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make(); print("g_burden_by_doubling: PNG+SVG written")
