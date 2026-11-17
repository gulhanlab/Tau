"""Fig 6 (top-hits) — strongest excess-burst hotspot chromosome ARMS. Excess amplicons are arm-scale, so
the natural unit is the arm: 2×2 (unit = sample) does a sample carry an excess burst OVERLAPPING this arm
vs the rest of the genome → OR + 95% CI, top arms as a horizontal forest annotated with oncogenes.
Table-driven: fig6_bursts.tsv (excess bursts) + tau_events (sample count). Mirrors dev
explore_genome_or_scan.py (arm section) + panel_top_hits.py. make() -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 6, "axes.linewidth": 0.5, "svg.fonttype": "none"})
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
NTOP, MIN_A = 5, 5
CHROMS = [str(i) for i in range(1, 23)] + ["X"]
CSIZE = {"1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260, "6": 171115067,
         "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
         "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753, "17": 81195210, "18": 78077248,
         "19": 59128983, "20": 63025520, "21": 48129895, "22": 51304566, "X": 155270560}
CENT = {"1": 125000000, "2": 93300000, "3": 91000000, "4": 50400000, "5": 48400000, "6": 61000000,
        "7": 59900000, "8": 45600000, "9": 49000000, "10": 40200000, "11": 53700000, "12": 35800000,
        "13": 17900000, "14": 17600000, "15": 19000000, "16": 36600000, "17": 24000000, "18": 17200000,
        "19": 26500000, "20": 27500000, "21": 13200000, "22": 14700000, "X": 60600000}
ACRO = {"13", "14", "15", "21", "22"}
ONC = {"MYC": ("8", 127735434), "EGFR": ("7", 55086725), "MET": ("7", 116312457), "CDK6": ("7", 92234236),
       "PIK3CA": ("3", 178866311), "MDM4": ("1", 204485509), "TERT": ("5", 1253167), "ERBB2": ("17", 37844393),
       "CCND1": ("11", 69165054), "CCNE1": ("19", 30302805), "MYCN": ("2", 16080683), "MDM2": ("12", 69201952),
       "SOX2": ("3", 181429711), "CDK4": ("12", 58141510), "KRAS": ("12", 25358180), "FGFR1": ("8", 38411656),
       "ZNF217": ("20", 52183609), "AURKA": ("20", 54944445), "NKX2-1": ("14", 36985603), "MCL1": ("1", 150547027)}


def _arm_or():
    N = load_events()["sample"].nunique()
    df = _wgd.load_bursts()
    eb = df[df["classification"].isin(BURST_CATS) & df["is_excess"] & df["chrom"].isin(CHROMS)]
    ARMS = []
    for c in CHROMS:
        if c not in ACRO:
            ARMS.append((c, "p", 0, CENT[c]))
        ARMS.append((c, "q", CENT[c], CSIZE[c]))
    NARM = len(ARMS)
    acov = {ai: set() for ai in range(NARM)}
    for _, r in eb.iterrows():
        for ai, (c, arm, s0, s1) in enumerate(ARMS):
            if r["chrom"] == c and r["start_bp"] < s1 and r["end_bp"] > s0:
                acov[ai].add(r["sample"])
    aa = np.array([len(acov[ai]) for ai in range(NARM)]); atot = aa.sum()
    rows = []
    for ai, (c, arm, s0, s1) in enumerate(ARMS):
        a = aa[ai]
        if a < MIN_A:
            continue
        bb = N - a; cc = atot - a; d = N * (NARM - 1) - cc
        ORa = (a * d) / (bb * cc); se = np.sqrt(1 / a + 1 / bb + 1 / cc + 1 / d)
        genes = [g for g, (gc, gp) in ONC.items() if gc == c and (("p" if gp < CENT[c] else "q") == arm)]
        rows.append({"armlabel": f"{c}{arm}", "OR": ORa, "lo": ORa * np.exp(-1.96 * se),
                     "hi": ORa * np.exp(1.96 * se), "a": int(a), "oncogenes": ",".join(genes)})
    return pd.DataFrame(rows).sort_values("OR", ascending=False).reset_index(drop=True)


def make(save=True):
    df = _arm_or().head(NTOP).iloc[::-1]
    xmax = df["hi"].max()
    fig, ax = plt.subplots(figsize=(54 * MM, 22 * MM))
    fig.subplots_adjust(left=0.34, right=0.97, top=0.86, bottom=0.24)
    y = range(len(df))
    ax.barh(y, df["OR"], color=TEAL[3], height=0.64, zorder=2)
    ax.errorbar(df["OR"], y, xerr=[df["OR"] - df["lo"], df["hi"] - df["OR"]], fmt="none",
                ecolor=TXT, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.axvline(1, color=GREY[3], lw=0.7, ls="--", zorder=1)
    ax.set_yticks(list(y)); ax.set_yticklabels([f"chr{r['armlabel']}" for _, r in df.iterrows()], fontsize=S.tick)
    for yi, (_, r) in zip(y, df.iterrows()):
        g = r["oncogenes"] if isinstance(r["oncogenes"], str) else ""
        if g:
            ax.text(r["hi"] + xmax * 0.03, yi, g.replace(",", "/"), fontsize=S.annot, color=TEAL[5],
                    fontweight="bold", va="center", ha="left")
    ax.set_xlabel("excess-burst OR (arm vs genome)", fontsize=S.label, color=TXT)
    ax.set_xlim(0, xmax * 1.55)
    ax.set_title("Excess-burst hotspot arms", fontsize=S.title, loc="left", color=TXT)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2)
    if save:
        fig.savefig(PANELS / "f_top_hits.png", dpi=300, bbox_inches="tight")
        fig.savefig(PANELS / "f_top_hits.svg", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    make(); print("top_hits: PNG+SVG written")
