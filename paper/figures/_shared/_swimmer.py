"""Shared per-gain swimmer: landscape figure panel b (MYC) and the extra_oncogenes supp (PIK3CA, EGFR).
Table-driven: reads figures/data/landscape/oncogene_gains.tsv.gz (gains at oncogene loci, from
tau_gains) + tau_events (WGD/ccPG times). One column per amplified sample (>= MIN_GAINS gains), sorted by
first-gain time within tumour type; each gain coloured by its own linkage: WGD teal / ccPG orange /
unlinked grey. Styled as in the assembled figure:
210 mm wide, type name + n above each block, "-> Sample" cue, legend centred below.
`make(gene)` -> fig + PNG/SVG.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

R = Path(__file__).resolve().parents[3]   # repo root; sys.path.insert(0, str(R))
from paper.config import cache_dir_of, panels_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.data_tables.tables import load_events

# The swimmer's cache lives in the landscape figure's data dir (figures/data/landscape); the
# calling panel script passes its own panels dir via make(..., panels_dir=...).
CACHE = Path(__file__).resolve().parents[1] / "data/landscape"
PANELS = None   # set per caller
MIN_GAINS, MIN_TT, SNAP = 3, 8, 0.10
WGD_COL, PGD_COL, NONE_COL, DOT = "#146a79", "#faa51a", "#b7b6b6", 7    # colours from the PDF
MM = 1 / 25.4
mpl.rcParams.update({"font.size": 7, "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"]})
_style_apply()   # the shared ladder wins over the sizes in the block above


def _linkage_times():
    ev = load_events()
    W = {s: sorted(g.event_time_frac.values) for s, g in ev[ev.classification == "WGD"].groupby("sample")}
    P = {s: sorted(g.event_time_frac.values) for s, g in ev[ev.class_strict == "PGD"].groupby("sample")}
    return W, P


def make(gene, save=True, panels_dir=None, stem=None):
    WGDT, PGDT = _linkage_times()
    g = pd.read_csv(CACHE / "oncogene_gains.tsv.gz", sep="\t")
    g = g[g.oncogene == gene]
    amps = []
    # per amplification (sample, seg_id): median gain time per gain_index -> sorted
    for (s, sid), gg in g.groupby(["sample", "seg_id"]):
        n_ev = gg.gain_index.max()
        if n_ev < MIN_GAINS:
            continue
        med = gg.groupby("gain_index").gain_time_frac.median().sort_index().values
        amps.append(dict(tt=gg.tumor_type.iloc[0], sample=str(s), gains=np.sort(med)))
    A = pd.DataFrame(amps)
    if not len(A):
        raise RuntimeError(f"no {gene} amplifications with >= {MIN_GAINS} gains")
    for a in amps:
        a["wgd"] = float(WGDT[a["sample"]][0]) if a["sample"] in WGDT else np.nan
        a["wgdt"] = list(WGDT.get(a["sample"], [])); a["pgdt"] = list(PGDT.get(a["sample"], []))
    A = pd.DataFrame(amps); A["mean_t"] = A.gains.apply(lambda x: float(np.mean(x)))
    cnt = A.tt.value_counts()
    tts = sorted([t for t in cnt.index if cnt[t] >= MIN_TT and t != "Unknown"],
                 key=lambda t: A[A.tt == t].mean_t.median())

    def gcol(gt, wt, pt):
        if any(abs(gt - t) < SNAP for t in wt): return WGD_COL
        if any(abs(gt - t) < SNAP for t in pt): return PGD_COL
        return NONE_COL

    W_MM, H_MM = 210.0, 62.0
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    fig.subplots_adjust(left=11.0 / W_MM, right=0.995, top=1 - 8.0 / H_MM, bottom=10.5 / H_MM)
    gs = fig.add_gridspec(1, len(tts), width_ratios=[cnt[t] for t in tts], wspace=0.08); ax0 = None
    for i, tt in enumerate(tts):
        ax = fig.add_subplot(gs[i], sharey=ax0); ax0 = ax0 or ax
        recs = sorted(A[A.tt == tt].to_dict("records"), key=lambda r: r["gains"][0])
        for x, r in enumerate(recs):
            ax.plot([x] * len(r["gains"]), r["gains"], color="#d9d9d9", lw=0.5, zorder=1)
            for gt in r["gains"]:
                ax.scatter(x, gt, s=DOT, color=gcol(gt, r["wgdt"], r["pgdt"]), edgecolors="white", linewidths=0.3,
                           zorder=3)
        ax.set_xlim(-1, len(recs)); ax.set_ylim(-0.02, 1.02); ax.set_xticks([])
        short = {"Stomach": "Stom"}.get(tt.split(".")[0], tt.split(".")[0])   # as abbreviated in the PDF
        ax.set_title(f"{short}\nn={len(recs)}", fontsize=S.title, pad=3)
        ax.grid(axis="y", color="#f2f2f2", lw=0.4)
        ax.spines["bottom"].set_linewidth(0.5)
        if i == 0:
            ax.set_ylabel("Molecular time", fontsize=S.label); ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=S.tick, length=2, width=0.5)
            ax.text(0.0, -0.08, "→ Sample", transform=ax.transAxes, ha="left", va="top", fontsize=S.title)
        else:
            plt.setp(ax.get_yticklabels(), visible=False); ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(left=False)
    hh = [mlines.Line2D([], [], marker="o", ls="None", color=WGD_COL, ms=4, label="gain WGD-linked"),
          mlines.Line2D([], [], marker="o", ls="None", color=PGD_COL, ms=4, label="gain ccPG-linked"),
          mlines.Line2D([], [], marker="o", ls="None", color=NONE_COL, ms=4, label="gain unlinked")]
    fig.legend(handles=hh, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=S.legend,
               handletextpad=0.3, columnspacing=1.8)
    if save:
        # the two callers live in different figures (landscape / extra_oncogenes), so each passes its
        # own panels dir and manuscript stem; the fallback keeps the old behaviour for ad-hoc use
        out_dir = Path(panels_dir) if panels_dir is not None else (PANELS or Path.cwd())
        stem = out_dir / (stem or f"{gene.lower()}_swimmer")
        fig.savefig(f"{stem}.png", dpi=300); fig.savefig(f"{stem}.svg")
    return fig
