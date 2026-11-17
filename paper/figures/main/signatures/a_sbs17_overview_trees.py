"""signatures figure (main), panel a — one esophageal genome (0250e6f7) with SBS17 activity relative to the
clock: LEFT the segment-level timing overview with every gain interval coloured by log2(SBS17/clock ratio ÷
genome baseline), RIGHT the same sample's pooled-cluster trees (the Fig 2b units) with their branches
coloured by the same quantity, one shared colourbar. This is the layout of the assembled figure,
which she built from two of our renders (the ratio-coloured overview and the cluster-tree strip); here it
is one script.

Left half is OBJECT-BASED (the per-segment ratios live in the sample's `sig_timing.json.gz` cache next to
its genome pickle); right half is table-driven (`per_sample/<id>.tau_sig_intervals.tsv`, drawn by
`_cluster_trees.py`). Colour scale: RdBu-like, clipped at +-2 (red = SBS17 outran the clock in that
interval, blue = fell behind). No cluster track, no CN-coloured trees.
"""
import copy, glob, gzip, json, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
R = HERE.parents[3]; sys.path.insert(0, str(R)); sys.path.insert(0, str(HERE))
from paper.config import panels_dir_of, OBJECTS_PCAWG, PER_SAMPLE  # noqa: E402
from paper.figures._style import apply as _style_apply, save as _save_panel, S, MM  # noqa: E402
_style_apply()
import _cluster_trees as CT  # noqa: E402
from tau.plotting import compute_offsets, normalize_chrom  # noqa: E402
from tau.utils import pick_best_key, extract_mat  # noqa: E402

SAMPLE = CT.SAMPLE                     # 0250e6f7 Eso.AdenoCA, as in Fig 2b
SIG, CLIP = CT.SIG, CT.CLIP
ROOT = OBJECTS_PCAWG
PER = PER_SAMPLE / SAMPLE
PANELS = panels_dir_of(__file__)
STEM = "a_sbs17_overview_trees"
CMAP = CT.LR_CMAP
NEUTRAL = (0.85, 0.85, 0.85, 1.0)
W_MM, H_MM = 183.0, 52.0


def _load(p):
    with gzip.open(p, "rb") as f:
        import pickle
        return pickle.load(f)


def _ratio_stack(ax, g, ratios, abund, norm):
    """plot_tau_stack with intervals coloured by the signature/clock ratio (from the dev renderer)."""
    segs, offsets = compute_offsets(g.segments)
    prev = start = last = None
    for seg in segs:
        chrom = normalize_chrom(seg.chrom)
        if chrom not in [str(i) for i in range(1, 23)]:
            continue
        xoff = offsets[seg.seg_id]; x0 = int(seg.start) + xoff; last = seg
        if chrom != prev:
            if prev is not None:
                ax.axvline(x0, color="black", ls="--", lw=0.5, zorder=6)
                ax.text((start + x0) / 2, -0.02, prev, ha="center", va="top", fontsize=S.tiny, clip_on=False)
            prev, start = chrom, x0
        key = pick_best_key(seg)
        if key is None:
            continue
        mat = extract_mat(seg, key, boot_id=0)
        if mat is None or mat.size == 0:
            continue
        K = mat.shape[1]; rr = ratios.get(seg.seg_id)
        cols = [NEUTRAL if (rr is None or i not in rr or abund <= 0 or rr[i] <= 0)
                else CMAP(norm(np.clip(np.log2(rr[i] / abund), -CLIP, CLIP))) for i in range(K)]
        x1 = int(seg.end) + xoff
        ax.stackplot(np.linspace(x0, x1, max(2, mat.shape[0])), *mat.T, colors=cols, baseline="zero", zorder=2)
    if prev is not None:
        ax.text((start + int(last.end) + offsets[last.seg_id]) / 2, -0.02, prev, ha="center", va="top",
                fontsize=S.tiny, clip_on=False)
    ax.margins(x=0); ax.set_ylim(0, 1); ax.set_xticks([])


def make(save=True):
    d = Path(glob.glob(f"{ROOT}/*/{SAMPLE}")[0]); tt = d.parent.name
    g = _load(d / f"{SAMPLE}.genome.pkl.gz")
    with gzip.open(d / f"{SAMPLE}.sig_timing.json.gz", "rt") as f:
        cache = json.load(f)
    abund = cache["genome_abund"][SIG]["ratio"]
    ratios = {sid: {r["interval"]: r["ratio"] for r in blk["rows"]} for sid, blk in cache["seg_activity"][SIG].items()}
    norm = Normalize(-CLIP, CLIP)
    import pandas as pd
    ev = pd.read_csv(PER / f"{SAMPLE}.tau_events.tsv", sep="\t")
    ev = ev[ev.classification.isin(["WGD", "PGD"])]

    rows = CT.tree_rows()                                     # the pooled units, coloured by ratio
    n = len(rows)

    # ── geometry: overview 88 mm | trees 78 mm | colourbar ──
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    L, T, B = 9.0, 6.0, 8.0                                    # mm
    ax_h = (H_MM - T - B) / H_MM
    ax_ov = fig.add_axes([L / W_MM, B / H_MM, 88.0 / W_MM, ax_h])
    _ratio_stack(ax_ov, g, ratios, abund, norm)
    ax_ov.set_ylabel("Molecular time", fontsize=S.label)
    ax_ov.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax_ov.tick_params(axis="y", labelsize=S.tick, length=2, width=0.5)
    ax_ov.set_xlabel("Genomic loci", fontsize=S.label, labelpad=8)
    for sp in ("top", "right"):
        ax_ov.spines[sp].set_visible(False)
    cols = {"WGD": "#0b6979", "PGD": "#cd9f2c"}
    for _, r in ev.iterrows():
        ax_ov.axhline(float(r.event_time_frac), color=cols[r.classification], ls="--", lw=0.9, alpha=0.9, zorder=7)

    x0 = (L + 88.0 + 6.0) / W_MM; tree_w = 68.0 / W_MM; gap = 0.35
    step = tree_w / n
    for i, r in enumerate(rows):
        ax = fig.add_axes([x0 + i * step, B / H_MM, step * (1 - gap / (1 + gap)), ax_h])
        CT._logratio_tree(ax, r["route"], r["cn"], r["bnds"][1:-1], r["bnds"], r["z"], norm)
        ax.set_xlim(CT.SP - 0.55, CT.SP * (r["cn"][0] + r["cn"][1]) + 0.55)
        ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        for _, e in ev.iterrows():
            ax.axhline(float(e.event_time_frac), color=cols[e.classification], ls="--", lw=0.6, alpha=0.6, zorder=0)
    cax = fig.add_axes([(W_MM - 10.0) / W_MM, B / H_MM + 0.15 * ax_h, 2.2 / W_MM, 0.7 * ax_h])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("log$_2$ signature/clock ratio", fontsize=S.tick); cb.ax.tick_params(labelsize=S.tick, length=1.5)
    cb.outline.set_linewidth(0.4)
    fig.text(L / W_MM, 1 - 1.5 / H_MM, f"{SIG} accumulation relative to clock signatures in {SAMPLE}",
             fontsize=S.title, ha="left", va="top")
    if save:
        _save_panel(fig, PANELS, STEM)
        print(f"  {n} trees; wrote {PANELS}/{STEM}.svg")
    return fig


if __name__ == "__main__":
    make()
