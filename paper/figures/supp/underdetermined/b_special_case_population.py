"""underdetermined figure (supp), panel b — special-case solutions place gains closer to the inferred WGD.

One cluster per (sample, copy-number state): the best-powered pooled cluster, so a sample contributing
several clusters of the same state cannot count twice (n=780, 464 samples). Each is underdetermined,
sits in a sample with a detected WGD, and has a route admitting a doubling.

Two points of the SAME solution space are compared, and the statistic is simply how far a gain sits
from the nearest inferred WGD:

    random      one arbitrary solution
    special     the solution in which gains coincide on multiple branches

The gains held to coincide are not chosen ad hoc: they are the route's enumerated DOUBLING BLOCKS
(paper/upstream/special_case/doubling_blocks.py) — the same presets Fig. 2 uses. Where a route admits more than one form
— (3,2) is the common case at 396/780 clusters, with a WGD-first and a gain-first-then-WGD reading —
forms are ranked by gains covered, then by proximity to the inferred WGD, exactly as the Fig. 2 preset
ranks them. Both arms are then measured on THAT form's gains, in the same solution space, so the
contrast is within-cluster and within-gain-set.

Measured as the per-cluster median distance of those gains to the nearest inferred WGD. NOT their
spread: the special case is DEFINED by their coinciding, so their spread collapses by construction and
would say nothing about the WGD. Where they land is free, so landing near it is a finding.

The effect tracks the interval width it has to resolve: nothing at all for near-pinned clusters (+3%),
rising steadily to +54% for the widest (overall +33%). That gradient is the result — a solution space
with no freedom has no special case to find.

Caveat, quantified and not acted on: the form available to a cluster depends on its CHOSEN route, and
route choice is by dist_rel, which is indifferent to whether a route can express a doubling. At (4,4) —
where route selection is weakest, 13-19 feasible routes per cluster and near-total ties — only 1/14
clusters gets the full two-doubling form under the top-1 pick, against 6/14 if any feasible route is
allowed, and the route achieving it is `4_4.4` every time. (4,4) is 14/780 clusters here, so the
headline is unaffected; the other states are not close to this regime.

Table-driven from <data>/frozen/supp4_dominant_34x.tsv (paper/upstream/special_case/scan_special_case.py, then the dominant
cluster per sample x state).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
from paper.config import FROZEN
SRC = FROZEN / "supp4_dominant_34x.tsv"     # upstream/special_case/scan_special_case.py
C_RND, C_SPEC = "#8a93a3", "#b0402f"
W_BINS = [0, 0.02, 0.05, 0.10, 0.20, 1.0]
W_LABS = ["<0.02", "0.02–0.05", "0.05–0.10", "0.10–0.20", ">0.20"]


def _blocks():
    """route -> its doubling blocks (start, length, implied_loss), length>=2 only."""
    import csv
    out = {}
    with open(FROZEN / "doubling_blocks.tsv") as f:      # upstream/special_case/doubling_blocks.py
        for r in csv.DictReader(f, delimiter="\t"):
            if not r["blocks"]:
                continue
            bs = [tuple(int(v) for v in b.replace("/", "+").split("+")) for b in r["blocks"].split(";")]
            bs = [b for b in bs if b[1] >= 2]
            if bs:
                out[r["route_key"]] = bs
    return out


def _collect(seed=0):
    """Apply the Fig. 2 preset to each cluster and measure the gains it co-times.

    Candidate forms are sets of non-overlapping DOUBLING BLOCKS of the cluster's route — structurally
    valid doublings, not merely adjacent gains that happen to coincide. For (4,4) in a two-WGD sample
    that is the single form covering all six gains: two bi-allelic gains at the first doubling and four
    balanced bi-allelic gains at the second. Forms are ranked by coverage, then by proximity of their
    doublings to the inferred WGD — the same tie-break the Fig. 2 preset uses.
    """
    import itertools
    d = pd.read_csv(SRC, sep="\t")
    DB = _blocks()
    rng = np.random.default_rng(seed)
    rows = []
    for r in d.itertuples():
        blocks = DB.get(r.route)
        if not blocks:
            continue
        w = np.array([float(x) for x in r.wgd.split(";")])
        allv = np.array([float(x) for x in r.all_ev.split(";")]).reshape(r.n_draws, r.n_gain)
        ts = np.diff(np.column_stack([np.zeros(r.n_draws), allv]), axis=1)   # interval widths
        best = None
        for k in range(1, min(len(w), 2) + 1):
            for combo in itertools.combinations(blocks, k):
                iv = sorted((a, a + L - 1) for a, L, _ in combo)
                if any(iv[j][1] >= iv[j + 1][0] for j in range(len(iv) - 1)):
                    continue
                sc = sum(ts[:, a:a + L - 1].sum(axis=1) for a, L, _ in combo)
                i = int(np.argmin(sc))
                if sc[i] > 0.01:
                    continue
                got = [float(allv[i, a - 1]) for a, L, _ in combo]
                cov = sum(L for _, L, _ in combo)
                dist = float(np.mean([np.min(np.abs(g - w)) for g in got]))
                cand = (-cov, dist, i, combo)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best is None:
            continue
        _, _, i, combo = best
        keep = np.zeros(r.n_gain, bool)
        for a, L, _ in combo:
            keep[a - 1:a - 1 + L] = True
        if keep.sum() < 2:
            continue
        rd = allv[rng.integers(r.n_draws)]
        med = lambda v: float(np.median(np.abs(v[keep] - w[np.argmin(np.abs(v[keep][:, None] - w[None, :]), axis=1)])))
        rows.append(dict(state=r.state, spread=float(np.mean(allv.max(0) - allv.min(0))),
                         n_doubling=len(combo), coverage=int(keep.sum()), n_gain=r.n_gain,
                         random=med(rd), special=med(allv[i])))
    return pd.DataFrame(rows), d["sample"].nunique()


def make(save=True):
    t, n_samp = _collect()
    t["bin"] = pd.cut(t.spread, W_BINS, labels=W_LABS)
    g = (t.groupby("bin", observed=True)
           .agg(n=("random", "size"), random=("random", "median"), special=("special", "median"))
           .reindex(W_LABS).dropna())

    fig, ax = plt.subplots(figsize=(62 * MM, 52 * MM))

    # the gradient IS the result: how much closer the special case sits, against how much freedom the
    # solution space gave it to be anywhere else
    y = np.arange(len(g))
    ax.hlines(y, g.special, g.random, color="0.78", lw=1.2, zorder=2)
    ax.plot(g.random, y, "o", ms=3.4, color=C_RND, zorder=4, label="random solution")
    ax.plot(g.special, y, "o", ms=3.4, color=C_SPEC, zorder=5, label="special case")
    for i, (nn, rr, ss) in enumerate(zip(g.n, g.random, g.special)):
        ax.text(rr + 0.006, i, f"{1 - ss/rr:.0%}", fontsize=S.tiny, color=C_SPEC,
                va="center", fontweight="bold")
        ax.text(-0.004, i - 0.30, f"n={int(nn)}", fontsize=S.tiny, color="0.5", va="center", ha="left")
    ax.set_yticks(y); ax.set_yticklabels(g.index); ax.set_ylim(-0.7, len(g) - 0.3)
    ax.set_xlim(0, float(g.random.max()) * 1.35)
    ax.set_xlabel("median distance to nearest inferred WGD")
    ax.set_ylabel("interval width")
    ax.legend(frameon=False, loc="lower right", handlelength=1.0, labelspacing=0.25)

    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.grid(lw=0.3, color="0.93"); ax.set_axisbelow(True)
    fig.text(0.005, 0.012, f"n={len(t)} dominant (3,x)/(4,x) clusters, {n_samp} samples",
             fontsize=S.tiny, color="0.45")
    fig.subplots_adjust(left=0.235, right=0.975, top=0.965, bottom=0.215)
    if save:
        _save_panel(fig, PANELS, "b_special_case_population")
    return fig


if __name__ == "__main__":
    make(); print(f"supp/underdetermined/b: written -> {PANELS}")
