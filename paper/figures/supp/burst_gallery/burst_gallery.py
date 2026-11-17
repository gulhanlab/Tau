"""burst_gallery figure (supp) — excess-burst exemplars, split into WGD-linked and WGD-unlinked.

Replaces four 183x170 mm pages of 16 with ONE page carrying both groups (mark the WGD-compatible
dots, split matching from non-matching, fit both on a single page, fewer and smaller).

    a  WGD-linked     — burst centre within +/-0.10 of a WGD in that sample            (n=783)
    b  WGD-unlinked   — outside that window, or the sample has no WGD at all           (n=648)

Individual GAINS are coloured by the same rule: a gain landing within +/-0.10 of the sample's WGD is
drawn in WGD teal, the rest in dark grey. So the reader can see which gains a doubling could account
for and which it cannot — in the WGD-linked block most dots are teal and sit on the dashed WGD line, in
the WGD-unlinked block they are grey and sit away from it.

Only TRUSTWORTHY calls are shown, as elsewhere in this figure: determined segments, plus underdetermined
ones whose burst holds under every fit. Both have a false-positive rate near zero against the matched
null (0.5% / 0.6%); the permissive "burst under some fit" class is excluded.

EXEMPLARS ARE THE UNAMBIGUOUS CASES. The blocks are defined on the burst CENTRE, but the dots are
coloured per GAIN, so a burst centred ~WGD_WIN from a doubling straddles the cut and renders half teal,
half grey — which reads as a colouring bug rather than a measurement. `_pick` therefore prefers segments
whose gains all fall the same side (12 straddlers skipped for this page). That makes the page cleaner
than the population: do NOT read it as every WGD-linked burst lying entirely inside the window.

Rendering is the shared FigHighCN helper, so these match the main burst figure exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of  # noqa: E402
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os, pickle
from paper.config import DATA, FROZEN, cache_dir_of
from _traj import (fetch, plot_traj, build_real_tree, draw_real_tree,
                   NATREV, TXT, MM, WGD_COL)
from paper.figures._style import save as _save_panel, S
from paper.figures._style import apply as _style_apply  # noqa: E402
_style_apply()

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
SRC = FROZEN / "burst_possible_necessary.tsv"     # paper/upstream/burst/burst_possible_necessary.py
# The 24 exemplars (their solutions and allele trees) are frozen here, so a normal render needs neither
# the per-solution analysis cache nor the cohort's per-sample outputs. TAU_REBUILD=1 re-selects them
# from those (see _traj.py for what that needs) and rewrites the file.
EXEMPLARS = cache_dir_of(__file__) / "exemplars.pkl"
PAGE_W, PAGE_H = 183, 170        # Nature double column; 170 leaves room for the caption on the page
NCOL, NROW = 4, 3                 # 12 per block, 24 on the page
WGD_WIN = 0.10
CENTROID_LO, CENTROID_HI = 0.15, 0.85
MUT_FLOOR_Q = 0.60
GREY = NATREV["grey"]

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.linewidth": 0.5, "svg.fonttype": "none",
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300,
    "savefig.transparent": True, "figure.facecolor": "none", "axes.facecolor": "none",
})
_style_apply()   # the shared ladder wins over the sizes in the block above


def _pool():
    x = pd.read_csv(SRC, sep="\t", dtype={"chrom": str})
    ev = pd.read_csv(DATA / "tau_events.tsv", sep="\t")
    wgd = ev[ev.classification == "WGD"].groupby("sample").event_time_frac.apply(list).to_dict()
    x["cls"] = np.where(x.determined & x.possible, "determined",
                        np.where(~x.determined & x.necessary, "every solution", "excluded"))
    e = x[x.cls != "excluded"].copy()
    e["d_wgd"] = [min((abs(r.centroid - t) for t in wgd.get(r.sample, [])), default=np.nan)
                  for r in e.itertuples()]
    e["wgd_match"] = e.d_wgd <= WGD_WIN
    return e, wgd


def _coherent(r, wgds, match):
    """Do this segment's gains colour consistently, or does the window cut through the burst?

    The blocks are defined on the burst CENTRE, but the dots are coloured per GAIN. A burst whose
    centre sits ~WGD_WIN from a doubling straddles the cut: one gain lands inside the window and its
    neighbours 0.005 later land outside, so an obviously co-timed burst renders half teal and half
    grey and reads as a colouring bug rather than a measurement. Prefer exemplars where every gain
    falls the same side of the cut. Returns (ok, times) — times is None if the tree cannot be built.
    """
    try:
        *_, times = build_real_tree(str(r.sample)[:8], str(r.chrom), int(r.start), bursty=True)
    except Exception:
        return False, None
    t = np.asarray(times, float)
    if not wgds:
        return (not match), t          # no WGD in the sample: every gain is "elsewhere"
    near = np.min(np.abs(t[:, None] - np.asarray(wgds)[None, :]), axis=1) <= WGD_WIN
    return (bool(near.all()) if match else bool(~near.any())), t


def _pick(e, match, n, taken, wgd):
    """Best-powered, interior-centroid exemplars, spread over tumour type and CN state.

    Passes 0-2 additionally require a coherent colouring (see `_coherent`); pass 3 drops that
    requirement so the block always fills even if too few clean cases exist.
    """
    floor = e.mutation_count.quantile(MUT_FLOOR_Q)
    q = e[(e.wgd_match == match) & (e.mutation_count >= floor)
          & (e.centroid > CENTROID_LO) & (e.centroid < CENTROID_HI)]
    q = q.sort_values("mutation_count", ascending=False)
    out, seen_t, seen_s = [], set(), set()
    n_straddle = 0
    for pass_ in range(4):
        for r in q.itertuples():
            if len(out) >= n:
                break
            st = (r.major_cn, r.minor_cn)
            if pass_ == 0 and (r.tumor_type in seen_t or st in seen_s):
                continue
            if pass_ == 1 and r.tumor_type in seen_t:
                continue
            if (r.sample, r.chrom, r.start) in taken:
                continue
            ok, times = _coherent(r, sorted(wgd.get(r.sample, [])), match)
            if times is None:                      # tree will not build — unusable either way
                continue
            if pass_ < 3 and not ok:
                n_straddle += 1
                continue
            out.append(r); taken.add((r.sample, r.chrom, r.start))
            seen_t.add(r.tumor_type); seen_s.add(st)
    print(f"  {'WGD-linked' if match else 'WGD-unlinked'}: {len(out)} exemplars "
          f"({n_straddle} straddlers skipped)")
    return out


def _draw(fig, gs, p, wgds):
    """`wgds` = the sample's WGD times from tau_events.tsv — the SAME source that defines the
    matching/non-matching split, so a panel can never be labelled 'no WGD' inside block a."""
    s = p.fetched
    maj, mn, M_, m_, times = p.tree
    axg = fig.add_subplot(gs[1])
    plot_traj(axg, s["draws"], s["ne"], None, show_median=False, show_wgd=False,
              wgd_label=False, yticklabels=True, highlight=False, ms=0, tick_fs=5.0, scale=0.85)
    dots = np.sort(times); yy = np.arange(1, s["ne"] + 1)
    axg.plot(dots, yy, color=TXT, lw=0.8, zorder=5)
    # colour each GAIN by whether a doubling could account for it
    if wgds:
        near = np.min(np.abs(dots[:, None] - np.asarray(wgds)[None, :]), axis=1) <= WGD_WIN
    else:
        near = np.zeros(len(dots), bool)
    axg.scatter(dots[near], yy[near], s=11, color=WGD_COL, zorder=6, linewidths=0)
    axg.scatter(dots[~near], yy[~near], s=11, color=GREY[5], zorder=6, linewidths=0)
    axg.set_xlim(-0.02, 1.02); axg.set_xticks([0, 0.5, 1.0])
    axg.set_xticklabels(["0", "0.5", "1"], fontsize=S.tick)
    if s["ne"] > 5:                       # G1..G7 collide at this panel height — label the ends only
        axg.set_yticks([1, s["ne"]]); axg.set_yticklabels([f"G1", f"G{s['ne']}"], fontsize=S.tick)
    axt = fig.add_subplot(gs[0])
    mspan = (0.99, 0.45) if s["major"] >= 6 else (0.94, 0.55)
    draw_real_tree(axt, maj, mn, None, show_minor=(m_ > 0), gain_marker="none",
                   show_allele_labels=False, maj_span=mspan)
    axt.set_xlim(-0.02, 1.02)
    for w in wgds:
        axg.add_artist(ConnectionPatch(xyA=(w, 0.4), coordsA=axg.transData,
                                       xyB=(w, 1.0), coordsB=axt.transData,
                                       color=WGD_COL, lw=0.9, ls=(0, (4, 2)), zorder=2))
    note = " · no WGD" if not wgds else ""
    axt.text(0.5, 1.16, f"{s['tumor_type']}{note}\n({s['major']},{s['minor']})  "
                        f"{int(p.max_burst_size)}-burst\nchr{p.chrom}:{p.start/1e6:.1f}Mb",
             transform=axt.transAxes, ha="center", va="bottom", fontsize=S.tiny,
             color=GREY[4], linespacing=1.2)
    return axg


def _Ex(r):
    """One frozen exemplar: the selected row's fields + its solutions (`fetched`) and allele tree.
    A SimpleNamespace of plain data (+ _traj.LiteNode trees) so the pickle loads from any entry point."""
    from types import SimpleNamespace
    sample, chrom, start = str(r.sample), str(r.chrom), int(r.start)
    return SimpleNamespace(sample=sample, chrom=chrom, start=start, max_burst_size=int(r.max_burst_size),
                           fetched=fetch(sample[:8], chrom, start),
                           tree=build_real_tree(sample[:8], chrom, start, bursty=True))


def _exemplars():
    """-> ([(label, is_wgd_linked, [exemplar, ...]) x2], {sample: [WGD times]})."""
    if EXEMPLARS.exists() and not os.environ.get("TAU_REBUILD"):
        with open(EXEMPLARS, "rb") as f:
            return pickle.load(f)
    e, wgd = _pool()
    taken = set()
    blocks = [("a   WGD-linked", True, [_Ex(r) for r in _pick(e, True, NCOL * NROW, taken, wgd)]),
              ("b   WGD-unlinked", False, [_Ex(r) for r in _pick(e, False, NCOL * NROW, taken, wgd)])]
    keep = {p.sample for _, _, rows in blocks for p in rows}
    out = (blocks, {k: v for k, v in wgd.items() if k in keep})
    with open(EXEMPLARS, "wb") as f:
        pickle.dump(out, f)
    return out


def make(save=True):
    blocks, wgd = _exemplars()

    fig = plt.figure(figsize=(PAGE_W * MM, PAGE_H * MM))
    outer = fig.add_gridspec(2, 1, left=0.055, right=0.99, top=0.918, bottom=0.055, hspace=0.30)
    for bi, (lab, _m, rows) in enumerate(blocks):
        # label sits clear above the block, so it cannot collide with the first column's title
        fig.text(0.005, outer[bi].get_position(fig).y1 + 0.042, lab, fontsize=S.title,
                 fontweight="bold", ha="left", va="bottom", color=TXT)
        inner = outer[bi].subgridspec(NROW, NCOL, hspace=0.72, wspace=0.42)
        for k, p in enumerate(rows):
            cell = inner[k // NCOL, k % NCOL].subgridspec(2, 1, height_ratios=[0.72, 1.5],
                                                          hspace=0.46)
            try:
                axg = _draw(fig, cell, p, sorted(wgd.get(p.sample, [])))
            except Exception as ex:
                print(f"  skip {str(p.sample)[:8]} chr{p.chrom}:{p.start} ({ex})"); continue
            if k % NCOL == 0:
                axg.set_ylabel("gain index", fontsize=S.label, color=TXT, labelpad=2)
    # one legend for the whole page — the dot colour is the point of the figure
    fig.scatter = None
    fig.text(0.62, 0.012, "●", color=WGD_COL, fontsize=S.label, ha="left", va="bottom")
    fig.text(0.635, 0.014, "gain within 0.10 of the WGD", fontsize=S.annot, ha="left", va="bottom",
             color=TXT)
    fig.text(0.83, 0.012, "●", color=GREY[5], fontsize=S.label, ha="left", va="bottom")
    fig.text(0.845, 0.014, "gain elsewhere", fontsize=S.annot, ha="left", va="bottom", color=TXT)
    fig.text(0.30, 0.014, "molecular time", ha="center", fontsize=S.annot, color=TXT)
    if save:
        _save_panel(fig, PANELS, "burst_gallery")
    print(f"burst_examples_wgd_split: {len(blocks[0][2])} matching + {len(blocks[1][2])} non-matching "
          f"on one {PAGE_W}x{PAGE_H} mm page")
    return fig


if __name__ == "__main__":
    make()
