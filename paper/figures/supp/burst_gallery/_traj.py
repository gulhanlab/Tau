#!/usr/bin/env python3
"""
_traj.py — trajectory/tree drawing helper for the burst gallery (vendored from the analysis tree,
dev/pcawg/FigHighCN/fighighcn_traj.py; only the paths were changed).

A "trajectory" panel shows one high-CN segment's gain timing:
  x = molecular time (0 = early, 1 = late), y = gain index G1..Gn.
  - every polytope solution as a thin grey line (the solution space)
  - the median solution as a bold line
  - the sample's WGD time as a vertical teal dashed line (where available)

Used by panel_A_exemplars.py (taxonomy grid) and panel_F_excess_vs_explained.py
(doubling-budget pair) so both panels render identically.
"""
from pathlib import Path
import re
import glob
import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from paper.config import TAU_OUTPUTS, ANALYSIS, cache_dir_of

# per-sample Tau output (all segments, all CN states, with mutation counts). Only needed when the
# exemplars are re-selected (TAU_REBUILD=1 in burst_gallery.py); a normal render reads the frozen
# exemplars and never touches these.
SEG_TIMES_DIR = TAU_OUTPUTS


def fix_svg_fonts(path):
    """Rewrite matplotlib's `font: [weight ]Npx family` CSS shorthand into longhand
    (font-weight / font-size / font-family). Illustrator mis-parses the shorthand when a
    weight token is present — bold text silently falls back to 12pt — but parses the
    longhand reliably, so bold labels keep BOTH their weight and their true point size."""
    p = Path(path)
    s = p.read_text()
    s = re.sub(r"font: (\d+) ([0-9.]+)px ",
               r"font-weight: \1; font-size: \2px; font-family: ", s)
    s = re.sub(r"font: ([0-9.]+)px ",
               r"font-size: \1px; font-family: ", s)
    p.write_text(s)

# upstream analysis outputs (paper/upstream/burst/): per-solution gain times of every high-CN segment
CACHE = ANALYSIS / "pcawg/misc_data_and_plots/genome_highcn_timing_cache_v2.tsv"
BURST_TABLE = ANALYSIS / "pcawg/misc_data_and_plots/burst_singleroute.tsv"
TAXO  = ANALYSIS / "pcawg/oncogenes/figures_genome_highcn_catalogue/wgd_burst_taxonomy_polytope.tsv"

NATREV = {"teal": ["#cee7ed", "#98d2d3", "#4cbdbd", "#0d9aa2", "#0b6979", "#07394d"],
          "grey": ["#e7e6ec", "#cacdd9", "#99a2b3", "#717c91", "#4b566d", "#273246"],
          "gold": ["#fdf3d0", "#fbe49a", "#f5cc4e", "#cd9f2c", "#9a6e12", "#5c3d06"]}
WGD_COL   = "#0b6979"          # WGD teal (matches the lead panel)
DRAW_COL  = NATREV["grey"][1]  # thin solution lines
MED_COL   = NATREV["grey"][5]  # bold median
TXT       = "#273246"
GCOLS     = [f"G{i}" for i in range(1, 10)]
MM = 1 / 25.4

_cache = None
_taxi  = None

# ── tree cache ──────────────────────────────────────────────────────────────────────
# build_real_tree's slow cost is importing tau.trees (~16s; it pulls in the Sage matrices via
# tau.utils). The trees themselves are tiny plain-Python structures, so we convert the tau Nodes to
# a tau-independent LiteNode, pickle them per segment, and replay them with a local sort_nodes.
# Re-renders then skip the tau import entirely. Set FIGHIGHCN_REBUILD=1 to force recomputation.
_TREECACHE = cache_dir_of(__file__) / "treecache"


class LiteNode:
    """tau-independent twin of tau.trees.Node — just the fields sort_nodes/draw_real_tree read."""
    def __init__(self, time=0.0, alignment=None, parent=None):
        self.time = float(time); self.branches = []
        self.alignment = alignment; self.parent = parent; self.mults = []; self.y = 0.0

    @property
    def mult(self):
        return self.mults[-1] if self.mults else None


def _to_lite(node, parent=None):
    """Deep-copy a tau Node tree into LiteNodes (so the pickle never references tau.trees)."""
    if node is None:
        return None
    ln = LiteNode(time=node.time, alignment=node.alignment, parent=parent)
    ln.mults = list(getattr(node, "mults", []))
    ln.branches = [_to_lite(b, ln) for b in node.branches]
    return ln


def sort_nodes(node):
    """Depth-first left/right ordering for plotting — local copy of tau.trees.sort_nodes (tau-free)."""
    if node is None:
        return []
    if len(node.branches) == 0:
        return [node]
    left = [b for br in node.branches if br.alignment == "L" for b in sort_nodes(br)]
    right = [b for br in node.branches if br.alignment == "R" for b in sort_nodes(br)]
    return [*left, node, *right]


def _load():
    """Load the solution cache, restricted to the SINGLE route the burst table used.

    The grey fan a panel draws must range over exactly the solutions the burst statistic searched, or a
    panel labelled "w0.00" can show a visible spread. v1 of the cache kept every route surviving
    dist_cut = 0.20 (a (5,2) segment carries a median of 12) and dropped the fit-quality columns, so the
    fan included routes fitting measurably worse than the segment's best.

    The burst table is built on ONE route per segment (tau_gains.is_best_route), matching Fig 6, and
    carries that key as `route_key` — so the fan is filtered to it here. Switching the analysis to the
    tied set means changing both together; `best_route` in the cache marks the whole tied set and is
    NOT the same filter.
    """
    global _cache, _taxi
    if _cache is None:
        _cache = pd.read_csv(CACHE, sep="\t", dtype={"chrom": str})
        bt = pd.read_csv(BURST_TABLE, sep="\t", dtype={"chrom": str},
                         usecols=["sample", "chrom", "start", "end", "route_key"])
        _cache = _cache.merge(bt, on=["sample", "chrom", "start", "end"], how="inner")
        _cache = _cache[_cache.key == _cache.route_key]
    if _taxi is None:
        t = pd.read_csv(TAXO, sep="\t", dtype={"chrom": str})
        _taxi = t.set_index(["sample", "chrom", "start"])
    return _cache, _taxi


BURST_THRESH = 0.05
BURST_COL    = "#ec7023"          # burst-solution highlight (NatRev orange sh.3)


def fetch(prefix, chrom, start):
    """Return a dict with draws, ne, wgd, major, minor, mut, tumor_type, end."""
    oc, taxi = _load()
    g = oc[(oc["sample"].str.startswith(prefix)) & (oc["chrom"] == chrom)
           & (oc["start"] == start)]
    if not len(g):
        raise ValueError(f"segment not found: {prefix} {chrom}:{start}")
    samp = g["sample"].iloc[0]
    ne   = int(g["n_events"].iloc[0])
    wgd = np.nan
    k = (samp, chrom, start)
    if k in taxi.index:
        row = taxi.loc[k]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        wgd = row["wgd_time"]
    return dict(draws=g[GCOLS[:ne]].values.astype(float), ne=ne,
                wgd=(float(wgd) if pd.notna(wgd) else None),
                major=int(g["major_cn"].iloc[0]), minor=int(g["minor_cn"].iloc[0]),
                mut=int(g["mutation_count"].iloc[0]), sample=samp,
                tumor_type=str(g["tumor_type"].iloc[0]), end=int(g["end"].iloc[0]))


def sample_gain_times(prefix):
    """All high-CN gain molecular-times for the sample (median over solutions per gain,
    pooled across every high-CN segment in that sample). Sorted ascending — ready for a CDF."""
    oc, _ = _load()
    hit = oc[oc["sample"].str.startswith(prefix)]
    samp = hit["sample"].iloc[0]
    g = oc[oc["sample"] == samp]
    times = []
    for _, seg in g.groupby(["chrom", "start", "end"]):
        ne = int(seg["n_events"].iloc[0])
        times.extend(np.median(seg[GCOLS[:ne]].values, axis=0).tolist())
    return np.sort(np.asarray(times, float))


def sample_gain_points(sample_id, weight_by="mut"):
    """Per-(segment, gain, polytope-draw) points for a sample's weighted gain CDF.
    Each gain carries weight base/n_gains, spread UNIFORMLY across its polytope draws — so an
    underdetermined gain smears over its feasible range (a ramp) while a determined gain
    (1 draw) stays a point. A segment's gains sum to its `base`.

    weight_by: 'mut' (segment mutation_count) or 'length' (segment genomic length →
    "fraction of amplified genome gained by t").

    Returns DataFrame[time, weight, determined] (one row per draw) or None.
    """
    hits = glob.glob(str(SEG_TIMES_DIR / "*" / sample_id / f"{sample_id}.segment_times.tsv"))
    if not hits:
        return None
    d = pd.read_csv(hits[0], sep="\t", usecols=["bootstrap_id", "best_route", "segment_id",
                    "draw_id", "time_point", "t", "mutation_count", "length"])
    d = d[(d["bootstrap_id"] == 0)                       # point estimate, not SNV resamples
          & d["best_route"].astype(str).str.lower().isin(["true", "1"])]
    if not len(d):
        return None
    # `t` is the per-INTERVAL mutation fraction; a gain's molecular time is the cumulative sum
    # of intervals up to it (per segment & draw, ordered by time_point) — matching the
    # trajectory's G values. Using raw `t` would place later gains far too early.
    d = d.sort_values(["segment_id", "draw_id", "time_point"])
    d["time"] = d.groupby(["segment_id", "draw_id"])["t"].cumsum()
    d["n_gains"] = d.groupby("segment_id")["time_point"].transform("nunique")
    d["n_draws"] = d.groupby(["segment_id", "time_point"])["t"].transform("size")
    base = d["length"] if weight_by == "length" else d["mutation_count"]
    d["weight"] = base / d["n_gains"] / d["n_draws"]
    d["determined"] = d["n_draws"] == 1
    return d[["time", "weight", "determined"]]


def sample_n_segments(sample_id):
    """Number of timed segments for a sample (best_route, point estimate), or None."""
    hits = glob.glob(str(SEG_TIMES_DIR / "*" / sample_id / f"{sample_id}.segment_times.tsv"))
    if not hits:
        return None
    d = pd.read_csv(hits[0], sep="\t", usecols=["bootstrap_id", "best_route", "segment_id"])
    d = d[(d["bootstrap_id"] == 0)
          & d["best_route"].astype(str).str.lower().isin(["true", "1"])]
    return int(d["segment_id"].nunique())


def _cdf_from_points(pts):
    """(times, cdf_0to1, total_weight) from a [time, weight] DataFrame, or None."""
    if pts is None or not len(pts):
        return None
    p = pts.sort_values("time")
    cw = np.cumsum(p["weight"].values)
    if cw[-1] <= 0:
        return None
    return p["time"].values, cw / cw[-1], float(cw[-1])


def sample_weighted_cdf(sample_id, subset="all", weight_by="mut"):
    """Weighted gain CDF across all segments of a sample (polytope-spread).
    subset: 'all' | 'det' (determined gains only) | 'undet' (underdetermined only).
    weight_by: 'mut' | 'length'.
    Returns (times, cdf_0to1, total_weight) or None."""
    pts = sample_gain_points(sample_id, weight_by=weight_by)
    if pts is None:
        return None
    if subset == "det":
        pts = pts[pts["determined"]]
    elif subset == "undet":
        pts = pts[~pts["determined"]]
    return _cdf_from_points(pts)


def draw_burst_span(draws, ne):
    """Per-draw tightest 3-consecutive-gain window span (inf if ne<3)."""
    if ne < 3:
        return np.full(len(draws), np.inf)
    best = np.full(len(draws), np.inf)
    for i in range(ne - 2):
        best = np.minimum(best, draws[:, i + 2] - draws[:, i])
    return best


def draw_is_burst(draws, ne, thresh=BURST_THRESH):
    """Per-draw mask: tightest 3-consecutive-gain window span < thresh."""
    return draw_burst_span(draws, ne) < thresh


def plot_traj(ax, draws, ne, wgd_time, *, med_color=MED_COL, draw_color=DRAW_COL,
              show_wgd=True, wgd_label=False, yticklabels=True, lw_med=1.6,
              highlight=False, burst_color=BURST_COL, show_median=True, tick_fs=6.5,
              scale=1.0, ms=9):
    """Render a single trajectory panel onto ax.

    highlight: False (none), 'all' (every burst solution orange), or 'burstiest'
      (only the single tightest burst solution highlighted, if one exists). Lets the
      burst hiding inside an underdetermined solution space be seen without recolouring
      the whole cloud.
    show_median: draw the bold median solution.
    scale: multiplier on line widths / marker sizes (use <1 for small panels).
    """
    ys = np.arange(1, ne + 1)
    alpha = max(0.05, min(0.5, 30.0 / max(len(draws), 1)))
    spans = draw_burst_span(draws, ne)
    hl_idx = np.array([], int)
    if highlight == "all":
        hl_idx = np.flatnonzero(spans < BURST_THRESH)
    elif highlight == "burstiest" and len(draws):
        j = int(np.argmin(spans))
        if spans[j] < BURST_THRESH:
            hl_idx = np.array([j])
    hl_set = set(hl_idx.tolist())
    # non-highlighted solutions first (faint grey)
    for i, d in enumerate(draws):
        if i in hl_set:
            continue
        ax.plot(d[:ne], ys, color=draw_color, alpha=alpha, lw=0.5 * scale, zorder=2,
                solid_capstyle="round")
    # highlighted burst solution(s) on top (orange) — markers sized by `ms`
    for i in hl_idx:
        lw = (1.4 if highlight == "burstiest" else 0.6) * scale
        ab = 0.95 if highlight == "burstiest" else min(0.55, alpha * 1.6)
        ax.plot(draws[i, :ne], ys, color=burst_color, alpha=ab, lw=lw, zorder=3,
                solid_capstyle="round")
        if ms > 0:
            fc = to_rgba(burst_color, max(0.55, min(0.95, ab * 1.4)))  # keep fill orange under black ring
            ax.scatter(draws[i, :ne], ys, s=ms * scale ** 2, facecolors=[fc],
                       edgecolors="black", linewidths=0.6 * scale, zorder=4, clip_on=False)
    # median solution
    if show_median:
        med = np.median(draws[:, :ne], axis=0)
        ax.plot(med, ys, color=med_color, lw=lw_med * scale, zorder=5, solid_capstyle="round")
        ax.scatter(med, ys, s=ms * scale ** 2 if ms > 0 else 12 * scale ** 2,
                   facecolors=[med_color], edgecolors="black", linewidths=0.6 * scale,
                   zorder=6, clip_on=False)
    # WGD landmark
    if show_wgd and wgd_time is not None:
        ax.axvline(wgd_time, color=WGD_COL, lw=1.1 * scale, ls=(0, (4, 2)), zorder=6)
        if wgd_label:
            ax.text(wgd_time, ne + 0.55, "WGD", color=WGD_COL, fontsize=6.5,
                    ha="center", va="bottom")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.4, ne + 0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"G{i}" for i in ys] if yticklabels else [], fontsize=tick_fs)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=tick_fs)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    return np.median(draws[:, :ne], axis=0)


# ── real allele trees (shared by panels a and c) ──────────────────────────────
def build_real_tree(prefix, chrom, start, collapse_wgd=None, n_burst=None, bursty=False):
    """Build the REAL allele tree for a segment from its chosen route topology.

    By default uses the median draw times. If ``bursty`` is set, picks the MOST-BURSTY polytope draw
    — the draw whose tightest >=3-gain window is smallest (same selection as the burst-at-WGD
    catalog), so the burst shows as in the catalog. If ``collapse_wgd`` (a WGD time) is given instead,
    picks the draw minimizing the span of the ``n_burst`` gains nearest WGD (the doubling co-times at
    WGD; any remaining gain is free). The polytope admits these draws; the median can blur the burst.
    Returns (maj_root, min_root, major_cn, minor_cn, times).

    Cached: the (maj, min, M, m, times) result is pickled per segment as tau-independent LiteNodes,
    so re-renders skip the ~16s tau.trees import. FIGHIGHCN_REBUILD=1 forces recomputation."""
    _mt = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in (CACHE, TAXO))  # invalidate on data change
    ck = hashlib.md5(repr((prefix, chrom, int(start),
                           None if collapse_wgd is None else round(float(collapse_wgd), 6),
                           n_burst, bool(bursty), _mt)).encode()).hexdigest()
    cpath = _TREECACHE / f"{ck}.pkl"
    if cpath.exists() and not os.environ.get("FIGHIGHCN_REBUILD"):
        with open(cpath, "rb") as fh:
            return pickle.load(fh)
    from tau.tree_polytope import _events_for_route
    from tau.trees import build_allele_trees
    oc, _ = _load()
    g = oc[(oc["sample"].str.startswith(prefix)) & (oc["chrom"] == chrom) & (oc["start"] == start)]
    key = g["key"].iloc[0]; ne = int(g["n_events"].iloc[0])
    M, m = int(g.major_cn.iloc[0]), int(g.minor_cn.iloc[0])
    V = g[GCOLS[:ne]].values.astype(float)
    if bursty and len(V) > 1:
        def _tw(row):                                  # tightest >=3-gain window span (catalog metric)
            s = np.sort(row); best = np.inf
            for kk in range(3, len(s) + 1):
                for j in range(len(s) - kk + 1):
                    best = min(best, s[j + kk - 1] - s[j])
            return best
        times = list(V[int(np.argmin([_tw(r) for r in V]))])
    elif collapse_wgd is not None and len(V) > 1:
        k = min(n_burst or ne, ne)
        best, bscore = None, np.inf
        for row in V:
            sel = row[np.argsort(np.abs(row - collapse_wgd))[:k]]   # k gains nearest WGD
            if abs(sel.mean() - collapse_wgd) > 0.15:               # the burst must sit at WGD
                continue
            score = sel.max() - sel.min()                           # tightness of those k gains
            if score < bscore:
                bscore, best = score, row
        times = list(best) if best is not None else list(np.median(V, axis=0))
    else:
        times = list(np.median(V, axis=0))
    times = [float(min(1.0, max(0.0, t))) for t in times]
    maj, mn = build_allele_trees(_events_for_route(key), M, m, times=times)
    result = (_to_lite(maj), _to_lite(mn), M, m, times)   # tau-independent → safe to unpickle without tau
    _TREECACHE.mkdir(parents=True, exist_ok=True)
    with open(cpath, "wb") as fh:
        pickle.dump(result, fh)
    return result


GAIN_DOT = dict(s=16, color=TXT, edgecolors="white", linewidths=0.6, zorder=6)  # uniform gain marker


def draw_real_tree(ax, maj_root, min_root, wgd, *, wgd_win=0.10, show_minor=True, show_band=True,
                   gain_marker="baseline", show_allele_labels=True,
                   maj_span=(0.94, 0.55), min_span=(0.34, 0.12)):
    """Render real allele trees in canonical Tau style (each copy a line to the present; connector
    at its gain time; major solid, minor dashed). show_minor=False for LOH (minor=0); show_band=False
    drops the ±wgd_win shaded WGD window. gain_marker: 'baseline' = legacy baseline dot + teal-at-WGD
    leader; 'branch' = uniform GAIN_DOT on the branch point; 'none' = no marker (e.g. when a leader is
    drawn down to the trajectory dot below). maj_span/min_span set the vertical extent the major/minor
    copies are spread over — widen maj_span for high-CN segments whose branches bunch up. Returns the
    list of gain branch points [(time, y), ...]."""
    G5 = NATREV["grey"][5]
    smaj = sort_nodes(maj_root)
    smin = sort_nodes(min_root) if show_minor else []
    for n, y in zip(smaj, np.linspace(maj_span[0], maj_span[1], len(smaj))): n.y = y
    for n, y in zip(smin, np.linspace(min_span[0], min_span[1], max(len(smin), 1))): n.y = y
    ax.set_xlim(0, 1.02); ax.set_ylim(0.05, 1.0); ax.axis("off")
    if wgd is not None:
        if show_band:
            ax.axvspan(wgd - wgd_win, wgd + wgd_win, color=NATREV["teal"][0], alpha=0.55, zorder=0)
        ax.axvline(wgd, color=WGD_COL, lw=1.0, ls=(0, (3, 2)), zorder=1)
        ax.text(wgd, 1.0, "WGD", color=WGD_COL, fontsize=5.0, ha="center", va="bottom")
    BASE_Y = 0.045
    gains = []

    def _draw(nodes, dashed):
        ls = (0, (2.2, 1.4)) if dashed else "-"
        for n in nodes:
            ax.plot([n.time, 1.0], [n.y, n.y], color=G5, lw=1.2, ls=ls,
                    solid_capstyle="round", zorder=3)
            if n.parent is not None:
                ax.plot([n.time, n.time], [n.y, n.parent.y], color=G5, lw=1.2, ls=ls, zorder=3)
                gains.append((n.time, n.y))
                if gain_marker == "branch":           # uniform dot on the branch point (gain)
                    ax.scatter([n.time], [n.y], **GAIN_DOT)
                elif gain_marker == "baseline":        # legacy baseline marker + leader
                    at = (wgd is not None and abs(n.time - wgd) < wgd_win)
                    c = WGD_COL if at else NATREV["grey"][4]
                    ax.plot([n.time, n.time], [BASE_Y, n.y], color=c, lw=0.4, ls=(0, (1.4, 1.4)),
                            alpha=0.7, zorder=1)
                    ax.plot([n.time], [BASE_Y], "o", color=c, ms=2.2, zorder=4)
    _draw(smaj, False)
    if show_minor:
        _draw(smin, True)
    if show_allele_labels:
        ax.text(0.0, float(np.mean([n.y for n in smaj])), "M", fontsize=5.2,
                color=NATREV["grey"][4], ha="right", va="center")
        if show_minor and smin:
            ax.text(0.0, float(np.mean([n.y for n in smin])), "m", fontsize=5.2,
                    color=NATREV["grey"][4], ha="right", va="center")
    return gains
