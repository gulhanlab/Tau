"""Table-driven allele-tree + trajectory helpers for Fig 6 exemplar panels a / b.

A segment's per-solution gain times come from tau_gains (base table); its allele-tree TOPOLOGY is
reconstructed from the shipped route topology (tau.tree_polytope._events_for_route + tau.trees.
build_allele_trees — reference geometry, NO genome pkl); its WGD time from tau_events. Rendering helpers
(sort_nodes / draw_real_tree / plot_traj) are self-contained copies of the canonical Tau tree style.
"""
import re
import sys
from pathlib import Path
import numpy as np, pandas as pd
from matplotlib.colors import to_rgba

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.data_tables.tables import load_gains, load_events, load_samples
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

NATREV = {"teal": ["#cee7ed", "#98d2d3", "#4cbdbd", "#0d9aa2", "#0b6979", "#07394d"],
          "grey": ["#e7e6ec", "#cacdd9", "#99a2b3", "#717c91", "#4b566d", "#273246"]}
WGD_COL, TXT, MM = "#0b6979", "#273246", 1 / 25.4
GCOLS_MAX = 9
GAIN_DOT = dict(s=16, color=TXT, edgecolors="white", linewidths=0.6, zorder=6)
BURST_THRESH = 0.05

_G = None; _WGD = None; _TT = None


def _load():
    """Lazy: slim best-route gains + per-sample WGD scalar + tumor type. Cached per process."""
    global _G, _WGD, _TT
    if _G is None:
        _G = load_gains(best_route_only=True,
                        columns=["sample", "seg_id", "chrom", "start_bp", "end_bp", "major_cn",
                                 "minor_cn", "route_key", "gain_index", "draw_id", "gain_time_frac"])
        _G["chrom"] = _G["chrom"].astype(str)
        samp = load_samples()
        _WGD = dict(zip(samp["sample"], samp.wgd_time_frac))
        _TT = dict(zip(samp["sample"], samp.tumor_type))
    return _G, _WGD, _TT


def _seg_rows(prefix, chrom, start):
    g, _, _ = _load()
    sub = g[g["sample"].str.startswith(prefix) & (g["chrom"] == str(chrom)) & (g["start_bp"] == start)]
    if not len(sub):
        raise ValueError(f"segment not found in tau_gains: {prefix} {chrom}:{start}")
    return sub


def _draw_matrix(sub):
    """(n_draws, n_events) matrix of gain_time_frac, sorted by gain_index; + n_events."""
    ne = int(sub["gain_index"].max())
    piv = (sub[sub.gain_index <= GCOLS_MAX]
           .pivot_table(index="draw_id", columns="gain_index", values="gain_time_frac", aggfunc="first")
           .reindex(columns=range(1, min(ne, GCOLS_MAX) + 1)))
    return piv.values.astype(float), min(ne, GCOLS_MAX)


def fetch(prefix, chrom, start):
    sub = _seg_rows(prefix, chrom, start)
    _, WGD, TT = _load()
    r0 = sub.iloc[0]; s = r0["sample"]
    V, ne = _draw_matrix(sub)
    wgd = WGD.get(s, np.nan)
    return dict(draws=V, ne=ne, wgd=(float(wgd) if pd.notna(wgd) else None),
                major=int(r0.major_cn), minor=int(r0.minor_cn), sample=s,
                tumor_type=str(TT.get(s, "")), end=int(r0.end_bp), key=str(r0.route_key))


# ── real allele trees ───────────────────────────────────────────────────────────────
class LiteNode:
    def __init__(self, time=0.0, alignment=None, parent=None):
        self.time = float(time); self.branches = []; self.alignment = alignment
        self.parent = parent; self.mults = []; self.y = 0.0


def _to_lite(node, parent=None):
    if node is None:
        return None
    ln = LiteNode(time=node.time, alignment=node.alignment, parent=parent)
    ln.mults = list(getattr(node, "mults", []))
    ln.branches = [_to_lite(b, ln) for b in node.branches]
    return ln


def sort_nodes(node):
    if node is None:
        return []
    if len(node.branches) == 0:
        return [node]
    left = [b for br in node.branches if br.alignment == "L" for b in sort_nodes(br)]
    right = [b for br in node.branches if br.alignment == "R" for b in sort_nodes(br)]
    return [*left, node, *right]


def _tightest3(row):
    s = np.sort(row); best = np.inf
    for k in range(3, len(s) + 1):
        for j in range(len(s) - k + 1):
            best = min(best, s[j + k - 1] - s[j])
    return best


def build_real_tree(prefix, chrom, start, collapse_wgd=None, n_burst=None, bursty=False):
    """Return (maj_root, min_root, M, m, times) — real allele tree at a chosen polytope solution."""
    from tau.tree_polytope import _events_for_route
    from tau.trees import build_allele_trees
    sub = _seg_rows(prefix, chrom, start)
    key = str(sub.iloc[0]["route_key"]); M, m = int(sub.iloc[0].major_cn), int(sub.iloc[0].minor_cn)
    V, ne = _draw_matrix(sub)
    if bursty and len(V) > 1:
        times = list(V[int(np.argmin([_tightest3(r) for r in V]))])
    elif collapse_wgd is not None and len(V) > 1:
        k = min(n_burst or ne, ne); best, bscore = None, np.inf
        for row in V:
            sel = row[np.argsort(np.abs(row - collapse_wgd))[:k]]
            if abs(sel.mean() - collapse_wgd) > 0.15:
                continue
            score = sel.max() - sel.min()
            if score < bscore:
                bscore, best = score, row
        times = list(best) if best is not None else list(np.median(V, axis=0))
    else:
        times = list(np.median(V, axis=0))
    times = [float(min(1.0, max(0.0, t))) for t in times]
    maj, mn = build_allele_trees(_events_for_route(key), M, m, times=times)
    return _to_lite(maj), _to_lite(mn), M, m, times


def draw_real_tree(ax, maj_root, min_root, wgd, *, wgd_win=0.10, show_minor=True, show_band=True,
                   gain_marker="none", show_allele_labels=False, maj_span=(0.94, 0.55), min_span=(0.34, 0.12)):
    G5 = NATREV["grey"][5]
    smaj = sort_nodes(maj_root); smin = sort_nodes(min_root) if show_minor else []
    for n, y in zip(smaj, np.linspace(maj_span[0], maj_span[1], len(smaj))):
        n.y = y
    for n, y in zip(smin, np.linspace(min_span[0], min_span[1], max(len(smin), 1))):
        n.y = y
    ax.set_xlim(0, 1.02); ax.set_ylim(0.05, 1.0); ax.axis("off")
    if wgd is not None:
        if show_band:
            ax.axvspan(wgd - wgd_win, wgd + wgd_win, color=NATREV["teal"][0], alpha=0.55, zorder=0)
        ax.axvline(wgd, color=WGD_COL, lw=1.0, ls=(0, (3, 2)), zorder=1)
        ax.text(wgd, 1.0, "WGD", color=WGD_COL, fontsize=S.tiny, ha="center", va="bottom")

    def _draw(nodes, dashed):
        ls = (0, (2.2, 1.4)) if dashed else "-"
        for n in nodes:
            ax.plot([n.time, 1.0], [n.y, n.y], color=G5, lw=1.2, ls=ls, solid_capstyle="round", zorder=3)
            if n.parent is not None:
                ax.plot([n.time, n.time], [n.y, n.parent.y], color=G5, lw=1.2, ls=ls, zorder=3)
                if gain_marker == "branch":
                    ax.scatter([n.time], [n.y], **GAIN_DOT)
    _draw(smaj, False)
    if show_minor:
        _draw(smin, True)


def plot_traj(ax, draws, ne, wgd_time, *, tick_fs=6, scale=0.85, yticklabels=True):
    """Faint grey solution fan (no median / WGD — drawn by the caller)."""
    ys = np.arange(1, ne + 1)
    alpha = max(0.05, min(0.5, 30.0 / max(len(draws), 1)))
    for d in draws:
        ax.plot(d[:ne], ys, color=NATREV["grey"][1], alpha=alpha, lw=0.5 * scale, zorder=2, solid_capstyle="round")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.4, ne + 0.7); ax.set_yticks(ys)
    ax.set_yticklabels([f"G{i}" for i in ys] if yticklabels else [], fontsize=tick_fs)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"], fontsize=tick_fs)
    ax.tick_params(length=2, width=0.5, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def fix_svg_fonts(path):
    p = Path(path); s = p.read_text()
    s = re.sub(r"font: (\d+) ([0-9.]+)px ", r"font-weight: \1; font-size: \2px; font-family: ", s)
    s = re.sub(r"font: ([0-9.]+)px ", r"font-size: \1px; font-family: ", s)
    p.write_text(s)
