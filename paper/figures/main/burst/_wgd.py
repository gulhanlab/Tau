"""Shared burst<->WGD helpers for Fig 6 panels c/d/e/f/g. Table-driven: fig6_bursts.tsv (per-amplification
burst descriptor from paper/figures/data/burst/build.py) + tau_events (per-sample WGD event times). No genome pkls.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.config import cache_dir_of
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.data_tables.tables import load_events

CACHE = cache_dir_of(__file__)
WGD_WIN = 0.10                       # burst-at-WGD window
BURST_CATS = ["complete_burst", "incomplete_burst", "burst_compatible"]
NATREV = {"teal": ["#cee7ed", "#98d2d3", "#4cbdbd", "#0d9aa2", "#0b6979", "#07394d"],
          "grey": ["#e7e6ec", "#cacdd9", "#99a2b3", "#717c91", "#4b566d", "#273246"]}
WGD_COL, NULL_COL, NULL_TICK, TXT = "#0b6979", NATREV["grey"][1], NATREV["grey"][4], "#273246"
MM = 1 / 25.4
# high-powered oncogenes (segment-overlap loci; matches figure_burst_wgd_lead.py)
ONC = {"MYC": ("8", 127735434, 127742951), "EGFR": ("7", 55086725, 55324313),
       "MET": ("7", 116312457, 116436282), "CDK6": ("7", 92234236, 92542718),
       "PIK3CA": ("3", 178866311, 179024555), "MDM4": ("1", 204485509, 204526217),
       "TERT": ("5", 1253167, 1295068), "ERBB2": ("17", 37844393, 37884915)}
OG_MID = {k: (c, (s + e) // 2) for k, (c, s, e) in ONC.items()}
HIGHPOWER = ["MYC", "MDM4", "PIK3CA", "EGFR", "CDK6", "MET", "TERT", "ERBB2"]


def load_bursts():
    return pd.read_csv(CACHE / "fig6_bursts.tsv", sep="\t", dtype={"chrom": str})


def wgd_event_sets():
    """{sample: np.array(WGD event times)} from tau_events (classification == WGD)."""
    ev = load_events()
    return {s: np.sort(g.event_time_frac.values)
            for s, g in ev[ev.classification == "WGD"].groupby("sample") if len(g)}


def excess_wgd_bursts(bursts, events):
    """EXCESS bursts (max_burst_size > D, any non-gradual tier) in WGD samples, with a valid centroid."""
    b = bursts[bursts["classification"].isin(BURST_CATS) & bursts["is_excess"]
               & bursts["centroid"].notna() & bursts["sample"].isin(events.keys())].copy()
    return b.reset_index(drop=True)


def near_matrix(b, events):
    """near[i, p] = burst i centroid within +-WGD_WIN of ANY WGD event of pool sample p; + own linkage."""
    pool = np.array(list(events.keys())); pool_ev = [events[s] for s in pool]
    P = len(pool); spos = {s: i for i, s in enumerate(pool)}
    w = b["centroid"].values
    near = np.zeros((len(b), P), bool)
    for p in range(P):
        near[:, p] = (np.abs(w[:, None] - pool_ev[p][None, :]) < WGD_WIN).any(axis=1)
    sidx = b["sample"].map(spos).values
    own = near[np.arange(len(b)), sidx]
    return near, sidx, P, pool_ev, pool


def enrich(mask, near, own, sidx, P, n_perm=20000, seed=42):
    idx = np.where(mask)[0]; n = len(idx)
    if n < 5:
        return None
    rng = np.random.default_rng(seed)
    obs = own[idx].mean(); si = sidx[idx]; sn = near[idx]
    perm = np.array([sn[np.arange(n), rng.permutation(P)[si]].mean() for _ in range(n_perm)])
    return dict(n=n, obs=obs, mu=perm.mean(), sd=perm.std(),
                p=(np.sum(perm >= obs) + 1) / (n_perm + 1))


def og_deltas(b, gene, events):
    """delta = burst centroid - nearest WGD event time, per excess burst overlapping the oncogene."""
    c, m = OG_MID[gene]
    mask = (b["chrom"].values == c) & (b["start_bp"].values <= m) & (b["end_bp"].values >= m)
    gw = b["centroid"].values[mask]; gs = b["sample"].values[mask]
    return np.array([(gw[i] - events[gs[i]])[np.argmin(np.abs(gw[i] - events[gs[i]]))]
                     for i in range(len(gw))])


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
