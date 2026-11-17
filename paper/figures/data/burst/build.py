"""Fig 6 cache: per-amplification BURST descriptors from the COMPOSITE TABLES (tau_gains + tau_events
+ tau_segments + tau_samples), NOT the genome pkls. Reproduces the dev burst caches
(build_genome_highcn_cache -> figure_burst_classified -> compute_max_burst_size ->
wgd_burst_polytope_analysis) as ONE table-driven builder.

BEST ROUTE ONLY (decision 2026-09-17). The Jun-2026 dev caches behind the assembled figure pooled the
solutions of ALL feasible routes (dev/pcawg/misc_data_and_plots/genome_highcn_timing_cache.tsv), so their
max_burst_size is >= this one and they call 6,076 excess bursts against 2,562 here (MYC n=87 vs 40 in
panel i, 7.1 vs 4.6 bursts per WGD-linked sample in panel d). Best-route is what every other partition in
the paper uses (Fig. 3a, the landscape caches, Supp. Fig. 12), so it is kept; the text must quote THESE numbers.

Per high-CN amplification (best route, n_gains>=3) we reduce its polytope gain-time solutions to a
burst descriptor: the four-tier timing archetype (complete_burst/incomplete_burst/burst_compatible/
gradual), the location-agnostic max_burst_size + excess (>D doubling budget), the burst centroid, and
its linkage to the sample's WGD/PGD. Emits fig6_bursts.tsv (one row per amplification) + a compact
fig6_perdraw.tsv.gz (bursting solutions, for the per-solution WGD-pileup panel c).

Run: python paper/figures/data/burst/build.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.data_tables.tables import load_gains, load_events, load_samples, load_segments

OUT = Path(__file__).resolve().parent
NG = 9                              # G1..G9 (dev caps at 9)
BURST_THRESH, GAP_THRESH, MAX_PREFIX = 0.05, 0.10, 2
WGD_WIN = 0.10                      # burst-at-WGD window (B / coarse label / linkage-near)
WGD_BERTH, PGD_BERTH = 0.20, 0.10  # linkage berths (Fig 5 convention)

# ── load gains (best route only, restricted cols), keep high-CN amplifications (n_gains>=3) ─────────
print("loading tau_gains (best route) ...", flush=True)
g = load_gains(best_route_only=True,
               columns=["sample", "seg_id", "chrom", "start_bp", "end_bp", "major_cn", "minor_cn",
                        "gain_index", "draw_id", "gain_time_frac"])
ne_seg = g.groupby(["sample", "seg_id"])["gain_index"].transform("max")
g = g[ne_seg >= 3].copy()
print(f"  high-CN gain rows (n_gains>=3): {len(g):,}", flush=True)

# ── pivot to a per-(seg,draw) G matrix: rows = seg-draws, cols = G1..G9 (NaN beyond n_gains) ────────
g = g[g.gain_index <= NG]
piv = (g.pivot_table(index=["sample", "seg_id", "draw_id"], columns="gain_index",
                     values="gain_time_frac", aggfunc="first")
       .reindex(columns=range(1, NG + 1)))
keys = piv.index.to_frame(index=False)          # sample, seg_id, draw_id per row
G = piv.values.astype(float)                     # (n_segdraw, 9), sorted G1..Gn, NaN beyond
meta = (g.drop_duplicates(["sample", "seg_id"])
        [["sample", "seg_id", "chrom", "start_bp", "end_bp", "major_cn", "minor_cn"]]
        .reset_index(drop=True))
# n_events (true) per segment, then clipped to 9 for the array math (matches dev)
ne_true = g.groupby(["sample", "seg_id"])["gain_index"].max().rename("n_events")
meta = meta.merge(ne_true, on=["sample", "seg_id"])
seg_ne = dict(zip(zip(meta["sample"], meta["seg_id"]), meta["n_events"]))
ne = np.array([min(int(seg_ne[(s, sd)]), NG) for s, sd in zip(keys["sample"], keys["seg_id"])])
print(f"  {len(keys):,} seg-draw solutions over {len(meta):,} amplifications", flush=True)

# ── WGD scalar (earliest WGD) + PGD times per sample from tau_events / tau_samples ─────────────────
ev = load_events()
WGDT = {s: np.sort(gg.event_time_frac.values) for s, gg in ev[ev.classification == "WGD"].groupby("sample")}
PGDT = {s: np.sort(gg.event_time_frac.values) for s, gg in ev[ev.classification == "PGD"].groupby("sample")}
samp = load_samples()
TT = dict(zip(samp["sample"], samp.tumor_type))
WGD_SCALAR = dict(zip(samp["sample"], samp.wgd_time_frac))     # earliest WGD event time (NaN if none)
wt_row = np.array([WGD_SCALAR.get(s, np.nan) for s in keys["sample"]], float)

# ── per-solution span statistics (vectorized, exactly as figure_burst_classified.py) ───────────────
ar = np.arange(len(G))
gn = G[ar, ne - 1]                               # last gain time
full_span = gn - G[:, 0]
prefix_span = {}; prefix_gap = {}
for k in range(1, MAX_PREFIX + 1):
    valid = ne >= k + 2
    prefix_span[k] = np.where(valid, gn - G[:, k], np.nan)
    prefix_gap[k] = np.where(valid, G[:, k] - G[:, k - 1], np.nan)

# tightest window of size >=3 (span + centroid) and min-3+-window span, per solution
min_win = np.full(len(G), np.inf)
tight3_span = np.full(len(G), np.inf); tight3_cen = np.full(len(G), np.nan)
with np.errstate(invalid="ignore"):
    for w in range(3, NG + 1):
        for j in range(NG - w + 1):
            valid = ne >= (j + w)
            span = G[:, j + w - 1] - G[:, j]
            np.minimum(min_win, np.where(valid, span, np.inf), out=min_win)
            upd = valid & (span < tight3_span)
            tight3_span[upd] = span[upd]
            tight3_cen[upd] = (G[upd, j + w - 1] + G[upd, j]) / 2
min_window_span = np.where(np.isinf(min_win), np.nan, min_win)

# per-solution B (gains within +-WGD_WIN of the scalar WGD time) and max_burst_size
maskv = np.arange(NG)[None, :] < ne[:, None]
within = (np.abs(G - wt_row[:, None]) <= WGD_WIN) & maskv
B_sol = np.where(np.isnan(wt_row), 0, within.sum(axis=1))
mbs = np.ones(len(G), np.int16)                  # largest set of consecutive gains within 0.05 span
with np.errstate(invalid="ignore"):
    for w in range(2, NG + 1):
        spans = G[:, w - 1:] - G[:, :NG - (w - 1)]
        tightw = np.nanmin(np.where(np.isnan(spans), np.inf, spans), axis=1)
        mbs[tightw < BURST_THRESH] = w

# ── aggregate per segment ──────────────────────────────────────────────────────────────────────────
S = pd.DataFrame({"sample": keys["sample"].values, "seg_id": keys["seg_id"].values,
                  "full_span": full_span, "min_window_span": min_window_span,
                  "tight3_span": tight3_span, "B_sol": B_sol, "mbs": mbs})
for k in range(1, MAX_PREFIX + 1):
    S[f"pspan{k}"] = prefix_span[k]; S[f"pgap{k}"] = prefix_gap[k]

grp = S.groupby(["sample", "seg_id"])
agg = grp.agg(max_span=("full_span", "max"),
              min_window_span_any=("min_window_span", lambda x: np.nanmin(x)),
              min_span=("tight3_span", lambda x: np.nanmin(x)),
              max_B=("B_sol", "max"),
              max_burst_size=("mbs", "max")).reset_index()
for k in range(1, MAX_PREFIX + 1):
    agg[f"max_pspan{k}"] = grp[f"pspan{k}"].apply(lambda x: np.nanmax(x)).values
    agg[f"min_pgap{k}"] = grp[f"pgap{k}"].apply(lambda x: np.nanmin(x)).values
agg["min_span"] = agg["min_span"].replace(np.inf, np.nan)

# winmid = tightest >=3-gain window CENTER on the MEDIAN Gi (per wgd_burst_polytope_analysis)
# median Gi via a groupby of the G matrix per segment
segidx = keys[["sample", "seg_id"]].copy()
Gm = (pd.DataFrame(G, columns=[f"G{i}" for i in range(1, NG + 1)])
      .assign(sample=segidx["sample"].values, seg_id=segidx["seg_id"].values)
      .groupby(["sample", "seg_id"]).median().reset_index())
agg = agg.merge(Gm, on=["sample", "seg_id"], how="left")
GM = agg[[f"G{i}" for i in range(1, NG + 1)]].values
nem = np.array([min(int(seg_ne[(s, sd)]), NG) for s, sd in zip(agg["sample"], agg["seg_id"])])
winmid = np.full(len(agg), np.nan); best = np.full(len(agg), np.inf)
with np.errstate(invalid="ignore"):
    for w in range(3, NG + 1):
        for j in range(NG - w + 1):
            valid = nem >= (j + w)
            span = GM[:, j + w - 1] - GM[:, j]
            cen = (GM[:, j + w - 1] + GM[:, j]) / 2
            bt = valid & (span < best)
            winmid[bt] = cen[bt]; best[bt] = span[bt]
fb = nem < 3
winmid[fb] = (GM[fb, 0] + GM[np.arange(len(GM))[fb], nem[fb] - 1]) / 2
agg["centroid"] = winmid

# ── attach metadata, classify tier, compute D / excess / coarse / linkage ──────────────────────────
agg = agg.merge(meta, on=["sample", "seg_id"], how="left")
agg["n_gains"] = agg["n_events"].astype(int)
agg["tumor_type"] = agg["sample"].map(TT)
agg["D"] = (agg["major_cn"] // 2 + agg["minor_cn"] // 2).astype(int)
agg["window_size"] = agg["max_burst_size"].astype(int)          # largest co-timed consecutive gains
agg["excess_k"] = (agg["max_burst_size"] - agg["D"]).astype(int)  # gains beyond the doubling budget
agg["is_excess"] = agg["max_burst_size"] > agg["D"]


def classify(r):
    ne_ = int(r["n_gains"])
    if r["max_span"] < BURST_THRESH:
        return "complete_burst"
    for k in range(MAX_PREFIX, 0, -1):
        if ne_ < k + 3:
            continue
        if (pd.notna(r[f"max_pspan{k}"]) and pd.notna(r[f"min_pgap{k}"])
                and r[f"max_pspan{k}"] < BURST_THRESH and r[f"min_pgap{k}"] > GAP_THRESH):
            return "incomplete_burst"
    if ne_ >= 3 and pd.notna(r["min_window_span_any"]) and r["min_window_span_any"] < BURST_THRESH:
        return "burst_compatible"
    return "gradual"


agg["classification"] = agg.apply(classify, axis=1)

agg["has_wgd"] = agg["sample"].isin(WGDT.keys())
agg["wgd_time"] = agg["sample"].map(WGD_SCALAR)


def coarse(r):                                                  # polytope-aware burst<->WGD label
    if not r["has_wgd"] or pd.isna(r["wgd_time"]):
        return "nonwgd_gradual" if r["classification"] == "gradual" else "nonwgd_burst"
    if r["classification"] == "gradual":
        return "gradual"
    if r["max_B"] >= 2:
        return "excess_burst" if r["max_B"] > r["D"] else "wgd_compatible"
    return "wgd_distinct"


agg["coarse"] = agg.apply(coarse, axis=1)


def _near_dist(times_map, s, c):
    t = times_map.get(s)
    return float(np.min(np.abs(c - t))) if (t is not None and len(t) and not np.isnan(c)) else np.nan


agg["dist_to_wgd"] = [_near_dist(WGDT, s, c) for s, c in zip(agg["sample"], agg["centroid"])]
agg["dist_to_pgd"] = [_near_dist(PGDT, s, c) for s, c in zip(agg["sample"], agg["centroid"])]
agg["is_wgd_linked"] = agg["dist_to_wgd"] <= WGD_BERTH
agg["is_pgd_linked"] = agg["dist_to_pgd"] <= PGD_BERTH

# ess from tau_segments
seglen = load_segments()[["sample", "seg_id", "ess"]]
agg = agg.merge(seglen, on=["sample", "seg_id"], how="left")

COLS = ["sample", "tumor_type", "seg_id", "chrom", "start_bp", "end_bp", "major_cn", "minor_cn",
        "D", "n_gains", "classification", "max_burst_size", "window_size", "excess_k", "is_excess",
        "min_span", "centroid", "max_B", "coarse", "has_wgd", "wgd_time",
        "dist_to_wgd", "is_wgd_linked", "dist_to_pgd", "is_pgd_linked", "ess"]
out = agg[COLS].sort_values(["sample", "chrom", "start_bp"]).reset_index(drop=True)
out.to_csv(OUT / "fig6_bursts.tsv", sep="\t", index=False, float_format="%.5f")
print(f"\nSaved fig6_bursts.tsv: {len(out):,} amplifications -> {OUT}", flush=True)
print("Tier counts (all high-CN amplifications):")
print(out["classification"].value_counts().reindex(
    ["complete_burst", "incomplete_burst", "burst_compatible", "gradual"]).to_string())
print(f"excess (max_burst_size>D): {int(out.is_excess.sum()):,}  "
      f"WGD-linked excess: {int((out.is_excess & out.is_wgd_linked).sum()):,}")

# ── compact per-solution burst table for panel c (per-draw WGD pileup) ──────────────────────────────
pd_burst = pd.DataFrame({"sample": keys["sample"].values, "seg_id": keys["seg_id"].values,
                         "draw_id": keys["draw_id"].values,
                         "centroid": tight3_cen, "span": tight3_span})
pd_burst = pd_burst[np.isfinite(pd_burst["span"]) & (pd_burst["span"] < BURST_THRESH)].copy()
pd_burst = pd_burst.merge(meta[["sample", "seg_id", "chrom", "start_bp", "end_bp"]],
                          on=["sample", "seg_id"], how="left")
pd_burst[["sample", "seg_id", "chrom", "start_bp", "end_bp", "draw_id", "centroid"]].to_csv(
    OUT / "fig6_perdraw.tsv.gz", sep="\t", index=False, compression="gzip", float_format="%.5f")
print(f"Saved fig6_perdraw.tsv.gz: {len(pd_burst):,} bursting solutions", flush=True)
