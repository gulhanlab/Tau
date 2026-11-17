#!/usr/bin/env python3
"""
classify_wgd_burst_all.py

Coarse burst↔WGD taxonomy for ALL high-CN segments (all CN states).

For a segment ending (M,m) with median gain times G1..Gn and (if WGD+) wgd_time:
  D = floor(M/2) + floor(m/2)        # largest single-WGD doubling block
  B = # gains within ±WGD_WIN of wgd_time   (median Gi across solutions)

WGD samples — cross the existing tier (complete/incomplete/burst_compatible/gradual)
with the burst↔WGD relationship:
  wgd_compatible : burst at WGD, B <= D   (consistent with one doubling)
  excess_burst   : burst at WGD, B >  D   (more co-timed than one doubling → genuine
                                           co-amplification, or needs >=2 doublings)
  wgd_distinct   : a burst exists but sits away from WGD (B < 2 near WGD)
  gradual        : no burst

Non-WGD samples — no WGD to compare against; any burst is genuine non-WGD
co-amplification by construction:
  nonwgd_burst   : complete/incomplete/burst_compatible
  nonwgd_gradual : gradual

Inputs:
  genome_highcn_timing_cache.tsv     (per-solution G values → median Gi)
  burst_classified_summary.tsv       (category tier)
  gain_progression_metadata.tsv      (has_wgd, wgd_time)
Outputs:
  wgd_burst_taxonomy.tsv
  wgd_burst_taxonomy_wgd.pdf/svg     (WGD samples, tier × relationship)
  wgd_burst_taxonomy_nonwgd.pdf/svg  (non-WGD samples)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE    = Path(os.environ.get("TAU_REPO", "."))
MISC    = BASE / "dev/pcawg/misc_data_and_plots"
OUT_DIR = BASE / "dev/pcawg/oncogenes/figures_genome_highcn_catalogue"

WGD_WIN = 0.10

NATREV = {
    "teal":   ["#cee7ed","#98d2d3","#4cbdbd","#0d9aa2","#0b6979","#07394d"],
    "grey":   ["#e7e6ec","#cacdd9","#99a2b3","#717c91","#4b566d","#273246"],
    "gold":   ["#fdf3d0","#fbe49a","#f5cc4e","#cd9f2c","#9a6e12","#5c3d06"],
    "orange": ["#fddfbc","#fdc07e","#f89a45","#ec7023","#b75026","#85341b"],
}
REL_ORDER  = ["excess_burst", "wgd_compatible", "wgd_distinct", "gradual"]
REL_COLOR  = {
    "excess_burst":   NATREV["orange"][3],
    "wgd_compatible": NATREV["teal"][3],
    "wgd_distinct":   NATREV["gold"][3],
    "gradual":        NATREV["grey"][2],
}
TIER_ORDER = ["complete_burst", "incomplete_burst", "burst_compatible", "gradual"]

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 7, "axes.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "figure.dpi": 300,
})
MM = 1 / 25.4
SEG   = ["sample", "chrom", "start", "end"]
gcols = [f"G{i}" for i in range(1, 10)]

# ── median Gi per segment ───────────────────────────────────────────────────────
print("Loading per-solution cache …", flush=True)
usecols = SEG + ["major_cn", "minor_cn", "n_events"] + gcols
oc = pd.read_csv(MISC / "genome_highcn_timing_cache.tsv", sep="\t",
                 dtype={"chrom": str}, usecols=usecols)
print(f"  {len(oc):,} solution-rows; aggregating median Gi …", flush=True)
agg = (oc.groupby(SEG)
         .agg({**{g: "median" for g in gcols},
               "major_cn": "first", "minor_cn": "first", "n_events": "first"})
         .reset_index())
print(f"  {len(agg):,} segments", flush=True)

# ── tier + WGD metadata ──────────────────────────────────────────────────────────
clf = pd.read_csv(OUT_DIR / "burst_classified_summary.tsv", sep="\t", dtype={"chrom": str})
agg = agg.merge(clf[SEG + ["category", "tumor_type"]], on=SEG, how="left")

wgd = pd.read_csv(MISC / "gain_progression_metadata.tsv", sep="\t")
wgd["has_wgd"]  = wgd["has_wgd"].astype(str).str.lower() == "true"
wgd["wgd_time"] = pd.to_numeric(wgd["wgd_time"], errors="coerce")
agg["has_wgd"]  = agg["sample"].map(wgd.set_index("sample")["has_wgd"]).fillna(False).astype(bool)
agg["wgd_time"] = agg["sample"].map(wgd.set_index("sample")["wgd_time"])

# ── B and D ──────────────────────────────────────────────────────────────────────
G  = agg[gcols].values
ne = agg["n_events"].clip(upper=len(gcols)).astype(int).values
wt = agg["wgd_time"].values
mask_valid = np.arange(len(gcols))[None, :] < ne[:, None]
within = (np.abs(G - wt[:, None]) <= WGD_WIN) & mask_valid
agg["B"] = np.where(np.isnan(wt), -1, within.sum(axis=1))
agg["D"] = (agg["major_cn"] // 2 + agg["minor_cn"] // 2).astype(int)

# ── coarse label ─────────────────────────────────────────────────────────────────
def label(r):
    if not r.has_wgd or pd.isna(r.wgd_time):
        return "nonwgd_gradual" if r.category == "gradual" else "nonwgd_burst"
    if r.category == "gradual":
        return "gradual"
    if r.B >= 2:
        return "excess_burst" if r.B > r.D else "wgd_compatible"
    return "wgd_distinct"

agg["coarse"] = agg.apply(label, axis=1)
agg.to_csv(OUT_DIR / "wgd_burst_taxonomy.tsv", sep="\t", index=False)
print(f"Saved → {OUT_DIR / 'wgd_burst_taxonomy.tsv'}", flush=True)

wgd_s    = agg[agg["has_wgd"] & agg["wgd_time"].notna()]
nonwgd_s = agg[~(agg["has_wgd"] & agg["wgd_time"].notna())]
print(f"\nWGD-sample segments: {len(wgd_s):,}   non-WGD: {len(nonwgd_s):,}", flush=True)

print("\nWGD samples — tier × burst↔WGD relationship:", flush=True)
print(f"{'tier':<18} {'excess':>7} {'wgd_compat':>11} {'wgd_distinct':>13} {'gradual':>8} {'n':>6}",
      flush=True)
for tier in TIER_ORDER:
    sub = wgd_s[wgd_s["category"] == tier]
    counts = sub["coarse"].value_counts()
    print(f"{tier:<18} {int(counts.get('excess_burst',0)):>7} "
          f"{int(counts.get('wgd_compatible',0)):>11} "
          f"{int(counts.get('wgd_distinct',0)):>13} "
          f"{int(counts.get('gradual',0)):>8} {len(sub):>6}", flush=True)

print("\nNon-WGD samples — tier counts:", flush=True)
for tier in TIER_ORDER:
    n = int((nonwgd_s["category"] == tier).sum())
    print(f"  {tier:<18} {n:>6}", flush=True)

# ── Plot 1: WGD samples, tier × relationship (stacked) ──────────────────────────
print("\nPlotting …", flush=True)
fig, ax = plt.subplots(figsize=(110*MM, 72*MM))
fig.subplots_adjust(left=0.12, right=0.74, top=0.88, bottom=0.22)
xs = np.arange(len(TIER_ORDER))
bottoms = np.zeros(len(TIER_ORDER))
for rel in REL_ORDER:
    vals = []
    for tier in TIER_ORDER:
        sub = wgd_s[wgd_s["category"] == tier]
        vals.append((sub["coarse"] == rel).sum())
    vals = np.array(vals)
    ax.bar(xs, vals, bottom=bottoms, color=REL_COLOR[rel], width=0.7,
           label=rel.replace("_", " "))
    bottoms += vals
ax.set_xticks(xs)
ax.set_xticklabels([t.replace("_", "\n") for t in TIER_ORDER], fontsize=6)
ax.set_ylabel("Segments (WGD samples)")
ax.set_title(f"Burst↔WGD taxonomy in WGD samples\nD=⌊M/2⌋+⌊m/2⌋, burst within ±{WGD_WIN} of WGD",
             fontsize=7.5)
ax.legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1),
          title="burst↔WGD", title_fontsize=6)
ax.spines[["top", "right"]].set_visible(False)
for ext in ("pdf", "svg"):
    fig.savefig(OUT_DIR / f"wgd_burst_taxonomy_wgd.{ext}", bbox_inches="tight")
    print(f"Saved → wgd_burst_taxonomy_wgd.{ext}", flush=True)
plt.close(fig)

# ── Plot 2: non-WGD samples, tier counts ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(80*MM, 70*MM))
fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.24)
ns = [int((nonwgd_s["category"] == t).sum()) for t in TIER_ORDER]
colors = [NATREV["teal"][4], NATREV["teal"][3], NATREV["teal"][1], NATREV["grey"][2]]
ax.bar(np.arange(len(TIER_ORDER)), ns, color=colors, width=0.7)
ax.set_xticks(np.arange(len(TIER_ORDER)))
ax.set_xticklabels([t.replace("_", "\n") for t in TIER_ORDER], fontsize=6)
ax.set_ylabel("Segments (non-WGD samples)")
ax.set_title("Non-WGD samples\n(all bursts are non-WGD by construction)", fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)
for ext in ("pdf", "svg"):
    fig.savefig(OUT_DIR / f"wgd_burst_taxonomy_nonwgd.{ext}", bbox_inches="tight")
    print(f"Saved → wgd_burst_taxonomy_nonwgd.{ext}", flush=True)
plt.close(fig)
print("\nDone.", flush=True)
