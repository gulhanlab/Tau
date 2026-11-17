#!/usr/bin/env python3
"""Panel A (worked example) — ground-truth vs Tau-estimate genome timing overviews, stacked.

Top: ground truth (each segment's stack built from the simulation's true t_values).
Bottom: Tau's estimate (standard overview stacks from the inferred timing).
Same x-axis (genome position) and y-axis (normalised time); horizontal lines = event times
(true on top, detected below). The two read near-identically -> whole-genome-accurate timing.

A truth "genome" is made by cloning the estimate genome and overwriting each segment's
timing_result with a single draw whose interval fractions = the true t_values (so the existing
stack renderer draws truth in the identical style). Output:
  notes_site/docs/figures/bench/fig3_A_overview_compare.png
"""
import os, gzip, pickle, json, copy, warnings
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nature_style import apply_nature, save_panel, MM
apply_nature()
warnings.simplefilter("ignore")
from matplotlib.patches import Rectangle
from tau.clustering import cluster_times_bottomup
from tau.plotting import plot_tau_stack, compute_offsets, _cmap_for_major

VER = os.environ.get("FIG3_VER", "v8")
SAMPLE = os.environ.get("EXAMPLE", "SIM_3744")   # clean 2-event (PGD 0.37 / WGD 0.83), no FP,
#                                                  and ALL 22 autosomes covered (no 0-SNV chrom)
sd = Path(f"dev/benchmarking/output_{VER}/{SAMPLE}")
WGD_C = "#0b6979"; PGD_C = "#cd9f2c"   # teal = WGD, gold = PGD (both panels)
EVCOL = {"WGD": WGD_C, "PGD": PGD_C}


def norm(sid):
    c, rng = sid.split(":"); a, b = rng.split("-")
    return (c.replace("chr", ""), int(a), int(b))


g = pickle.load(gzip.open(sd / f"{SAMPLE}.genome.pkl.gz", "rb"))
ct, scid, ot, summ = cluster_times_bottomup(g)
truth = json.load(open(sd / f"{SAMPLE}.truth.json"))
true_ev = sorted((e["type"], float(e["time"])) for e in truth["genome_events"]
                 if e["type"] in ("WGD", "PGD"))
true_times = [t for _, t in true_ev]
tmap = {norm(ts["seg_id"]): list(ts["t_values"]) for ts in truth.get("segments", [])
        if len(ts.get("t_values", [])) >= 2}

# clone genome, overwrite each segment timing with the TRUE interval fractions
tg = copy.deepcopy(g)
for seg in tg.segments:
    tv = tmap.get(norm(seg.seg_id))
    if tv:
        tdict = {f"t{i+1}": float(v) for i, v in enumerate(tv)}
        seg.timing_result = {"truth": {"qc": [{"dist_rel": 0.0}],
                                       "draws": [{"t": tdict, "boot_id": 0}],
                                       "meta": {"free_var_count": 1}}}
    elif int(seg.major_cn) == 1:
        # single major copy (no gain to time): flat grey block, matching the estimate track
        seg.timing_result = {"flat": {"qc": [{"dist_rel": 0.0}],
                                      "draws": [{"t": {"t1": 1.0}, "boot_id": 0}],
                                      "meta": {"free_var_count": 1}}}
    else:
        seg.timing_result = {}

est_ord, off = compute_offsets(g.segments)
tru_ord, off_t = compute_offsets(tg.segments)

# slightly narrower than full double-column; left column holds the legend
fig = plt.figure(figsize=(89 * MM, 45 * MM))   # single column (full width)
gs = fig.add_gridspec(3, 1, height_ratios=[0.13, 0.435, 0.435], hspace=0.30)
axL = fig.add_subplot(gs[0]); axL.axis("off")   # horizontal legend strip on top
axT = fig.add_subplot(gs[1])
axB = fig.add_subplot(gs[2])

# truth events: solid lines, teal=WGD / gold=PGD.
# ess_thresh=0: ground truth is not subject to Tau's mutation-count detection limit, so show
# every true segment (incl. the low-mutation (2,2) that the estimate cannot time).
plot_tau_stack(axT, tru_ord, off_t, cluster_times=[], color_by_cluster=False, ess_thresh=0)
for typ, t in true_ev:
    axT.axhline(t, color=EVCOL.get(typ, "black"), ls="-", lw=1.4, alpha=0.95, zorder=6)
axT.set_title(f"Ground truth — {SAMPLE}", loc="left", fontsize=6, pad=2)
axT.tick_params(labelbottom=False)   # chrom numbers shown once, on the bottom track

# estimated events: dashed lines, same teal/gold by detected class (suppress tab10 default)
plot_tau_stack(axB, est_ord, off, cluster_times=[], color_by_cluster=False)
for _, r in summ.iterrows():
    if r["classification"] in EVCOL:
        axB.axhline(float(r["time"]), color=EVCOL[r["classification"]], ls="--",
                    lw=1.4, alpha=0.95, zorder=6)
axB.set_title("Tau estimate", loc="left", fontsize=6, pad=2)

for a in (axT, axB):   # plot_tau_stack hardcodes the y-label at 8 pt; shorten for the short tracks
    a.set_ylabel("norm. time", fontsize=6)

# chrom labels: plot_tau_stack draws them as 5 pt text — keep them ONLY on the bottom track and
# make them smaller (they were colliding with the "Tau estimate" title)
for t in list(axT.texts):
    t.remove()
for t in axB.texts:
    t.set_fontsize(3.6)

# ── horizontal legend strip (top): CN-state colour (light→dark = early→late) + event lines ──
axL.set_xlim(0, 1); axL.set_ylim(0, 1)
N = 5
x = 0.005; sw = 0.075          # swatch-block width per CN state
for maj in [2, 3, 4, 5]:
    cmap = _cmap_for_major(maj)
    for j in range(N):
        axL.add_patch(Rectangle((x + sw * j / N, 0.40), sw / N, 0.45,
                                facecolor=cmap((j + 1) / N), edgecolor="none"))
    axL.text(x + sw / 2, 0.10, f"CN{maj}", ha="center", va="top", fontsize=6)
    x += sw + 0.035
for typ, lab, col in [("WGD", "WGD", WGD_C), ("PGD", "ccPG", PGD_C)]:   # event lines
    axL.plot([x, x + 0.05], [0.62, 0.62], color=col, lw=1.6)          # typ = stored class, lab = paper name
    axL.text(x + 0.062, 0.62, lab, va="center", ha="left", fontsize=6)
    x += 0.155
axL.text(1.0, 0.62, "interval: early to late", fontsize=5.5, color="#888", ha="right", va="center")

fig.subplots_adjust(left=0.11, right=0.995, top=0.985, bottom=0.085, hspace=0.52)
out = Path("notes_site/docs/figures/bench/fig3_a_example")
save_panel(fig, out)
fig.savefig("notes_site/docs/figures/bench/fig3_A_overview_compare.png", dpi=300,
            bbox_inches="tight")   # keep legacy name for the page
print("true events:", [round(t, 3) for t in true_times])
print("detected:", [(r.classification, round(float(r.time), 3)) for _, r in summ.iterrows()])
print("truth segs injected:", sum(1 for s in tg.segments if s.timing_result),
      "/ est timed:", sum(1 for s in g.segments if s.timing_result))
print("saved", out)
