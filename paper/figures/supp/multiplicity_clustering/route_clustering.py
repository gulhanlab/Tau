"""Supp — multiplicity-based route clustering, demonstrated on both Fig 2 example tumours.

One figure per sample; three panels each, following the inference:
  a  the CN-state track exactly as the sample's overview (Fig 2) draws it -- one row per state,
     colour by major CN, cluster by fill hatch -- with the focus state boxed and recoloured by
     multiplicity cluster
  b  the multiplicity profiles those clusters are formed from
  c  the pooled-segment tree each cluster yields -- distinct histories from one CN state

COLOURS ARE TIED TO FIG 2. tau/overview.py:467 colours every tree by `_cmap_for_major(major)(0.6)`,
so a CN state has one colour family throughout Fig 2 ((2,x) green, (3,x) orange, (4,x) purple,
(5,x) blue) and clusters are separated by hatch. Here clusters take SHADES of that same family, so
the state's identity still reads across from Fig 2 while the clusters stay distinguishable.

PANEL B DEPENDS ON THE STATE. A normalised profile of a major_cn=K state lives on a (K-1)-simplex,
so PCA is only meaningful for K>=3; for K=2 the profile is one number and panel b is drawn as a
1-D strip instead. Both show the same thing: individual segments separating into clusters.

Chosen states (from a scan of every multi-cluster state in each sample, ranked by profile
separation and by how far apart the clusters' first gains fall):
  0250e6f7 Eso.AdenoCA  (3,0)  chr3 gains at the ccPG (0.23); chr7/10/11/17 at the WGD (0.87)
                               (2,0) separates even harder but is 1-D; its chr15 cluster is the
                               ccPG's other arm, recovered independently -- worth a caption line
  dc537fcf Stomach      (2,2)  two large clusters, 74 vs 23 segments, first gains 0.72 vs 0.30,
                               straddling the two WGDs of this double-WGD genome
"""
import gzip, json, pickle, sys
from pathlib import Path

import numpy as np, pandas as pd
from paper.config import panels_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
from tau.clustering import cluster_segment_multiplicities, make_clustered_genome
from tau.timing import _get_route_env
from tau.utils import pick_best_key, extract_mat
from tau.trees import _build_tree_for_cluster, plot_route, _get_final_segment_clusters
from tau.plotting import (_cmap_for_major, compute_offsets, plot_tau_stack,
                          segment_cluster_plotting)
from tau.overview import _row_color

R = Path(__file__).resolve().parents[4]   # repo root
OUT = panels_dir_of(__file__)
GREY = "#c8c8c8"


def _event_style(r):
    """Event name + Fig 2's own colour for it (tau/overview.py::_row_color: WGD #0b6979 teal,
    ccPG/PGD #cd9f2c gold), so the lines here match the ones running across the Fig 2 panels."""
    cls = str(r.classification)
    lbl = "ccPG" if cls.upper() in ("PGD", "CCPG") else cls
    return lbl, _row_color("ccPG" if lbl == "ccPG" else cls)
FOCUS_H = 1.55
HATCHES = ["", "////////", "........", "xxxxxxxx", "--------"]
# Clusters of one CN state are the SAME COLOUR in Fig 2 -- the track separates them by hatch, and
# the light->dark gradient there is t_early->t_late WITHIN a block, not cluster identity. So keep the
# hue and vary only lightness, staying near the paper's tree colour cmap(0.6). Do NOT go past ~0.75:
# the (3,x) map is #973003 (brown) at 0.92, which reads as a second hue rather than a shade.
# Clusters are listed in index order (0, 1, 2, ...) and shaded dark -> light in that same order,
# so the legend, the trees and the colour ramp all run the same way. Shades stay inside one CN-state
# family (see the module docstring); do not exceed ~0.8, where the (3,x) map turns brown.
SHADES = [0.80, 0.55, 0.34, 0.95]

CONFIGS = [
    # eps=0.10 (NOT the shipped default of 0.07). At 0.07 the contiguous chr10 block is left
    # unattached and becomes a third cluster whose gains (0.85, 0.89) are indistinguishable from
    # cluster 0's (0.87, 0.95) -- a difference far below Tau's ~0.15 temporal resolution, i.e. one
    # event split in two. 0.10 merges them while leaving the chr3 ccPG cluster separate all the way
    # to eps=0.30, and cohort-wide it is the best-supported value for this state (same-arm NN median
    # 0.078; chi2 GOF: +0.5pp heterogeneity against 36 over-split pairs fixed). Because eps differs
    # from the run on disk, the pooled genome is REBUILT in memory rather than read from the pickle.
    dict(short="eso", tt="Eso.AdenoCA", sample="0250e6f7-2538-4d1f-a8ff-ca34a7057dde",
         state=(3, 0), show=[0, 1], fig="Fig 2b", eps=0.10, separate_panels=True),
    dict(short="stomach", tt="Stomach.AdenoCA", sample="dc537fcf-d910-4c4b-8af9-e7da429f2633",
         state=(2, 2), show=[0, 1], fig="Fig 2a", eps=None),
]

# Double-column (183 mm) width -- the genome overview in panel a needs it, as it does in Fig 2.
# ALL type stays on a 6-8 pt scale; nothing below 6.
# Everything the borrowed tau plotting code sets for itself (chromosome names at 4 pt, CN-state row
# labels at 7 pt) is overridden after the fact in panel a.
SINGLE_W = 3.46          # inches == 88 mm (kept for reference)
DOUBLE_W = 7.20          # inches == 183 mm
FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = S.title, S.label, S.tick, S.annot
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": FS_TICK, "axes.labelsize": FS_LABEL,
    "axes.linewidth": 0.5, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_TICK, "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "svg.fonttype": "none", "figure.dpi": 300,
})
_style_apply()   # the shared ladder wins over the sizes in the block above


def _chrom_offsets():
    sizes = {}
    with open(R / "tau/data/hg19.chrom.sizes") as f:
        for line in f:
            c, n = line.split()[:2]
            c = c.replace("chr", "")
            if c.isdigit():
                sizes[c] = int(n)
    off, cum = {}, 0
    for c in sorted(sizes, key=int):
        off[c] = cum; cum += sizes[c]
    return off, sizes, cum


def build(cfg, save=True, panel=None):
    """panel=None draws the combined single-column figure; panel in {'a','b','c'} draws ONLY that
    panel into its own figure, so the exported SVG contains that panel's artists and nothing else.
    (Cropping the combined figure with bbox_inches does NOT do this -- it clips the canvas but every
    artist stays in the file, which is why the first attempt produced three ~670 KB SVGs each
    carrying all 66 text elements.) Leader lines span panels, so they are drawn only when
    panel is None.
    """
    STATE, SHOW = cfg["state"], cfg["show"]
    from paper.config import OBJECTS
    RUN = OBJECTS / "pcawg_v2" / cfg["tt"] / cfg["sample"]   # re-emitted (v2) objects of the two examples
    S = cfg["sample"]
    cmap = _cmap_for_major(STATE[0])
    COL = {lab: cmap(SHADES[i % len(SHADES)]) for i, lab in enumerate(SHOW)}

    g = pickle.load(gzip.open(RUN / f"{S}.genome.pkl.gz", "rb"))
    events = pd.read_csv(RUN / f"{S}.tau_events.tsv", sep="\t")
    eps = cfg.get("eps")
    if eps is None:
        cg = pickle.load(gzip.open(RUN / f"{S}.clustered_genome.pkl.gz", "rb"))
        scd = cluster_segment_multiplicities(g, output_plot_file=None)
    else:
        # non-default eps: recluster and rebuild the pooled genome exactly as tau/cli.py does,
        # so the track, the profiles and the trees all describe the SAME clustering
        scd = cluster_segment_multiplicities(g, output_plot_file=None, eps=eps)
        cg = make_clustered_genome(g, scd)
        cg.calculate_multiplicities(bootstrap_B=1, random_state=42)
        cg.time_segments(env=_get_route_env())
    mk = scd[(scd.major_cn == STATE[0]) & (scd.minor_cn == STATE[1])].copy()

    byid = {s.seg_id: s for s in g.segments}
    prof, keep = [], []
    for r in mk.itertuples():
        s = byid.get(r.seg_id)
        if s is None or s.N_counts is None:
            continue
        n = np.asarray(s.N_counts, float)
        if n.sum() <= 0:
            continue
        prof.append(n / n.sum()); keep.append(r)
    P = np.array(prof); mk = pd.DataFrame(keep)
    K = P.shape[1]

    SOLO = {"a": (DOUBLE_W, 2.55), "b": (3.30, 2.45), "c": (3.10, 2.45)}
    axA = axB = axC = None
    if panel is None:
        fig = plt.figure(figsize=(DOUBLE_W, 4.9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.00], width_ratios=[1.0, 0.85],
                              hspace=0.55, wspace=0.26)
        axA = fig.add_subplot(gs[0, :]); axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=SOLO[panel])
        ax = fig.add_subplot(111)
        axA, axB, axC = (ax if panel == k else None for k in ("a", "b", "c"))
    inset = None; focus_y = None; cent = {}

    if axA is not None:
     # ---- a: the Fig 2 overview panel itself ------------------------------
     # Built by calling the SAME functions Fig 2 calls, in the same order, from
     # dev/pcawg/figure2_signature/make_overview_trees.py:74-76 -- so geometry, colours, hatches
     # and row order are identical by construction rather than by imitation. The only edits
     # afterwards are (i) recolouring the focus state's blocks by multiplicity cluster and
     # (ii) boxing that row.
     segs_ord, offsets = compute_offsets(g.segments)
     axA.set_ylim(0, 1)
     plot_tau_stack(axA, segs_ord, offsets, ess_thresh=0, cluster_times=[],
                    segment_cluster_ids={}, color_by_cluster=False)
     fc_sel = _get_final_segment_clusters(cg)
     segment_cluster_plotting(axA, cg, segs_ord, offsets, final_clusters=fc_sel)
     inset = axA.child_axes[-1]          # the CN-state track segment_cluster_plotting just added

     # Reproduce the function's own iteration order so patch i <-> segment i.
     seg_by_cluster, states, seen, cl_index = {}, [], {}, {}
     for name, mseg in fc_sel.items():
         cnst = (int(mseg.major_cn), int(mseg.minor_cn))
         if cnst not in seen:
             seen[cnst] = 0; states.append(cnst)
         seen[cnst] += 1
         cl_index[name] = seen[cnst]
         for sid in getattr(mseg, "seg_ids", []):
             seg_by_cluster[sid] = name
     ordered = [sg for sg in segs_ord if sg.seg_id in seg_by_cluster]
     clab_of = {r.seg_id: int(r.cluster_label) for r in mk.itertuples()}

     focus_y = None
     for sg, patch in zip(ordered, inset.patches):
         if (int(sg.major_cn), int(sg.minor_cn)) != STATE:
             patch.set_alpha(0.30)
             continue
         lab = clab_of.get(sg.seg_id)
         patch.set_facecolor(COL.get(lab, GREY) if lab is not None else GREY)
         patch.set_alpha(1.0); patch.set_zorder(5)
         focus_y = patch.get_y() + patch.get_height() / 2

     if focus_y is not None:
         x0, x1 = inset.get_xlim()
         box = FancyBboxPatch((x0, focus_y - 0.5), x1 - x0, 1.0,
                              boxstyle="round,pad=0,rounding_size=0.12",
                              fill=False, edgecolor="#111", lw=0.9, zorder=8)
         box.set_clip_on(False); inset.add_patch(box)
         for t in inset.get_yticklabels():
             if t.get_text() == f"({STATE[0]},{STATE[1]})":
                 t.set_fontweight("bold"); t.set_color("#111")
             else:
                 t.set_color("#999")

     # At 89 mm the borrowed overview's own type is too large: shrink the CN-state row labels, the
     # chromosome ticks (odd numbers only) and the Cluster-hatch legend it adds to the figure.
     for t in inset.get_yticklabels():
         t.set_fontsize(FS_TICK)
     for t in axA.get_xticklabels():
         t.set_fontsize(FS_TICK)
     # plot_tau_stack draws chromosome names as ax.text artists and clears the xticks
     # (tau/plotting.py:234,246), so they must be thinned as Text, not as tick labels. Past chr12
     # the bands are too narrow at 89 mm for every label to fit.
     for t in axA.texts:
         txt = t.get_text()
         if txt.isdigit():
             # at 183 mm all 22 fit at 6 pt (they did not at 89 mm)
             t.set_fontsize(FS_TICK)
     for t in axA.get_yticklabels():
         t.set_fontsize(FS_TICK)
     axA.yaxis.label.set_size(FS_LABEL)
     for lg in list(fig.legends):
         for t in lg.get_texts():
             t.set_fontsize(FS_TICK)
         lg.set_bbox_to_anchor((0.5, 1.045)); lg._loc = 9
     # WGD / ccPG lines across the stack, exactly as Fig 2 draws them
     # (make_overview_trees.py:141 -- ls="--", lw=0.9, alpha=0.85, colour by classification)
     for r in events.itertuples():
         lbl, c = _event_style(r)
         axA.axhline(r.event_time_frac, color=c, ls="--", lw=0.9, alpha=0.85, zorder=6)
         axA.text(1.006, r.event_time_frac, f"{lbl} {r.event_time_frac:.2f}",
                  transform=axA.get_yaxis_transform(), fontsize=FS_ANNOT,
                  va="center", ha="left", color=c, clip_on=False)
     axA.set_title(f"a   {cfg['fig']} overview; ({STATE[0]},{STATE[1]}) row by cluster",
                   loc="left", fontsize=FS_TITLE, fontweight="bold", pad=26)

    if axB is not None:
     # ---- b: profiles -- raw multiplicity proportions ---------------------
     labs = mk.cluster_label.astype(int).values
     other = ~np.isin(labs, list(COL))
     cent = {}
     if K >= 3:
         # Plot the RAW multiplicity proportions, not principal components. For a major_cn=3 state
         # this IS the space DBSCAN clusters in -- _two_step_records uses block_pool[b][:major-1],
         # i.e. exactly (m1, m2) -- so the scatter is the feature space itself rather than a rotation
         # of it, the axes carry units a reader can interpret, and both are bounded to [0,1] so no
         # outlier can stretch them. (For major_cn>3 these are the first two of major-1 dimensions.)
         xy = P[:, :2]
         # Size by weighted mutation count (sum of N_counts, the quantity the code calls ESS).
         # The points at the far edges are not outliers in any meaningful sense -- they are
         # segments with almost no mutations (0.3-5 against a median of 16), so their
         # "profile" is one or two reads landing in a single bin. Sizing shows that rather than
         # letting them dominate axes that are already bounded to [0,1].
         ess = np.array([float(np.sum(byid[r.seg_id].N_counts)) for r in mk.itertuples()])
         sz = np.clip(ess * 0.40, 1.5, 30)
         axB.scatter(xy[other, 0], xy[other, 1], s=sz[other], c=GREY, lw=0, alpha=0.7,
                     label="other clusters", zorder=1)
         for lab in SHOW:
             sel = labs == lab
             if not sel.any():
                 continue
             axB.scatter(xy[sel, 0], xy[sel, 1], s=sz[sel], color=COL[lab], lw=0, alpha=0.9,
                         zorder=3, label=f"cluster {lab} (n={int(sel.sum())})")
             # count-weighted centroid, so the leader line points at the cluster's real mass
             w = ess[sel]
             cent[lab] = (xy[sel] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
         axB.set_xlabel("proportion at multiplicity 1")
         axB.set_ylabel("proportion at multiplicity 2")
     else:
         # one free coordinate: show it directly, jittered, sized by weighted mutation count
         rng = np.random.default_rng(0)
         x = P[:, 0]
         jit = rng.uniform(-0.35, 0.35, len(x))
         ess = np.array([float(np.sum(byid[r.seg_id].N_counts)) for r in mk.itertuples()])
         axB.scatter(x[other], jit[other], s=np.clip(ess[other] / 3, 5, 55), c=GREY, lw=0,
                     alpha=0.7, label="other clusters", zorder=1)
         for lab in SHOW:
             sel = labs == lab
             if not sel.any():
                 continue
             axB.scatter(x[sel], jit[sel], s=np.clip(ess[sel] / 3, 5, 55), color=COL[lab], lw=0,
                         alpha=0.9, zorder=3, label=f"cluster {lab} (n={int(sel.sum())})")
             cent[lab] = np.array([float(np.mean(x[sel])), float(np.mean(jit[sel]))])
         axB.set_xlabel("proportion of mutations at multiplicity 1")
         axB.set_yticks([]); axB.set_ylim(-0.62, 0.62)
         for sp_ in ("left",):
             axB.spines[sp_].set_visible(False)
     axB.set_title("b   multiplicity profiles", loc="left", fontsize=FS_TITLE, fontweight="bold")
     axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False,
                handletextpad=0.3, labelspacing=0.25, borderpad=0.0)
     for sp_ in ("top", "right"):
         axB.spines[sp_].set_visible(False)

    if axC is not None:
     # ---- c: pooled allele trees -----------------------------------------
     rt = json.load(gzip.open(R / "tau/data/route_topology.json.gz"))
     meta = {}
     for k, v in rt.items():
         meta.setdefault((v["major_cn"], v["minor_cn"]), {})[k] = v
     units = {s.seg_id: s for s in cg.segments}
     xoff, drawn = 0, []
     for lab in SHOW:
         seg = units.get(f"merged_{STATE[0]}_{STATE[1]}_{lab}")
         if seg is None:
             continue
         t = _build_tree_for_cluster(seg, meta)
         if t is None:
             continue
         plot_route(axC, t[0], t[1], x_offset=xoff, spacing=0.55,
                    major_color=COL[lab], minor_color=COL[lab], linewidth_scale=1.0)
         k = pick_best_key(seg)
         ev = np.median(np.cumsum(extract_mat(seg, k, 0), axis=1)[:, :-1], axis=0)
         n = np.asarray(seg.N_counts, float); n = n / n.sum()
         drawn.append((lab, xoff, n, ev)); xoff += 6.2
     # Time runs UPWARD here, matching panel a and plot_route's own convention
     # ("y=0 at bottom (early), y=1 at top (late) -- matches overview orientation", tau/trees.py).
     for lab, xo, n, ev in drawn:
         axC.text(xo + 1.0, 1.22, f"cluster {lab}", ha="center", fontsize=FS_LABEL,
                  fontweight="bold", color=COL[lab])
         # N split across two lines: as one line it is wider than the tree spacing and the
         # neighbouring clusters' labels run into each other.
         axC.text(xo + 1.0, -0.15, " ".join(f"{v:.2f}" for v in ev),
                  ha="center", fontsize=FS_ANNOT, color="#4b566d")
     # event lines span the panel; labels sit in a reserved right margin so they cannot land on a tree
     RMARGIN = 1.9
     for r in events.itertuples():
         lbl, c = _event_style(r)
         axC.axhline(r.event_time_frac, color=c, ls="--", lw=0.9, alpha=0.85, zorder=0)
         axC.text(xoff - 0.9 + RMARGIN, r.event_time_frac + 0.012,
                  f"{lbl} {r.event_time_frac:.2f}",
                  fontsize=FS_ANNOT, color=c, va="bottom", ha="right")
     axC.set_ylim(-0.26, 1.26); axC.set_xlim(-1.6, xoff - 0.9 + RMARGIN)   # left pad: gain labels are centred on the
     #                                            first tree and ran into the y-axis
     axC.set_ylabel("molecular time", fontsize=FS_LABEL); axC.set_xticks([]); axC.set_yticks([0, 0.5, 1])
     axC.set_title("c   pooled-segment trees", loc="left", fontsize=FS_TITLE, fontweight="bold")
     for sp_ in ("top", "right", "bottom"):
         axC.spines[sp_].set_visible(False)

    # Leader lines span panels, so only in the combined figure.
    # Anchor each leader line on a REAL block of its cluster -- the widest one -- not on the mean
    # genomic position. A cluster spread over chr7/10/11/17 has its mean near chr9, where it owns
    # nothing, so the arrow would point at empty track.
    for lab in (SHOW if panel is None else []):
        if lab not in cent:
            continue
        best = None
        for sid in mk[mk.cluster_label == lab].seg_id:
            if sid not in offsets:
                continue
            _c, se = sid.split(":"); a, b = se.split("-")
            w = int(b) - int(a)
            if best is None or w > best[0]:
                best = (w, offsets[sid] + (int(a) + int(b)) / 2)
        if best is None:
            continue
        con = ConnectionPatch(
            xyA=(best[1], (focus_y - 0.5) if focus_y is not None else 0.0),
            coordsA=inset.transData,
            xyB=(cent[lab][0], cent[lab][1]), coordsB=axB.transData,
            color=COL[lab], lw=0.6, ls=(0, (2, 2)), alpha=0.8, zorder=0)
        con.set_clip_on(False); axB.add_artist(con)

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        # svg.fonttype="none" is set in rcParams, so text stays editable in Illustrator.
        # NOTE: the CN-state track rectangles carry rasterized=True from segment_cluster_plotting
        # (its own comment: SVG cannot tile a hatch onto a sub-point-wide sliver and renders it
        # blank), so that strip is a raster layer inside the SVG by design.
        # manuscript names: a_overview / b_profiles / c_trees for the eso panels; combined figures
        # and the stomach variant are alternatives
        _PANEL_STEM = {"a": "a_overview", "b": "b_profiles", "c": "c_trees"}
        stem = (_PANEL_STEM[panel] if (panel and cfg["short"] == "eso")
                else f"alt_combined_{cfg['short']}" + (f"_panel_{panel}" if panel else ""))
        exts = ("pdf", "png", "svg") if panel is None else ("svg", "pdf")
        for ext in exts:
            fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
        print(f"wrote {OUT}/{stem}.{'/'.join(exts)}")
    return fig


def make(save=True):
    figs = [build(c, save=save) for c in CONFIGS]
    for c in CONFIGS:
        if c.get("separate_panels"):
            for pn in ("a", "b", "c"):
                figs.append(build(c, save=save, panel=pn))
    return figs


if __name__ == "__main__":
    make()
