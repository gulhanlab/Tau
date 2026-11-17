"""burst_characteristics figure (supp) — excess bursts split by how tightly the polytope constrains them.

Emits FOUR separate panel files, one per panel:

    determinacy_b_counts        how many excess bursts are well vs poorly constrained
    determinacy_c_enrichment    WGD enrichment per class, observed vs a WGD-shuffle null
    determinacy_d_tight         five worked examples, tightly bound   (tree over gain trajectory)
    determinacy_e_rest          five worked examples, poorly constrained

Panel letters START AT b: this figure is Supp. Fig. 12, whose panel a is a_burst_null.py (burst vs matched
null). Letters here must not restart at a or two different panels answer to "a".

TIED-BEST POOLING — the thing that changed
------------------------------------------
Tau emits every route surviving dist_cut = 0.20, and v1 of the timing cache dropped the fit-quality
columns. So the burst statistic (a MAX over solutions) and the grey fan both ranged over routes fitting
measurably worse than the segment's best — a (5,2) segment carries a median of 12 routes, typically
only a few tied at the minimum. Everything here now uses `best_route`: the routes at that minimum, the
ones the data cannot separate.

It matters. Excess bursts fall 5,848 -> 3,045, with 2,902 lost and 0 gained (structural: removing
solutions can only lower a maximum, so every burst dropped needed a worse-fitting route to exist). The
Bladder.TCC (5,3) chr5:12.5Mb example previously shown at "w0.68" is a SINGLE solution of width 0.000
once the 26 worse routes are excluded.

What survives untouched is the result itself: WGD enrichment 3.00 pool-all vs 3.16 tied-best. Discard
half the excess bursts and the effect does not move — that robustness is worth more than the count.

TWO CLASSES, not three
----------------------
The old three-way split is gone because under fair pooling it stops being a gradient: moderate (2.83)
and weak (2.93) sit on top of each other and in the wrong order. What holds is a dichotomy —

    tightly bound (width <= 0.10)   fold 3.47  [3.34, 3.59]   n=1,549
    rest          (width >  0.10)   fold 2.85  [2.74, 2.96]   n=1,496
    difference +0.61 [+0.44, +0.78], bootstrap p = 1e-4

Splitting "tight" again into determined (width exactly 0, n=736) and tightly bound was tested and is
deliberately NOT shown: determined comes out nominally LOWER (3.31 vs 3.60), a difference that does not
survive correcting for three pairwise tests. Reporting it would invite a story the data will not carry.

Enrichment being HIGHER where there is less freedom is the opposite of what an artefact produces —
indeterminacy adds timing noise, which can only dilute alignment with the WGD. So this answers the
"a burst is just a choice among solutions" objection rather than needing defence from it.

Excess is the published, WGD-agnostic definition (max_burst_size > D, D = floor(M/2)+floor(m/2),
computed without reference to any WGD), so the enrichment panel is not circular.

Examples are rendered with the shared FigHighCN helper, which now filters to tied-best routes too — so
the fan a reader sees is exactly the solution set the burst statistic searched.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "burst_gallery"))   # the shared trajectory helper
from _traj import (fetch, plot_traj, build_real_tree, draw_real_tree,
                            GAIN_DOT, NATREV, TXT, MM, WGD_COL)
from paper.figures._style import S, save as _save_panel, SINGLE, DOUBLE
from paper.figures._style import apply as _style_apply  # noqa: E402
_style_apply()

PANELS = panels_dir_of(__file__)
PANELS.mkdir(parents=True, exist_ok=True)
from paper.config import FROZEN
SRC = FROZEN / "burst_possible_necessary.tsv"     # paper/upstream/burst/burst_possible_necessary.py
NCOL = 5
WGD_WIN, N_PERM, N_BOOT = 0.10, 20000, 10000
CENTROID_LO, CENTROID_HI = 0.15, 0.85
MUT_FLOOR_Q = 0.60          # examples must be better-powered than most of the pool: at ~15 weighted
                            # mutations a wide fan reflects no data rather than real indeterminacy
PAGE_W, PAGE_H = 183, 170          # full text width; height leaves room for the caption.
                                   # 247 mm is the page maximum INCLUDING the legend, so artwork
                                   # at 247 leaves nowhere for it to go.
PAGE_NCOL, PAGE_NROW = 4, 4        # 16 per page; 4 columns give 25% more width per exemplar than
                                   # 5 did, which is what lets the labels reach the 5 pt floor
PAGES_PER_CLASS = 2                # -> 32 exemplars per class, 64 in total
PAGE_N = PAGE_NCOL * PAGE_NROW * PAGES_PER_CLASS
cls_n = {}                     # class -> total excess count, filled in make() for page headers
# TRUSTWORTHY CALLS ONLY, matching what panel a now reports. Both classes have a false-positive rate
# near zero against the matched null (0.5% and 0.6%); the permissive class — a burst under SOME fit but
# not all — has an 11.8% false-positive rate against a 16.3% signal and is deliberately not shown.
# THREE classes for the count/enrichment panels, TWO for the exemplar pages.
# The third — a burst under some fit but not all — is NOT a defensible burst call: its false-positive
# rate against the matched null (11.8%) is close to its own signal (16.3%). It is kept in panels b/c
# anyway because it makes a separate point: those segments are still enriched for WGD proximity. The
# burst CALL is unreliable while the burst POSITION is not, since the position is inherited from the
# segment's actual gain times. It gets no exemplar page — nothing there is safe to show as a burst.
CLASSES = ["determined", "every solution", "compatible"]
PAGE_CLASSES = ["determined", "every solution"]
LABEL = {"determined": "determined", "every solution": "burst under\nevery fit",
         "compatible": "compatible\nwith a burst"}
EDGE = {"determined": "a single solution — the timing is not a choice",
        "every solution": "several solutions, all of them bursty",
        "compatible": "bursty under some fits only"}
COLS = {"determined": NATREV["teal"][5], "every solution": NATREV["teal"][3],
        "compatible": NATREV["grey"][3]}
STEM = {"determined": "burst_d{i}_determined", "every solution": "burst_e{i}_everyfit"}
LETTER = {"determined": "d", "every solution": "e"}
TEAL, GREY = NATREV["teal"], NATREV["grey"]

# Explicit rather than _style.apply(): the example panels must match the main FigHighCN burst figure,
# which sets its own sizes. Vector-text settings are aligned with _style so the panels stay editable.
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7, "axes.linewidth": 0.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300,
    "savefig.transparent": True, "figure.facecolor": "none", "axes.facecolor": "none",
})
_style_apply()   # the shared ladder wins over the sizes in the block above


def _stats(x, wgd, pool, rng):
    """Per class: observed near-WGD fraction, its null, the fold, and a bootstrap CI on the fold.

    The CI is on the FOLD rather than the observed fraction because each class has its own null (its
    bursts sit at different molecular times, so a shuffled WGD lands in the window at different rates).
    Comparing raw fractions across classes would not be comparing like with like.
    """
    out = {}
    for c in CLASSES:
        s = x[(x["bin"] == c) & x.excess]
        b = s[s.centroid.notna() & s["sample"].isin(wgd)]
        cc = b.centroid.to_numpy(float)
        near = np.array([min(abs(cc[i] - t) for t in wgd[b["sample"].iloc[i]]) <= WGD_WIN
                         for i in range(len(b))])
        perm = np.array([(np.abs(cc - pool[rng.integers(0, len(pool), len(b))]) <= WGD_WIN).mean()
                         for _ in range(N_PERM)])
        null = perm.mean()
        idx = rng.integers(0, len(b), (N_BOOT, len(b)))
        fb = near[idx].mean(axis=1) / null
        out[c] = dict(n=len(s), n_wgd=len(b), obs=near.mean(), null=null, sd=perm.std(),
                      fold=near.mean() / null, boot=fb,
                      ci=tuple(np.percentile(fb, [2.5, 97.5])),
                      n_exact=int((s.width_max <= 1e-9).sum()),
                      p=(np.sum(perm >= near.mean()) + 1) / (N_PERM + 1))
    d = out["determined"]["boot"] - out["every solution"]["boot"]
    out["_diff"] = dict(mean=d.mean(), ci=tuple(np.percentile(d, [2.5, 97.5])),
                        p=max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / N_BOOT))
    return out


def _examples(x, n_per_class=NCOL):
    """Pick `n_per_class` exemplars per class, spread over tumour types and CN states.

    Three passes, loosening as it goes: first demand a new (tumour type, CN state), then a new tumour
    type, then take anything. Ordering within a pass is by mutation count, so the best-powered
    segments are preferred — a wide fan on 15 mutations reflects no data rather than real
    indeterminacy. `taken` is shared across classes so no segment appears on two pages.
    """
    floor = x.loc[x.excess, "mutation_count"].quantile(MUT_FLOOR_Q)
    picks, taken = {}, set()
    for c in PAGE_CLASSES:
        q = x[(x["bin"] == c) & x.excess & (x.mutation_count >= floor)
              & (x.centroid > CENTROID_LO) & (x.centroid < CENTROID_HI)]
        q = q.sort_values("mutation_count", ascending=False)
        row, seen_t, seen_ts = [], set(), set()
        for pass_ in range(3):
            for r in q.itertuples():
                if len(row) >= n_per_class:
                    break
                st = (r.major_cn, r.minor_cn)
                if pass_ == 0 and (r.tumor_type, st) in seen_ts:
                    continue
                if pass_ == 1 and r.tumor_type in seen_t:
                    continue
                if (r.sample, r.chrom, r.start) in taken:
                    continue
                # a tree that cannot be built (bursty draw conflicting with the route's event order)
                # is SKIPPED here rather than leaving a hole in the figure
                try:
                    build_real_tree(str(r.sample)[:8], str(r.chrom), int(r.start), bursty=True)
                except Exception:
                    continue
                row.append(r); taken.add((r.sample, r.chrom, r.start))
                seen_t.add(r.tumor_type); seen_ts.add((r.tumor_type, st))
        picks[c] = row
    return picks


def _draw_example(fig, gs_tree, gs_traj, p):
    """One exemplar: allele tree above its gain trajectory, x-axes aligned, WGD drawn through both."""
    prefix = str(p.sample)[:8]
    s = fetch(prefix, str(p.chrom), int(p.start))
    maj, mn, M_, m_, times = build_real_tree(prefix, str(p.chrom), int(p.start), bursty=True)
    axg = fig.add_subplot(gs_traj)
    plot_traj(axg, s["draws"], s["ne"], s["wgd"], show_median=False, show_wgd=False,
              wgd_label=False, yticklabels=True, highlight=False, ms=0, tick_fs=5.0, scale=0.85)
    dots = np.sort(times); yy = np.arange(1, s["ne"] + 1)
    axg.plot(dots, yy, color=TXT, lw=0.8, zorder=5)
    axg.scatter(dots, yy, **GAIN_DOT)
    axg.set_xlim(-0.02, 1.02); axg.set_xticks([0, 0.5, 1.0])
    axg.set_xticklabels(["0", "0.5", "1"], fontsize=S.tick)
    axt = fig.add_subplot(gs_tree)
    mspan = (0.99, 0.45) if s["major"] >= 6 else (0.94, 0.55)
    draw_real_tree(axt, maj, mn, None, show_minor=(m_ > 0), gain_marker="none",
                   show_allele_labels=False, maj_span=mspan)
    axt.set_xlim(-0.02, 1.02)
    if s["wgd"] is not None:
        axg.add_artist(ConnectionPatch(
            xyA=(s["wgd"], 0.4), coordsA=axg.transData, xyB=(s["wgd"], 1.0),
            coordsB=axt.transData, color=WGD_COL, lw=0.9, ls=(0, (4, 2)), zorder=2))
    note = " · no WGD" if s["wgd"] is None else ""
    axt.text(0.5, 1.16,
             f"{s['tumor_type']}{note}\n({s['major']},{s['minor']})  "
             f"{int(p.max_burst_size)}-burst  w{p.width_max:.2f}\n"
             f"chr{p.chrom}:{p.start/1e6:.1f}Mb",
             transform=axt.transAxes, ha="center", va="bottom", fontsize=S.tiny,
             color=GREY[4], linespacing=1.2)
    return axg


def page_examples(cls, rows, stem, letter, ncol=PAGE_NCOL, save=True):
    """A FULL PAGE of exemplars for one class — the expanded d / e pages.

    Same rendering as the 5-example panel (shared helper, so it cannot drift from the main burst
    figure); only the grid and type sizes change. Page is 183 x 247 mm — full text width rather than
    an 89 mm column, because each exemplar is a tree stacked on a trajectory and five of those across
    an 89 mm strip would be unreadable.
    """
    nrow = int(np.ceil(len(rows) / ncol))
    fig = plt.figure(figsize=(PAGE_W * MM, PAGE_H * MM))
    # no page header (Nature style — panel letter and description live in the caption), so the
    # grid can run higher; row-1 exemplar titles still need their three lines of clearance
    outer = fig.add_gridspec(nrow, 1, left=0.055, right=0.99, top=0.955, bottom=0.035,
                             hspace=0.72)
    n_ok = 0
    for ri in range(nrow):
        gs = outer[ri].subgridspec(2, ncol, height_ratios=[0.72, 1.5], hspace=0.46, wspace=0.42)
        for ci in range(ncol):
            k = ri * ncol + ci
            if k >= len(rows):
                break
            p = rows[k]
            try:
                axg = _draw_example(fig, gs[0, ci], gs[1, ci], p)
            except Exception as e:
                print(f"  skip {str(p.sample)[:8]} chr{p.chrom}:{p.start} ({e})"); continue
            if ci == 0:
                axg.set_ylabel("gain index", fontsize=S.label, color=TXT, labelpad=2)
            n_ok += 1
    fig.text(0.53, 0.008, "molecular time", ha="center", fontsize=S.annot, color=TXT)
    if save:
        _save_panel(fig, PANELS, stem)
    print(f"{stem}: {n_ok} exemplars ({nrow}x{ncol})")
    plt.close(fig)
    return n_ok


def _ticks():
    """Short names only. EDGE holds the explanatory clause, which is caption material — at three
    classes across an 89 mm panel a full sentence per tick is unreadable."""
    return [LABEL[c] for c in CLASSES]


def panel_counts(st, save=True, ax=None, fs=1.0):
    """ax: draw into a caller's axes (page composition). fs scales type for the denser page layout."""
    if ax is None:
        fig = plt.figure(figsize=(SINGLE, 52 * MM))
        ax = fig.add_subplot(111)
    else:
        fig, save = ax.figure, False
    n = [st[c]["n"] for c in CLASSES]
    ax.bar(range(len(CLASSES)), n, color=[COLS[c] for c in CLASSES], width=0.6, zorder=3)
    for i, v in enumerate(n):
        ax.text(i, v + max(n) * 0.02, f"{v:,}", ha="center", va="bottom", fontsize=S.annot * fs)
    # the fully determined bursts are a real subgroup, reported here rather than as a third class:
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(_ticks(), fontsize=S.tick * fs)
    ax.set_ylabel("excess bursts", fontsize=S.label * fs); ax.set_ylim(0, max(n) * 1.16)
    ax.tick_params(axis="y", labelsize=S.tick * fs)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    if save:
        fig.subplots_adjust(left=0.165, right=0.985, top=0.965, bottom=0.20)
    if save:
        _save_panel(fig, PANELS, "alt_excess_counts")
    return fig


def panel_enrichment(st, save=True, bx=None, fs=1.0):
    """bx: draw into a caller's axes (page composition). fs scales type for the denser page layout."""
    if bx is None:
        fig = plt.figure(figsize=(SINGLE, 52 * MM))
        bx = fig.add_subplot(111)
    else:
        fig, save = bx.figure, False
    w = 0.30
    for i, c in enumerate(CLASSES):
        bx.bar(i - w / 2, 100 * st[c]["obs"], w, color=COLS[c], zorder=3,
               label="observed" if i == 0 else None)
        bx.bar(i + w / 2, 100 * st[c]["null"], w, color="0.82", zorder=3,
               label="WGD-shuffle null" if i == 0 else None)
        bx.errorbar(i + w / 2, 100 * st[c]["null"], yerr=100 * st[c]["sd"], color="0.45",
                    lw=0.7, capsize=1.5, zorder=4)
        lo, hi = st[c]["ci"]
        bx.text(i, 100 * st[c]["obs"] + 3.0, f"{st[c]['fold']:.2f}×", ha="center", va="bottom",
                fontsize=S.annot * fs, fontweight="bold", color=COLS[c])
        bx.text(i, 100 * st[c]["obs"] + 9.0, f"[{lo:.2f}–{hi:.2f}]", ha="center", va="bottom",
                fontsize=S.tiny * fs, color=COLS[c])   # 5 pt floor: was 4.4, sub-floor at any fs <= 1
    bx.set_xticks(range(len(CLASSES))); bx.set_xticklabels(_ticks(), fontsize=S.tick * fs)
    bx.set_ylabel("% of bursts within\n0.10 of a WGD", fontsize=S.label * fs); bx.set_ylim(0, 108)
    bx.tick_params(axis="y", labelsize=S.tick * fs)
    bx.legend(frameon=False, fontsize=S.legend * fs, loc="upper center", handlelength=0.8,
              bbox_to_anchor=(0.5, 1.02), ncol=2, columnspacing=1.0)
    # The old title claimed enrichment was HIGHER where bounds are tight, with p<1e-3. That was true
    # of the previous determined-vs-permissive split; between the two trustworthy classes the
    # difference is Δ+0.16 [-0.12,+0.44], p=0.26 — not a difference. The finding now is that all three
    # classes are enriched, INCLUDING the one whose burst call is not defensible: the call is
    # unreliable there, but the burst's position is inherited from real gain times and is not.
    for s_ in ("top", "right"):
        bx.spines[s_].set_visible(False)
    if save:
        fig.subplots_adjust(left=0.165, right=0.985, top=0.965, bottom=0.20)
    if save:
        _save_panel(fig, PANELS, "alt_excess_enrichment")
    return fig


def panel_examples(cls, row, save=True):
    """One determinacy class: NCOL segments, allele tree above its gain trajectory."""
    fig = plt.figure(figsize=(DOUBLE, 62 * MM))
    gs = fig.add_gridspec(2, NCOL, height_ratios=[0.78, 1.5], hspace=0.50, wspace=0.42,
                          left=0.055, right=0.99, top=0.775, bottom=0.155)
    fig.text(0.005, 0.985, f"{LETTER[cls]}   {LABEL[cls]}  ({EDGE[cls]})", ha="left", va="top",
             fontsize=S.label, color=COLS[cls], fontweight="bold")
    n_ok = 0
    for ci, p in enumerate(row):
        prefix = str(p.sample)[:8]
        try:
            s = fetch(prefix, str(p.chrom), int(p.start))
            maj, mn, M_, m_, times = build_real_tree(prefix, str(p.chrom), int(p.start), bursty=True)
        except Exception as e:
            print(f"  skip {prefix} chr{p.chrom}:{p.start} ({e})"); continue
        axg = fig.add_subplot(gs[1, ci])
        plot_traj(axg, s["draws"], s["ne"], s["wgd"], show_median=False, show_wgd=False,
                  wgd_label=False, yticklabels=True, highlight=False, ms=0, tick_fs=5, scale=0.85)
        dots = np.sort(times); yy = np.arange(1, s["ne"] + 1)
        axg.plot(dots, yy, color=TXT, lw=0.9, zorder=5)
        axg.scatter(dots, yy, **GAIN_DOT)
        axg.set_xlim(-0.02, 1.02); axg.set_xticks([0, 0.5, 1.0])
        axg.set_xticklabels(["0", "0.5", "1"], fontsize=S.tick)
        axt = fig.add_subplot(gs[0, ci])
        mspan = (0.99, 0.45) if s["major"] >= 6 else (0.94, 0.55)
        draw_real_tree(axt, maj, mn, None, show_minor=(m_ > 0), gain_marker="none",
                       show_allele_labels=False, maj_span=mspan)
        axt.set_xlim(-0.02, 1.02)
        if s["wgd"] is not None:
            axg.add_artist(ConnectionPatch(
                xyA=(s["wgd"], 0.4), coordsA=axg.transData, xyB=(s["wgd"], 1.0),
                coordsB=axt.transData, color=WGD_COL, lw=1.0, ls=(0, (4, 2)), zorder=2))
        note = " · no WGD" if s["wgd"] is None else ""
        # each panel states its own width: the class is a range, and the whole figure turns on the
        # fan matching the number, so the reader can check that pairing panel by panel
        axt.text(0.5, 1.15,
                 f"{s['tumor_type']}{note}\n({s['major']},{s['minor']})  "
                 f"{int(p.max_burst_size)}-burst  w{p.width_max:.2f}\n"
                 f"chr{p.chrom}:{p.start/1e6:.1f}Mb",
                 transform=axt.transAxes, ha="center", va="bottom", fontsize=S.tiny,
                 color=GREY[4], linespacing=1.15)
        if ci == 0:
            axg.set_ylabel("gain index", fontsize=S.label, color=TXT, labelpad=2)
        n_ok += 1
    fig.text(0.53, 0.025, "molecular time", ha="center", fontsize=S.label, color=TXT)
    if save:
        _save_panel(fig, PANELS, STEM[cls])
    return fig, n_ok


def make(save=False, pages=False):
    """Compute the stats and print them; write nothing unless asked.

    This module is now used as a BUILDER — `burst_abc_solo.py` calls `_stats`, `panel_counts` and
    `panel_enrichment` and draws into its own axes. The files this used to emit (`burst_b_counts`,
    `burst_c_enrichment`, and the four `burst_d*/burst_e*` exemplar pages) are superseded by
    `burst_solo_b/c` and by the single-page `burst_examples_wgd_split`, so they default to OFF —
    otherwise running this module would put stale duplicates back into panels/.

    save=True restores the standalone b/c files; pages=True restores the four exemplar pages.
    """
    x = pd.read_csv(SRC, sep="\t", dtype={"chrom": str})
    # determined: one solution, so possible == necessary. underdetermined: keep only those whose burst
    # survives EVERY solution. Segments that burst under some fit but not all are dropped entirely.
    x["bin"] = np.where(x.determined & x.possible, "determined",
                        np.where(~x.determined & x.necessary, "every solution",
                                 np.where(~x.determined & x.possible, "compatible", "excluded")))
    x["excess"] = x["bin"] != "excluded"
    print(f"determined {int((x['bin']=='determined').sum()):,} | "
          f"every-solution {int((x['bin']=='every solution').sum()):,} | "
          f"dropped (some-fit-only) {int((x['bin']=='excluded').sum() - (~x.possible).sum()):,}")
    ev = pd.read_csv(R / "paper/data_tables/tau_events.tsv", sep="\t")
    wgd = ev[ev.classification == "WGD"].groupby("sample").event_time_frac.apply(list).to_dict()
    pool = np.array([t for v in wgd.values() for t in v], float)
    st = _stats(x, wgd, pool, np.random.default_rng(0))
    for c in CLASSES:
        cls_n[c] = st[c]["n"]
    ex_big = _examples(x, PAGE_N)          # full pages (PAGE_CLASSES only)
    ex = {c: v[:NCOL] for c, v in ex_big.items()}

    # PAGES ONLY. The individual b/c/d/e panel files were superseded by the multi-page layout, so
    # emitting them would leave stale duplicates in panels/ that disagree with the pages after an edit.
    figs = {"alt_excess_counts": panel_counts(st, save),
            "alt_excess_enrichment": panel_enrichment(st, save)}
    for c in CLASSES:
        lo, hi = st[c]["ci"]
        print(f"{LABEL[c]:<20} n={st[c]['n']:<6} obs={st[c]['obs']:.3f} null={st[c]['null']:.3f} "
              f"fold={st[c]['fold']:.2f} [{lo:.2f},{hi:.2f}] p={st[c]['p']:.1e}")
    d = st["_diff"]
    print(f"tight - rest: {d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] p={d['p']:.4f}")
    # RETIRED: four full pages of exemplars, superseded by burst_examples_wgd_split (one page, split
    # by whether the burst matches a WGD). Kept behind a flag rather than deleted — the layout code is
    # the only thing that can rebuild them if the single-page version turns out not to be enough.
    if pages:
        per = PAGE_NCOL * PAGE_NROW
        for c in PAGE_CLASSES:
            for i in range(PAGES_PER_CLASS):
                chunk = ex_big[c][i * per:(i + 1) * per]
                if not chunk:
                    continue
                page_examples(c, chunk, STEM[c].format(i=i + 1), LETTER[c], save=True)
    print(f"wrote {'standalone b/c' if save else 'no files'}"
          f"{' + 4 exemplar pages' if pages else ''} to {PANELS}")
    return figs


if __name__ == "__main__":
    make()
