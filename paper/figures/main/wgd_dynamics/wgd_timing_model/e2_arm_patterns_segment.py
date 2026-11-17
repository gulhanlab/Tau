"""Fig 4e2 — chromosome-ARM programmes, segment-length-weighted variant of Joseph's
`paper/figures/main/punctuated/panel_scripts/e_arm_patterns.py`, run once per event class: PGD, WGD, and
"neither" (the `chrom_specific` category — single-chromosome gains that are neither a WGD nor a PGD).
Those three `tau_events.classification` values partition ALL clustered events, so the three figures
together cover the whole cohort exactly once.

That original panel reads a per-EVENT arm-membership cache: for each event, an arm is either "reported"
or not (binary), and dot size = % of a type's events reporting the arm. That collapses away how MUCH of
the arm was actually gained — an event where CNV calling oversegmented one true amplification into five
adjacent tiny segments on an arm looks identical to an event that gained the whole arm in one segment,
since both just set membership[arm] = True once. This matters most for the `chrom_specific` class: those
events can carry up to ~160 member segments on a SINGLE chromosome (vs. exactly 1 for a true single-copy
gain), i.e. exactly the oversegmentation the length-weighting is meant to correct for.

This variant works at the segment level instead: for each qualifying event, sum the physical bp length of
its member segments (`tau_segments`, joined on `event_id`) that fall on each arm, and normalise by that
arm's hg19 length. A segment straddling the centromere is CLIPPED to each side by actual overlap (not
counted as full-length on both), so fragmentation never inflates coverage past the arm's true size. Dot
size is then the mean fraction of the arm covered per qualifying event of that type (0-100%), and dot
colour is a coverage-weighted mean event time (events that barely clip an arm barely move its timing
estimate), instead of an unweighted mean over events.

Needs no Tier-2 cache: built entirely from the Tier-1 tables (`tau_segments`, `tau_events`) via
`paper.data_tables.tables`, plus the packaged hg19 centromere/chrom-size reference used by
`tau.clustering.assign_chrom_arm`. Reads from the Tau_github repo READ-ONLY; writes only under this
folder (`./output/`), never back into Joseph's `paper/figures/main/punctuated/`.

Run:
    PATH=".pixi/envs/default/bin:$PATH" .pixi/envs/default/bin/python e2_arm_patterns_segment.py
(or plain `python e2_arm_patterns_segment.py` if pandas/matplotlib are already on PATH)
Writes e2_arm_patterns_segment_{pgd,wgd,neither}.{png,svg} to ./output/.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TAU_REPO = Path(__file__).resolve().parents[5]   # repo root (wgd_timing_model/wgd_dynamics/main/figures/paper/<repo>)
sys.path.insert(0, str(TAU_REPO))
from paper.data_tables.tables import load_events, load_segments, load_samples  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MM = 1 / 25.4
TXT = "#273246"
mpl_rc = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "font.size": 6.0, "axes.titlesize": 7.0, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
    "figure.titlesize": 7.0, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "patch.linewidth": 0.4,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 300, "savefig.dpi": 450,
}
matplotlib.rcParams.update(mpl_rc)

MIN_CELL = 2     # >= this many nonzero-coverage events to colour a (type, arm) dot by timing
AUTOSOMES = [str(i) for i in range(1, 23)]
ARMS = [f"{c}{a}" for c in range(1, 23) for a in "pq"]
EXCLUDE_ARMS = {"15p", "21p", "22p"}   # acrocentric short arms with negligible true content — drop from all figures
TCMAP = plt.get_cmap("plasma")   # same low=dark-blue/purple -> high=yellow direction as Joseph's viridis
                                  # (plasma's natural direction already satisfies this, no "_r" reversal needed)
LAB = {"CNS.GBM": "CNS GBM", "CNS.Medullo": "Medulloblastoma", "Liver.HCC": "Liver",
       "Kidney.RCC": "Kidney", "Panc.AdenoCA": "Pancreatic", "Panc.Endocrine": "Panc. endocrine",
       "Breast.AdenoCA": "Breast", "Lymph.BNHL": "Lymphoma B-NHL", "Head.SCC": "Head/neck SCC",
       "Stomach.AdenoCA": "Stomach", "ColoRect.AdenoCA": "Colorectal", "Eso.AdenoCA": "Esophageal"}


def _reps():
    """Representative-aliquot samples, or None if the column is unpopulated (no dedup possible).
    Mirrors `paper/figures/main/punctuated/panel_scripts/_wgdpgd_common.py::_reps`."""
    s = load_samples()
    if "is_representative_aliquot" not in s.columns:
        return None
    m = s["is_representative_aliquot"]
    if m.notna().sum() == 0:
        warnings.warn("tau_samples.is_representative_aliquot is all-NA; multi-aliquot donor dedup "
                      "skipped (using the full cohort).", stacklevel=2)
        return None
    return set(s.loc[m.fillna(False).astype(bool), "sample"])


def _events_deduped():
    ev = load_events().rename(columns={"event_time_frac": "time", "genome_frac": "gf"})
    ev = ev.astype({"sample": str, "tumor_type": str, "classification": str})
    reps = _reps()
    if reps is not None:
        ev = ev[ev["sample"].isin(reps)]
    return ev.reset_index(drop=True)


def _sample_counts():
    """Total COHORT size per tumor type (all samples, whether or not they have a qualifying event) —
    the denominator for a per-sample effective frequency. Same donor-dedup as `_events_deduped`."""
    s = load_samples().astype({"sample": str, "tumor_type": str})
    reps = _reps()
    if reps is not None:
        s = s[s["sample"].isin(reps)]
    return s["tumor_type"].value_counts()


# Dropped from every figure's row set — mostly ultra-rare/benign/non-carcinoma subtypes whose n=1-3
# cohorts produced noisy, uninformative rows (huge dots from a single sample, spurious extreme Z-scores).
EXCLUDE_TYPES = {
    "Unknown", "Breast.DCIS", "CNS.Oligo", "Cervix.AdenoCA", "Breast.LobularCA", "Bone.Benign",
    "Bone.Epith", "CNS.PiloAstro", "Myeloid.AML", "Myeloid.MDS", "Myeloid.MPN", "Thy.AdenoCA",
}


# Additionally dropped from WGD-based figures ONLY (gain WGD variants, and every loss figure — loss
# timing is defined purely in terms of WGD samples): too few WGD events/samples to show a meaningful
# WGD-conditioned row, even though these types have plenty of PGD/chrom-specific data and so still get a
# row in those figures. Lymph.CLL has 0 WGD events; Prost.AdenoCA has only 4 (out of 211 samples).
WGD_SPARSE_TYPES = {"Lymph.CLL", "Prost.AdenoCA"}


def _all_types(exclude=None):
    """Every tumor type in the deduped cohort except `EXCLUDE_TYPES` (and, if passed, `exclude` on top —
    e.g. `WGD_SPARSE_TYPES` for WGD/loss figures), alphabetically sorted by its DISPLAY label. This is the
    fixed row set every arm-pattern figure (gain PGD/WGD/chrom_specific, and the loss figures) should use,
    so a type with few qualifying events/samples (e.g. CNS.GBM) still gets a row — possibly an empty one —
    instead of silently vanishing, and every figure lists the exact same tumor types in the exact same
    order for direct comparison (modulo the extra `exclude` some figures apply)."""
    drop = EXCLUDE_TYPES | set(exclude or ())
    cohort = _sample_counts()
    return sorted([t for t in cohort.index if t not in drop], key=lambda t: LAB.get(t, t).lower())


def _centromeres():
    # Read the packaged hg19 reference file directly rather than `import tau` — the `tau` package
    # __init__ pulls in `tau.timing`, which requires a SageMath/Singular install we don't have here.
    df = pd.read_csv(TAU_REPO / "tau/data/hg19.centromeres.tsv", sep="\t")
    df["chrom"] = df["chrom"].astype(str)
    return df.set_index("chrom")


def _chrom_sizes():
    df = pd.read_csv(TAU_REPO / "tau/data/hg19.chrom.sizes", sep="\t", header=None, names=["chrom", "length"])
    df["chrom"] = df["chrom"].astype(str)
    return df.set_index("chrom")["length"]


def _arm_lengths(cen, sizes):
    """hg19 p/q arm lengths in bp (p arm := [0, p_end), q arm := [q_start, chrom_end))."""
    out = {}
    for c in AUTOSOMES:
        out[f"{c}p"] = int(cen.loc[c, "p_end"])
        out[f"{c}q"] = int(sizes[c] - cen.loc[c, "q_start"])
    return out


def _arm_timed_bp():
    """Per (sample, arm): total bp of `is_timed` segments overlapping that arm, REGARDLESS of which
    event (if any) they were clustered into. This is the "how much of this arm could we actually assess"
    denominator — smaller, lower-mutation-burden chromosomes (17, 19, 20, 21, 22) have systematically
    less of their length successfully timed (SNV count scales with bp, so less physical sequence means
    less power to time a segment), so dividing by the fixed theoretical arm length instead of this
    systematically penalises small chromosomes with a technical, not biological, deficit. Used in place
    of `_arm_lengths()` for the `weighted=True` fraction."""
    seg = load_segments()
    seg = seg[seg["chrom"].isin(AUTOSOMES) & seg["is_timed"].fillna(False)].copy()
    reps = _reps()
    if reps is not None:
        seg = seg[seg["sample"].isin(reps)]

    cen = _centromeres()
    p_end = seg["chrom"].map(cen["p_end"]).astype(float)
    q_start = seg["chrom"].map(cen["q_start"]).astype(float)
    p_ov = (np.minimum(seg["end_bp"], p_end) - seg["start_bp"]).clip(lower=0)
    q_ov = (seg["end_bp"] - np.maximum(seg["start_bp"], q_start)).clip(lower=0)

    parts = []
    pm = p_ov > 0
    if pm.any():
        parts.append(pd.DataFrame({"sample": seg["sample"][pm], "arm": seg["chrom"][pm] + "p", "bp": p_ov[pm]}))
    qm = q_ov > 0
    if qm.any():
        parts.append(pd.DataFrame({"sample": seg["sample"][qm], "arm": seg["chrom"][qm] + "q", "bp": q_ov[qm]}))
    tb = pd.concat(parts, ignore_index=True)
    return tb.groupby(["sample", "arm"], as_index=False)["bp"].sum().rename(columns={"bp": "timed_bp"})


def _arm_coverage(classification, weighted=True):
    """One row per (sample, event_id, arm) covered by a qualifying event's member segments, with `bp`
    covering length and `frac` the per-event arm contribution. Joined with tumor_type + event time.

    `classification` selects which `tau_events.classification` rows qualify — either one of the three
    exact categories {"WGD", "PGD", "chrom_specific"}, or a set/list of them (e.g. the complement of
    {"WGD", "PGD"} for the "neither" figure).

    `weighted=True` (default): `frac` = bp / arm_length (capped at 1.0) — the segment-length-weighted
    metric this script was built around. `weighted=False` reproduces Joseph's ORIGINAL
    `e_arm_patterns.py` methodology instead: `frac` = 1.0 whenever the event touches the arm AT ALL
    (binary presence, regardless of how much of the arm or how many member segments), so multiple
    small oversegmented calls on one arm count exactly the same as one call spanning the whole arm."""
    want = {classification} if isinstance(classification, str) else set(classification)
    ev = _events_deduped()
    ev = ev[ev.classification.isin(want)].copy()
    ev["time"] = ev["time"].astype(float)

    seg = load_segments()
    reps = set(ev["sample"].unique()) if _reps() is None else _reps()
    seg = seg[seg["sample"].isin(reps) & seg["chrom"].isin(AUTOSOMES) & seg["event_id"].notna()].copy()
    seg["event_id"] = seg["event_id"].astype("int64")
    seg = seg.merge(ev[["sample", "event_id"]].drop_duplicates(), on=["sample", "event_id"], how="inner")

    cen = _centromeres()
    p_end = seg["chrom"].map(cen["p_end"]).astype(float)
    q_start = seg["chrom"].map(cen["q_start"]).astype(float)
    p_ov = (np.minimum(seg["end_bp"], p_end) - seg["start_bp"]).clip(lower=0)
    q_ov = (seg["end_bp"] - np.maximum(seg["start_bp"], q_start)).clip(lower=0)

    parts = []
    pm = p_ov > 0
    if pm.any():
        parts.append(pd.DataFrame({"sample": seg["sample"][pm], "event_id": seg["event_id"][pm],
                                    "arm": seg["chrom"][pm] + "p", "bp": p_ov[pm]}))
    qm = q_ov > 0
    if qm.any():
        parts.append(pd.DataFrame({"sample": seg["sample"][qm], "event_id": seg["event_id"][qm],
                                    "arm": seg["chrom"][qm] + "q", "bp": q_ov[qm]}))
    cov = pd.concat(parts, ignore_index=True)
    cov = cov.groupby(["sample", "event_id", "arm"], as_index=False)["bp"].sum()

    if weighted:
        # Denominator is the bp actually TIMED on this arm for this sample, not the fixed theoretical arm
        # length — see `_arm_timed_bp()`. Using the fixed length would systematically under-count small,
        # low-ESS chromosomes (less timing power there is a technical deficit, not a biological one).
        cov = cov.merge(_arm_timed_bp(), on=["sample", "arm"], how="left")
        cov["frac"] = (cov["bp"] / cov["timed_bp"]).clip(upper=1.0)
    else:
        cov["frac"] = 1.0   # binary presence: this event touched the arm, full stop

    cov = cov.merge(ev[["sample", "event_id", "tumor_type", "time"]], on=["sample", "event_id"], how="left")
    return cov, ev


def _round_nice(x):
    if x < 1:
        return round(x, 2)
    if x < 10:
        return round(x, 1)
    return round(x)


def _bubble_plot(eff, color_val, arms, row_labels, title, stem, color_by="timing",
                 size_label="effective frequency / sample", save=True,
                 diverging_label="row Z-score (relative frequency)",
                 diverging_words=("depleted", "enriched")):
    """Shared bubble-chart renderer (used by both the gain and loss arm-pattern scripts): dot size comes
    from `eff` (a [types x arms] matrix), dot colour from `color_val` of the same shape — either a 0-1
    "timing" value (plasma) or a signed, symmetric-about-0 diverging value (RdBu_r; `color_by` in
    {"zscore", "diverging"} are the same rendering, just with a customisable colourbar label/endpoint
    words via `diverging_label`/`diverging_words`, since not every diverging metric is a row Z-score —
    e.g. the WGD1-vs-WGD2 loss-timing ratio is diverging but not a Z-score).
    `row_labels` are the pre-formatted y-tick strings (tumour type + whatever counts matter)."""
    n_types = eff.shape[0]
    # Row height is fixed (mm/row); top/bottom margins (title+size-legend, xlabel+rotated arm labels) are
    # fixed in mm too, so figures with very different row counts (9 vs all 37 cohort types) stay legible
    # instead of squashing more rows into the same fixed canvas.
    top_mm, bottom_mm, row_mm = 20.0, 28.0, 4.2
    height_mm = top_mm + bottom_mm + row_mm * max(n_types, 1)
    fig, ax = plt.subplots(figsize=(150 * MM, height_mm * MM))
    fig.subplots_adjust(left=0.17, right=0.87, top=1 - top_mm / height_mm, bottom=bottom_mm / height_mm)

    mx = float(eff.max()) if eff.max() > 0 else 1.0   # scales dot size + the size-legend ticks below
    xx, yy = np.meshgrid(np.arange(len(arms)), np.arange(n_types))
    m = eff > 0
    sizes = (eff[m] / mx) * 45 + 2
    xm, ym, cv = xx[m], yy[m], color_val[m]
    nan = np.isnan(cv)
    ax.scatter(xm[nan], ym[nan], s=sizes[nan], c="#bbbbbb", edgecolor="#3a3a3a", lw=0.6, zorder=2)
    diverging = color_by in ("zscore", "diverging")
    if diverging:
        zmax = _round_nice(np.nanmax(np.abs(cv))) or 1.0
        cmap, vmin, vmax = plt.get_cmap("RdBu_r"), -zmax, zmax
    else:
        cmap, vmin, vmax = TCMAP, 0, 1
    sc = ax.scatter(xm[~nan], ym[~nan], s=sizes[~nan], c=cv[~nan], cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolor="#3a3a3a", lw=0.6, zorder=3)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, fontsize=5, rotation=90)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(row_labels, fontsize=6, color=TXT)
    ax.set_xlabel("Chromosome arm", fontsize=7, color=TXT)
    ax.set_title(title, fontsize=7, color=TXT, loc="left", pad=28)
    ax.set_xlim(-0.6, len(arms) - 0.4); ax.set_ylim(-0.6, n_types - 0.4)
    ax.invert_yaxis(); ax.tick_params(length=0)
    ax.set_axisbelow(True); ax.grid(True, color="#eeeeee", lw=0.3)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(sc, ax=ax, fraction=0.022, pad=0.01)
    cb.ax.tick_params(labelsize=6, length=0); cb.outline.set_visible(False)
    if diverging:
        cb.set_label(diverging_label, fontsize=6, color=TXT)
        cb.set_ticks([-zmax, 0, zmax])
        lo, hi = diverging_words
        cb.set_ticklabels([f"-{zmax:g} {lo}", "0", f"+{zmax:g} {hi}"])
    else:
        cb.set_label("coverage-weighted mean timing", fontsize=6, color=TXT)
        cb.set_ticks([0, 0.5, 1.0]); cb.set_ticklabels(["0 early", "0.5", "1 late"])
    leg_vals = sorted({_round_nice(mx * f) for f in (0.25, 0.5, 1.0)})
    hands = [Line2D([0], [0], marker="o", ls="", mfc="#bbbbbb", mec="#3a3a3a", mew=0.6,
                    ms=np.sqrt((v / mx) * 45 + 2), label=f"{v:g}") for v in leg_vals]
    ax.legend(handles=hands, title=size_label, fontsize=6, title_fontsize=6,
              frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
              handletextpad=0.2, columnspacing=0.7)

    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    return fig


def compute(classification="PGD", weighted=True, order_by="alpha", denom_mode="cohort", color_by="timing",
           extra_exclude=None):
    """The math behind `make()`, returned as matrices instead of a rendered figure (so other scripts —
    e.g. a combined WGD-vs-loss comparison — can reuse it directly). See `make()`'s docstring for what
    each parameter means. Returns a dict: label, types, arms, eff, tim, color_val, n_ev_by_type,
    n_pos_by_type, cohort, n (total qualifying events). `types`/`eff`/`tim`/`color_val` are already
    reordered per `order_by`.

    extra_exclude: additional tumor types to drop from the row set on top of `EXCLUDE_TYPES` (e.g.
    `WGD_SPARSE_TYPES` for WGD-based variants — see that constant's docstring)."""
    label = classification if isinstance(classification, str) else "+".join(sorted(classification))
    cov, ev = _arm_coverage(classification, weighted=weighted)
    n = len(ev)   # total qualifying events (post donor-dedup)

    cohort = _sample_counts()                          # total samples per type
    n_ev_by_type = ev.groupby("tumor_type").size()      # row-label numerator only; 0 for untouched types
    n_pos_by_type = ev.groupby("tumor_type")["sample"].nunique()   # samples with >=1 qualifying event
    denom = cohort if denom_mode == "cohort" else n_pos_by_type   # -> normalises eff below
    types = _all_types(exclude=extra_exclude)           # see `make()` docstring
    arms = [a for a in ARMS if a in set(cov["arm"].unique()) and a not in EXCLUDE_ARMS]

    # effective frequency per (type, arm): each event contributes `frac` for that arm — either its
    # length-weighted coverage (weighted=True) or a flat 1.0 for merely touching it (weighted=False, the
    # Joseph-style binary count). Summing that across every event/sample of the type, then dividing by
    # `denom[t]` (see denom_mode above), gives a per-sample rate — cohort-size-corrected and (when
    # weighted) oversegmentation-proof, so both methods are normalised identically and only differ in
    # the per-event contribution.
    eff = np.zeros((len(types), len(arms)))          # per-sample effective frequency -> dot size
    tim = np.full((len(types), len(arms)), np.nan)   # coverage-weighted mean timing -> dot colour (timing mode)
    for ti, t in enumerate(types):
        d = denom.get(t, 0)
        if d == 0:
            continue
        sub = cov[cov.tumor_type == t]
        for ai, arm in enumerate(arms):
            garm = sub[sub["arm"] == arm]
            if garm.empty:
                continue
            eff[ti, ai] = garm["frac"].sum() / d
            if len(garm) >= MIN_CELL:
                w = garm["frac"].to_numpy(float); tvals = garm["time"].to_numpy(float)
                tim[ti, ai] = float(np.sum(w * tvals) / np.sum(w))

    if order_by == "alpha":
        ordr = sorted(range(len(types)), key=lambda i: LAB.get(types[i], types[i]).lower())
    else:
        # row order by dominant arm position, then its value (matches e_arm_patterns.py)
        ordr = sorted(range(len(types)), key=lambda i: (int(np.argmax(eff[i])), -eff[i].max()))
    types = [types[i] for i in ordr]; eff = eff[ordr]; tim = tim[ordr]

    if color_by == "zscore":
        # Row-wise z-score: for each tumor type, standardise its 44 (well, len(arms)) arm frequencies
        # against that type's OWN mean/std "over all the genome" — so this asks "which arms are unusually
        # enriched/depleted FOR THIS TYPE", not "which types have more events than others" (that's what
        # the raw effective-frequency size already shows).
        row_mean = eff.mean(axis=1, keepdims=True)
        row_std = eff.std(axis=1, keepdims=True)
        row_std[row_std == 0] = np.nan
        color_val = (eff - row_mean) / row_std
    else:
        color_val = tim

    return dict(label=label, types=types, arms=arms, eff=eff, tim=tim, color_val=color_val,
                n_ev_by_type=n_ev_by_type, n_pos_by_type=n_pos_by_type, cohort=cohort, n=n)


def make(classification="PGD", title=None, stem=None, save=True, color_by="timing", weighted=True,
         order_by="alpha", denom_mode="cohort", extra_exclude=None):
    """classification: "PGD" | "WGD" | "chrom_specific", or an iterable of those (e.g. the complement
    set for the "neither WGD nor PGD" figure). `title`/`stem` default from `classification` if omitted.

    color_by: "timing" (default) colours dots by the coverage-weighted mean event time (plasma, 0-1).
    "zscore" instead colours dots by each row's (tumor type's) OWN z-score of effective frequency across
    all arms — i.e. "relative to this type's genome-wide average, is this arm hit more or less than
    usual" — on a diverging red-blue (ColorBrewer RdBu) scale, red = enriched, blue = depleted.

    weighted: True (default) = segment-length-weighted arm coverage (this script's whole point).
    False = Joseph's ORIGINAL `e_arm_patterns.py` metric — binary "did this event touch the arm at
    all" — recomputed here from the same Tier-1 tables + per-sample cohort normalisation, purely so the
    two methodologies can be compared side by side on identical axes/scales.

    order_by: "alpha" (default) sorts rows alphabetically by the displayed tumor-type label — and every
    variant always shows ALL tumor types in the deduped cohort (via `_all_types()`, excluding "Unknown"),
    even ones with zero qualifying events for this classification (an empty row, not a dropped one), so
    every figure this script produces lists the exact same rows in the exact same order. "dominant_arm"
    instead sorts rows by each type's biggest dot (matches the original `e_arm_patterns.py` ordering).

    denom_mode: "cohort" (default) divides by the type's TOTAL sample count — this conflates "how common
    is this event class in this tumor type" with "which arms does it prefer" (a type where the event is
    rare will show small dots everywhere even if, WHEN it happens, it's extremely arm-selective).
    "event_positive" instead divides by the number of samples of that type with >=1 qualifying event —
    "given this sample HAS a PGD/WGD/chrom-specific event, how much of arm X does it cover" — isolating
    arm-selectivity from event prevalence. Row-label denominator switches to match.

    extra_exclude: additional tumor types to drop from the row set (on top of `EXCLUDE_TYPES`) — pass
    `WGD_SPARSE_TYPES` for WGD variants (Lymph.CLL/Prost.AdenoCA have too few WGD events to show a
    meaningful row here, even though they keep their row in the PGD/chrom_specific figures)."""
    method = "segment-weighted" if weighted else "Joseph-style, unweighted (binary presence)"
    r = compute(classification, weighted=weighted, order_by=order_by, denom_mode=denom_mode, color_by=color_by,
               extra_exclude=extra_exclude)
    label, types, arms, eff, color_val = r["label"], r["types"], r["arms"], r["eff"], r["color_val"]
    cohort, n_ev_by_type, n_pos_by_type, n = r["cohort"], r["n_ev_by_type"], r["n_pos_by_type"], r["n"]

    if title is None:
        cond = " — given event present" if denom_mode == "event_positive" else ""
        title = f"{label} arm programmes ({method}){cond}"
    if stem is None:
        stem = (f"e2_arm_patterns_segment_{label.lower().replace('+', '_')}" + ("" if weighted else "_binary")
                + ("" if denom_mode == "cohort" else "_conditional"))

    row_denom = cohort if denom_mode == "cohort" else n_pos_by_type
    row_labels = [f"{LAB.get(t, t)} ({int(n_ev_by_type.get(t, 0))}/{int(row_denom.get(t, 0))})" for t in types]
    size_label = "effective frequency / sample" if weighted else "event frequency / sample"
    if denom_mode == "event_positive":
        size_label += " w/ event"
    fig = _bubble_plot(eff, color_val, arms, row_labels, title, stem, color_by, size_label, save)
    print(f"  [{label}, {method}, denom={denom_mode}] {n} events, {len(types)} types shown, {len(arms)} arms")
    return fig


# The three classifications in tau_events.classification partition ALL clustered events, so these three
# figures together cover every gain event in the cohort exactly once.
VARIANTS = [
    dict(classification="PGD", stem="e2_arm_patterns_segment_pgd",
         title="PGD arm programmes (segment-weighted)"),
    dict(classification="WGD", stem="e2_arm_patterns_segment_wgd",
         title="WGD arm programmes (segment-weighted)", order_by="alpha", extra_exclude=WGD_SPARSE_TYPES),
    dict(classification=["chrom_specific"], stem="e2_arm_patterns_segment_neither",
         title="Non-WGD/PGD (chrom-specific) gain arm programmes (segment-weighted)"),
    # Joseph-style (binary-presence) counterpart for the "neither" class, same treatment (timing colour,
    # not Z-score) as the segment-weighted version above and as the original PGD panel it's ported from —
    # directly comparable to `e2_arm_patterns_segment_neither` since only `weighted` differs.
    dict(classification=["chrom_specific"], stem="e2_arm_patterns_segment_neither_binary",
         title="Non-WGD/PGD (chrom-specific) gain arm programmes (Joseph-style, unweighted)",
         weighted=False),
    # WGD hits ~the whole genome every time, so the timing-coloured version is nearly one flat colour;
    # this variant instead colours by each type's OWN row Z-score, surfacing arms relatively over/under-
    # represented within its WGD events (partial retention/loss patterns) rather than absolute timing.
    dict(classification="WGD", stem="e2_arm_patterns_segment_wgd_zscore", order_by="alpha",
         title="WGD arm programmes — relative frequency (segment-weighted)", color_by="zscore",
         extra_exclude=WGD_SPARSE_TYPES),
    # Same enrichment plot, but with `weighted=False` swapping in Joseph's ORIGINAL binary-presence
    # metric instead of the length-weighted one — directly comparable (same rows, same cohort
    # normalisation, same Z-score/colour scale) so the two methodologies can be eyeballed side by side.
    dict(classification="WGD", stem="e2_arm_patterns_segment_wgd_zscore_binary", order_by="alpha",
         title="WGD arm programmes — relative frequency (Joseph-style, unweighted)",
         color_by="zscore", weighted=False, extra_exclude=WGD_SPARSE_TYPES),
    # Same Z-score enrichment plot again, but conditioned on the event: divides by samples that actually
    # HAD a WGD instead of the type's whole cohort, isolating "which arms does WGD prefer" from "how
    # common is WGD in this tumour type" (the latter is what the un-conditioned dot size mixes in).
    dict(classification="WGD", stem="e2_arm_patterns_segment_wgd_zscore_conditional", order_by="alpha",
         title="WGD arm programmes — relative frequency, given WGD present (segment-weighted)",
         color_by="zscore", denom_mode="event_positive", extra_exclude=WGD_SPARSE_TYPES),
    # Same conditional-Z-score treatment for PGD and "neither" (chrom_specific): divide by samples that
    # actually HAD that event class instead of the type's whole cohort, so dot size/colour isolate arm
    # selectivity from how common PGD (resp. chrom-specific gains) is in that tumour type.
    dict(classification="PGD", stem="e2_arm_patterns_segment_pgd_zscore_conditional", order_by="alpha",
         title="PGD arm programmes — relative frequency, given PGD present (segment-weighted)",
         color_by="zscore", denom_mode="event_positive"),
    dict(classification=["chrom_specific"], stem="e2_arm_patterns_segment_neither_zscore_conditional",
         order_by="alpha", denom_mode="event_positive", color_by="zscore",
         title="Non-WGD/PGD (chrom-specific) arm programmes — relative frequency, "
               "given event present (segment-weighted)"),
]

if __name__ == "__main__":
    for v in VARIANTS:
        make(**v)
    print("e2_arm_patterns_segment: PNG+SVG written for PGD, WGD, WGD-zscore (weighted + binary), "
          "and neither (chrom-specific)")
