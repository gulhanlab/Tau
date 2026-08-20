#!/usr/bin/env python3
"""Interactive, self-contained HTML viewer for Tau's per-sample timing overview.

``genome_to_html`` reproduces the static :func:`tau.plotting.plot_overview`
as an interactive Plotly figure written to a single self-contained ``.html``
file (Plotly JS inlined, so it opens offline and is shareable).  The main panel
draws each timed segment as a **filled vertical stacked band** coloured by its
copy-number family (light = early, dark = late), exactly like the static
``stackplot`` overview — *not* as scatter markers.  Around it sit the same
companion elements as the static figure:

* a thin **multiplicity-cluster track** above the main panel, one row per CN
  state, with ``(major,minor)`` row labels and a cluster legend;
* a **left CN colour legend** (one column per major CN, ``t_early`` bottom →
  ``t_late`` top);
* horizontal **event lines** (teal WGD / gold PGD) with ``t`` + genome-fraction
  labels; and
* a **right timepoint-cluster histogram** (ESS-weighted, coloured by event).

Clicking a segment opens a detail panel with its multiplicity histogram, VAF
distribution, and a table of every polytope (route) solution.  The ``tree_top_k``
lowest-``dist_rel`` routes are made interactively explorable: expanding such a
route row reveals its allele tree plus one slider per free variable (a "tree
solution with knobs", ported from ``examples/tree_explorer_demo.html``).  Dialing
a knob moves the solution through the polytope — node heights (cumulative event
times) and the t-intervals update live from affine maps precomputed once via
:func:`tau.tree_polytope.build_tree_polytope_payload`, and the other knobs are
re-clamped so the solution never leaves the feasible region.  Determined routes
(0 free variables) show a static tree; routes beyond the top-K are dimmed and
non-expandable.

The molecular-time convention matches the rest of Tau: ``y = 0`` is EARLY
(few mutations before the event) and ``y = 1`` is LATE (many mutations before
it).  Entries of ``segment.timing_result`` are *polytope solutions* (equally
valid points in an underdetermined solution space) and are labelled as such in
the UI — never as draws, posteriors, or a distribution.
"""

from __future__ import annotations

import gzip
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tau.plotting import (
    HG19_ORDER,
    HG19_SIZES,
    GREY_RGBA,
    compute_offsets,
    normalize_chrom,
    _cmap_for_major,
    _colors_for_segment,
    _event_color,
    WGD_COLOR,
    PGD_COLOR,
)
from tau.utils import pick_best_key, extract_mat, order_t

_SEG_ID_RE = re.compile(r"^(.+):(\d+)-(\d+)$")

# Fill shades used to distinguish clusters within a single CN state in the
# top track (indexed by 1-based cluster_idx).  Two clusters of the same CN
# state (e.g. two (2,2) clusters) get different shades of the same colour map
# so they can be told apart; combined with the existing dashed-edge styles.
_CLUSTER_SHADES = [0.62, 0.88, 0.42, 0.98, 0.30]

# Max sampled polytope-solution vectors kept per tree payload (faint overlay).
_TREE_MAX_SAMPLES = 15
_AUTOSOMES = [str(i) for i in range(1, 23)]

# tab10 palette as CSS strings, for within-segment cluster-boundary lines.
_TAB10 = [
    "rgb(31,119,180)", "rgb(255,127,14)", "rgb(44,160,44)", "rgb(214,39,40)",
    "rgb(148,103,189)", "rgb(140,86,75)", "rgb(227,119,194)", "rgb(127,127,127)",
    "rgb(188,189,34)", "rgb(23,190,207)",
]

# ── SBS96 mutational-spectrum constants ─────────────────────────────────────
# Canonical 96-channel layout: 6 substitution types, each with the 16 flanking
# contexts (5' base × 3' base) in sorted order, giving channel strings of the
# form ``A[T>G]T``.  Channels are ordered exactly as the COSMIC/SigProfiler
# reference profile plots.
_SBS6_TYPES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
_SBS6_COLORS = {
    "C>A": "#03bcee", "C>G": "#000000", "C>T": "#e32925",
    "T>A": "#cac9c9", "T>C": "#a1ce63", "T>G": "#ebc6c4",
}
_SBS_BASES = ["A", "C", "G", "T"]


def _sbs96_channels():
    """Return the 96 SBS channel strings in canonical COSMIC order."""
    chans = []
    for sub in _SBS6_TYPES:
        ref = sub[0]
        for five in _SBS_BASES:
            for three in _SBS_BASES:
                chans.append(f"{five}[{sub}]{three}")
    return chans


_SBS96_CHANNELS = _sbs96_channels()
_SBS96_INDEX = {c: i for i, c in enumerate(_SBS96_CHANNELS)}
# Minimum grouped-signature sample share to earn its own selector option.
_SIG_SHARE_MIN = 0.10
# Validation-gate threshold: above this mean-abs-error between the
# reconstructed clock posterior and the stored exact ``mut_w`` we fall back to
# HARD (assigned) spectra for the non-clock signature options.
_SOFT_RECON_MAE_CUT = 0.10
_SIG_RE = re.compile(r"^(SBS\d+)[a-z]?$")


def _group_sig(sig):
    """Group a COSMIC signature name by its numeric stem (SBS17a/b → SBS17)."""
    m = _SIG_RE.match(str(sig))
    return m.group(1) if m else str(sig)


def _pool_snv_tables(genome):
    """Concatenate every segment's ``snv_table`` (those that carry SBS96 data)."""
    tabs = []
    for seg in getattr(genome, "segments", []) or []:
        st = getattr(seg, "snv_table", None)
        if st is not None and len(st) and "sbs96" in st.columns:
            tabs.append(st)
    if not tabs:
        return None
    return pd.concat(tabs, ignore_index=True)


def _build_spectrum_setup(genome):
    """Sample-wide signature setup for the SBS96 spectrum selector.

    Pools every segment's ``snv_table`` and derives, self-contained from the
    table (no exposures stored on the genome, no Sage):

    * ``ehat`` — per-individual-signature exposure estimate ``ê[sig]`` = the
      fraction of SNVs whose ``hard_sig_choice`` equals that signature;
    * the selector ``options`` — ``Unweighted``, ``Clock (SBS1+SBS5)``, and one
      per *grouped* signature (SBS17a/b → SBS17) whose pooled share exceeds
      :data:`_SIG_SHARE_MIN`;
    * a **validation gate**: the clock posterior is reconstructed the same soft
      way it would be for any signature group (Σ over SBS1, SBS5 of
      ``C[ctx]·ê``, normalised over all signatures) and compared to the stored
      exact ``mut_w``.  If the mean-abs-error exceeds :data:`_SOFT_RECON_MAE_CUT`
      the non-clock signature options fall back to HARD (assigned) spectra.

    Returns ``(setup, info)`` where *setup* is a dict consumed by
    :func:`_spectrum_for_table` (``ehat``, ``sig_cols``, ``options``,
    ``soft_path``) or ``None`` when no SBS96 data is present, and *info* is a
    human-readable dict for the generation log.
    """
    pool = _pool_snv_tables(genome)
    if pool is None or not len(pool):
        return None, dict(reason="no snv_table with sbs96")

    n_total = len(pool)
    has_hard = "hard_sig_choice" in pool.columns
    has_mw = "mut_w" in pool.columns

    # Per-individual-signature exposure estimate ê[sig] (hard-assignment frac).
    ehat = {}
    if has_hard:
        ehat = pool["hard_sig_choice"].value_counts(normalize=True).to_dict()
    # Signature C-columns actually present in the table.
    sig_cols = [c for c in pool.columns if _SIG_RE.match(c)]
    # Restrict the exposure vector to signatures that have a C column.
    sig_in_C = [c for c in sig_cols if ehat.get(c, 0.0) > 0]

    # Grouped shares (for the selector + labels).
    if has_hard:
        grp = pool["hard_sig_choice"].map(_group_sig)
        grp_share = grp.value_counts(normalize=True).to_dict()
    else:
        grp_share = {}

    # ── validation gate: reconstruct the clock posterior the soft way ───────
    soft_path = True
    val = dict(available=False)
    clock_variants = [c for c in ("SBS1", "SBS5") if c in sig_cols]
    if has_mw and sig_in_C and clock_variants:
        C = pool[sig_in_C].to_numpy(float)
        e = np.array([ehat[c] for c in sig_in_C], float)
        num = C * e[None, :]
        denom = num.sum(axis=1)
        cidx = [i for i, c in enumerate(sig_in_C) if c in clock_variants]
        with np.errstate(invalid="ignore", divide="ignore"):
            recon = num[:, cidx].sum(axis=1) / np.where(denom > 0, denom, np.nan)
        mw = pd.to_numeric(pool["mut_w"], errors="coerce").to_numpy(float)
        m = np.isfinite(recon) & np.isfinite(mw)
        if m.any():
            mae = float(np.mean(np.abs(recon[m] - mw[m])))
            corr = (float(np.corrcoef(recon[m], mw[m])[0, 1])
                    if m.sum() > 1 and np.std(recon[m]) > 0 and np.std(mw[m]) > 0
                    else float("nan"))
            val = dict(available=True, mae=mae, corr=corr, n=int(m.sum()))
            soft_path = mae <= _SOFT_RECON_MAE_CUT

    # ── selector options ────────────────────────────────────────────────────
    # Each option: dict(key, label, kind, sig (grouped), variants (C-cols)).
    options = [dict(key="unweighted", label="Unweighted", kind="unweighted",
                    sig=None, variants=[], share=None)]
    if clock_variants and has_mw:
        options.append(dict(key="clock", label="Clock (SBS1+SBS5)", kind="clock",
                            sig="clock", variants=clock_variants,
                            share=sum(grp_share.get(g, 0.0)
                                      for g in ("SBS1", "SBS5"))))
    # grouped signatures over the share threshold (skip the clock pair itself).
    sig_word = "weighted" if soft_path else "assigned"
    for g, sh in sorted(grp_share.items(), key=lambda kv: -kv[1]):
        if sh <= _SIG_SHARE_MIN:
            continue
        # SBS1/SBS5 are ALSO offered combined as the Clock option above, but a
        # single clock signature over threshold still earns its own (individual)
        # spectrum option here.
        variants = [c for c in sig_in_C if _group_sig(c) == g]
        if not variants:
            continue
        options.append(dict(key=g, label=f"{g} ({sig_word})", kind="sig",
                            sig=g, variants=variants, share=float(sh)))

    setup = dict(
        ehat=ehat,
        sig_in_C=sig_in_C,
        options=options,
        soft_path=soft_path,
        grp_share=grp_share,
        channels=_SBS96_CHANNELS,
    )
    info = dict(
        n_total=int(n_total),
        soft_path=soft_path,
        validation=val,
        options=[o["label"] for o in options],
        grp_share={g: round(s, 4) for g, s in grp_share.items() if s > _SIG_SHARE_MIN},
    )
    return setup, info


def _spectrum_for_table(st, setup):
    """Per-channel weighted sums for one ``snv_table`` and every selector option.

    Returns a dict mapping each option ``key`` → a length-96 list of weighted
    channel sums (weight summed per ``sbs96`` channel), aligned to
    :data:`_SBS96_CHANNELS`.  Per-SNV weights:

    * ``unweighted`` → 1 per SNV;
    * ``clock``      → the exact stored ``mut_w``;
    * signature ``g`` → soft posterior ``P(g|SNV)`` reconstructed self-contained
      as ``(Σ_{s∈g} C[s][ctx]·ê[s]) / (Σ_{s'} C[s'][ctx]·ê[s'])`` when the
      validation gate passed (``soft_path``); otherwise the HARD spectrum
      (weight 1 for SNVs whose ``hard_sig_choice`` groups to ``g``, else 0).

    Channels not present in this table simply receive 0.
    """
    out = {}
    if st is None or not len(st) or "sbs96" not in st.columns:
        return {o["key"]: [0.0] * 96 for o in setup["options"]}

    chan = st["sbs96"].astype(str).to_numpy()
    # channel index per SNV (drop unknown channels)
    ci = np.array([_SBS96_INDEX.get(c, -1) for c in chan])
    valid = ci >= 0

    ehat = setup["ehat"]
    sig_in_C = setup["sig_in_C"]
    soft_path = setup["soft_path"]

    # Precompute the soft normaliser denom = Σ_s' C[s'][ctx]·ê[s'] once.
    denom = None
    Cnum = None
    if sig_in_C:
        e = np.array([ehat[c] for c in sig_in_C], float)
        Ccols = st[sig_in_C].to_numpy(float)
        Cnum = Ccols * e[None, :]              # (N, n_sig) numerator terms
        denom = Cnum.sum(axis=1)               # (N,)

    hard = (st["hard_sig_choice"].astype(str).map(_group_sig).to_numpy()
            if "hard_sig_choice" in st.columns else None)

    def _accumulate(weights):
        w = np.asarray(weights, float)
        w = np.where(np.isfinite(w), w, 0.0)
        sums = np.zeros(96, float)
        np.add.at(sums, ci[valid], w[valid])
        return [round(float(x), 4) for x in sums]

    for opt in setup["options"]:
        kind = opt["kind"]
        if kind == "unweighted":
            w = np.ones(len(st), float)
        elif kind == "clock":
            w = (pd.to_numeric(st["mut_w"], errors="coerce").to_numpy(float)
                 if "mut_w" in st.columns else np.ones(len(st), float))
        else:  # signature group
            if soft_path and denom is not None and opt["variants"]:
                idx = [i for i, c in enumerate(sig_in_C) if c in opt["variants"]]
                with np.errstate(invalid="ignore", divide="ignore"):
                    w = Cnum[:, idx].sum(axis=1) / np.where(denom > 0, denom, np.nan)
            elif hard is not None:
                w = (hard == opt["sig"]).astype(float)
            else:
                w = np.zeros(len(st), float)
        out[opt["key"]] = _accumulate(w)
    return out


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def load_genome(path):
    """Load a (clustered) genome the same way ``scripts/run_tau.py`` writes it."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _load_sig_timing(sig_timing):
    """Resolve the per-signature timing cache into a compact JS-embed dict.

    *sig_timing* may be a path to a ``<sample>.sig_timing.json.gz`` file, an
    already-loaded dict in that schema, or ``None``.  Returns a compact dict
    ::

        {"signatures": [{"name", "share"}, ...],          # clock first
         "segment_times": {sig: {seg_id: [cum event times]}},
         "segment_mats": {sig: {seg_id: [[...], ...]}},    # full timing matrices
         "activity": {sig: [{seg_id, cn_state, interval, clock_time, ratio}]}}

    ready for ``json.dumps`` + ``SIG_TIMING`` embedding, or ``None`` when no
    usable cache is available (the signature features then stay OFF and the
    viewer is unchanged).  The cache holds *polytope solutions* collapsed to a
    point estimate per segment — never draws or a distribution.
    """
    if sig_timing is None:
        return None
    if isinstance(sig_timing, (str, Path)):
        p = Path(sig_timing)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt") as f:
                data = json.load(f)
        except Exception:
            return None
    elif isinstance(sig_timing, dict):
        data = sig_timing
    else:
        return None

    seg_times = data.get("segment_times") or {}
    if not seg_times:
        return None

    # Order signatures: clock first, then the rest by descending share.
    sigs_meta = data.get("signatures") or []
    share_by = {str(s.get("name")): float(s.get("share") or 0.0) for s in sigs_meta}
    present = [s for s in seg_times.keys()]
    rest = sorted([s for s in present if s != "clock"],
                  key=lambda s: -share_by.get(s, 0.0))
    ordered = (["clock"] if "clock" in present else []) + rest

    def _round_list(v):
        return [round(float(x), 4) for x in v]

    out_segtimes = {}
    for sig in ordered:
        out_segtimes[sig] = {
            str(k): _round_list(v) for k, v in (seg_times.get(sig) or {}).items()
        }

    # Full per-segment timing matrices (n_x x K) per signature.  These drive the
    # band-geometry overlay so each signature renders with the SAME stacked-band
    # geometry as the clock overview.  Pass through (already rounded upstream);
    # absent for older caches, in which case the toggle has no full-geometry
    # bands to swap (and silently falls back to clock-only).
    seg_mats_in = data.get("segment_mats") or {}
    out_segmats = {}
    for sig in ordered:
        sm = seg_mats_in.get(sig) or {}
        if not sm:
            continue
        out_segmats[sig] = {
            str(k): [[round(float(x), 5) for x in row] for row in mat]
            for k, mat in sm.items()
        }

    activity = data.get("activity") or {}
    out_activity = {}
    for sig, rows in activity.items():
        if sig not in out_segtimes:
            continue
        compact = []
        for r in rows:
            ra = r.get("ratio")
            # clock-time SPAN of the interval; fall back to a point (old caches that
            # only stored the right-edge clock_time) so they still render.
            t1 = r.get("t1", r.get("clock_time"))
            t0 = r.get("t0", t1)
            if t1 is None or ra is None:
                continue
            try:
                t0 = float(t0); t1 = float(t1); ra = float(ra)
                ess = float(r.get("ess", 1.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if t0 != t0 or t1 != t1 or ra != ra:
                continue
            # [t0, t1, ratio, ess] — the interval's clock-time span + ESS weight; the
            # aggregate curve spreads the ratio across [t0, t1] (see renderSigActivity).
            compact.append([round(t0, 4), round(t1, 4), round(ra, 5), round(ess, 4)])
        if compact:
            out_activity[sig] = compact

    # PER-SEGMENT activity: {sig: {seg_id: {abund, rows:[{interval, clock_time,
    # ratio, ct, st, ok}, ...]}}} — ALL intervals (gated and not), for the click
    # detail panel.  Independent of the overview toggle / aggregate `activity`.
    seg_activity_in = data.get("seg_activity") or {}
    out_seg_activity = {}
    for sig in ordered:
        if sig == "clock":
            continue
        blk = seg_activity_in.get(sig) or {}
        if not blk:
            continue
        out_blk = {}
        for sid, sv in blk.items():
            rows = sv.get("rows") or []
            ab = sv.get("abund")
            try:
                ab = float(ab)
            except (TypeError, ValueError):
                ab = None
            if ab is None or ab != ab:
                continue
            crows = []
            for r in rows:
                ctime = r.get("clock_time"); ra = r.get("ratio")
                if ctime is None or ra is None:
                    continue
                try:
                    ctime = float(ctime); ra = float(ra)
                except (TypeError, ValueError):
                    continue
                if ctime != ctime or ra != ra:
                    continue
                # clock-time SPAN of the interval (point fallback for old caches).
                t1 = float(r.get("t1", ctime))
                t0 = float(r.get("t0", t1))
                crows.append(dict(
                    interval=int(r.get("interval", 0)),
                    t0=round(t0, 4), t1=round(t1, 4),
                    clock_time=round(ctime, 4), ratio=round(ra, 5),
                    ct=round(float(r.get("ct", 0.0)), 5),
                    st=round(float(r.get("st", 0.0)), 5),
                    ok=bool(r.get("ok", False))))
            if crows:
                out_blk[str(sid)] = dict(abund=round(ab, 4), rows=crows)
        if out_blk:
            out_seg_activity[sig] = out_blk

    # genome-wide base rate per signature: {sig: {D_sig, D_clock, ratio, n_seg}}
    ga_in = data.get("genome_abund") or {}
    out_genome_abund = {}
    for sig, gv in ga_in.items():
        if sig == "clock" or not isinstance(gv, dict):
            continue
        rr = gv.get("ratio")
        if rr is None:
            continue
        out_genome_abund[sig] = dict(
            D_sig=float(gv.get("D_sig", 0.0)), D_clock=float(gv.get("D_clock", 0.0)),
            ratio=float(rr), n_seg=int(gv.get("n_seg", 0)))

    return dict(
        signatures=[dict(name=s, share=round(share_by.get(s, 0.0), 4))
                    for s in ordered],
        segment_times=out_segtimes,
        segment_mats=out_segmats,
        activity=out_activity,
        seg_activity=out_seg_activity,
        genome_abund=out_genome_abund,
    )


def _rgba_to_css(rgba):
    """Matplotlib RGBA tuple (0-1 floats) → CSS ``rgba(...)`` string."""
    r, g, b = (int(round(255 * c)) for c in rgba[:3])
    a = rgba[3] if len(rgba) > 3 else 1.0
    return f"rgba({r},{g},{b},{a:.3f})"


def _genome_offsets():
    """Cumulative genome-position offset for the start of each chromosome."""
    offsets, running = {}, 0
    for c in HG19_ORDER:
        offsets[c] = running
        running += HG19_SIZES.get(c, 0)
    autosome_len = sum(HG19_SIZES[c] for c in _AUTOSOMES)
    return offsets, running, autosome_len


def _route_solution_summary(info):
    """Summarise one route's polytope solutions into per-interval time bounds.

    Returns ``(dist_rel, loglik, free_var_count, intervals)`` where *intervals*
    is a list of dicts with ``min``/``max``/``mid`` cumulative event times — a
    point estimate when determined, a [min, max] bound when underdetermined.
    """
    qc = info.get("qc", []) or []
    qc0 = qc[0] if qc else {}
    dist_rel = qc0.get("dist_rel", float("nan"))
    loglik = qc0.get("loglik", float("nan"))
    free_var_count = (info.get("meta", {}) or {}).get("free_var_count", None)

    draws = info.get("draws", []) or []
    cums = []
    for d in draws:
        ts = order_t(d.get("t", {}))
        if ts.sum() <= 0:
            continue
        cum = np.cumsum(ts)[:-1]  # event-boundary times (K-1)
        if cum.size:
            cums.append(cum)
    intervals = []
    if cums:
        n_int = min(len(c) for c in cums)
        mat = np.vstack([c[:n_int] for c in cums])
        for j in range(n_int):
            col = mat[:, j]
            intervals.append(
                dict(min=float(col.min()), max=float(col.max()), mid=float(np.median(col)))
            )
    return dist_rel, loglik, free_var_count, intervals


def _violated_conditions(key, info, env, major, minor):
    """Human-readable N-inequalities the raw counts violated for this route.

    These are the projected/active constraints (qc ``active_constraints``) mapped
    back to the route's symbolic ``N_ineqs`` in the SAME order ``_Ab_from_N_ineqs``
    builds the halfspaces (a ``>=`` constraint → one row; an ``==`` constraint →
    two rows, ``>=`` then ``<=``). Empty when the route is exactly feasible
    (dist_rel≈0, nothing projected) or when env/Sage is unavailable.
    """
    if env is None:
        return []
    qc = info.get("qc", []) or []
    active = (qc[0].get("active_constraints", []) if qc else []) or []
    if not active:
        return []
    try:
        import operator as _op
        from tau.timing import prepare_route
        _, _, N_ineqs, *_ = prepare_route(env, key, major, minor)
    except Exception:
        return []
    labels = []
    for e in N_ineqs:
        op = e.operator()
        lhs, rhs = str(e.lhs()), str(e.rhs())
        if op is _op.ge:
            labels.append(f"{lhs} ≥ {rhs}")
        elif op is _op.eq:
            labels.append(f"{lhs} ≥ {rhs}")
            labels.append(f"{lhs} ≤ {rhs}")
    out = []
    for idx in active:
        if 0 <= idx < len(labels) and labels[idx] not in out:
            out.append(labels[idx])
    return out


def _segment_detail(seg, best_key, env=None):
    """Build the click-detail payload for one segment (multiplicity, VAF, routes)."""
    timing = seg.timing_result

    # All-route polytope-solution table for the detail panel.
    route_rows = []
    for key, info in timing.items():
        dist_rel, loglik, fvc, intervals = _route_solution_summary(info)
        interval_strs = []
        for iv in intervals:
            if iv["max"] - iv["min"] < 1e-6:
                interval_strs.append(f"{iv['mid']:.3f}")
            else:
                interval_strs.append(f"[{iv['min']:.3f}, {iv['max']:.3f}]")
        route_rows.append(
            dict(
                key=key,
                dist_rel=None if dist_rel != dist_rel else round(float(dist_rel), 4),
                conditions_violated=_violated_conditions(
                    key, info, env, seg.major_cn, seg.minor_cn),
                free_var_count=fvc,
                intervals=interval_strs,
                is_best=(key == best_key),
            )
        )

    # Best-fitting routes first: ascending dist_rel (unknown/None last).
    route_rows.sort(
        key=lambda r: (r["dist_rel"] is None,
                       r["dist_rel"] if r["dist_rel"] is not None else 0.0)
    )

    nc = getattr(seg, "N_counts", None)
    N_counts = np.asarray([] if nc is None else nc, float)
    if N_counts.size and N_counts.sum() > 0:
        props = (N_counts / N_counts.sum()).tolist()
    else:
        props = []

    vaf = []
    st = getattr(seg, "snv_table", None)
    if st is not None and len(st) and {"nalt", "nref"} <= set(st.columns):
        nalt = pd.to_numeric(st["nalt"], errors="coerce").to_numpy(float)
        nref = pd.to_numeric(st["nref"], errors="coerce").to_numpy(float)
        depth = nalt + nref
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(depth > 0, nalt / depth, np.nan)
        vaf = [float(x) for x in v if x == x]

    return route_rows, props, vaf


# ──────────────────────────────────────────────────────────────────────────
# interactive "tree solution with knobs" payloads
# ──────────────────────────────────────────────────────────────────────────
def _round_floats(obj, sig=6):
    """Recursively round all floats in a JSON-serialisable structure to *sig*
    significant figures, to keep the embedded tree payloads compact."""
    if isinstance(obj, float):
        if obj == 0.0 or not np.isfinite(obj):
            return obj
        from math import floor, log10
        return round(obj, -int(floor(log10(abs(obj)))) + (sig - 1))
    if isinstance(obj, dict):
        return {k: _round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, sig) for v in obj]
    return obj


def _build_tree_payloads(genome, tree_targets, env, top_k):
    """Build per-(segment, route) "tree solution with knobs" payloads.

    Parameters
    ----------
    genome : Genome
        Timed genome (passed straight to ``build_tree_polytope_payload``).
    tree_targets : list of (ri, seg_id, [route_key, ...])
        For each drawn segment: its detail-record index ``ri`` and the up-to-K
        route keys (lowest dist_rel first) to make explorable.
    env : tau.timing.RouteEnv or None
        Shared route environment (caches per-route symbolic prep).  If ``None``
        no trees are built.
    top_k : int
        Number of routes per segment to attempt (already applied to the lists).

    Returns
    -------
    (tree_payloads, expandable, failures)
        ``tree_payloads`` maps ``str(ri)`` -> {route_key: payload}; ``expandable``
        maps ``str(ri)`` -> [route_key, ...] (those that built successfully);
        ``failures`` is a count of routes whose extraction raised.
    """
    from tau.tree_polytope import build_tree_polytope_payload

    tree_payloads: dict[str, dict] = {}
    expandable: dict[str, list] = {}
    failures = 0
    if env is None or top_k <= 0:
        return tree_payloads, expandable, failures

    for ri, seg_id, route_keys in tree_targets:
        for rk in route_keys[:top_k]:
            try:
                payload = build_tree_polytope_payload(
                    genome, seg_id, route_key=rk, n_samples=_TREE_MAX_SAMPLES, env=env
                )
            except Exception:
                failures += 1
                continue
            # trim the faint overlay + round floats so the HTML stays manageable
            ss = payload.get("sampled_solutions") or []
            if len(ss) > _TREE_MAX_SAMPLES:
                payload["sampled_solutions"] = ss[:_TREE_MAX_SAMPLES]
            payload = _round_floats(payload, sig=6)
            tree_payloads.setdefault(str(ri), {})[rk] = payload
            expandable.setdefault(str(ri), []).append(rk)
    return tree_payloads, expandable, failures


# ──────────────────────────────────────────────────────────────────────────
# segment / band geometry
# ──────────────────────────────────────────────────────────────────────────
def _band_polys_for_mat(x0, x1, major_cn, minor_cn, mat):
    """Filled-band polygons for ONE segment from its timing matrix ``mat``.

    This is the single source of truth for stacked-band geometry, shared by the
    clock overview and every per-signature overlay so they render IDENTICALLY.

    ``mat`` has shape ``(n_x, K)`` (the ``extract_mat`` polytope-solution array);
    we accumulate it (``cumsum`` over axis 1), lay an x grid of ``max(2, n_x)``
    points across ``[x0, x1]``, and emit one closed polygon per multiplicity band
    ``b`` spanning ``cum[:, b-1] → cum[:, b]``.  Colours come from
    ``_colors_for_segment`` (flat GREY when ``major_cn == 1``).

    Returns ``{b: (xs, ys, color)}`` where ``xs``/``ys`` are the (single-polygon)
    vertex lists, closed back to the first vertex — exactly the per-segment slice
    that :func:`_segment_bands` concatenates into the genome-wide band traces.
    The point ordering (lower edge left→right, upper edge right→left, close)
    matches the clock build so a JS restyle can swap one for another verbatim.
    """
    mat = np.asarray(mat, float)
    out = {}
    if mat.size == 0:
        return out
    x0 = float(x0)
    x1 = float(x1)
    if x1 <= x0:
        x1 = x0 + 1  # guarantee a visible width
    maj = int(major_cn)
    K = mat.shape[1]
    n_x = mat.shape[0]
    cmap = _cmap_for_major(maj)
    cols = _colors_for_segment(maj, K, cmap)
    cum = np.cumsum(mat, axis=1)  # (n_x, K)
    x_vec = np.linspace(x0, x1, max(2, n_x))
    for b in range(K):
        upper = cum[:, b]
        lower = cum[:, b - 1] if b > 0 else np.zeros(n_x)
        if n_x == 1:
            xs = [x0, x1, x1, x0]
            ys = [float(lower[0]), float(lower[0]),
                  float(upper[0]), float(upper[0])]
        else:
            xs = list(x_vec) + list(x_vec[::-1])
            ys = list(lower) + list(upper[::-1])
        xs = [float(v) for v in xs]
        ys = [float(v) for v in ys]
        xs.append(xs[0])
        ys.append(ys[0])
        out[b] = (xs, ys, _rgba_to_css(cols[b]))
    return out


def _segment_bands(genome, chrom_off, ess_thresh=10, tree_top_k=0, spectrum_setup=None,
                   env=None):
    """Build filled-band polygons + per-segment hit/detail records.

    Mirrors :func:`tau.plotting.plot_tau_stack`: for each timed autosomal
    segment we extract ``mat`` (n_x, K), accumulate it, and emit one filled
    polygon per multiplicity band ``b`` spanning the segment's genome span with
    the ``_colors_for_segment`` ramp colour.  Polygons are grouped by
    ``(major_cn, band)`` so a handful of traces cover the whole genome.

    When ``tree_top_k > 0`` also collects, per drawn segment, the up-to-K route
    keys with the lowest ``dist_rel`` (for the interactive tree precompute) and
    the set of CN states present (so a single shared route env can be loaded).

    Returns ``(band_polys, boundary_lines, hit_records, detail_records,
    tree_targets, cn_states_present)``.
    """
    segs_ordered, offsets = compute_offsets(genome.segments)
    tree_targets = []          # (ri, seg_id, [top-k route keys])
    cn_states_present = set()   # (major, minor) of drawn segments
    state_nbands = {}           # (major, minor) -> K bands actually drawn (legend colours)

    # band_polys[(major, band)] = list of (xs, ys) closed polygons
    band_polys: dict[tuple[int, int], dict] = {}
    # within-segment cluster-boundary lines, grouped by colour
    boundary_lines: dict[str, list] = {}
    hit_records = []   # one per segment: bar geometry + customdata index
    detail_records = []  # one per segment: detail-panel payload

    seg_cluster_ids = getattr(genome, "segment_cluster_ids", {}) or {}

    for seg in segs_ordered:
        chrom = normalize_chrom(seg.chrom)
        if chrom not in _AUTOSOMES:
            continue

        nc = getattr(seg, "N_counts", None)
        ess = float(np.nansum(np.asarray([] if nc is None else nc, float)))
        if ess < ess_thresh:
            continue

        key = pick_best_key(seg)
        if key is None:
            continue

        mat = extract_mat(seg, key, boot_id=0)
        if mat is None or mat.size == 0:
            continue

        maj = int(seg.major_cn)
        minor = int(seg.minor_cn)
        xoff = chrom_off.get(chrom)
        if xoff is None:
            continue
        x0 = int(seg.start) + xoff
        x1 = int(seg.end) + xoff
        if x1 <= x0:
            x1 = x0 + 1  # guarantee a visible width

        K = mat.shape[1]
        n_x = mat.shape[0]
        cum = np.cumsum(mat, axis=1)  # (n_x, K)  (kept for boundary lines below)

        # Filled bands (stackplot equivalent): built through the shared helper so
        # the clock overview and every per-signature overlay use IDENTICAL
        # geometry.  Each band b's polygon spans lower→upper edge over the same
        # linspace x grid.
        seg_polys = _band_polys_for_mat(x0, x1, maj, minor, mat)
        for b in range(K):
            xs, ys, color = seg_polys[b]
            # key by (major, K, band): a Plotly trace has ONE fill colour, and the
            # band colour depends on the segment's K (_colors_for_segment), so
            # states with the same major but different K (e.g. (2,1) K=2 vs (2,2)
            # K=3) MUST NOT share a trace — else (2,2)'s gradient collapses to the
            # K=2 colours and looks like 2 colours instead of 3.
            grp = band_polys.setdefault(
                (maj, K, b),
                dict(color=color, xs=[], ys=[],
                     chrom=[], cn=[], seg=[], band=[]),
            )
            if grp["xs"]:
                grp["xs"].append(None)
                grp["ys"].append(None)
            grp["xs"].extend(xs)
            grp["ys"].extend(ys)
            grp["chrom"].append(chrom)
            grp["cn"].append(maj)
            grp["seg"].append(str(seg.seg_id))
            # Per-polygon (seg_id, band) so the JS signature toggle can swap in
            # another signature's prebuilt band polygons (see SIG_BANDS).
            # Polygons within this trace are separated in xs/ys by a single
            # ``None`` sentinel, in this order.
            grp["band"].append(b)

        # Within-segment cluster-boundary lines (cum[:, b] for b in 0..K-2).
        cids = seg_cluster_ids.get(seg.seg_id, None)
        if cids is not None:
            for b in range(K - 1):
                if b >= len(cids):
                    continue
                cid = int(cids[b])
                colour = "rgb(0,0,0)" if cid == -1 else _TAB10[cid % len(_TAB10)]
                y = cum[:, b]
                if n_x == 1:
                    lx, ly = [x0, x1], [float(y[0]), float(y[0])]
                else:
                    lx, ly = list(np.linspace(x0, x1, n_x)), list(y)
                grp = boundary_lines.setdefault(colour, dict(xs=[], ys=[]))
                if grp["xs"]:
                    grp["xs"].append(None)
                    grp["ys"].append(None)
                grp["xs"].extend(lx)
                grp["ys"].extend(ly)

        # Best-route event times for the hover summary.
        b_dist, _, _, b_intervals = _route_solution_summary(seg.timing_result[key])
        event_times = [iv["mid"] for iv in b_intervals]
        n_mut = int(round(ess))

        ri = len(detail_records)
        route_rows, props, vaf = _segment_detail(seg, key, env=env)
        try:
            purity = float(getattr(seg, "purity", float("nan")))
        except (TypeError, ValueError):
            purity = float("nan")
        try:
            total_cn = int(seg.total_cn)
        except (TypeError, ValueError, AttributeError):
            total_cn = maj + minor
        rec = dict(
            ri=ri,
            seg_id=str(seg.seg_id),
            major_cn=maj,
            minor_cn=minor,
            n_mut=n_mut,
            best_key=key,
            best_dist_rel=None if b_dist != b_dist else round(float(b_dist), 4),
            route_rows=route_rows,
            props=props,
            vaf=vaf,
            purity=None if purity != purity else round(purity, 6),
            total_cn=total_cn,
        )
        if spectrum_setup is not None:
            rec["spectra"] = _spectrum_for_table(
                getattr(seg, "snv_table", None), spectrum_setup
            )
        spatial = _segment_spatial(seg)
        if spatial is not None:
            rec["spatial"] = spatial
        detail_records.append(rec)

        # Track every drawn (major,minor) state + its band count K — used for the
        # left CN legend (independent of whether interactive trees are enabled).
        cn_states_present.add((maj, minor))
        state_nbands[(maj, minor)] = max(state_nbands.get((maj, minor), 0), K)

        # Interactive-tree targets: the K lowest-dist_rel routes (route_rows are
        # already dist_rel-sorted ascending) become explorable.
        if tree_top_k > 0:
            top_keys = [r["key"] for r in route_rows[:tree_top_k]]
            tree_targets.append((ri, str(seg.seg_id), top_keys))
        hit_records.append(
            dict(
                center=(x0 + x1) / 2.0,
                width=(x1 - x0),
                chrom=chrom,
                major_cn=maj,
                minor_cn=minor,
                ri=ri,
                hover=(
                    f"<b>{seg.seg_id}</b><br>"
                    f"chr{chrom}<br>"
                    f"CN ({maj},{minor})<br>"
                    f"{n_mut} mutations<br>"
                    f"route {key} (dist_rel="
                    f"{'NA' if b_dist != b_dist else round(float(b_dist),4)})<br>"
                    f"event time(s) t="
                    + (", ".join(f"{t:.2f}" for t in event_times) if event_times else "NA")
                ),
            )
        )

    return (band_polys, boundary_lines, hit_records, detail_records,
            tree_targets, sorted(cn_states_present), state_nbands)


def _segment_coords(genome, chrom_off):
    """Per-segment genome-x extent + CN, keyed by seg_id (autosomes only).

    Returns ``{seg_id: (x0, x1, major_cn, minor_cn)}`` using the SAME offset
    convention as :func:`_segment_bands` so per-signature band polygons land on
    exactly the clock x positions.
    """
    coords = {}
    for seg in genome.segments:
        chrom = normalize_chrom(seg.chrom)
        if chrom not in _AUTOSOMES:
            continue
        xoff = chrom_off.get(chrom)
        if xoff is None:
            continue
        x0 = int(seg.start) + xoff
        x1 = int(seg.end) + xoff
        if x1 <= x0:
            x1 = x0 + 1
        coords[str(seg.seg_id)] = (x0, x1, int(seg.major_cn), int(seg.minor_cn))
    return coords


def _cn_profile_data(genome, chrom_off, cap=6):
    """Genome-wide major/minor copy-number step lines for the CN-profile row.

    Returns parallel x/y arrays (``None`` breaks between segments) for a major
    and a minor step line across all autosomal segments. The DRAWN y is clamped
    to ``cap + 1`` (a "cap+1 plus" overflow level) so a few very-high-CN segments
    don't squash the axis; the TRUE copy number is carried in the ``*_cd`` arrays
    for the hover tooltip. Uses the SAME x-offset convention as the main panel so
    the profile registers with the segment bands and cluster track below it.
    """
    overflow = cap + 1
    coords = _segment_coords(genome, chrom_off)
    maj_x, maj_y, maj_cd = [], [], []
    min_x, min_y, min_cd = [], [], []
    for (x0, x1, maj, minor) in sorted(coords.values()):
        my = maj if maj <= cap else overflow      # clamp for drawing
        ny = minor if minor <= cap else overflow
        maj_x += [x0, x1, None]; maj_y += [my, my, None]; maj_cd += [maj, maj, None]
        min_x += [x0, x1, None]; min_y += [ny, ny, None]; min_cd += [minor, minor, None]
    return dict(maj_x=maj_x, maj_y=maj_y, maj_cd=maj_cd,
                min_x=min_x, min_y=min_y, min_cd=min_cd,
                cap=int(cap), overflow=int(overflow))


def _sig_band_polys(genome, chrom_off, segment_mats):
    """Prebuilt per-signature band-trace polygon sets for the JS toggle.

    For every signature in ``segment_mats`` (``{sig: {seg_id: mat}}``) we rebuild
    each segment's stacked bands with the SHARED :func:`_band_polys_for_mat`
    helper — IDENTICAL geometry to the clock overview — and concatenate them into
    genome-wide arrays keyed by the band-trace name ``f"maj{maj}_K{K}_b{b}"``
    (matching the clock band traces; K = #bands keeps different-K states apart).  Polygons within a trace are ``None``-separated, just
    like clock.  Segments not timed under a signature are simply absent from that
    signature's arrays (so they vanish when that signature is active).

    Returns ``{sig: {trace_name: {"x": [...], "y": [...]}}}``.
    """
    coords = _segment_coords(genome, chrom_off)
    out = {}
    for sig, seg_mats in (segment_mats or {}).items():
        traces = {}
        for seg_id, mat in seg_mats.items():
            c = coords.get(str(seg_id))
            if c is None:
                continue
            x0, x1, maj, minor = c
            polys = _band_polys_for_mat(x0, x1, maj, minor, mat)
            K = len(polys)   # match the (major, K, band) trace naming of the clock bands
            for b, (xs, ys, _color) in polys.items():
                name = f"maj{maj}_K{K}_b{b}"
                tr = traces.setdefault(name, {"x": [], "y": []})
                if tr["x"]:
                    tr["x"].append(None)
                    tr["y"].append(None)
                tr["x"].extend(xs)
                tr["y"].extend(ys)
        out[sig] = traces
    return out


# Max SNVs embedded per segment for the spatial (rainfall / VAF-vs-position)
# panel; larger segments are evenly subsampled to bound the HTML size.
_SPATIAL_MAX_SNV = 1500

# Map the 6 SBS substitution strings to an index 0..5 (C>A..T>G), matching
# _SBS6_TYPES / _SBS6_COLORS.
_SBS6_INDEX = {sub: i for i, sub in enumerate(_SBS6_TYPES)}


def _segment_spatial(seg):
    """Compact per-SNV spatial arrays for one segment's detail panel.

    Returns a dict with parallel arrays (ints/floats), capped at
    :data:`_SPATIAL_MAX_SNV` SNVs (even subsample when larger):

    * ``pos`` — int genomic offset from ``seg.start`` (add ``seg.start`` back in
      JS to get absolute position; keeps the embedded ints small);
    * ``vaf`` — VAF (``vaf`` column if present else ``nalt/(nalt+nref)``), 3dp;
    * ``mult`` — ``MAP_multiplicity`` if present, else nearest integer;
    * ``t6``  — SBS6 substitution-type index 0..5 from the ``sbs96`` channel;
    * ``start`` — ``seg.start`` (so JS reconstructs absolute positions);
    * ``n_total`` — total SNVs before any cap; ``capped`` — bool.

    Returns ``None`` when the segment has no usable SNV table.
    """
    st = getattr(seg, "snv_table", None)
    if st is None or not len(st) or "pos" not in st.columns:
        return None

    pos = pd.to_numeric(st["pos"], errors="coerce").to_numpy(float)

    # VAF: prefer an explicit column, else nalt/(nalt+nref).
    if "vaf" in st.columns:
        vaf = pd.to_numeric(st["vaf"], errors="coerce").to_numpy(float)
    elif {"nalt", "nref"} <= set(st.columns):
        nalt = pd.to_numeric(st["nalt"], errors="coerce").to_numpy(float)
        nref = pd.to_numeric(st["nref"], errors="coerce").to_numpy(float)
        depth = nalt + nref
        with np.errstate(invalid="ignore", divide="ignore"):
            vaf = np.where(depth > 0, nalt / depth, np.nan)
    else:
        vaf = np.full(pos.shape, np.nan)

    # Multiplicity: MAP_multiplicity if present else nearest integer of it.
    if "MAP_multiplicity" in st.columns:
        mult = pd.to_numeric(st["MAP_multiplicity"], errors="coerce").to_numpy(float)
    else:
        mult = np.full(pos.shape, np.nan)

    # SBS6 type index from the sbs96 channel (middle substitution).
    if "sbs96" in st.columns:
        chan = st["sbs96"].astype(str).to_numpy()
        t6 = np.array([
            _SBS6_INDEX.get(c[2:5], -1) if (len(c) >= 5 and c[1] == "[") else -1
            for c in chan
        ])
    else:
        t6 = np.full(pos.shape, -1, int)

    keep = np.isfinite(pos)
    pos = pos[keep]; vaf = vaf[keep]; mult = mult[keep]; t6 = t6[keep]
    n_total = int(pos.size)
    if n_total == 0:
        return None

    # Sort by genomic position (rainfall needs sorted order anyway).
    order = np.argsort(pos, kind="stable")
    pos = pos[order]; vaf = vaf[order]; mult = mult[order]; t6 = t6[order]

    capped = n_total > _SPATIAL_MAX_SNV
    if capped:
        idx = np.linspace(0, n_total - 1, _SPATIAL_MAX_SNV).round().astype(int)
        idx = np.unique(idx)
        pos = pos[idx]; vaf = vaf[idx]; mult = mult[idx]; t6 = t6[idx]

    start = int(seg.start)
    pos_off = (pos - start).round().astype(int).tolist()
    vaf_l = [None if not np.isfinite(v) else round(float(v), 3) for v in vaf]
    mult_l = [None if not np.isfinite(m) else int(round(float(m))) for m in mult]
    t6_l = [int(x) for x in t6]
    return dict(
        start=start,
        pos=pos_off,
        vaf=vaf_l,
        mult=mult_l,
        t6=t6_l,
        n_total=n_total,
        capped=bool(capped),
    )


def _vaf_from_snv_table(st):
    """Extract VAF = nalt / (nalt + nref) values from a segment's ``snv_table``."""
    if st is None or not len(st) or not ({"nalt", "nref"} <= set(st.columns)):
        return []
    nalt = pd.to_numeric(st["nalt"], errors="coerce").to_numpy(float)
    nref = pd.to_numeric(st["nref"], errors="coerce").to_numpy(float)
    depth = nalt + nref
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(depth > 0, nalt / depth, np.nan)
    return [float(x) for x in v if x == x]


def _cnstate_vaf(genome, n_bins=50):
    """Pool per-SNV VAFs across all segments of each (major, minor) CN state.

    For every CN state present in *genome* this collects the VAF of every SNV
    (from each segment's ``snv_table`` — the ``vaf`` column when present, else
    ``nalt / (nalt + nref)``), bins the pooled values into a fixed
    ``[0, 1]``-domain histogram (so only bin counts + edges are embedded, not
    the raw VAF list), and records the per-multiplicity *expected* VAF lines

        p_m = m * rho / (total_cn * rho + 2 * (1 - rho))   for m = 1..major_cn

    where ``rho`` is the sample-wide tumour purity.  Returns a list of per-state
    dicts sorted by descending SNV count (most-populated state first); CN states
    with ``major_cn < 1`` or no SNVs are skipped.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    edges_list = edges.tolist()

    # Sample-wide purity: take it from any segment that carries a valid value.
    sample_purity = None
    pooled = {}  # (major, minor) -> {"vafs": [...], "n_segments": int}
    for seg in getattr(genome, "segments", []) or []:
        try:
            maj = int(seg.major_cn)
            minor = int(seg.minor_cn)
        except (TypeError, ValueError, AttributeError):
            continue
        if sample_purity is None:
            try:
                p = float(getattr(seg, "purity", float("nan")))
            except (TypeError, ValueError):
                p = float("nan")
            if p == p and 0.0 < p <= 1.0:
                sample_purity = p

        st = getattr(seg, "snv_table", None)
        if st is not None and len(st) and "vaf" in getattr(st, "columns", []):
            v = pd.to_numeric(st["vaf"], errors="coerce").to_numpy(float)
            vafs = [float(x) for x in v if x == x]
        else:
            vafs = _vaf_from_snv_table(st)

        bucket = pooled.setdefault((maj, minor), {"vafs": [], "n_segments": 0})
        bucket["n_segments"] += 1
        if vafs:
            bucket["vafs"].extend(vafs)

    out = []
    for (maj, minor), bucket in pooled.items():
        if maj < 1 or not bucket["vafs"]:
            continue
        total_cn = maj + minor
        vafs = np.asarray(bucket["vafs"], float)
        counts, _ = np.histogram(vafs, bins=edges)

        expected = []
        if (sample_purity is not None and 0.0 < sample_purity <= 1.0
                and total_cn > 0):
            denom = total_cn * sample_purity + 2.0 * (1.0 - sample_purity)
            if denom > 0:
                for m in range(1, maj + 1):
                    pm = m * sample_purity / denom
                    if 0.0 <= pm <= 1.0:
                        expected.append(round(float(pm), 6))

        out.append(dict(
            major=maj,
            minor=minor,
            total_cn=total_cn,
            n_segments=int(bucket["n_segments"]),
            n_snvs=int(vafs.size),
            purity=(None if sample_purity is None
                    else round(float(sample_purity), 6)),
            counts=[int(c) for c in counts],
            bin_centers=centers,
            bin_edges=edges_list,
            expected_vafs=expected,
        ))

    out.sort(key=lambda d: d["n_snvs"], reverse=True)
    return out


def _cluster_track(clustered_genome, chrom_off, detail_records=None,
                   spectrum_setup=None, orig_genome=None, tree_env=None,
                   tree_top_k=0):
    """Build the multiplicity-cluster track rectangles, row labels + payloads.

    Mirrors :func:`tau.plotting.segment_cluster_plotting`.  Returns
    ``(track_shapes, cn_states, max_cluster_idx, cluster_payloads)``.

    *track_shapes* is a list of dicts (x0, x1, row, color, cluster_idx,
    payload_idx) — one per member segment of every cluster; ``payload_idx``
    indexes into *cluster_payloads* so every rectangle of a cluster maps back to
    that same cluster's detail payload.

    *cluster_payloads* is a list of per-cluster dicts (aggregate VAF + pooled
    multiplicity proportions + member-segment records).  Member records carry a
    ``ri`` cross-reference into *detail_records* (the embedded per-segment
    detail array) when the member is drawn on the main panel, or ``None`` when
    the member was filtered out (below ESS threshold / untimed) so the cluster
    panel can still list it as "not individually timed".
    """
    from tau.trees import _get_final_segment_clusters

    if clustered_genome is None:
        return [], [], 0, []

    final = _get_final_segment_clusters(clustered_genome)
    if not final:
        return [], [], 0, []

    # Map original member seg_id -> index in the embedded per-segment array.
    ri_by_seg = {}
    if detail_records:
        for i, rec in enumerate(detail_records):
            ri_by_seg[str(rec["seg_id"])] = i

    # Stable ordering: enumerate clusters once; payload_idx == enumeration order.
    cluster_names = list(final.keys())
    payload_idx_of = {name: i for i, name in enumerate(cluster_names)}

    seg_by_cluster = {}
    for cluster_name, seg in final.items():
        for sid in getattr(seg, "seg_ids", []):
            seg_by_cluster[str(sid)] = cluster_name

    cn_states, seen_cn, cn_cluster_num = [], {}, {}
    for name, seg in final.items():
        cn = (int(seg.major_cn), int(seg.minor_cn))
        if cn not in seen_cn:
            seen_cn[cn] = 0
            cn_states.append(cn)
        seen_cn[cn] += 1
        cn_cluster_num[name] = seen_cn[cn]

    cn_row = {cn: i for i, cn in enumerate(cn_states)}

    track_shapes = []
    for sid, cluster_name in seg_by_cluster.items():
        m = _SEG_ID_RE.match(sid)
        if not m:
            continue
        chrom = normalize_chrom(m.group(1))
        if chrom not in chrom_off:
            continue
        start, end = int(m.group(2)), int(m.group(3))
        seg = final[cluster_name]
        cn = (int(seg.major_cn), int(seg.minor_cn))
        row = cn_row[cn]
        x0 = chrom_off[chrom] + start
        x1 = chrom_off[chrom] + end
        if x1 <= x0:
            x1 = x0 + 1
        cidx = cn_cluster_num[cluster_name]  # 1-based within its CN state
        shade = _CLUSTER_SHADES[min(cidx - 1, len(_CLUSTER_SHADES) - 1)]
        track_shapes.append(
            dict(x0=x0, x1=x1, row=row,
                 color=_rgba_to_css(_cmap_for_major(cn[0])(shade)),
                 cluster_idx=cidx, payload_idx=payload_idx_of[cluster_name])
        )

    # snv_table lookup over the ORIGINAL genome, so a cluster that has no pooled
    # merged snv_table can still aggregate over its member segments' tables.
    orig_st_by_seg = {}
    if spectrum_setup is not None and orig_genome is not None:
        for oseg in getattr(orig_genome, "segments", []) or []:
            st = getattr(oseg, "snv_table", None)
            if st is not None and len(st) and "sbs96" in st.columns:
                orig_st_by_seg[str(oseg.seg_id)] = st

    # ── per-cluster payloads ────────────────────────────────────────────────
    cluster_payloads = []
    for name in cluster_names:
        seg = final[name]
        maj = int(seg.major_cn)
        minor = int(seg.minor_cn)

        # pooled multiplicity proportions (merged N_counts normalised)
        nc = getattr(seg, "N_counts", None)
        N_counts = np.asarray([] if nc is None else nc, float)
        if N_counts.size and N_counts.sum() > 0:
            props = (N_counts / N_counts.sum()).tolist()
            total_muts = int(round(float(N_counts.sum())))
        else:
            props = []
            total_muts = 0

        # aggregate VAF from the pooled (merged) snv_table
        vaf = _vaf_from_snv_table(getattr(seg, "snv_table", None))

        # member-segment records: cross-reference each into detail_records.
        members = []
        for sid in getattr(seg, "seg_ids", []):
            sid = str(sid)
            length = None
            mm = _SEG_ID_RE.match(sid)
            if mm:
                length = int(mm.group(3)) - int(mm.group(2))
            ri = ri_by_seg.get(sid)
            n_mut = detail_records[ri]["n_mut"] if (ri is not None and detail_records) else None
            members.append(dict(seg_id=sid, length=length, n_mut=n_mut, ri=ri))
        # longest segments first (most informative members at the top)
        members.sort(key=lambda d: (d["length"] is not None, d["length"] or 0), reverse=True)

        # ── cluster timing: route-solutions table from the merged segment ────
        # The merged segment carries its own ``timing_result``; build a route
        # table identical in form to the per-segment one (dist_rel-sorted),
        # plus the best route's per-interval cumulative event times.
        best_key = pick_best_key(seg)
        route_rows = []
        timing = getattr(seg, "timing_result", {}) or {}
        for key, info in timing.items():
            dist_rel, loglik, fvc, intervals = _route_solution_summary(info)
            interval_strs = []
            for iv in intervals:
                if iv["max"] - iv["min"] < 1e-6:
                    interval_strs.append(f"{iv['mid']:.3f}")
                else:
                    interval_strs.append(f"[{iv['min']:.3f}, {iv['max']:.3f}]")
            route_rows.append(dict(
                key=key,
                dist_rel=None if dist_rel != dist_rel else round(float(dist_rel), 4),
                conditions_violated=_violated_conditions(key, info, tree_env, maj, minor),
                free_var_count=fvc,
                intervals=interval_strs,
                is_best=(key == best_key),
            ))
        route_rows.sort(
            key=lambda r: (r["dist_rel"] is None,
                           r["dist_rel"] if r["dist_rel"] is not None else 0.0)
        )
        best_event_times = []
        if best_key is not None and best_key in timing:
            _, _, _, b_iv = _route_solution_summary(timing[best_key])
            best_event_times = [round(float(iv["mid"]), 4) for iv in b_iv]

        try:
            cl_purity = float(getattr(seg, "purity", float("nan")))
        except (TypeError, ValueError):
            cl_purity = float("nan")
        try:
            cl_total_cn = int(seg.total_cn)
        except (TypeError, ValueError, AttributeError):
            cl_total_cn = maj + minor

        cl_payload = dict(
            cn=[maj, minor],
            cluster_idx=cn_cluster_num[name],
            n_segments=int(getattr(seg, "num_segs", len(getattr(seg, "seg_ids", [])))),
            total_muts=total_muts,
            props=props,
            vaf=vaf,
            members=members,
            major_cn=maj,
            purity=None if cl_purity != cl_purity else round(cl_purity, 6),
            total_cn=cl_total_cn,
            routes=route_rows,
            best_key=best_key,
            best_event_times=best_event_times,
        )

        # ── cluster tree-knobs: build trees for MULTIPLE routes of the merged
        #    segment — all exactly-feasible (dist_rel==0) routes, or the top
        #    `tree_top_k` if fewer than that are feasible. route_rows is
        #    dist_rel-sorted ascending, so we just take the first k_cl. ──
        if tree_env is not None and tree_top_k > 0 and route_rows:
            n0 = sum(1 for r in route_rows
                     if r["dist_rel"] is not None and r["dist_rel"] <= 1e-9)
            k_cl = max(tree_top_k, n0)
            cl_trees = {}
            for r in route_rows[:k_cl]:
                try:
                    from tau.tree_polytope import build_tree_polytope_payload
                    tp = build_tree_polytope_payload(
                        clustered_genome, str(seg.seg_id), route_key=r["key"],
                        n_samples=_TREE_MAX_SAMPLES, env=tree_env,
                    )
                    ss = tp.get("sampled_solutions") or []
                    if len(ss) > _TREE_MAX_SAMPLES:
                        tp["sampled_solutions"] = ss[:_TREE_MAX_SAMPLES]
                    cl_trees[r["key"]] = _round_floats(tp, sig=6)
                except Exception:
                    pass  # merged seg not resolvable / extraction failed → skip
            if cl_trees:
                cl_payload["trees"] = cl_trees
                cl_payload["tree_expandable"] = [
                    r["key"] for r in route_rows[:k_cl] if r["key"] in cl_trees]
        if spectrum_setup is not None:
            # Prefer the merged segment's pooled snv_table; else pool member
            # original segments' tables.
            merged_st = getattr(seg, "snv_table", None)
            if merged_st is not None and len(merged_st) and "sbs96" in merged_st.columns:
                cl_st = merged_st
            else:
                parts = [orig_st_by_seg[str(sid)]
                         for sid in getattr(seg, "seg_ids", [])
                         if str(sid) in orig_st_by_seg]
                cl_st = pd.concat(parts, ignore_index=True) if parts else None
            cl_payload["spectra"] = _spectrum_for_table(cl_st, spectrum_setup)
        cluster_payloads.append(cl_payload)

    max_idx = max(cn_cluster_num.values()) if cn_cluster_num else 0
    return track_shapes, cn_states, max_idx, cluster_payloads


def _timing_hist(genome, cluster_df):
    """ESS-weighted horizontal timing histogram per event.

    Adapts :func:`tau.plotting._add_timing_histogram` to the column layout of
    ``genome.times_to_df`` (sample, seg_id, key, boot_id, time_interval, time,
    eff_N, ...).  Returns a list of per-event dicts with bar ``y`` (bin
    centres), ``x`` (summed ESS weight), colour and event time/label.
    """
    if cluster_df is None or not len(cluster_df):
        return []
    try:
        df = genome.times_to_df("S")
    except Exception:
        return []
    if df is None or df.empty:
        return []

    df = df[~df["is_bootstrap"].astype(bool)]
    if df.empty:
        return []

    # Restrict to each segment's best route (matches the static figure's
    # best_route==True filter).
    best = {}
    for seg in genome.segments:
        k = pick_best_key(seg)
        if k is not None:
            best[str(seg.seg_id)] = k
    df = df[df.apply(lambda r: best.get(str(r["seg_id"])) == r["key"], axis=1)]
    if df.empty:
        return []

    grp = (
        df.groupby(["seg_id", "time_interval"])
        .agg(
            t_mean=("time", "mean"),
            t_range=("time", lambda x: float(x.max() - x.min())),
            ess=("eff_N", "first"),
        )
        .reset_index()
    )
    grp = grp[grp["t_range"] < 0.10].copy()
    if grp.empty:
        return []

    events_sorted = cluster_df.sort_values("time").reset_index(drop=True)
    ev_times = events_sorted["time"].to_numpy(float)

    def _nearest(t):
        d = np.abs(ev_times - t)
        i = int(d.argmin())
        return i if d[i] < 0.20 else -1

    grp["event_idx"] = grp["t_mean"].apply(_nearest)

    bins = np.linspace(0, 1, 26)
    centres = 0.5 * (bins[:-1] + bins[1:])
    out = []
    for i, row in events_sorted.iterrows():
        sub = grp[grp["event_idx"] == i]
        if sub.empty:
            continue
        weights, _ = np.histogram(sub["t_mean"], bins=bins, weights=sub["ess"])
        pct = row.get("pct_of_theoretical_genome", row.get("gf", 0) * 100)
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            pct = float("nan")
        out.append(
            dict(
                y=list(centres),
                x=[float(w) for w in weights],
                color=_event_color(row),
                time=float(row["time"]),
                label=f" t={float(row['time']):.2f}  ({pct:.1f}%)",
            )
        )
    return out


# ──────────────────────────────────────────────────────────────────────────
# figure construction
# ──────────────────────────────────────────────────────────────────────────
def _build_figure(
    band_polys, boundary_lines, hit_records, track_shapes, cn_states, max_cluster_idx,
    cluster_df, hist_data, chrom_off, autosome_len, sample, cn_profile=None,
    legend_states=None, legend_nbands=None,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_track_rows = max(len(cn_states), 1)

    # ── Left CN legend columns ──────────────────────────────────────────────
    # One column per distinct COLOUR SET among the drawn states. A state's colours
    # are fully determined by (major → palette, K = #bands → #shades), so states
    # that share both collapse into a single column labelled "(maj, m1/m2/…)"
    # (e.g. (2,0) and (2,1) are both 2 greens → "(2, 0/1)"). Width adapts to the
    # column count; the main panel absorbs the difference (histogram fixed 0.14).
    legend_states = sorted(legend_states or [])
    _nbands = legend_nbands or {}
    _groups = {}   # (major, K) -> [minors]
    for (maj, mn) in legend_states:
        K = int(_nbands.get((maj, mn)) or max(int(maj), 1))
        _groups.setdefault((maj, K), []).append(mn)

    def _glabel(maj, minors):
        ms = sorted(set(minors))
        return f"({maj},{ms[0]})" if len(ms) == 1 else f"({maj},{'/'.join(map(str, ms))})"

    legend_cols = [((maj, K), _glabel(maj, mns)) for (maj, K), mns in sorted(_groups.items())]
    _legend_w = min(0.16, 0.05 + 0.008 * len(legend_cols)) if legend_cols else 0.07
    _hist_w = 0.14
    _main_w = 1.0 - _legend_w - _hist_w

    fig = make_subplots(
        rows=3,
        cols=3,
        # Top row = genome-wide CN profile, then the multiplicity-cluster track,
        # then the main panel / CN legend / histogram.
        column_widths=[_legend_w, _main_w, _hist_w],
        row_heights=[0.11, 0.15, 0.74],
        # placeholder spacing — the x-domains are set EXPLICITLY below so the two
        # column gaps can differ (a single horizontal_spacing can't do that).
        horizontal_spacing=0.012,
        vertical_spacing=0.02,
        shared_xaxes=False,
        shared_yaxes=False,
        specs=[
            [None, {}, None],
            [None, {}, None],
            [{}, {}, {}],
        ],
    )
    # Axis map after make_subplots (row-major over non-None cells):
    #   row1 col2 -> xaxis/yaxis   (genome-wide CN profile)
    #   row2 col2 -> xaxis2/yaxis2 (multiplicity-cluster track)
    #   row3 col1 -> xaxis3/yaxis3 (left timing-interval legend)
    #   row3 col2 -> xaxis4/yaxis4 (main panel)
    #   row3 col3 -> xaxis5/yaxis5 (right histogram)
    CNPROF = dict(row=1, col=2)
    TRACK = dict(row=2, col=2)
    LEGEND = dict(row=3, col=1)
    MAIN = dict(row=3, col=2)
    HIST = dict(row=3, col=3)

    # Explicit x-domains so the two column gaps differ: a WIDE gap before the main
    # panel (room for the "Molecular time" y-axis title/ticks without overlapping
    # the CN legend) but the timing-density histogram HUGS the main panel (tiny
    # gap). The CN profile + cluster track share the main panel's x-domain so they
    # stay column-aligned.
    _g1, _g2 = 0.055, 0.0                   # legend|main gap, main|hist gap (flush)
    _mw = 1.0 - _legend_w - _hist_w - _g1 - _g2
    _x_main = [_legend_w + _g1, _legend_w + _g1 + _mw]
    fig.update_xaxes(domain=[0.0, _legend_w], **LEGEND)
    fig.update_xaxes(domain=_x_main, **CNPROF)
    fig.update_xaxes(domain=_x_main, **TRACK)
    fig.update_xaxes(domain=_x_main, **MAIN)
    fig.update_xaxes(domain=[_x_main[1] + _g2, 1.0], **HIST)

    # ── MAIN PANEL: filled stacked bands ────────────────────────────────────
    for (maj, K, b), grp in sorted(band_polys.items()):
        fig.add_trace(
            go.Scatter(
                x=grp["xs"],
                y=grp["ys"],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor=grp["color"],
                hoverinfo="skip",
                showlegend=False,
                # tag the trace so the JS filter can match it back to segments.
                # ``segs``/``bands`` are aligned to the polygons of this trace
                # (one (seg_id, band) per ``None``-separated polygon in x/y) so
                # the signature toggle can recompute the y heights per segment.
                meta=dict(kind="band", major=maj, band=b, K=K, chroms=grp["chrom"],
                          segs=grp["seg"], bands=grp["band"]),
                name=f"maj{maj}_K{K}_b{b}",
            ),
            **MAIN,
        )

    # within-segment cluster-boundary dotted lines
    for colour, grp in boundary_lines.items():
        fig.add_trace(
            go.Scatter(
                x=grp["xs"], y=grp["ys"], mode="lines",
                line=dict(color=colour, width=1, dash="dot"),
                hoverinfo="skip", showlegend=False,
                name="boundary",
            ),
            **MAIN,
        )

    # event horizontal dashed lines + labels.  This is the SINGLE source of event
    # text labels (the right histogram's duplicate annotation was removed) — the
    # histogram can be empty even when events exist, whereas this iterates
    # cluster_df directly.  Labels are nudged apart vertically so closely-timed
    # events don't overlap, while each dashed line stays at its true time t.
    if cluster_df is not None and len(cluster_df):
        _ev_rows = list(cluster_df.sort_values("time").iterrows())
        _LBL_GAP = 0.035          # min vertical separation between labels (0..1)
        _last_y = -1.0
        for _, row in _ev_rows:
            colour = _event_color(row)
            t = float(row["time"])
            pct = row.get("pct_of_theoretical_genome", row.get("gf", 0) * 100)
            try:
                pct = float(pct)
            except (TypeError, ValueError):
                pct = float("nan")
            cls = row.get("classification", "")
            # dashed event line drawn across BOTH the main panel and the histogram
            # (here, not in the hist loop) so it runs unbroken to the far right even
            # when the histogram is empty; the bold label sits at that far-right end.
            fig.add_shape(
                type="line", x0=0, x1=autosome_len, y0=t, y1=t,
                line=dict(color=colour, width=1.4, dash="dash"),
                **MAIN,
            )
            fig.add_shape(
                type="line", x0=0, x1=1, y0=t, y1=t, xref="x5 domain",
                line=dict(color=colour, width=1.4, dash="dash"),
                **HIST,
            )
            ly = t if (t - _last_y) >= _LBL_GAP else _last_y + _LBL_GAP
            ly = min(ly, 1.0)
            _last_y = ly
            pct_txt = "" if pct != pct else f" ({pct:.0f}%)"
            fig.add_annotation(
                x=1.0, y=ly, xref="x5 domain", xanchor="left", yanchor="middle",
                text=f"<b>{cls} t={t:.2f}{pct_txt}</b>",
                showarrow=False, font=dict(size=11, color=colour),
                **HIST,
            )

    # chromosome boundary dashed lines + numeric labels
    tickvals, ticktext = [], []
    for c in _AUTOSOMES:
        size = HG19_SIZES.get(c, 0)
        if size == 0:
            continue
        start = chrom_off[c]
        tickvals.append(start + size / 2.0)
        ticktext.append(c)
        if start > 0:
            fig.add_shape(
                type="line", x0=start, x1=start, y0=0, y1=1,
                line=dict(color="black", width=0.5, dash="dash"),
                **MAIN,
            )

    # ── INTERACTION LAYER: transparent full-height hit bars ─────────────────
    fig.add_trace(
        go.Bar(
            x=[r["center"] for r in hit_records],
            y=[1.0] * len(hit_records),
            width=[r["width"] for r in hit_records],
            base=0,
            marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)),
            opacity=0,
            customdata=[[r["ri"], r["chrom"], r["major_cn"]] for r in hit_records],
            text=[r["hover"] for r in hit_records],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="segments",
            meta=dict(kind="hit"),
        ),
        **MAIN,
    )

    # ── SELECT LAYER: transparent scattergl for box/lasso multi-select ──────
    # Bar traces do not reliably emit plotly_selected, so overlay one ~invisible
    # marker per drawn segment at its bar center on the MAIN axes (x3/y3).  The
    # box/lasso modebar tools select these points; customdata carries each
    # segment's RECORD index (ri) for the comparison view.
    if hit_records:
        fig.add_trace(
            go.Scattergl(
                x=[r["center"] for r in hit_records],
                y=[0.5] * len(hit_records),
                mode="markers",
                marker=dict(color="rgba(0,0,0,0)", size=10, line=dict(width=0)),
                customdata=[[r["ri"]] for r in hit_records],
                hoverinfo="skip",
                showlegend=False,
                name="select-layer",
                meta=dict(kind="selhit"),
            ),
            **MAIN,
        )

    # ── TOP TRACK: multiplicity clusters ────────────────────────────────────
    # Clusters within the same CN-state row are distinguished by both a varying
    # fill SHADE (sh["color"], indexed by cluster_idx via _CLUSTER_SHADES) and a
    # cluster-index dash style for the rectangle edge (legend swatches below).
    dash_for_idx = ["solid", "dash", "dot", "dashdot", "longdash"]
    for sh in track_shapes:
        ls = dash_for_idx[min(sh["cluster_idx"] - 1, len(dash_for_idx) - 1)]
        fig.add_shape(
            type="rect",
            x0=sh["x0"], x1=sh["x1"],
            y0=sh["row"] - 0.4, y1=sh["row"] + 0.4,
            fillcolor=sh["color"],
            line=dict(color="black", width=0.7, dash=ls),
            **TRACK,
        )
    # legend swatches for cluster index (drawn as dummy scatter traces)
    for ci in range(1, max_cluster_idx + 1):
        ls = dash_for_idx[min(ci - 1, len(dash_for_idx) - 1)]
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="grey", width=2, dash=ls),
                name=f"Cluster {ci}", showlegend=True,
            ),
            **TRACK,
        )

    # ── TRACK INTERACTION LAYER: transparent hit bars (one per member rect) ──
    # Layout shapes are not clickable, so overlay an opacity-0 bar on the track
    # axes.  customdata carries each rect's payload_idx, so clicking ANY rect of
    # a cluster opens that same cluster's detail payload.
    if track_shapes:
        fig.add_trace(
            go.Bar(
                x=[(sh["x0"] + sh["x1"]) / 2.0 for sh in track_shapes],
                y=[0.8] * len(track_shapes),
                width=[(sh["x1"] - sh["x0"]) for sh in track_shapes],
                base=[sh["row"] - 0.4 for sh in track_shapes],
                marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)),
                opacity=0,
                customdata=[[sh["payload_idx"]] for sh in track_shapes],
                hovertemplate="cluster %{customdata[0]} — click to inspect<extra></extra>",
                showlegend=False,
                name="cluster_hits",
                meta=dict(kind="cluster_hit"),
            ),
            **TRACK,
        )

    # ── GENOME-WIDE CN PROFILE: major / minor copy-number step lines ─────────
    if cn_profile and cn_profile.get("maj_x"):
        # minor drawn AFTER major so it stays visible where they coincide
        # (balanced regions); LOH shows the minor line dropping to 0.
        fig.add_trace(
            go.Scatter(
                x=cn_profile["maj_x"], y=cn_profile["maj_y"], mode="lines",
                customdata=cn_profile["maj_cd"],
                line=dict(color="#cb3e3e", width=1.7), connectgaps=False,
                name="major CN", legendgroup="cn", showlegend=True,
                hovertemplate="major CN = %{customdata}<extra></extra>",
            ),
            **CNPROF,
        )
        fig.add_trace(
            go.Scatter(
                x=cn_profile["min_x"], y=cn_profile["min_y"], mode="lines",
                customdata=cn_profile["min_cd"],
                line=dict(color="#1072b2", width=1.4), connectgaps=False,
                name="minor CN", legendgroup="cn", showlegend=True,
                hovertemplate="minor CN = %{customdata}<extra></extra>",
            ),
            **CNPROF,
        )

    # ── LEFT CN LEGEND: one column per distinct colour set (see legend_cols) ──
    # Each shows the EXACT band colours used on the main panel — K shades from the
    # major's palette via `_colors_for_segment` (CN=1 → flat grey). Bottom = early
    # (light) → top = late (dark). Colour-identical states are merged.
    for i, ((maj, K), _label) in enumerate(legend_cols):
        cmap = _cmap_for_major(maj)
        cols = _colors_for_segment(int(maj), int(K), cmap)
        for j, c in enumerate(cols):
            fig.add_shape(
                type="rect",
                x0=i - 0.45, x1=i + 0.45,
                y0=j / K, y1=(j + 1) / K,
                fillcolor=_rgba_to_css(c),
                line=dict(width=0),
                **LEGEND,
            )

    # ── RIGHT HISTOGRAM: ESS-weighted timing bars ───────────────────────────
    for h in hist_data:
        fig.add_trace(
            go.Bar(
                x=h["x"], y=h["y"], orientation="h",
                width=(1.0 / 25) * 0.92,
                marker=dict(color=h["color"], line=dict(color="white", width=0.3)),
                opacity=0.7, showlegend=False, hoverinfo="skip",
            ),
            **HIST,
        )
        # (event dashed line + bold far-right label are drawn from cluster_df above,
        # spanning main panel + histogram, so they exist even if hist_data is empty.)

    # ── AXES ────────────────────────────────────────────────────────────────
    # main panel
    fig.update_xaxes(
        range=[0, autosome_len], showgrid=False, zeroline=False,
        tickvals=tickvals, ticktext=ticktext, tickfont=dict(size=8),
        title_text="Chromosome", title_font=dict(size=9), **MAIN,
    )
    fig.update_yaxes(
        range=[0, 1], showgrid=False, zeroline=False,
        title_text="Molecular time", title_font=dict(size=10),
        tickfont=dict(size=9), **MAIN,
    )

    # cluster track (shares main x-range). matches="x4" links the track x-axis
    # to the MAIN panel x-axis (xaxis4) so zoom/pan move them together over the
    # same genomic window; both already use the [0, autosome_len] coordinate
    # system so chromosome boundaries + cluster rects stay registered.
    fig.update_xaxes(
        range=[0, autosome_len], showgrid=False, zeroline=False,
        showticklabels=False, matches="x4", **TRACK,
    )
    fig.update_yaxes(
        range=[-0.5, n_track_rows - 0.5],
        tickvals=list(range(n_track_rows)),
        ticktext=[f"({maj},{mn})" for maj, mn in cn_states] or [""],
        tickfont=dict(size=7), showgrid=False, zeroline=False, **TRACK,
    )

    # genome-wide CN profile (top row; shares the main panel's x-range so it
    # zooms/pans together via matches="x4").
    fig.update_xaxes(
        range=[0, autosome_len], showgrid=False, zeroline=False,
        showticklabels=False, matches="x4", **CNPROF,
    )
    _cap = (cn_profile or {}).get("cap", 6)
    _ov = (cn_profile or {}).get("overflow", _cap + 1)
    fig.update_yaxes(
        range=[-0.4, _ov + 0.5], tickfont=dict(size=7),
        tickmode="array", tickvals=list(range(0, _ov + 1)),
        ticktext=[str(i) for i in range(0, _cap + 1)] + [f"{_ov}+"],
        title_text="CN", title_font=dict(size=9),
        showgrid=True, gridcolor="#eef0f3", zeroline=False, **CNPROF,
    )

    # left CN legend (timing-interval colour ramp) — one labelled column per
    # (major,minor) state; vertical labels + tiny font once there are many.
    _n_cols = len(legend_cols)
    fig.update_xaxes(
        range=[-0.5, max(_n_cols - 0.5, 0.5)],
        tickvals=list(range(_n_cols)),
        ticktext=[lab for _k, lab in legend_cols],
        tickangle=(-90 if _n_cols > 10 else -45),
        tickfont=dict(size=(5 if _n_cols > 10 else 6)),
        showgrid=False, zeroline=False, side="bottom", **LEGEND,
    )
    fig.update_yaxes(
        range=[0, 1], tickvals=[0.1, 0.9], ticktext=["t_early", "t_late"],
        tickfont=dict(size=7), side="left", showgrid=False, zeroline=False,
        **LEGEND,
    )

    # right histogram (shares the 0..1 time range)
    fig.update_xaxes(
        range=[0, None], showticklabels=False, showgrid=False, zeroline=False,
        title_text="Timing density", title_font=dict(size=8), **HIST,
    )
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False,
                     zeroline=False, **HIST)

    title = "Tau timing overview"
    if sample:
        title += f" — {sample}"
    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0.01, xanchor="left", y=0.99),
        template="plotly_white",
        height=580,
        bargap=0,
        # wider right margin so the WGD/PGD histogram event labels (anchored near
        # the right edge of the timing-density column) aren't clipped; a touch
        # more left margin for the t_early/t_late legend now on the outer side.
        margin=dict(l=78, r=140, t=86, b=50),
        font=dict(size=9),
        legend=dict(orientation="h", yanchor="bottom", y=1.045, xanchor="right",
                    x=0.86, font=dict(size=8),
                    title_text="Multiplicity clusters"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# client-side JS (detail panel + filters)
# ──────────────────────────────────────────────────────────────────────────
_POST_SCRIPT = r"""
(function() {
  var gd = document.getElementById('__DIV_ID__');
  if (!gd) { return; }
  var RECORDS = __RECORDS_JSON__;
  var HITS = __HITS_JSON__;
  var CLUSTERS = __CLUSTERS_JSON__;
  var TREES = __TREES_JSON__;        // {ri: {route_key: payload}}
  var EXPANDABLE = __EXPANDABLE_JSON__;  // {ri: [route_key, ...]}
  var SPECTRUM = __SPECTRUM_JSON__;  // SBS96 selector meta + channel layout
  var TREE_EVENTS = __EVENTS_JSON__;  // [{t, cls:WGD/PGD, gf}] landmark lines on trees
  var CNSTATE_VAF = __CNSTATE_VAF_JSON__;  // per-CN-state pooled VAF histograms
  var SIG_TIMING = __SIG_TIMING_JSON__;  // {signatures, segment_times, activity} or null
  // SIG_BANDS[sig][traceName] = {x:[...], y:[...]} — prebuilt full band polygons
  // per signature, IDENTICAL geometry to the clock overview (same shared Python
  // helper), keyed by band-trace name "maj{major}_b{band}".  Empty when off.
  var SIG_BANDS = __SIG_BANDS_JSON__;
  // SEG_ACTIVITY[sig][seg_id] = {abund, rows:[{interval, clock_time, ratio, ct,
  // st, ok}]} — PER-SEGMENT signature-activity vs clock, ALL intervals (ok flags
  // determinacy).  Drives the click-detail panel independent of the band toggle.
  var SEG_ACTIVITY = __SEG_ACTIVITY_JSON__;
  var UID = 0;  // unique-id counter for inline Plotly mount points

  // ---- expected-VAF lines ----------------------------------------------
  // Tau's per-multiplicity expected VAF: p_m = m*purity / (total_cn*purity +
  // 2*(1-purity)) for m = 1..major_cn (exactly the p_j used in tau/core.py).
  // Returns {shapes, annotations} to overlay on a VAF histogram, or empty
  // arrays when purity/total_cn are out of range.
  function expectedVafOverlay(purity, total_cn, major_cn) {
    var shapes = [], anns = [];
    if (purity == null || !(purity > 0) || !(purity <= 1) || !(total_cn > 0)) {
      return {shapes: shapes, annotations: anns};
    }
    var denom = total_cn * purity + 2 * (1 - purity);
    if (!(denom > 0)) { return {shapes: shapes, annotations: anns}; }
    var M = Math.max(1, major_cn || 1);
    for (var m = 1; m <= M; m++) {
      var pm = m * purity / denom;
      if (!(pm >= 0 && pm <= 1)) { continue; }
      shapes.push({type:'line', xref:'x', yref:'paper', x0:pm, x1:pm,
        y0:0, y1:1, line:{color:'#444', width:1, dash:'dash'}});
      anns.push({x:pm, y:1.0, xref:'x', yref:'paper', text:'×'+m,
        showarrow:false, yanchor:'bottom', xanchor:'center',
        font:{size:9, color:'#444'}});
    }
    return {shapes: shapes, annotations: anns};
  }

  // ---- SBS96 mutational-spectrum panel ---------------------------------
  // `spectra` (on a segment or cluster record) maps each selector-option key to
  // a length-96 array of weighted channel sums (same order as SPECTRUM.channels).
  // Renders a 96-bar profile + a <select> weighting toggle into `host`.
  function renderSpectrumInto(host, spectra) {
    if (!SPECTRUM || !SPECTRUM.enabled || !spectra) { return; }
    var opts = SPECTRUM.options.filter(function(o){ return spectra[o.key]; });
    if (!opts.length) { return; }
    var wrap = document.createElement('div');
    wrap.style.cssText = 'margin-top:18px;max-width:1000px;';
    var ctrl = document.createElement('div');
    ctrl.style.cssText = 'font-size:13px;margin-bottom:4px;';
    var sel = document.createElement('select');
    sel.style.cssText = 'margin-left:8px;padding:2px 6px;';
    opts.forEach(function(o){
      var op = document.createElement('option');
      op.value = o.key; op.textContent = o.label;
      sel.appendChild(op);
    });
    var def = SPECTRUM.default;
    if (!opts.some(function(o){ return o.key === def; })) { def = opts[0].key; }
    sel.value = def;
    var b = document.createElement('b'); b.textContent = 'SBS96 weighting:';
    ctrl.appendChild(b); ctrl.appendChild(sel);
    var note = document.createElement('span');
    note.style.cssText = 'margin-left:12px;color:#777;';
    ctrl.appendChild(note);
    wrap.appendChild(ctrl);
    var plotId = 'tau-sbs96-' + (UID++);
    var plotDiv = document.createElement('div');
    plotDiv.id = plotId;
    plotDiv.style.cssText = 'width:980px;max-width:100%;height:260px;';
    wrap.appendChild(plotDiv);
    host.appendChild(wrap);

    function shareTxt(o){
      if (o.share == null) { return ''; }
      return '  ·  sample share ' + (100*o.share).toFixed(1) + '%';
    }
    function render(key){
      var o = opts.filter(function(x){ return x.key === key; })[0] || opts[0];
      var raw = spectra[key] || [];
      var tot = 0; for (var i=0;i<raw.length;i++){ tot += raw[i]; }
      var frac = raw.map(function(v){ return tot>0 ? v/tot : 0; });
      var xs = SPECTRUM.channels;
      // trinucleotide context per channel for the x ticks: "A[C>A]A" -> "ACA"
      // (5' flank + ref base + 3' flank).  Full channel name stays on hover.
      var ctx = xs.map(function(c){
        return (c.length >= 7) ? (c.charAt(0)+c.charAt(2)+c.charAt(6)) : c;
      });
      var idx = xs.map(function(_, i){ return i; });
      var trace = {
        x: idx,
        y: frac, type: 'bar',
        marker: { color: SPECTRUM.chan_colors },
        hovertext: xs, hovertemplate: '%{hovertext}<br>%{y:.3f}<extra></extra>',
        width: 0.82
      };
      // 6-type colour bands above the axis (annotations as substitution labels).
      var shapes = [], anns = [];
      for (var t=0; t<SPECTRUM.sub_types.length; t++){
        var x0 = t*16 - 0.5, x1 = t*16 + 15.5;
        shapes.push({type:'rect', xref:'x', yref:'paper', x0:x0, x1:x1,
          y0:1.0, y1:1.05, fillcolor:SPECTRUM.sub_colors[SPECTRUM.sub_types[t]],
          line:{width:0}});
        anns.push({x:(x0+x1)/2, y:1.075, xref:'x', yref:'paper',
          text:SPECTRUM.sub_types[t], showarrow:false,
          font:{size:11, color:'#333'}});
      }
      var lab = SPECTRUM.soft_path ? 'soft posterior weighting'
                                   : 'hard-assignment (validation gate fell back)';
      Plotly.newPlot(plotId, [trace], {
        title:{text:'SBS96 spectrum — '+o.label+shareTxt(o), font:{size:12}},
        margin:{l:46,r:10,t:46,b:54},
        xaxis:{tickmode:'array', tickvals:idx, ticktext:ctx, tickangle:90,
               tickfont:{size:6, family:'monospace'}, range:[-0.5, 95.5],
               fixedrange:true},
        yaxis:{title:'fraction', rangemode:'tozero', fixedrange:true},
        shapes:shapes, annotations:anns, height:280, bargap:0.1
      }, {displayModeBar:false, responsive:true});
      note.textContent = lab + (o.kind==='clock'
        ? '  (exact mut_w)' : (o.kind==='unweighted' ? '  (every SNV = 1)' : ''));
    }
    sel.onchange = function(){ render(sel.value); };
    render(def);
  }

  // ---- interactive "tree solution with knobs" (ported from the proof) -------
  // Renders one route's allele tree + one slider per free variable into `host`.
  // All recomputation is plain arithmetic on affine coefficients (PAYLOAD), so
  // dialing a knob re-clamps the others and never leaves the polytope.
  function renderTreeInto(host, PAYLOAD, opts) {
    host.innerHTML = '';
    opts = opts || {};
    // opts.treeW/treeH shrink the SVG; opts.compact drops the interval/event
    // tables (keeps tree + knobs) and narrows the knob panel — used by the
    // cluster-timeline section to fit more trees side by side.
    var W=opts.treeW||300, H=opts.treeH||300;
    var PADX=opts.compact?30:40, PADT=opts.compact?14:20, PADB=opts.compact?20:26;
    var TOP=PADT, BOT=H-PADB;
    function yOf(t){ return BOT - t*(BOT-TOP); }
    function evalAffine(m, free){ var v=m.const; for(var k=0;k<free.length;k++) v+=m.coef[k]*free[k]; return v; }
    function feasibleInterval(ki, free){
      var lo=-Infinity, hi=Infinity;
      var cons = PAYLOAD.constraints.bounds.concat(PAYLOAD.constraints.interval_nonneg);
      for(var i=0;i<cons.length;i++){ var c=cons[i];
        var base=c.const; for(var k=0;k<free.length;k++){ if(k!==ki) base+=c.coef[k]*free[k]; }
        var a=c.coef[ki];
        if(Math.abs(a)<1e-12){ continue; }
        if(a>0){ lo=Math.max(lo, -base/a); } else { hi=Math.min(hi, -base/a); }
      }
      return [lo,hi];
    }
    function clampAll(free){
      for(var k=0;k<free.length;k++){
        var iv=feasibleInterval(k,free);
        if(isFinite(iv[0])) free[k]=Math.max(free[k],iv[0]);
        if(isFinite(iv[1])) free[k]=Math.min(free[k],iv[1]);
      }
      return free;
    }
    function isFeasible(free){
      var cons = PAYLOAD.constraints.bounds.concat(PAYLOAD.constraints.interval_nonneg);
      for(var i=0;i<cons.length;i++){ if(evalAffine(cons[i],free) < -1e-6) return false; }
      return true;
    }
    var maj=PAYLOAD.nodes.filter(function(n){return n.allele==='major';});
    var min=PAYLOAD.nodes.filter(function(n){return n.allele==='minor';});
    var order=maj.concat(min);
    var XS={};
    order.forEach(function(n,i){ XS[n.index]= PADX + (i+0.5)*((W-2*PADX)/Math.max(order.length,1)); });
    var COL="#0b6979";
    function nodeHeight(n, free){
      if(n.event_index<0) return 0.0;
      var v=evalAffine(PAYLOAD.event_time_maps[n.event_index], free);
      return Math.max(0, Math.min(1, v));
    }
    function drawTree(treeDiv, free){
      var ns=PAYLOAD.nodes;
      var svg='<svg width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'">';
      svg+='<line x1="'+(PADX-14)+'" y1="'+TOP+'" x2="'+(PADX-14)+'" y2="'+BOT+'" stroke="#c5cbd4"/>';
      if(opts.axisLabels!==false){
        svg+='<text x="'+(PADX-18)+'" y="'+(TOP+4)+'" font-size="10" fill="#8a93a3" text-anchor="end">late 1</text>';
        svg+='<text x="'+(PADX-18)+'" y="'+BOT+'" font-size="10" fill="#8a93a3" text-anchor="end">early 0</text>';
      }
      // WGD/PGD landmark lines (compare gain heights to these large-scale events)
      (TREE_EVENTS||[]).forEach(function(ev){
        var ey=yOf(Math.max(0,Math.min(1,ev.t)));
        var ecol = ev.cls==='WGD' ? '#0b6979' : '#cd9f2c';
        svg+='<line x1="'+PADX+'" y1="'+ey+'" x2="'+(W-2)+'" y2="'+ey+'" stroke="'+ecol+'" stroke-width="1.1" stroke-dasharray="5 3" stroke-opacity="0.9"/>';
        svg+='<text x="'+(W-3)+'" y="'+(ey-2)+'" font-size="9" fill="'+ecol+'" text-anchor="end">'+ev.cls+' '+ev.t.toFixed(2)+'</text>';
      });
      var sols=PAYLOAD.sampled_solutions||[];
      for(var s=0;s<sols.length;s++){ var fv=sols[s];
        for(var i=0;i<ns.length;i++){ var n=ns[i]; if(n.parent_index<0) continue;
          var x=XS[n.index], h=nodeHeight(n,fv);
          svg+='<line x1="'+x+'" y1="'+yOf(h)+'" x2="'+x+'" y2="'+yOf(1)+'" stroke="'+COL+'" stroke-opacity="0.07" stroke-width="2"/>';
        }
      }
      for(var i=0;i<ns.length;i++){ var n=ns[i];
        var x=XS[n.index]; var h=nodeHeight(n,free);
        var dash = n.allele==='minor' ? 'stroke-dasharray="5 4"' : '';
        svg+='<line x1="'+x+'" y1="'+yOf(h)+'" x2="'+x+'" y2="'+yOf(1)+'" stroke="'+COL+'" stroke-width="2.4" '+dash+'/>';
        if(n.parent_index>=0){ var px=XS[n.parent_index];
          svg+='<line x1="'+x+'" y1="'+yOf(h)+'" x2="'+px+'" y2="'+yOf(h)+'" stroke="'+COL+'" stroke-width="2.4" '+dash+'/>';
        }
      }
      svg+='</svg>';
      treeDiv.innerHTML=svg;
    }

    // layout: tree on the left, knobs + tables on the right
    var wrap=document.createElement('div');
    wrap.style.cssText = opts.stack
      ? 'display:flex;flex-direction:column;gap:6px;align-items:flex-start;'
      : 'display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;';
    var treeDiv=document.createElement('div');
    var right=document.createElement('div');
    right.style.cssText='min-width:'+(opts.stack?W:260)+'px;font-size:12px;';
    wrap.appendChild(treeDiv); wrap.appendChild(right);
    host.appendChild(wrap);
    var leg=document.createElement('div');
    leg.style.cssText='font-size:11px;color:#555;margin-top:8px;clear:both;';
    leg.innerHTML='<span style="color:#0b6979">&#9472; major</span>'
      + '&nbsp;&nbsp;<span style="color:#0b6979">&#8211;&#8211; minor</span>'
      + '&nbsp;&nbsp;&middot; faint = polytope solutions';
    host.appendChild(leg);

    var free = PAYLOAD.init_free.slice();
    clampAll(free);
    var _S=(PAYLOAD.seg&&PAYLOAD.seg.total_sum)||1;   // raw free-var counts -> molecular-time fractions

    if(PAYLOAD.free_vars.length===0){
      right.innerHTML='';
      leg.innerHTML += '&nbsp;&nbsp;&middot; <b>determined</b>';
      drawTree(treeDiv, free);
      return;
    }

    var knobsDiv=document.createElement('div');
    var feasDiv=document.createElement('div');
    feasDiv.style.cssText='margin:6px 0 10px;';
    var ttblDiv=document.createElement('div');
    var evtblDiv=document.createElement('div');
    evtblDiv.style.cssText='margin-top:8px;';
    right.appendChild(knobsDiv); right.appendChild(feasDiv);
    if(!opts.compact){ right.appendChild(ttblDiv); right.appendChild(evtblDiv); }

    var khtml='';
    PAYLOAD.free_vars.forEach(function(name,k){
      // compact (cluster timeline): slider only — hide the value/range numbers that
      // clutter the small stacked trees.
      var rngSpan = opts.compact ? '' :
        ' <span style="font-weight:400;color:#8a93a3" id="rng'+UID+'_'+k+'"></span>';
      var vvSpan = opts.compact ? '' :
        '<span id="vv'+UID+'_'+k+'" style="min-width:58px;text-align:right;'
        + 'font-variant-numeric:tabular-nums;color:#2d3340;"></span>';
      khtml+='<div style="margin:8px 0 12px;">'
        + '<label style="display:block;font-weight:600;margin-bottom:3px;">'
        + name + rngSpan + '</label>'
        + '<div style="display:flex;align-items:center;gap:8px;">'
        + '<input type="range" id="kn'+UID+'_'+k+'" style="flex:1;"/>'
        + vvSpan + '</div></div>';
    });
    var myUID=UID;
    knobsDiv.innerHTML=khtml;

    function syncSlider(k){
      var iv=feasibleInterval(k,free);
      var L=isFinite(iv[0])?iv[0]:0, Hh=isFinite(iv[1])?iv[1]:1;
      var sl=document.getElementById('kn'+myUID+'_'+k);
      sl.min=L; sl.max=Hh; sl.step=Math.max((Hh-L)/1000,1e-6); sl.value=free[k];
      var vv=document.getElementById('vv'+myUID+'_'+k);
      if(vv) vv.textContent=(free[k]/_S).toFixed(4);
      var rng=document.getElementById('rng'+myUID+'_'+k);
      if(rng) rng.textContent='['+(L/_S).toFixed(3)+', '+(Hh/_S).toFixed(3)+']';
    }
    function refresh(){
      drawTree(treeDiv, free);
      PAYLOAD.free_vars.forEach(function(_,k){ syncSlider(k); });
      if(!opts.compact){
      var rows='';
      PAYLOAD.t_vars.forEach(function(tn){
        var v=evalAffine(PAYLOAD.interval_maps[tn], free);
        rows+='<tr><td style="padding:2px 8px">'+tn+'</td>'
          + '<td style="padding:2px 8px;text-align:right">'+v.toFixed(4)+'</td></tr>';
      });
      ttblDiv.innerHTML='<table style="border-collapse:collapse;font-size:12px;"><thead>'
        + '<tr><th style="padding:2px 8px;text-align:left;color:#5a6473">interval</th>'
        + '<th style="padding:2px 8px;text-align:right;color:#5a6473">t (fraction)</th></tr>'
        + '</thead><tbody>'+rows+'</tbody></table>';
      var ev='<table style="border-collapse:collapse;font-size:12px;"><thead><tr>'
        + '<th style="padding:2px 8px;text-align:left;color:#5a6473">event</th>'
        + '<th style="padding:2px 8px;text-align:right;color:#5a6473">cumulative time</th></tr></thead><tbody>';
      PAYLOAD.events.forEach(function(e,j){
        var t=evalAffine(PAYLOAD.event_time_maps[j], free);
        ev+='<tr><td style="padding:2px 8px">'+(j+1)+'</td>'
          + '<td style="padding:2px 8px;text-align:right">'+t.toFixed(4)+'</td></tr>';
      });
      ev+='</tbody></table>';
      evtblDiv.innerHTML=ev;
      }
      if(!opts.compact){
        var ok=isFeasible(free);
        feasDiv.innerHTML='Feasible: <span style="display:inline-block;padding:2px 9px;'
          + 'border-radius:11px;font-size:12px;font-weight:600;'
          + (ok?'background:#e3f5ea;color:#1f8b4c;">yes':'background:#fdecea;color:#c0392b;">no')
          + '</span>';
      }
    }
    PAYLOAD.free_vars.forEach(function(_,k){
      document.getElementById('kn'+myUID+'_'+k).addEventListener('input', function(e){
        free[k]=parseFloat(e.target.value);
        clampAll(free);
        refresh();
      });
    });
    UID++;  // reserve this UID block for the knob element ids above
    refresh();
  }

  // ---- detail panel ----------------------------------------------------
  var panel = document.createElement('div');
  panel.id = 'tau-detail';
  panel.style.cssText = 'font-family:sans-serif;max-width:1100px;margin:16px auto;'
    + 'padding:12px;border-top:2px solid #ddd;color:#222;';
  panel.innerHTML = '<em style="color:#888">Click a segment (or a top-track '
    + 'cluster) for details.</em>';
  gd.parentNode.insertBefore(panel, gd.nextSibling);

  // ---- per-segment spatial distribution (rainfall <-> VAF-vs-position) ------
  // `rec.spatial` carries compact parallel arrays (pos offsets from seg.start,
  // vaf, mult, t6 = SBS6 type index 0..5).  Renders a toggle + an inline Plotly
  // subplot.  Rainfall: x = abs position, y = log10(distance to previous
  // mutation), coloured by SBS6 type.  VAF-vs-position: x = abs position,
  // y = VAF, coloured by multiplicity, with dashed expected-VAF reference lines.
  var _SBS6_TYPES = (SPECTRUM && SPECTRUM.sub_types)
    ? SPECTRUM.sub_types
    : ['C>A','C>G','C>T','T>A','T>C','T>G'];
  var _SBS6_COLORS = (SPECTRUM && SPECTRUM.sub_colors)
    ? SPECTRUM.sub_colors
    : {'C>A':'#03bcee','C>G':'#000000','C>T':'#e32925',
       'T>A':'#cac9c9','T>C':'#a1ce63','T>G':'#ebc6c4'};

  function renderSpatialInto(host, rec) {
    var sp = rec.spatial;
    if (!sp || !sp.pos || !sp.pos.length) { return; }
    var wrap = document.createElement('div');
    wrap.style.cssText = 'margin-top:18px;max-width:1000px;';
    var ctrl = document.createElement('div');
    ctrl.style.cssText = 'font-size:13px;margin-bottom:4px;';
    var b = document.createElement('b'); b.textContent = 'Spatial distribution:';
    var sel = document.createElement('select');
    sel.style.cssText = 'margin-left:8px;padding:2px 6px;';
    [['rainfall','Rainfall (inter-mutation distance)'],
     ['vaf','VAF vs position']].forEach(function(o){
      var op = document.createElement('option');
      op.value = o[0]; op.textContent = o[1];
      sel.appendChild(op);
    });
    var note = document.createElement('span');
    note.style.cssText = 'margin-left:12px;color:#777;font-size:12px;';
    note.textContent = sp.n_total + ' SNVs'
      + (sp.capped ? '  (showing ' + sp.pos.length
                     + ' evenly subsampled, cap ' + sp.pos.length + ')' : '');
    ctrl.appendChild(b); ctrl.appendChild(sel); ctrl.appendChild(note);
    wrap.appendChild(ctrl);

    var plotId = 'tau-spatial-' + (UID++);
    var plotDiv = document.createElement('div');
    plotDiv.id = plotId;
    plotDiv.style.cssText = 'width:720px;max-width:100%;height:280px;';
    wrap.appendChild(plotDiv);
    host.appendChild(wrap);

    var absPos = sp.pos.map(function(p){ return sp.start + p; });

    function renderRainfall() {
      // distance to previous mutation by sorted position (arrays are sorted).
      var traces = [];
      for (var t = 0; t < _SBS6_TYPES.length; t++) {
        var xs = [], ys = [];
        for (var i = 1; i < absPos.length; i++) {
          if (sp.t6[i] !== t) { continue; }
          var d = absPos[i] - absPos[i-1];
          if (!(d > 0)) { d = 1; }
          xs.push(absPos[i]); ys.push(Math.log10(d));
        }
        traces.push({
          x: xs, y: ys, mode: 'markers', type: 'scattergl',
          name: _SBS6_TYPES[t],
          marker: {color: _SBS6_COLORS[_SBS6_TYPES[t]], size: 5, opacity: 0.8},
          hovertemplate: _SBS6_TYPES[t] + '<br>pos %{x}<br>log10(dist) %{y:.2f}<extra></extra>'
        });
      }
      Plotly.newPlot(plotId, traces, {
        title: {text:'Rainfall — inter-mutation distance', font:{size:12}},
        margin:{l:50,r:10,t:34,b:36},
        xaxis:{title:'genomic position (bp)'},
        yaxis:{title:'log10(distance to prev)'},
        legend:{font:{size:9}, orientation:'h', y:1.12},
        height:280, showlegend:true
      }, {displayModeBar:false, responsive:true});
    }

    function renderVafPos() {
      var xs = [], ys = [], cs = [], tx = [];
      for (var i = 0; i < absPos.length; i++) {
        if (sp.vaf[i] == null) { continue; }
        xs.push(absPos[i]); ys.push(sp.vaf[i]);
        cs.push(sp.mult[i] == null ? 0 : sp.mult[i]);
        tx.push('×' + (sp.mult[i] == null ? '?' : sp.mult[i]));
      }
      var trace = {
        x: xs, y: ys, mode: 'markers', type: 'scattergl',
        marker: {color: cs, colorscale:'Viridis', size:6, opacity:0.85,
                 showscale:true, colorbar:{title:'mult', thickness:10, len:0.8}},
        text: tx,
        hovertemplate: 'pos %{x}<br>VAF %{y:.3f}<br>%{text}<extra></extra>'
      };
      // expected-VAF dashed reference lines p_m = m*purity/denom (m=1..major_cn).
      var ov = expectedVafOverlay(rec.purity, rec.total_cn, rec.major_cn);
      var shapes = ov.shapes.map(function(s){
        return {type:'line', xref:'paper', yref:'y', x0:0, x1:1,
                y0:s.x0, y1:s.x0, line:{color:'#444', width:1, dash:'dash'}};
      });
      var anns = ov.annotations.map(function(a){
        return {x:1.0, y:a.x, xref:'paper', yref:'y', text:a.text,
                showarrow:false, xanchor:'left', yanchor:'middle',
                font:{size:9, color:'#444'}};
      });
      Plotly.newPlot(plotId, [trace], {
        title: {text:'VAF vs position', font:{size:12}},
        margin:{l:50,r:50,t:34,b:36},
        xaxis:{title:'genomic position (bp)'},
        yaxis:{title:'VAF', range:[0,1]},
        shapes:shapes, annotations:anns, height:280
      }, {displayModeBar:false, responsive:true});
    }

    function render(kind){ if (kind === 'vaf') { renderVafPos(); } else { renderRainfall(); } }
    sel.onchange = function(){ render(sel.value); };
    render('rainfall');  // default to Rainfall
  }

  // Render one segment's detail (mult bar, VAF hist, route table) into `host`.
  // Returns nothing; mounts inline Plotly charts using fresh element ids.
  function renderSegmentDetailInto(host, rec, opts) {
    opts = opts || {};
    host.innerHTML = '';
    if (!opts.noHeader) {
      var h = document.createElement('h3');
      h.textContent = rec.seg_id + '   CN (' + rec.major_cn + ',' + rec.minor_cn
        + ')   ' + rec.n_mut + ' mutations';
      h.style.margin = '4px 0';
      host.appendChild(h);
    }

    var multId = 'tau-mult-' + (UID++);
    var vafId = 'tau-vaf-' + (UID++);
    var charts = document.createElement('div');
    charts.style.cssText = 'display:flex;flex-wrap:wrap;gap:24px;';
    var c1 = document.createElement('div'); c1.id = multId;
    c1.style.cssText = 'width:340px;height:240px;';
    var c2 = document.createElement('div'); c2.id = vafId;
    c2.style.cssText = 'width:340px;height:240px;';
    charts.appendChild(c2); charts.appendChild(c1);  // VAF left, multiplicity right
    host.appendChild(charts);

    if (rec.props && rec.props.length) {
      var xs = rec.props.map(function(_, i){ return '×' + (i+1); });
      Plotly.newPlot(multId, [{
        x: xs, y: rec.props, type: 'bar', marker: {color: '#4c78a8'}
      }], {
        title: {text:'Multiplicity', font:{size:12}},
        margin:{l:40,r:10,t:30,b:30}, yaxis:{range:[0,1]}, height:240
      }, {displayModeBar:false, responsive:true});
    } else { c1.innerHTML = '<em style="color:#888">no multiplicity data</em>'; }

    if (rec.vaf && rec.vaf.length) {
      var ov = expectedVafOverlay(rec.purity, rec.total_cn, rec.major_cn);
      Plotly.newPlot(vafId, [{
        x: rec.vaf, type: 'histogram',
        xbins:{start:0,end:1,size:0.025}, marker:{color:'#e45756'}
      }], {
        title:{text:'VAF', font:{size:12}},
        margin:{l:40,r:10,t:30,b:30}, xaxis:{range:[0,1], title:'VAF'}, height:240,
        shapes:ov.shapes, annotations:ov.annotations
      }, {displayModeBar:false, responsive:true});
    } else { c2.innerHTML = '<em style="color:#888">no VAF data</em>'; }

    // Which routes of this segment have a precomputed interactive tree?
    var riKey = (rec.ri != null) ? String(rec.ri) : null;
    var treeMap = (riKey != null && TREES[riKey]) ? TREES[riKey] : {};
    var expandable = (riKey != null && EXPANDABLE[riKey]) ? EXPANDABLE[riKey] : [];
    var anyTree = expandable.length > 0;

    var tbl = document.createElement('table');
    tbl.style.cssText = 'border-collapse:collapse;margin-top:14px;font-size:13px;width:100%;max-width:980px;';
    var thead = '<tr style="background:#f2f2f2;text-align:left">'
      + '<th style="padding:4px 10px">route</th>'
      + '<th style="padding:4px 10px" title="Euclidean distance from the route’s '
        + 'feasibility boundary (0 = exactly feasible; larger = counts projected '
        + 'further onto the constraint region)">distance</th>'
      + '<th style="padding:4px 10px" title="N-inequalities the raw counts violated">violated</th>'
      + '<th style="padding:4px 10px">free vars</th>'
      + '<th style="padding:4px 10px" title="point if determined, [min,max] if underdetermined">event time(s)</th>'
      + '</tr>';
    var tb = document.createElement('table');
    tb.style.cssText = tbl.style.cssText;
    tb.innerHTML = '<thead>' + thead + '</thead>';
    var tbody = document.createElement('tbody');
    tb.appendChild(tbody);

    rec.route_rows.forEach(function(r){
      var hasTree = (expandable.indexOf(r.key) >= 0);
      var bg = r.is_best ? 'background:#fff7e6;' : '';
      var arrow = hasTree ? '<span style="color:#0b6979">&#9656; </span>' : '';
      var name = arrow + r.key + (r.is_best ? ' <span style="color:#cc8400">(best)</span>' : '');
      var tr = document.createElement('tr');
      // rows beyond the precomputed top-K (when trees are enabled at all) are
      // dimmed + non-expandable.
      var dim = (anyTree && !hasTree) ? 'color:#aaa;' : '';
      tr.style.cssText = bg + dim + 'border-top:1px solid #eee;'
        + (hasTree ? 'cursor:pointer;' : '');
      var condStr = (r.conditions_violated && r.conditions_violated.length)
        ? r.conditions_violated.join('; ') : '—';
      tr.innerHTML =
          '<td style="padding:4px 10px">' + name + '</td>'
        + '<td style="padding:4px 10px">' + (r.dist_rel==null?'NA':r.dist_rel) + '</td>'
        + '<td style="padding:4px 10px;font-size:12px;color:#9d2825">' + condStr + '</td>'
        + '<td style="padding:4px 10px">' + (r.free_var_count==null?'NA':r.free_var_count) + '</td>'
        + '<td style="padding:4px 10px">' + (r.intervals.join(',  ') || '—')
        + (anyTree && !hasTree ? ' <span style="font-size:11px;color:#bbb">(no tree)</span>' : '')
        + '</td>';
      tbody.appendChild(tr);

      if (hasTree) {
        var detailTr = document.createElement('tr');
        var td = document.createElement('td');
        td.colSpan = 5;
        td.style.cssText = 'padding:0;background:#fafbfc;';
        var body = document.createElement('div');
        body.style.cssText = 'display:none;padding:10px 14px;border-top:1px solid #eef0f3;';
        td.appendChild(body);
        detailTr.appendChild(td);
        tbody.appendChild(detailTr);
        var loaded = false;
        tr.onclick = function(){
          var open = (body.style.display === 'none');
          body.style.display = open ? 'block' : 'none';
          tr.querySelector('td span') &&
            (tr.querySelector('td span').innerHTML = open ? '&#9662; ' : '&#9656; ');
          if (open && !loaded){ renderTreeInto(body, treeMap[r.key]); loaded = true; }
        };
      }
    });
    var cap = document.createElement('div');
    cap.style.cssText = 'font-size:12px;color:#888;margin-top:8px';
    cap.textContent = 'Routes are polytope solutions — equally valid, not draws.'
      + (anyTree ? ' Click an arrowed route to explore it.' : '');
    host.appendChild(tb);
    host.appendChild(cap);

    // Per-segment signature activity vs clock (independent of the band toggle).
    renderSegSigActivityInto(host, rec);

    // SBS96 mutational spectrum + weighting toggle for this segment.
    renderSpectrumInto(host, rec.spectra);

    // Per-segment spatial distribution (rainfall <-> VAF-vs-position toggle).
    renderSpatialInto(host, rec);
  }

  // ---- per-segment signature-activity-vs-clock plot (click detail) -------
  // SEG_ACTIVITY[sig][seg_id] = {abund, rows:[{interval, clock_time, ratio, ct,
  // st, ok}]}.  Shows the activity ratio per gain interval for this exact
  // segment/route, with the segment's abundance baseline (D_sig/D_clock) and a
  // clock-parity (1×) reference, so a user sees whether early values sit ≈baseline
  // (not enrichment).  Determined intervals (ok) solid, undetermined hollow/faint.
  function renderSegSigActivityInto(host, rec){
    if (!SEG_ACTIVITY) return;
    var sid = rec.seg_id;
    // which non-clock signatures have data for THIS segment?
    var sigs = [];
    Object.keys(SEG_ACTIVITY).forEach(function(sig){
      if (sig === 'clock') return;
      var blk = SEG_ACTIVITY[sig];
      if (blk && blk[sid] && blk[sid].rows && blk[sid].rows.length) sigs.push(sig);
    });
    if (!sigs.length) return;

    var wrap = document.createElement('div');
    wrap.style.cssText = 'margin-top:18px;';
    var h = document.createElement('h4');
    h.style.cssText = 'margin:0 0 4px 0;';
    h.textContent = 'Signature activity vs clock';
    wrap.appendChild(h);
    var sub = document.createElement('div');
    sub.style.cssText = 'font-size:11px;color:#888;margin-bottom:6px;';
    sub.textContent = 'CN (' + rec.major_cn + ',' + rec.minor_cn + ')'
      + (rec.best_key ? '  route ' + rec.best_key : '')
      + '  —  this segment timed under both clock and the signature on the same route.';
    wrap.appendChild(sub);

    var plotId = 'tau-segsigact-' + (UID++);
    var plotDiv = document.createElement('div');
    plotDiv.id = plotId;
    plotDiv.style.cssText = 'width:560px;max-width:100%;height:300px;';

    // signature selector (only if >1 sig has data for this segment)
    var sel = null;
    if (sigs.length > 1){
      var lbl = document.createElement('label');
      lbl.style.cssText = 'font-size:12px;margin-bottom:4px;display:inline-block;';
      lbl.textContent = 'signature: ';
      sel = document.createElement('select');
      sel.style.cssText = 'margin-left:4px;padding:2px 6px;';
      sigs.forEach(function(s){
        var o = document.createElement('option'); o.value = s; o.textContent = s;
        sel.appendChild(o);
      });
      lbl.appendChild(sel);
      wrap.appendChild(lbl);
    }
    wrap.appendChild(plotDiv);
    var cap = document.createElement('div');
    cap.style.cssText = 'font-size:11px;color:#666;max-width:560px;margin-top:6px;'
      + 'line-height:1.5;';
    cap.innerHTML = 'ratio = (sig t / clock t) × abundance, drawn as a line across each '
      + 'gain interval&rsquo;s clock-time span; segments above the dashed baseline = '
      + 'signature enriched in that interval, below = depleted.';
    wrap.appendChild(cap);
    host.appendChild(wrap);

    function draw(sig){
      var blk = SEG_ACTIVITY[sig][sid];
      var abund = blk.abund;
      var rows = blk.rows;
      // Each interval is a horizontal LINE across its clock-time span [t0,t1] at
      // y=ratio (the ratio is a constant density over the interval), NOT a point.
      // Determined intervals solid, underdetermined faded/dotted.  `null` breaks
      // the line between intervals.  Markers at both ends make narrow intervals
      // visible and give a hover target.
      var det = {x:[], y:[], txt:[]}, und = {x:[], y:[], txt:[]};
      var yvals = [];
      rows.forEach(function(r){
        var t0 = (r.t0 != null) ? r.t0 : r.clock_time;
        var t1 = (r.t1 != null) ? r.t1 : r.clock_time;
        var tip = 'interval ' + r.interval + '<br>clock-time ' + t0.toFixed(2) + '–'
          + t1.toFixed(2) + '<br>ratio ' + r.ratio.toFixed(2) + '×'
          + '<br>sig t ' + r.st.toFixed(3) + ' / clock t ' + r.ct.toFixed(3)
          + (r.ok ? '' : '<br>(underdetermined interval)');
        var g = r.ok ? det : und;
        g.x.push(t0, t1, null); g.y.push(r.ratio, r.ratio, null);
        g.txt.push(tip, tip, '');
        if (r.ratio > 0 && isFinite(r.ratio)) yvals.push(r.ratio);
      });
      var traces = [];
      if (det.x.length) traces.push({
        x: det.x, y: det.y, mode: 'lines+markers', name: 'determined',
        line: {color: '#7a5aaa', width: 3}, connectgaps: false,
        marker: {size: 5, color: '#7a5aaa'},
        text: det.txt, hovertemplate: '%{text}<extra></extra>' });
      if (und.x.length) traces.push({
        x: und.x, y: und.y, mode: 'lines+markers', name: 'underdetermined',
        line: {color: '#7a5aaa', width: 2, dash: 'dot'}, connectgaps: false,
        opacity: 0.5, marker: {size: 4, color: '#7a5aaa', symbol: 'circle-open'},
        text: und.txt, hovertemplate: '%{text}<extra></extra>' });
      // log-axis range from the plotted ratios + the abundance & parity lines
      // (NOT rangemode:'tozero' — invalid on log, blows up to ~10k).
      var yv2 = yvals.slice();
      yv2.push(1); if (abund > 0) yv2.push(abund);
      var yr2 = [Math.log10(Math.min.apply(null, yv2)) - 0.12,
                 Math.log10(Math.max.apply(null, yv2)) + 0.12];
      Plotly.newPlot(plotId, traces, {
        margin: {l: 54, r: 14, t: 12, b: 42},
        xaxis: {title: 'molecular time (clock)', range: [0, 1]},
        yaxis: {title: 'activity ratio (×clock)', type: 'log', range: yr2},
        shapes: [
          {type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y',
           y0: abund, y1: abund,
           line: {color: '#cd9f2c', width: 1.5, dash: 'dash'}},
          {type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y',
           y0: 1, y1: 1, line: {color: '#888', width: 1, dash: 'dot'}}
        ],
        annotations: [
          {x: 0.01, y: abund, xref: 'paper', yref: 'y',
           text: 'abundance baseline (D<sub>sig</sub>/D<sub>clock</sub> = '
             + abund.toFixed(1) + '×)', showarrow: false, xanchor: 'left',
           yanchor: 'bottom', font: {size: 10, color: '#a07d1e'}},
          {x: 0.01, y: 1, xref: 'paper', yref: 'y', text: 'clock parity (1×)',
           showarrow: false, xanchor: 'left', yanchor: 'bottom',
           font: {size: 10, color: '#888'}}
        ],
        height: 300, legend: {font: {size: 10}, orientation: 'h', y: 1.1}
      }, {displayModeBar: false, responsive: true});
    }
    draw(sigs[0]);
    if (sel) sel.onchange = function(){ draw(sel.value); };
  }

  function renderDetail(rec) {
    panel.innerHTML = '';
    renderSegmentDetailInto(panel, rec, {});
  }

  // ---- cluster-mode detail panel --------------------------------------
  function renderCluster(cl) {
    panel.innerHTML = '';
    var maj = cl.cn[0], minr = cl.cn[1];
    var h = document.createElement('h3');
    h.textContent = 'CN (' + maj + ',' + minr + ')   Cluster ' + cl.cluster_idx
      + '   —   ' + cl.n_segments + ' segments, ' + cl.total_muts + ' mutations';
    h.style.margin = '4px 0';
    panel.appendChild(h);

    var multId = 'tau-cl-mult-' + (UID++);
    var vafId = 'tau-cl-vaf-' + (UID++);
    var charts = document.createElement('div');
    charts.style.cssText = 'display:flex;flex-wrap:wrap;gap:24px;';
    var c1 = document.createElement('div'); c1.id = multId;
    c1.style.cssText = 'width:340px;height:240px;';
    var c2 = document.createElement('div'); c2.id = vafId;
    c2.style.cssText = 'width:340px;height:240px;';
    charts.appendChild(c2); charts.appendChild(c1);  // VAF left, multiplicity right
    panel.appendChild(charts);

    if (cl.props && cl.props.length) {
      var xs = cl.props.map(function(_, i){ return '×' + (i+1); });
      Plotly.newPlot(multId, [{
        x: xs, y: cl.props, type: 'bar', marker: {color: '#4c78a8'}
      }], {
        title: {text:'Multiplicity (pooled)', font:{size:12}},
        margin:{l:40,r:10,t:30,b:30}, yaxis:{range:[0,1]}, height:240
      }, {displayModeBar:false, responsive:true});
    } else { c1.innerHTML = '<em style="color:#888">no multiplicity data</em>'; }

    // Aggregate VAF histogram — SAME bins as the per-segment VAF hist.
    if (cl.vaf && cl.vaf.length) {
      var clov = expectedVafOverlay(cl.purity, cl.total_cn, cl.major_cn);
      Plotly.newPlot(vafId, [{
        x: cl.vaf, type: 'histogram',
        xbins:{start:0,end:1,size:0.025}, marker:{color:'#e45756'}
      }], {
        title:{text:'VAF (pooled)', font:{size:12}},
        margin:{l:40,r:10,t:30,b:30}, xaxis:{range:[0,1], title:'VAF'}, height:240,
        shapes:clov.shapes, annotations:clov.annotations
      }, {displayModeBar:false, responsive:true});
    } else { c2.innerHTML = '<em style="color:#888">no VAF data</em>'; }

    var lbl = document.createElement('div');
    lbl.style.cssText = 'font-weight:bold;margin:14px 0 4px';
    lbl.textContent = 'Member segments (' + cl.members.length + ')';
    panel.appendChild(lbl);

    var listWrap = document.createElement('div');
    listWrap.style.cssText = 'max-height:260px;overflow-y:auto;border:1px solid #eee;'
      + 'border-radius:4px;';
    cl.members.forEach(function(m){
      var rowDiv = document.createElement('div');
      rowDiv.style.cssText = 'border-bottom:1px solid #f0f0f0;';
      var head = document.createElement('div');
      var timed = (m.ri != null);
      head.style.cssText = 'padding:5px 10px;font-size:13px;display:flex;'
        + 'justify-content:space-between;' + (timed ? 'cursor:pointer;' : 'color:#999;');
      var lenTxt = (m.length != null)
        ? (m.length >= 1e6 ? (m.length/1e6).toFixed(2)+' Mb'
                           : (m.length/1e3).toFixed(0)+' kb')
        : '—';
      var mutTxt = timed ? (m.n_mut + ' muts') : 'not individually timed';
      head.innerHTML = '<span>' + (timed ? '▸ ' : '') + m.seg_id + '</span>'
        + '<span style="color:#666">' + lenTxt + '  ·  ' + mutTxt + '</span>';
      rowDiv.appendChild(head);

      var body = document.createElement('div');
      body.style.cssText = 'display:none;padding:8px 12px;background:#fafafa;';
      rowDiv.appendChild(body);

      if (timed) {
        var loaded = false, arrow = head.querySelector('span');
        head.onclick = function(){
          var open = (body.style.display === 'none');
          body.style.display = open ? 'block' : 'none';
          arrow.textContent = (open ? '▾ ' : '▸ ') + m.seg_id;
          if (open && !loaded) {
            renderSegmentDetailInto(body, RECORDS[m.ri], {noHeader:true});
            loaded = true;
          }
        };
      }
      listWrap.appendChild(rowDiv);
    });
    panel.appendChild(listWrap);

    var cap = document.createElement('div');
    cap.style.cssText = 'font-size:12px;color:#888;margin-top:8px';
    cap.textContent = 'VAF + multiplicity pooled over member segments. '
      + 'Routes are polytope solutions — equally valid, not draws.';
    panel.appendChild(cap);

    // ── Cluster timing (polytope solutions) ──────────────────────────────
    // The merged segment's own route-solution table + best route's event times,
    // identical in form to the per-segment route table.
    if (cl.routes && cl.routes.length) {
      var thdr = document.createElement('div');
      thdr.style.cssText = 'font-weight:bold;margin:18px 0 4px';
      thdr.textContent = 'Cluster timing';
      panel.appendChild(thdr);

      if (cl.best_event_times && cl.best_event_times.length) {
        var bt = document.createElement('div');
        bt.style.cssText = 'font-size:13px;color:#444;margin-bottom:6px';
        bt.innerHTML = 'Best route <b>' + (cl.best_key || '?')
          + '</b> — event time(s) t = '
          + cl.best_event_times.map(function(t){ return t.toFixed(3); }).join(',  ');
        panel.appendChild(bt);
      }

      // Trees are precomputed for multiple routes of the merged segment (all
      // dist_rel==0, or the top few if fewer are feasible) — arrowed routes are
      // expandable inline, exactly like the per-segment route table.
      var clTrees = cl.trees || {};
      var clExp = cl.tree_expandable || [];
      var clAnyTree = clExp.length > 0;
      var ctb = document.createElement('table');
      ctb.style.cssText = 'border-collapse:collapse;margin-top:4px;font-size:13px;'
        + 'width:100%;max-width:980px;';
      var cthead = '<tr style="background:#f2f2f2;text-align:left">'
        + '<th style="padding:4px 10px">route</th>'
        + '<th style="padding:4px 10px" title="Euclidean distance from the route’s '
          + 'feasibility boundary (0 = exactly feasible)">distance</th>'
        + '<th style="padding:4px 10px" title="N-inequalities the raw counts violated">violated</th>'
        + '<th style="padding:4px 10px">free vars</th>'
        + '<th style="padding:4px 10px" title="point if determined, [min,max] if underdetermined">event time(s)</th>'
        + '</tr>';
      ctb.innerHTML = '<thead>' + cthead + '</thead>';
      var ctbody = document.createElement('tbody');
      ctb.appendChild(ctbody);
      cl.routes.forEach(function(r){
        var hasTree = (clExp.indexOf(r.key) >= 0);
        var bg = r.is_best ? 'background:#fff7e6;' : '';
        var dim = (clAnyTree && !hasTree) ? 'color:#aaa;' : '';
        var arrow = hasTree ? '<span style="color:#0b6979">&#9656; </span>' : '';
        var name = arrow + r.key
          + (r.is_best ? ' <span style="color:#cc8400">(best)</span>' : '');
        var condStr = (r.conditions_violated && r.conditions_violated.length)
          ? r.conditions_violated.join('; ') : '—';
        var tr = document.createElement('tr');
        tr.style.cssText = bg + dim + 'border-top:1px solid #eee;'
          + (hasTree ? 'cursor:pointer;' : '');
        tr.innerHTML =
            '<td style="padding:4px 10px">' + name + '</td>'
          + '<td style="padding:4px 10px">' + (r.dist_rel==null?'NA':r.dist_rel) + '</td>'
          + '<td style="padding:4px 10px;font-size:12px;color:#9d2825">' + condStr + '</td>'
          + '<td style="padding:4px 10px">' + (r.free_var_count==null?'NA':r.free_var_count) + '</td>'
          + '<td style="padding:4px 10px">' + (r.intervals.join(',  ') || '—')
          + (clAnyTree && !hasTree ? ' <span style="font-size:11px;color:#bbb">(no tree)</span>' : '')
          + '</td>';
        ctbody.appendChild(tr);
        if (hasTree) {
          var detailTr = document.createElement('tr');
          var td = document.createElement('td');
          td.colSpan = 5; td.style.cssText = 'padding:0;background:#fafbfc;';
          var tbody2 = document.createElement('div');
          tbody2.style.cssText = 'display:none;padding:10px 14px;border-top:1px solid #eef0f3;';
          td.appendChild(tbody2); detailTr.appendChild(td); ctbody.appendChild(detailTr);
          var clLoaded = false;
          tr.onclick = function(){
            var open = (tbody2.style.display === 'none');
            tbody2.style.display = open ? 'block' : 'none';
            tr.querySelector('td span') &&
              (tr.querySelector('td span').innerHTML = open ? '&#9662; ' : '&#9656; ');
            if (open && !clLoaded){ renderTreeInto(tbody2, clTrees[r.key]); clLoaded = true; }
          };
        }
      });
      panel.appendChild(ctb);
      if (clAnyTree) {
        var clcap = document.createElement('div');
        clcap.style.cssText = 'font-size:12px;color:#888;margin-top:6px';
        clcap.textContent = 'Click an arrowed route to explore it.';
        panel.appendChild(clcap);
      }
    }

    // Aggregate SBS96 mutational spectrum for this cluster.
    renderSpectrumInto(panel, cl.spectra);
  }

  // ── VAF DISTRIBUTION BY CN STATE ──────────────────────────────────────
  // A browsable reference section (mounted once, below the plot).  The user
  // picks a (major,minor) CN state and sees the POOLED VAF distribution of all
  // SNVs across every segment of that state in the sample, overlaid with the
  // dashed expected-VAF lines for each multiplicity.  Built entirely from the
  // embedded CNSTATE_VAF histograms (bin counts + edges, no raw VAFs).
  var cnvafPanel = document.createElement('div');
  cnvafPanel.id = 'tau-cnvaf';
  cnvafPanel.style.cssText = 'font-family:sans-serif;max-width:1100px;margin:16px auto;'
    + 'padding:12px;border-top:2px solid #9fb8c9;color:#222;';

  function renderCnVafState(host, st){
    var plotId = 'tau-cnvaf-plot-' + (UID++);
    var div = document.createElement('div');
    div.id = plotId;
    div.style.cssText = 'width:640px;max-width:100%;height:320px;';
    host.appendChild(div);
    if (!st || !st.counts || !st.counts.length) {
      div.innerHTML = '<em style="color:#888">no VAF data for this state</em>';
      return;
    }
    var total = 0; for (var i=0;i<st.counts.length;i++){ total += st.counts[i]; }
    var frac = st.counts.map(function(c){ return total>0 ? c/total : 0; });
    var trace = {
      x: st.bin_centers, y: frac, type: 'bar',
      marker:{color:'#e45756'}, width: (1.0/st.counts.length)*0.95,
      text: st.counts, hovertemplate:'VAF %{x:.3f}<br>fraction %{y:.3f}'
        + '<br>count %{text}<extra></extra>'
    };
    // Reuse the shared expected-VAF helper (vertical dashed lines + ×m labels).
    var ov = expectedVafOverlay(st.purity, st.total_cn, st.major);
    Plotly.newPlot(plotId, [trace], {
      title:{text:'Pooled VAF — (' + st.major + ',' + st.minor + '),  '
        + st.n_segments + ' seg' + (st.n_segments===1?'':'s') + ',  '
        + st.n_snvs + ' SNVs'
        + (st.purity!=null ? ',  ρ = ' + st.purity.toFixed(3) : ''),
        font:{size:13}},
      margin:{l:50,r:14,t:40,b:40},
      xaxis:{range:[0,1], title:'VAF'},
      yaxis:{title:'fraction', rangemode:'tozero'},
      shapes:ov.shapes, annotations:ov.annotations, height:320, bargap:0.04
    }, {displayModeBar:false, responsive:true});
  }

  function renderCnVafSection(){
    cnvafPanel.innerHTML = '';
    var h = document.createElement('h3');
    h.style.cssText = 'margin:0 0 8px 0;';
    h.textContent = 'VAF by CN state';
    cnvafPanel.appendChild(h);

    if (!CNSTATE_VAF || !CNSTATE_VAF.length) {
      var none = document.createElement('em');
      none.style.cssText = 'color:#888;';
      none.textContent = 'No per-SNV VAF data available in this sample.';
      cnvafPanel.appendChild(none);
      return;
    }

    var ctrl = document.createElement('div');
    ctrl.style.cssText = 'font-size:13px;margin-bottom:8px;';
    var b = document.createElement('b'); b.textContent = 'CN state:';
    var sel = document.createElement('select');
    sel.style.cssText = 'margin-left:8px;padding:2px 6px;';
    CNSTATE_VAF.forEach(function(s, i){
      var op = document.createElement('option');
      op.value = String(i);
      op.textContent = '(' + s.major + ',' + s.minor + ') — '
        + s.n_segments + ' seg' + (s.n_segments===1?'':'s') + ', '
        + s.n_snvs + ' SNVs';
      sel.appendChild(op);
    });
    ctrl.appendChild(b); ctrl.appendChild(sel);
    cnvafPanel.appendChild(ctrl);

    var plotHost = document.createElement('div');
    cnvafPanel.appendChild(plotHost);

    var cap = document.createElement('div');
    cap.style.cssText = 'font-size:12px;color:#666;max-width:660px;margin-top:10px;'
      + 'line-height:1.5;';
    cap.innerHTML = 'Dashed lines = expected VAF per multiplicity m: '
      + '<b>VAF<sub>m</sub> = m·ρ / (CN<sub>tot</sub>·ρ + 2·(1−ρ))</b> '
      + '(ρ = purity). Pooled over all segments of the selected CN state.';
    cnvafPanel.appendChild(cap);

    sel.onchange = function(){
      plotHost.innerHTML = '';
      renderCnVafState(plotHost, CNSTATE_VAF[parseInt(sel.value, 10) || 0]);
    };
    sel.value = '0';
    renderCnVafState(plotHost, CNSTATE_VAF[0]);
  }

  // ── MULTI-SEGMENT COMPARISON ──────────────────────────────────────────
  // state.compare holds an ordered, deduped list of RECORD indices (ri).
  // Shift/Ctrl/Cmd-click a hit bar or box/lasso-select on the select layer
  // adds segments; the panel below the plot overlays their VAF + SBS96 and a
  // per-segment stats table.  Each compared segment gets a stable categorical
  // colour by its position in state.compare.
  var COMPARE_PALETTE = ['#4c78a8','#f58518','#54a24b','#e45756','#72b7b2',
    '#b279a2','#ff9da6','#9d755d','#bab0ac','#1b9e77','#d95f02','#7570b3'];
  function compareColor(pos){ return COMPARE_PALETTE[pos % COMPARE_PALETTE.length]; }

  // Dedicated comparison container, created once and reused.
  var cmpPanel = document.createElement('div');
  cmpPanel.id = 'tau-compare';
  cmpPanel.style.cssText = 'font-family:sans-serif;max-width:1100px;margin:16px auto;'
    + 'padding:12px;border-top:2px solid #c9a3a3;color:#222;display:none;';

  function bestRoute(rec){
    var rows = rec.route_rows || [];
    for (var i=0;i<rows.length;i++){ if (rows[i].is_best) return rows[i]; }
    return rows[0] || null;
  }

  function toggleCompare(ri){
    var pos = state.compare.indexOf(ri);
    if (pos >= 0) { state.compare.splice(pos, 1); }
    else { state.compare.push(ri); }
    renderCompare();
  }
  function addCompare(ri){
    if (state.compare.indexOf(ri) < 0) { state.compare.push(ri); }
  }
  function clearCompare(){ state.compare = []; renderCompare(); }

  // Shared, fixed weighting key for the overlaid SBS96 (changed via a <select>).
  var cmpSpecKey = null;
  function spectraOptionsFor(ris){
    // Intersection-ish: options present in SPECTRUM that at least one segment has.
    if (!SPECTRUM || !SPECTRUM.enabled) return [];
    var present = {};
    ris.forEach(function(ri){
      var sp = RECORDS[ri] && RECORDS[ri].spectra;
      if (sp) { for (var k in sp) { present[k] = true; } }
    });
    return SPECTRUM.options.filter(function(o){ return present[o.key]; });
  }

  function renderCompareStatsTable(host, ris){
    var tb = document.createElement('table');
    tb.style.cssText = 'border-collapse:collapse;font-size:13px;width:100%;max-width:1080px;';
    var thead = '<tr style="background:#f2f2f2;text-align:left">'
      + '<th style="padding:4px 8px"></th>'
      + '<th style="padding:4px 8px">seg_id</th>'
      + '<th style="padding:4px 8px">CN</th>'
      + '<th style="padding:4px 8px">#mut</th>'
      + '<th style="padding:4px 8px">best route</th>'
      + '<th style="padding:4px 8px">dist_rel</th>'
      + '<th style="padding:4px 8px">free_vars</th>'
      + '<th style="padding:4px 8px">event time(s)</th>'
      + '<th style="padding:4px 8px"></th></tr>';
    tb.innerHTML = '<thead>' + thead + '</thead>';
    var body = document.createElement('tbody');
    tb.appendChild(body);
    ris.forEach(function(ri, pos){
      var rec = RECORDS[ri]; if (!rec) return;
      var col = compareColor(pos);
      var br = bestRoute(rec);
      var nmut = (rec.n_mut != null) ? rec.n_mut
        : ((rec.vaf && rec.vaf.length) ? rec.vaf.length : '—');
      var tr = document.createElement('tr');
      tr.style.cssText = 'border-top:1px solid #eee;';
      tr.innerHTML =
          '<td style="padding:4px 8px"><span style="display:inline-block;width:14px;'
          + 'height:14px;border-radius:3px;background:' + col + '"></span></td>'
        + '<td style="padding:4px 8px;font-weight:600">' + rec.seg_id + '</td>'
        + '<td style="padding:4px 8px">(' + rec.major_cn + ',' + rec.minor_cn + ')</td>'
        + '<td style="padding:4px 8px">' + nmut + '</td>'
        + '<td style="padding:4px 8px">' + (br ? br.key : '—') + '</td>'
        + '<td style="padding:4px 8px">' + (br && br.dist_rel != null ? br.dist_rel : 'NA') + '</td>'
        + '<td style="padding:4px 8px">' + (br && br.free_var_count != null ? br.free_var_count : 'NA') + '</td>'
        + '<td style="padding:4px 8px">' + (br && br.intervals ? (br.intervals.join(',  ') || '—') : '—') + '</td>';
      var xtd = document.createElement('td');
      xtd.style.cssText = 'padding:4px 8px;color:#c0392b;cursor:pointer;font-weight:700;';
      xtd.textContent = '✕';
      xtd.title = 'remove from comparison';
      xtd.onclick = function(){ toggleCompare(ri); };
      tr.appendChild(xtd);
      body.appendChild(tr);
    });
    host.appendChild(tb);
  }

  function renderCompareVAF(host, ris){
    var plotId = 'tau-cmp-vaf-' + (UID++);
    var div = document.createElement('div');
    div.id = plotId;
    div.style.cssText = 'width:520px;max-width:100%;height:300px;';
    host.appendChild(div);
    var traces = [];
    ris.forEach(function(ri, pos){
      var rec = RECORDS[ri]; if (!rec || !rec.vaf || !rec.vaf.length) return;
      traces.push({
        x: rec.vaf, type: 'histogram', name: rec.seg_id,
        opacity: 0.55, marker: {color: compareColor(pos)},
        xbins: {start:0, end:1, size:0.025}, histnorm: 'probability'
      });
    });
    if (!traces.length) { div.innerHTML = '<em style="color:#888">no VAF data</em>'; return; }
    Plotly.newPlot(plotId, traces, {
      title: {text:'Overlaid VAF', font:{size:12}},
      barmode: 'overlay', margin:{l:46,r:10,t:30,b:34},
      xaxis:{range:[0,1], title:'VAF'}, yaxis:{title:'fraction'},
      height:300, legend:{font:{size:10}}
    }, {displayModeBar:false, responsive:true});
  }

  function renderCompareSpectrum(host, ris, specKey){
    if (!SPECTRUM || !SPECTRUM.enabled) { return; }
    var opts = spectraOptionsFor(ris);
    if (!opts.length) { return; }
    var wrap = document.createElement('div');
    wrap.style.cssText = 'flex:1;min-width:560px;';
    var ctrl = document.createElement('div');
    ctrl.style.cssText = 'font-size:13px;margin-bottom:4px;';
    var b = document.createElement('b'); b.textContent = 'SBS96 weighting:';
    var sel = document.createElement('select');
    sel.style.cssText = 'margin-left:8px;padding:2px 6px;';
    opts.forEach(function(o){
      var op = document.createElement('option'); op.value = o.key; op.textContent = o.label;
      sel.appendChild(op);
    });
    var key = specKey;
    if (!opts.some(function(o){ return o.key === key; })) {
      var def = SPECTRUM.default;
      key = opts.some(function(o){ return o.key === def; }) ? def : opts[0].key;
    }
    sel.value = key; cmpSpecKey = key;
    ctrl.appendChild(b); ctrl.appendChild(sel);
    wrap.appendChild(ctrl);
    var plotId = 'tau-cmp-sbs-' + (UID++);
    var plotDiv = document.createElement('div');
    plotDiv.id = plotId;
    plotDiv.style.cssText = 'width:100%;min-width:540px;height:300px;';
    wrap.appendChild(plotDiv);
    host.appendChild(wrap);

    function draw(k){
      cmpSpecKey = k;
      var traces = [];
      ris.forEach(function(ri, pos){
        var rec = RECORDS[ri]; if (!rec || !rec.spectra) return;
        var raw = rec.spectra[k]; if (!raw) return;
        var tot = 0; for (var i=0;i<raw.length;i++){ tot += raw[i]; }
        var frac = raw.map(function(v){ return tot>0 ? v/tot : 0; });
        traces.push({
          x: SPECTRUM.channels.map(function(_, i){ return i; }),
          y: frac, type:'scatter', mode:'lines', name: rec.seg_id,
          line:{color: compareColor(pos), width:1.4},
          text: SPECTRUM.channels,
          hovertemplate:'%{text}<br>%{y:.3f}<extra>'+rec.seg_id+'</extra>'
        });
      });
      var shapes = [], anns = [];
      for (var t=0; t<SPECTRUM.sub_types.length; t++){
        var x0 = t*16 - 0.5, x1 = t*16 + 15.5;
        shapes.push({type:'rect', xref:'x', yref:'paper', x0:x0, x1:x1,
          y0:1.0, y1:1.05, fillcolor:SPECTRUM.sub_colors[SPECTRUM.sub_types[t]],
          line:{width:0}});
        anns.push({x:(x0+x1)/2, y:1.075, xref:'x', yref:'paper',
          text:SPECTRUM.sub_types[t], showarrow:false, font:{size:11, color:'#333'}});
      }
      Plotly.newPlot(plotId, traces, {
        title:{text:'Overlaid SBS96 spectra', font:{size:12}},
        margin:{l:46,r:10,t:46,b:30},
        xaxis:{showticklabels:false, range:[-0.5,95.5], fixedrange:true},
        yaxis:{title:'fraction', rangemode:'tozero', fixedrange:true},
        shapes:shapes, annotations:anns, height:300, legend:{font:{size:10}}
      }, {displayModeBar:false, responsive:true});
    }
    sel.onchange = function(){ draw(sel.value); };
    draw(key);
  }

  function renderCompare(){
    var ris = state.compare;
    cmpPanel.innerHTML = '';
    if (!ris || ris.length < 1) { cmpPanel.style.display = 'none'; return; }
    cmpPanel.style.display = 'block';

    var header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;gap:16px;margin-bottom:10px;';
    var h = document.createElement('h3');
    h.style.cssText = 'margin:0;';
    h.textContent = 'Segment comparison — ' + ris.length + ' selected';
    var clearBtn = document.createElement('button');
    clearBtn.textContent = 'Clear all';
    clearBtn.style.cssText = 'padding:4px 12px;cursor:pointer;border:1px solid #c0392b;'
      + 'background:#fff;color:#c0392b;border-radius:4px;font-size:13px;';
    clearBtn.onclick = function(){ clearCompare(); };
    header.appendChild(h); header.appendChild(clearBtn);
    var hint = document.createElement('span');
    hint.style.cssText = 'color:#888;font-size:12px;';
    hint.textContent = 'Shift/Ctrl/Cmd-click a segment to toggle it, or box/lasso-select on the plot.';
    header.appendChild(hint);
    cmpPanel.appendChild(header);

    renderCompareStatsTable(cmpPanel, ris);

    var charts = document.createElement('div');
    charts.style.cssText = 'display:flex;flex-wrap:wrap;gap:24px;margin-top:14px;'
      + 'align-items:flex-start;';
    cmpPanel.appendChild(charts);
    renderCompareVAF(charts, ris);
    renderCompareSpectrum(charts, ris, cmpSpecKey);

    var cap = document.createElement('div');
    cap.style.cssText = 'font-size:12px;color:#888;margin-top:10px;';
    cap.textContent = 'Multiplicity is omitted from the overlay (bin count varies '
      + 'by CN state); see each segment’s single-click panel. Routes are '
      + 'polytope solutions — equally valid points in an underdetermined space.';
    cmpPanel.appendChild(cap);
  }

  gd.on('plotly_click', function(evt) {
    if (!evt || !evt.points || !evt.points.length) { return; }
    var p = evt.points[0];
    var cd = p.customdata;
    if (cd == null) { return; }
    var idx = Array.isArray(cd) ? cd[0] : cd;
    var kind = (p.data && p.data.meta) ? p.data.meta.kind : null;
    // Modifier-click on a segment hit bar (or select layer) → toggle that
    // segment in the multi-segment comparison view WITHOUT disturbing the
    // single-click detail panel.  Plotly forwards the DOM event as evt.event.
    var de = evt.event || (window.event || null);
    var modified = de && (de.shiftKey || de.ctrlKey || de.metaKey);
    if (modified && (kind === 'hit' || kind === 'selhit')) {
      toggleCompare(idx);
      return;
    }
    if (kind === 'cluster_hit') {
      var cl = CLUSTERS[idx];
      if (cl) { renderCluster(cl); }
    } else {
      var rec = RECORDS[idx];
      if (rec) { renderDetail(rec); }
    }
  });

  // ---- chromosome dropdown + CN-state checkboxes -----------------------
  var CHROM_LIST = __CHROM_LIST__;
  var CHROM_SPANS = __CHROM_SPANS__;  // {chrom: [x_start, x_end]} genome window
  var AUTO_LEN = __AUTO_LEN__;        // total autosome length (full x-range)
  var cnVals = [];
  HITS.forEach(function(r){ if (cnVals.indexOf(r.major_cn)<0) cnVals.push(r.major_cn); });
  cnVals.sort(function(a,b){return a-b;});

  var state = {chrom:'all', cnHidden:[], compare:[]};

  // ── COMBINED CLUSTER TIMELINE ─────────────────────────────────────────────
  // All major multiplicity clusters' allele trees drawn TOGETHER on one shared
  // molecular-time axis (early 0 → late 1).  Each column = one cluster with a
  // candidate-route <select> and (for underdetermined routes) free-variable
  // knobs; reuses renderTreeInto, so trees redraw live and you can see how the
  // clustered amplifications line up in time.  Trees come from CLUSTERS[i].trees
  // (precomputed for the top routes of each cluster's merged segment).
  var clTLPanel = document.createElement('div');
  clTLPanel.id = 'tau-cluster-timeline';
  clTLPanel.style.cssText = 'font-family:sans-serif;max-width:100%;margin:16px auto;'
    + 'padding:12px;border-top:2px solid #9fb8c9;color:#222;';

  function renderClusterTimeline(){
    clTLPanel.innerHTML = '';
    var h = document.createElement('h3');
    h.style.cssText = 'margin:0 0 4px;';
    h.textContent = 'Cluster timeline — all major clusters on one molecular-time axis';
    clTLPanel.appendChild(h);
    var cap = document.createElement('div');
    cap.style.cssText = 'font-size:12px;color:#666;margin-bottom:10px;max-width:920px;line-height:1.5;';
    cap.innerHTML = 'Each column is one multiplicity cluster’s allele tree on the SAME '
      + 'molecular-time axis (early 0 → late 1). Pick a candidate route per cluster; for '
      + 'underdetermined routes, drag the knobs to move its events through the polytope — '
      + 'the tree redraws live, so you can see how the clustered amplifications line up in time. '
      + 'Teal/gold dashed lines = WGD/PGD landmarks (shared across all columns). Sorted by '
      + 'number of member segments (most prominent first).';
    clTLPanel.appendChild(cap);
    var cls = (CLUSTERS || []).filter(function(c){
      return c.trees && c.tree_expandable && c.tree_expandable.length; });
    cls.sort(function(a,b){ return (b.n_segments||0) - (a.n_segments||0); });
    if (!cls.length){
      var none = document.createElement('em');
      none.style.cssText = 'color:#888;';
      none.textContent = 'No clusters with precomputed trees in this sample '
        + '(regenerate the viewer with tree-top-k > 0).';
      clTLPanel.appendChild(none); return;
    }
    // tight, gapless row so the trees read like one continuous plot on the
    // shared timeline; each tree is compact (no tables) with its knobs stacked
    // directly below it, so you can still dial every cluster's free variables.
    var TLW = 150;  // per-tree column width
    var rowWrap = document.createElement('div');
    rowWrap.style.cssText = 'display:flex;gap:0;overflow-x:auto;align-items:flex-start;'
      + 'padding-bottom:10px;';
    clTLPanel.appendChild(rowWrap);
    cls.forEach(function(c, ci){
      var col = document.createElement('div');
      col.style.cssText = 'flex:0 0 ' + TLW + 'px;max-width:' + TLW + 'px;'
        + (ci ? 'border-left:1px solid #eef0f3;' : '');
      var title = document.createElement('div');
      title.style.cssText = 'font-weight:600;font-size:11px;margin:0 0 2px 4px;line-height:1.25;';
      title.innerHTML = '(' + c.cn[0] + ',' + c.cn[1] + ') cl' + c.cluster_idx
        + ' <span style="font-weight:400;color:#888">' + (c.n_segments||0) + 's</span>';
      col.appendChild(title);
      var sel = document.createElement('select');
      sel.style.cssText = 'font-size:11px;margin:0 0 2px 4px;padding:1px 2px;max-width:' + (TLW-8) + 'px;';
      c.tree_expandable.forEach(function(k){
        var o = document.createElement('option'); o.value = k;
        o.textContent = k + (k === c.best_key ? ' ★' : '');
        sel.appendChild(o);
      });
      sel.value = (c.tree_expandable.indexOf(c.best_key) >= 0)
        ? c.best_key : c.tree_expandable[0];
      col.appendChild(sel);
      var mount = document.createElement('div');
      col.appendChild(mount);
      // Column MUST be in the live DOM before renderTreeInto runs — it wires the
      // knob sliders via getElementById, which returns null on a detached node.
      rowWrap.appendChild(col);
      function drawCl(k){
        try {
          // only the first column draws the early0/late1 axis labels (shared axis).
          renderTreeInto(mount, c.trees[k], {
            treeW: TLW, treeH: 300, compact: true, stack: true,
            axisLabels: (ci === 0)});
        } catch (e) {
          mount.innerHTML = '<em style="color:#9d2825;font-size:11px">tree render '
            + 'failed: ' + (e && e.message ? e.message : e) + '</em>';
        }
      }
      sel.onchange = function(){ drawCl(sel.value); };
      drawCl(sel.value);
    });
  }

  // Mount the browsable "VAF by CN state" reference section just below the
  // single-click detail panel, and the comparison container below that.
  panel.parentNode.insertBefore(cnvafPanel, panel.nextSibling);
  cnvafPanel.parentNode.insertBefore(cmpPanel, cnvafPanel.nextSibling);
  // Cluster timeline goes directly below the main plot (above the CN-VAF section).
  panel.parentNode.insertBefore(clTLPanel, panel.nextSibling);
  renderClusterTimeline();
  renderCnVafSection();

  // Box/lasso selection on the transparent select layer (or the hit bars) adds
  // every selected segment's RECORD index to the comparison view.
  gd.on('plotly_selected', function(evt){
    if (!evt || !evt.points || !evt.points.length) { return; }
    var added = false;
    evt.points.forEach(function(p){
      var k = (p.data && p.data.meta) ? p.data.meta.kind : null;
      if (k !== 'selhit' && k !== 'hit') { return; }
      var cd = p.customdata;
      if (cd == null) { return; }
      var ri = Array.isArray(cd) ? cd[0] : cd;
      if (ri != null) { addCompare(ri); added = true; }
    });
    if (added) { renderCompare(); }
  });

  // Identify the hit-bar trace and the per-segment fill-band traces.
  var hitTraceIdx = -1;
  var bandTraces = [];   // {idx, major, chroms:[...] }
  for (var ti=0; ti<gd.data.length; ti++) {
    var meta = gd.data[ti].meta;
    if (!meta) continue;
    if (meta.kind === 'hit') hitTraceIdx = ti;
    else if (meta.kind === 'band') bandTraces.push({idx:ti, major:meta.major, band:meta.band, name:gd.data[ti].name, chroms:meta.chroms});
  }

  function cnVisible(cn) {
    return state.cnHidden.indexOf(cn) < 0;
  }

  // CN-checkbox filtering only (the chromosome dropdown zooms the x-axis via
  // applyChromZoom instead of hiding traces — see sel.onchange below).
  function applyFilter() {
    // Hit bars: toggle per-point opacity so hover/click follows the CN filter.
    if (hitTraceIdx >= 0) {
      var op = HITS.map(function(r){ return cnVisible(r.major_cn) ? 1 : 0; });
      Plotly.restyle(gd, {'marker.opacity':[op]}, [hitTraceIdx]);
    }
    // Fill bands: a band trace is per (major, band); hide the whole trace when
    // its major CN is filtered out.  ONE batched restyle over all band traces
    // (was one restyle call PER trace → hundreds of redraws → very laggy).
    var vis = [], idxs = [];
    bandTraces.forEach(function(bt){
      vis.push(cnVisible(bt.major) ? true : 'legendonly');
      idxs.push(bt.idx);
    });
    if (idxs.length){ Plotly.restyle(gd, {visible: vis}, idxs); }
  }

  // Zoom the main-panel x-axis (xaxis4) to a chromosome's genomic window; the
  // cluster track follows automatically (its x-axis matches="x4").
  function applyChromZoom() {
    if (state.chrom === 'all' || !CHROM_SPANS[state.chrom]) {
      Plotly.relayout(gd, {'xaxis4.range': [0, AUTO_LEN]});
    } else {
      Plotly.relayout(gd, {'xaxis4.range': CHROM_SPANS[state.chrom]});
    }
  }

  var bar = document.createElement('div');
  bar.style.cssText = 'font-family:sans-serif;text-align:center;margin:8px;font-size:13px;';

  var sel = document.createElement('select');
  sel.style.cssText = 'margin:0 14px;padding:2px 6px;';
  var optAll = document.createElement('option');
  optAll.value = 'all'; optAll.textContent = 'All chromosomes';
  sel.appendChild(optAll);
  CHROM_LIST.forEach(function(c){
    var o = document.createElement('option');
    o.value = c; o.textContent = 'chr' + c;
    sel.appendChild(o);
  });
  sel.onchange = function(){ state.chrom = sel.value; applyChromZoom(); };
  var selLbl = document.createElement('b');
  selLbl.textContent = 'Chromosome: ';
  bar.appendChild(selLbl); bar.appendChild(sel);

  var cnLbl = document.createElement('b');
  cnLbl.textContent = '   Show major CN: ';
  bar.appendChild(cnLbl);
  cnVals.forEach(function(cn){
    var lbl = document.createElement('label');
    lbl.style.cssText = 'margin:0 6px;cursor:pointer;';
    var cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = true; cb.value = cn;
    cb.onchange = function(){
      state.cnHidden = [];
      bar.querySelectorAll('input[type=checkbox]').forEach(function(x){
        if (!x.checked) state.cnHidden.push(parseInt(x.value,10));
      });
      applyFilter();
    };
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(' ' + cn));
    bar.appendChild(lbl);
  });

  // ── SIGNATURE TOGGLE + SIGNATURE-ACTIVITY CURVE ───────────────────────────
  // SIG_TIMING (when present, >1 signature) carries per-signature CUMULATIVE
  // event times per segment and per-non-clock-signature activity rows.  The
  // toggle re-renders ONLY the main-panel filled bands so each segment's
  // stacked-band heights use the selected signature's event times — SAME x
  // positions, SAME copy-number colours.  The plot DIMENSIONS are pinned
  // (xaxis4.range=[0,AUTO_LEN], yaxis4.range=[0,1], autorange off) so flipping
  // the toggle only moves band heights, making the timing shift directly
  // visible.  These are polytope solutions collapsed to a point estimate —
  // never draws or a distribution.  Cluster track / CN legend / event lines /
  // histogram stay as the CLOCK reference (see caption).
  if (SIG_TIMING && SIG_TIMING.signatures && SIG_TIMING.signatures.length > 1) {
    var SIG_NAMES = SIG_TIMING.signatures.map(function(s){ return s.name; });
    var SHARE_BY = {};
    SIG_TIMING.signatures.forEach(function(s){ SHARE_BY[s.name] = s.share; });
    var activeSig = 'clock';

    // Cache the ORIGINAL clock band x/y arrays (the mat-based polygons) so
    // switching back to clock restores them EXACTLY.
    var clockBandXY = {};
    bandTraces.forEach(function(bt){
      clockBandXY[bt.idx] = {
        x: gd.data[bt.idx].x.slice(),
        y: gd.data[bt.idx].y.slice()
      };
    });

    // Swap one band trace to signature `sig`'s prebuilt FULL polygon set.  Both
    // x and y are replaced so each segment's stacked bands use the signature's
    // own polytope-solution geometry (same x-spread, same colours, same
    // structure as clock — built by the SAME Python helper).  Segments not timed
    // under `sig` are absent from SIG_BANDS[sig] and so vanish; if the trace's
    // (major,band) is missing entirely it collapses to an empty polygon.
    function bandXYForSig(bt, sig){
      if (sig === 'clock'){
        var clk = clockBandXY[bt.idx];
        return {x: clk.x.slice(), y: clk.y.slice()};
      }
      var bySig = SIG_BANDS[sig] || {};
      var tr = bySig[bt.name];
      if (tr && tr.x && tr.x.length){
        return {x: tr.x.slice(), y: tr.y.slice()};
      }
      return {x: [], y: []};  // no segments of this (major,band) under sig
    }

    function renderBandsForSig(sig){
      var xs = [], ys = [], idxs = [];
      bandTraces.forEach(function(bt){
        var xy = bandXYForSig(bt, sig);
        xs.push(xy.x);
        ys.push(xy.y);
        idxs.push(bt.idx);
      });
      // Pin axes BEFORE restyle so the dimensions never change between sigs.
      Plotly.relayout(gd, {
        'xaxis4.autorange': false, 'xaxis4.range': [0, AUTO_LEN],
        'yaxis4.autorange': false, 'yaxis4.range': [0, 1]
      });
      if (idxs.length){ Plotly.restyle(gd, {x: xs, y: ys}, idxs); }
      // Re-apply CN-filter visibility (restyle does not touch it) and chrom zoom.
      applyFilter();
      applyChromZoom();
    }

    // ---- signature-activity curve (per-sample) ----------------------------
    // For a non-clock signature S: bin activity rows' clock_time into 0.05
    // windows; plot the MEDIAN unnormalized interval ratio (ws/wc·D_sig/D_clock)
    // per bin as a line with an IQR (25–75th pct) band; ratio=1 reference line.
    var sigActPanel = document.createElement('div');
    sigActPanel.id = 'tau-sigact';
    sigActPanel.style.cssText = 'font-family:sans-serif;max-width:1100px;'
      + 'margin:16px auto;padding:12px;border-top:2px solid #b3a3c9;color:#222;'
      + 'display:none;';

    function quantile(sorted, q){
      if (!sorted.length) return null;
      var pos = (sorted.length - 1) * q;
      var lo = Math.floor(pos), hi = Math.ceil(pos);
      if (lo === hi) return sorted[lo];
      return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }
    // weighted quantile: pairs = [[value, weight], ...] sorted by value.
    function wquantile(pairs, q){
      if (!pairs.length) return null;
      var tot = 0, i;
      for (i = 0; i < pairs.length; i++) tot += pairs[i][1];
      if (!(tot > 0)) {  // no weights -> plain quantile on values
        var vals = pairs.map(function(p){ return p[0]; });
        return quantile(vals, q);
      }
      var target = q * tot, cum = 0;
      for (i = 0; i < pairs.length; i++){
        cum += pairs[i][1];
        if (cum >= target) return pairs[i][0];
      }
      return pairs[pairs.length - 1][0];
    }

    function renderSigActivity(sig){
      sigActPanel.innerHTML = '';
      var rows = (SIG_TIMING.activity && SIG_TIMING.activity[sig]) || [];
      var h = document.createElement('h3');
      h.style.cssText = 'margin:0 0 6px 0;';
      h.textContent = sig + ' activity relative to clock';
      sigActPanel.appendChild(h);
      if (!rows.length){
        var none = document.createElement('em');
        none.style.cssText = 'color:#888;';
        none.textContent = 'No activity rows for ' + sig + ' in this sample.';
        sigActPanel.appendChild(none);
        sigActPanel.style.display = 'block';
        return;
      }
      // Each row is an interval with a clock-time SPAN [t0,t1] (row[0],row[1]),
      // a ratio (row[2]) and an ESS weight (row[3]).  The ratio is a constant
      // density over that span, so we SPREAD it: add [ratio, weight] to every
      // 0.05 bin the span overlaps (full weight per bin — the interval is present
      // across the whole window).  Then each bin's weighted median/IQR over all
      // overlapping intervals is the curve.
      var NB = 20, BW = 0.05, bins = [];
      for (var i = 0; i < NB; i++) bins.push([]);
      rows.forEach(function(r){
        var t0 = r[0], t1 = r[1], ra = r[2], w = (r.length > 3 && r[3] > 0) ? r[3] : 1;
        if (!(ra > 0)) return;
        if (t1 < t0){ var tmp = t0; t0 = t1; t1 = tmp; }   // safety
        t0 = Math.max(0, Math.min(1, t0));
        t1 = Math.max(0, Math.min(1, t1));
        var b0 = Math.min(NB - 1, Math.floor(t0 / BW));
        // upper bin: nudge a hair below t1 so a span ending exactly on a boundary
        // doesn't bleed into the next (empty) bin; zero-width spans -> single bin.
        var hi = (t1 > t0) ? t1 - 1e-9 : t1;
        var b1 = Math.min(NB - 1, Math.floor(hi / BW));
        for (var b = b0; b <= b1; b++) bins[b].push([ra, w]);   // [ratio, ESS weight]
      });
      var xs = [], med = [], q25 = [], q75 = [];
      for (var b = 0; b < NB; b++){
        if (!bins[b].length) continue;
        bins[b].sort(function(a,c){ return a[0] - c[0]; });
        xs.push((b + 0.5) * 0.05);
        med.push(wquantile(bins[b], 0.5));   // ESS-weighted median + IQR
        q25.push(wquantile(bins[b], 0.25));
        q75.push(wquantile(bins[b], 0.75));
      }
      var plotId = 'tau-sigact-plot-' + (UID++);
      var div = document.createElement('div');
      div.id = plotId;
      div.style.cssText = 'width:680px;max-width:100%;height:320px;';
      sigActPanel.appendChild(div);
      var traces = [
        { x: xs.concat(xs.slice().reverse()),
          y: q75.concat(q25.slice().reverse()),
          fill: 'toself', fillcolor: 'rgba(122,90,170,0.18)',
          line: {width: 0}, mode: 'lines', hoverinfo: 'skip',
          name: 'IQR (25–75%)', showlegend: true },
        { x: xs, y: med, mode: 'lines+markers', name: 'median ratio',
          line: {color: '#7a5aaa', width: 2}, marker: {size: 5},
          hovertemplate: 'clock-time %{x:.2f}<br>ratio %{y:.2f}×<extra></extra>' }
      ];
      var ga = (SIG_TIMING.genome_abund && SIG_TIMING.genome_abund[sig]) || null;
      var shapes = [{type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y',
                     y0: 1, y1: 1, line: {color: '#888', width: 1.2, dash: 'dash'}}];
      var annos = [{x: 0.01, y: 1, xref: 'paper', yref: 'y',
        text: 'clock parity (1×)', showarrow: false, xanchor: 'left',
        yanchor: 'bottom', font: {size: 10, color: '#888'}}];
      if (ga && ga.ratio > 0){
        shapes.push({type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y',
          y0: ga.ratio, y1: ga.ratio, line: {color: '#cd9f2c', width: 1.6, dash: 'dot'}});
        annos.push({x: 0.99, y: ga.ratio, xref: 'paper', yref: 'y',
          text: 'genome base rate (' + ga.ratio.toFixed(2) + '×)', showarrow: false,
          xanchor: 'right', yanchor: 'bottom', font: {size: 10, color: '#cd9f2c'}});
      }
      // y-range: fit the actual plotted values (median, IQR band, base-rate line,
      // and clock parity=1 so it's always visible), padded in LOG space.  NOTE:
      // never use rangemode:'tozero' on a log axis — log(0)=-inf makes Plotly
      // auto-range to absurd upper bounds (~10k).
      var yv = med.concat(q25, q75).filter(function(v){ return v > 0 && isFinite(v); });
      yv.push(1);                                    // clock parity
      if (ga && ga.ratio > 0) yv.push(ga.ratio);     // genome base rate
      var ylo = Math.min.apply(null, yv), yhi = Math.max.apply(null, yv);
      var yrange = [Math.log10(ylo) - 0.12, Math.log10(yhi) + 0.12];
      Plotly.newPlot(plotId, traces, {
        title: {text: sig + ' / clock — unnormalized interval ratio '
          + '(w<sub>s</sub>/w<sub>c</sub> · D<sub>sig</sub>/D<sub>clock</sub>)',
          font: {size: 13}},
        margin: {l: 56, r: 14, t: 42, b: 44},
        xaxis: {title: 'molecular time (clock)', range: [0, 1]},
        yaxis: {title: 'activity ratio (×clock)', type: 'log', range: yrange},
        shapes: shapes,
        annotations: annos,
        height: 320, legend: {font: {size: 10}, orientation: 'h', y: 1.12}
      }, {displayModeBar: false, responsive: true});
      var cap = document.createElement('div');
      cap.style.cssText = 'font-size:12px;color:#666;max-width:680px;margin-top:8px;'
        + 'line-height:1.5;';
      cap.innerHTML = 'Per-sample ' + sig + ' activity vs the clock, as the '
        + 'unnormalized interval ratio binned into 0.05 windows of molecular '
        + 'time (median line, IQR band over that bin’s segment-intervals). '
        + (ga ? 'Gold line = genome-wide base rate D<sub>sig</sub>/D<sub>clock</sub> = '
            + ga.ratio.toFixed(2) + '× (D<sub>sig</sub>=' + Math.round(ga.D_sig)
            + ', D<sub>clock</sub>=' + Math.round(ga.D_clock) + ', ' + ga.n_seg
            + ' segs) — the abundance baseline; curve below it = ' + sig
            + ' depleted in that window, above = enriched. ' : '')
        + 'These are polytope solutions collapsed to a point estimate, not draws.';
      sigActPanel.appendChild(cap);
      sigActPanel.style.display = 'block';
    }

    function selectSig(sig){
      activeSig = sig;
      renderBandsForSig(sig);
      if (sig === 'clock'){ sigActPanel.style.display = 'none'; }
      else { renderSigActivity(sig); }
    }

    // toggle UI in the filter bar
    var sigWrap = document.createElement('span');
    sigWrap.style.cssText = 'margin-left:18px;';
    var sigLbl = document.createElement('b');
    sigLbl.textContent = 'Band timing weighting: ';
    var sigSel = document.createElement('select');
    sigSel.style.cssText = 'margin:0 6px;padding:2px 6px;';
    SIG_NAMES.forEach(function(s){
      var o = document.createElement('option');
      o.value = s;
      var sh = SHARE_BY[s];
      o.textContent = (s === 'clock' ? 'Clock' : s)
        + (sh != null && sh > 0 ? ' (' + (100*sh).toFixed(0) + '%)' : '');
      sigSel.appendChild(o);
    });
    sigSel.value = 'clock';
    sigSel.onchange = function(){ selectSig(sigSel.value); };
    var sigNote = document.createElement('span');
    sigNote.style.cssText = 'color:#888;font-size:11px;margin-left:6px;';
    sigNote.textContent = '(bands only; track/legend/events stay clock)';
    sigWrap.appendChild(sigLbl); sigWrap.appendChild(sigSel);
    sigWrap.appendChild(sigNote);
    bar.appendChild(sigWrap);

    // mount the activity panel below the comparison container.
    cmpPanel.parentNode.insertBefore(sigActPanel, cmpPanel.nextSibling);
  }

  gd.parentNode.insertBefore(bar, gd);
})();
"""


def genome_to_html(genome, cluster_df=None, output_html=None, sample=None,
                   clustered_genome=None, ess_thresh=10, tree_top_k=5, env=None,
                   show_spectra=True, sig_timing=None):
    """Write a self-contained interactive HTML timing overview for *genome*.

    Parameters
    ----------
    genome : Genome
        The (original) Tau genome.  Filled timing bands, the CN legend and the
        right histogram are built from its timed segments.  Segments below
        ``ess_thresh`` effective mutations or without a best route are skipped.
    cluster_df : pandas.DataFrame, optional
        Event/cluster table (``*.time_clusters.tsv``) with at least ``time``
        and ``classification`` columns; used for event lines and the histogram.
    output_html : str or pathlib.Path
        Destination ``.html`` path (Plotly JS inlined → opens offline).
    sample : str, optional
        Sample id for the title.
    clustered_genome : Genome, optional
        The clustered genome; drives the top multiplicity-cluster track.
        Defaults to *genome* (so passing a loaded ``*.clustered_genome.pkl.gz``
        as *genome* populates the track automatically).
    ess_thresh : float
        Minimum summed ``N_counts`` for a segment to be drawn (default 10).
    tree_top_k : int
        Number of routes per timed segment to make interactively explorable
        (the K with the lowest ``dist_rel``).  Each gets a precomputed
        "tree solution with knobs" payload embedded in the HTML; expanding a
        route row shows its allele tree + one slider per free variable.
        ``0`` skips all tree precompute (fast path, no Sage).  Default 5.
    env : tau.timing.RouteEnv, optional
        Pre-loaded route environment shared across all tree payloads (caches
        per-route symbolic prep).  If ``None`` and ``tree_top_k > 0`` one is
        loaded for the CN states present (requires Sage); on failure the trees
        are skipped and the rest of the viewer still generates.
    show_spectra : bool
        When ``True`` (default) add a 96-channel mutational-spectrum (SBS96)
        panel with a signature-weighting ``<select>`` toggle to both the segment
        detail panel and the cluster panel.  Built entirely from each segment's
        ``snv_table`` (``sbs96``/``mut_w``/``hard_sig_choice`` + per-context
        signature emission columns) — no external files and no Sage.  The toggle
        offers Unweighted, Clock (SBS1+SBS5), and each grouped COSMIC signature
        whose sample share exceeds 10%.  Per-option 96-channel weighted sums are
        precomputed and embedded (96 × n_options floats per segment/cluster — no
        per-SNV rows).  Automatically disabled when no ``snv_table`` carries
        SBS96 data.
    sig_timing : str, pathlib.Path or dict, optional
        Per-signature timing cache (``<sample>.sig_timing.json.gz`` path or an
        already-loaded dict in that schema).  When present and carrying more
        than one signature, a signature toggle is added near the filter bar:
        switching to a signature S re-renders the main-panel filled bands so
        each segment's stacked-band heights use that signature's cumulative
        event times (same x positions, same CN colours, fixed axes), and a
        per-sample signature-activity curve (interval ratio vs clock-time) is
        shown.  The values are *polytope solutions* collapsed to a point
        estimate — never draws or a distribution.  ``None`` (default) leaves
        the viewer unchanged.

    Returns
    -------
    str
        The path to the written HTML file.
    """
    import time as _time
    import plotly.io as pio

    if output_html is None:
        raise ValueError("output_html is required")

    _t0 = _time.time()

    # The multiplicity-cluster track needs a clustered genome.  When the caller
    # passes a clustered genome (the common case — loading *.clustered_genome.pkl.gz),
    # reuse it so the track populates without requiring a separate argument.
    # An unclustered genome degrades gracefully to an empty track.
    if clustered_genome is None:
        clustered_genome = genome

    chrom_off, _genome_len, autosome_len = _genome_offsets()

    # ── sample-wide SBS96 spectrum setup (self-contained from snv_tables) ────
    spectrum_setup, spectrum_options, spectrum_meta = None, [], {}
    if show_spectra:
        spectrum_setup, spec_info = _build_spectrum_setup(genome)
        if spectrum_setup is not None:
            spectrum_options = spectrum_setup["options"]
            val = spec_info.get("validation", {})
            spectrum_meta = dict(
                soft_path=bool(spec_info["soft_path"]),
                validation=val,
                grp_share=spec_info.get("grp_share", {}),
            )
            path = "soft-reconstruction" if spec_info["soft_path"] else "HARD-assignment fallback"
            if val.get("available"):
                print(f"[viz] SBS96 spectrum: {len(spectrum_options)} option(s) "
                      f"{spec_info['options']}; weighting path = {path} "
                      f"(clock recon vs mut_w: MAE={val['mae']:.4f}, "
                      f"corr={val['corr']:.4f}, n={val['n']})")
            else:
                print(f"[viz] SBS96 spectrum: {len(spectrum_options)} option(s) "
                      f"{spec_info['options']}; weighting path = {path} "
                      f"(validation gate unavailable — no mut_w/clock cols)")
        else:
            print(f"[viz] SBS96 spectrum disabled ({spec_info.get('reason')})")

    (band_polys, boundary_lines, hit_records, detail_records,
     tree_targets, cn_states_present, state_nbands) = _segment_bands(
        genome, chrom_off, ess_thresh=ess_thresh, tree_top_k=tree_top_k,
        spectrum_setup=spectrum_setup, env=env,
    )

    # ── interactive tree payloads (top-K routes per segment) ────────────────
    tree_payloads, expandable = {}, {}
    if tree_top_k > 0 and tree_targets:
        if env is None and cn_states_present:
            try:
                from tau.timing import load_routes_for_states
                env = load_routes_for_states(list(cn_states_present))
            except Exception as exc:  # Sage unavailable / load failed → no trees
                print(f"[viz] tree precompute disabled (route env load failed: {exc})")
                env = None
        if env is not None:
            _te = _time.time()
            tree_payloads, expandable, n_fail = _build_tree_payloads(
                genome, tree_targets, env, tree_top_k
            )
            n_routes = sum(len(v) for v in tree_payloads.values())
            print(f"[viz] built {n_routes} tree payload(s) for "
                  f"{len(tree_payloads)} segment(s) in {_time.time()-_te:.1f}s "
                  f"(top_k={tree_top_k}, {n_fail} route extraction(s) failed/skipped)")

    track_shapes, cn_states, max_cluster_idx, cluster_payloads = _cluster_track(
        clustered_genome, chrom_off, detail_records=detail_records,
        spectrum_setup=spectrum_setup, orig_genome=genome,
        tree_env=env, tree_top_k=tree_top_k,
    )
    hist_data = _timing_hist(genome, cluster_df)

    fig = _build_figure(
        band_polys, boundary_lines, hit_records, track_shapes, cn_states,
        max_cluster_idx, cluster_df, hist_data, chrom_off, autosome_len, sample,
        cn_profile=_cn_profile_data(genome, chrom_off),
        legend_states=cn_states_present, legend_nbands=state_nbands,
    )

    div_id = "tau-overview"
    records_json = json.dumps(detail_records)
    hits_json = json.dumps(
        [dict(chrom=r["chrom"], major_cn=r["major_cn"]) for r in hit_records]
    )
    chrom_list = json.dumps(_AUTOSOMES)
    # Per-chromosome genomic window [x_start, x_end] for the chrom-zoom dropdown.
    chrom_spans = {
        c: [chrom_off[c], chrom_off[c] + HG19_SIZES[c]]
        for c in _AUTOSOMES if HG19_SIZES.get(c, 0) > 0
    }
    chrom_spans_json = json.dumps(chrom_spans)
    auto_len_json = json.dumps(autosome_len)
    clusters_json = json.dumps(cluster_payloads)
    trees_json = json.dumps(tree_payloads)
    expandable_json = json.dumps(expandable)

    # ── SBS96 spectrum embed: option metadata + channel layout/colours ──────
    # Per-channel colour string (one per the 96 ordered channels).
    chan_colors = [_SBS6_COLORS[c.split("[")[1].split("]")[0]]
                   for c in _SBS96_CHANNELS]
    spectrum_embed = dict(
        enabled=bool(spectrum_setup is not None),
        options=[
            dict(key=o["key"], label=o["label"], kind=o["kind"],
                 sig=o["sig"], share=o["share"])
            for o in spectrum_options
        ],
        default=("clock" if any(o["key"] == "clock" for o in spectrum_options)
                 else (spectrum_options[0]["key"] if spectrum_options else None)),
        channels=_SBS96_CHANNELS,
        chan_colors=chan_colors,
        sub_types=_SBS6_TYPES,
        sub_colors=_SBS6_COLORS,
        soft_path=bool(spectrum_meta.get("soft_path", True)),
        validation=spectrum_meta.get("validation", {}),
    )
    spectrum_json = json.dumps(spectrum_embed)

    # ── per-CN-state pooled VAF histograms (for the "VAF by CN state" section) ─
    cnstate_vaf_json = json.dumps(_cnstate_vaf(genome))

    # WGD/PGD event landmarks for the tree panels (tree y-axis == molecular time,
    # same 0..1 scale as event times, so each landmark is a horizontal line).
    _ev_rows = []
    if cluster_df is not None and not cluster_df.empty:
        for _, _row in cluster_df.iterrows():
            _cls = str(_row.get("classification", ""))
            if _cls not in ("WGD", "ccPG", "PGD"):
                continue
            _t = _row.get("time")
            if _t is None or (isinstance(_t, float) and _t != _t):
                continue
            _gf = _row.get("pct_of_theoretical_genome")
            if _gf is None and _row.get("gf") is not None:
                _gf = float(_row.get("gf")) * 100.0
            _ev_rows.append(dict(t=float(_t), cls=_cls,
                                 gf=(None if _gf is None else round(float(_gf), 1))))
    events_json = json.dumps(sorted(_ev_rows, key=lambda r: r["t"]))

    # ── per-signature timing cache (signature toggle + activity curve) ──────
    sig_embed = _load_sig_timing(sig_timing)
    sig_bands = {}
    # PER-SEGMENT activity is embedded as its own global SEG_ACTIVITY and drives the
    # click-detail panel INDEPENDENTLY of the overview band toggle (which needs >1
    # signature).  Pull it out before sig_embed may be reset to None below.
    seg_activity = (sig_embed or {}).get("seg_activity") or {}
    if sig_embed is not None and len(sig_embed.get("signatures", [])) > 1:
        n_sig = len(sig_embed["signatures"])
        n_act = len(sig_embed.get("activity", {}))
        n_seg = {s: len(sig_embed["segment_times"].get(s, {}))
                 for s in sig_embed["segment_times"]}
        # Prebuild each signature's full band-polygon set with the SHARED helper
        # so the toggle swaps in geometry IDENTICAL to the clock overview
        # (multi-point x-spans for underdetermined segments, not flat rectangles).
        sig_bands = _sig_band_polys(
            genome, chrom_off, sig_embed.get("segment_mats") or {}
        )
        n_band = {s: len(sig_bands.get(s, {})) for s in sig_bands}
        print(f"[viz] signature timing: {n_sig} signature(s) "
              f"{[s['name'] for s in sig_embed['signatures']]}; "
              f"segments timed {n_seg}; band-trace sets {n_band}; "
              f"activity curves for {n_act} signature(s)")
        # The full mats only fed the Python polygon build; drop them from the
        # embed (the JS reads prebuilt SIG_BANDS instead) to keep SIG_TIMING small.
        # seg_activity has its own SEG_ACTIVITY global; drop from SIG_TIMING too.
        sig_embed = {k: v for k, v in sig_embed.items()
                     if k not in ("segment_mats", "seg_activity")}
    else:
        sig_embed = None  # <=1 signature → nothing to toggle, feature OFF
        sig_bands = {}
    sig_timing_json = json.dumps(sig_embed)
    sig_bands_json = json.dumps(sig_bands)
    seg_activity_json = json.dumps(seg_activity)

    post_script = (
        _POST_SCRIPT
        .replace("__DIV_ID__", div_id)
        .replace("__RECORDS_JSON__", records_json)
        .replace("__HITS_JSON__", hits_json)
        .replace("__CLUSTERS_JSON__", clusters_json)
        .replace("__TREES_JSON__", trees_json)
        .replace("__EXPANDABLE_JSON__", expandable_json)
        .replace("__SPECTRUM_JSON__", spectrum_json)
        .replace("__CNSTATE_VAF_JSON__", cnstate_vaf_json)
        .replace("__CHROM_LIST__", chrom_list)
        .replace("__CHROM_SPANS__", chrom_spans_json)
        .replace("__AUTO_LEN__", auto_len_json)
        .replace("__EVENTS_JSON__", events_json)
        .replace("__SIG_TIMING_JSON__", sig_timing_json)
        .replace("__SIG_BANDS_JSON__", sig_bands_json)
        .replace("__SEG_ACTIVITY_JSON__", seg_activity_json)
    )

    html = pio.to_html(
        fig,
        include_plotlyjs="inline",
        full_html=True,
        div_id=div_id,
        post_script=post_script,
        config={"responsive": True, "displaylogo": False,
                "modeBarButtonsToAdd": ["select2d", "lasso2d"]},
    )

    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    size_mb = out.stat().st_size / 1e6
    print(f"[viz] wrote {out}  ({size_mb:.2f} MB, "
          f"total generation time {_time.time()-_t0:.1f}s)")
    return str(out)


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Build an interactive Tau timing overview HTML.")
    ap.add_argument("--genome", required=True, help="genome.pkl.gz")
    ap.add_argument("--clustered", default=None, help="clustered_genome.pkl.gz")
    ap.add_argument("--clusters", default=None, help="time_clusters.tsv")
    ap.add_argument("--out", required=True, help="output .html")
    ap.add_argument("--sample", default=None)
    ap.add_argument(
        "--tree-top-k", type=int, default=5,
        help="number of lowest-dist_rel routes per segment to make interactively "
             "explorable (tree + knobs); 0 disables tree precompute (default 5)",
    )
    ap.add_argument(
        "--no-spectra", action="store_true",
        help="disable the SBS96 mutational-spectrum panel + weighting toggle",
    )
    ap.add_argument(
        "--sig-timing", default=None,
        help="per-signature timing cache (<sample>.sig_timing.json.gz) for the "
             "signature toggle + activity curve; defaults to the sibling file "
             "next to --genome if present",
    )
    ap.add_argument(
        "--no-sig-timing", action="store_true",
        help="disable the signature toggle + activity curve even if a sibling "
             "<sample>.sig_timing.json.gz exists",
    )
    args = ap.parse_args()

    g = load_genome(args.genome)
    cg = load_genome(args.clustered) if args.clustered else None
    cdf = pd.read_csv(args.clusters, sep="\t") if args.clusters else None

    # Resolve the per-signature timing cache: explicit --sig-timing wins; else
    # auto-find the sibling <stem>.sig_timing.json.gz next to --genome (the stem
    # strips the ``.genome.pkl.gz`` / ``.clustered_genome.pkl.gz`` suffix).
    sig_timing = None
    if not args.no_sig_timing:
        if args.sig_timing:
            sig_timing = args.sig_timing
        else:
            gp = Path(args.genome)
            stem = gp.name
            for suf in (".genome.pkl.gz", ".clustered_genome.pkl.gz",
                        ".pkl.gz", ".pkl"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            cand = gp.parent / f"{stem}.sig_timing.json.gz"
            if cand.exists():
                sig_timing = str(cand)
                print(f"[viz] auto-found signature timing cache: {cand}")

    path = genome_to_html(
        g, cluster_df=cdf, output_html=args.out, sample=args.sample,
        clustered_genome=cg, tree_top_k=args.tree_top_k,
        show_spectra=not args.no_spectra, sig_timing=sig_timing,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    _cli()
