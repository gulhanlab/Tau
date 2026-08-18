#!/usr/bin/env python3
"""
tau/sig_timing.py - Per-signature re-timing engine for Tau.

Re-times a single sample under a chosen mutational-signature weighting, using
ONLY the information already stored inside one ``genome.pkl.gz`` (no external
exposure files, no signature catalogue). The viewer can therefore toggle
between signatures by calling :func:`retime_under_signature` on demand.

Why this works
--------------
The per-SNV weight ``mut_w`` in ``seg.snv_table`` is exactly the EM weight read
by :meth:`tau.core.Segment.estimate_pi_em`. EM sets
``N_counts = pi * w.sum()`` where ``pi = _em_pi(L, w)`` and ``L`` is the
per-SNV multiplicity-likelihood matrix (``snv_table["multiplicity_likelihoods"]``),
which depends only on VAF / depth / purity, NOT on signature. So replacing
``mut_w`` with a signature-specific soft posterior and re-running the *real*
EM + timing pipeline yields signature-specific ``N_counts`` (hence timing)
while leaving ``L`` untouched.

Method (mirrors dev/pcawg/signatures/ + the signature-activity memory)
----------------------------------------------------------------------
* Exposure estimate ``e_hat[s]`` per ungrouped COSMIC signature = fraction of
  the sample's SNVs whose ``hard_sig_choice == s`` (pooled over segments). This
  is a validated proxy (reconstructed clock soft-weight vs stored ``mut_w``
  correlation ~0.997).
* Signature grouping collapses letter-suffixed variants
  (SBS7a/b/c/d -> SBS7, SBS17a/b -> SBS17, SBS10a.. -> SBS10) and pools
  SBS2+SBS13 -> APOBEC. "clock" = {SBS1, SBS5}.
* Soft posterior weight for a signature group g, per SNV:
  ``mut_w_g = (sum_{s in g} C[s][ctx] * e_hat[s]) / (sum_{all s'} C[s'][ctx] * e_hat[s'])``
  using the snv_table emission columns ``C`` (= COSMIC emission per context,
  identical for SNVs sharing a context) and the exposure estimate.

Terminology
-----------
The per-route entries returned by the timing pipeline are POLYTOPE SOLUTIONS,
not draws / samples / a posterior. We mean the equally-valid points of an
underdetermined solution space.
"""
from __future__ import annotations

import copy
import re
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tau.core import Genome, _em_pi
from tau.timing import _segment_time_segment
from tau.utils import order_t, pick_best_key, extract_mat

# ---------------------------------------------------------------------------
# Signature grouping
# ---------------------------------------------------------------------------

# Signatures whose lettered variants collapse to a single biological group.
CLOCK_SIGNATURES = ("SBS1", "SBS5")

# Hard-coded multi-signature groups (members that do NOT share a common prefix).
_NAMED_GROUPS: Dict[str, tuple] = {
    "APOBEC": ("SBS2", "SBS13"),
}

# matches a COSMIC signature with an optional trailing lowercase letter group,
# e.g. SBS7a -> base "SBS7", SBS17b -> base "SBS17", SBS5 -> base "SBS5".
_SIG_LETTER = re.compile(r"^(SBS\d+)[a-z]?$")


def _base_signature(name: str) -> str:
    """Collapse a lettered COSMIC variant to its base name (SBS7a -> SBS7)."""
    m = _SIG_LETTER.match(str(name))
    return m.group(1) if m else str(name)


def signature_columns(genome: Genome) -> List[str]:
    """All COSMIC signature emission columns present in the genome's snv_table."""
    for seg in genome.segments:
        if seg.n_snvs:
            return [c for c in seg.snv_table.columns
                    if re.fullmatch(r"SBS\d+[a-z]?", str(c))]
    return []


def group_members(genome: Genome, group_name: str) -> List[str]:
    """Resolve a grouped signature name to the COSMIC columns present in *genome*.

    ``"clock"`` -> {SBS1, SBS5}; ``"APOBEC"`` -> {SBS2, SBS13};
    ``"SBS17"`` -> all SBS17* columns; an explicit column name -> itself.
    """
    cols = set(signature_columns(genome))
    if group_name == "clock":
        wanted = set(CLOCK_SIGNATURES)
    elif group_name in _NAMED_GROUPS:
        wanted = set(_NAMED_GROUPS[group_name])
    else:
        base = _base_signature(group_name)
        wanted = {c for c in cols if _base_signature(c) == base}
        wanted.add(group_name)
    return sorted(c for c in cols if c in wanted)


# ---------------------------------------------------------------------------
# Exposure estimate
# ---------------------------------------------------------------------------

def sample_exposure_estimate(genome: Genome) -> Dict[str, float]:
    """Per-signature exposures, normalized to fractions (keys = COSMIC names).

    Prefers the REAL MuSiCal exposures stored on the genome as
    ``genome.exposures`` (a ``{signature: value}`` mapping) — these are the exact
    soft exposures the sample was timed with (validated: they reproduce the stored
    clock ``mut_w`` exactly). Falls back to a hard-assignment estimate from
    ``hard_sig_choice`` fractions only when no exposures are stored on the genome.
    """
    stored = getattr(genome, "exposures", None)
    if stored:
        d = {str(k): float(v) for k, v in dict(stored).items() if float(v) > 0}
        tot = sum(d.values())
        if tot > 0:
            return {k: v / tot for k, v in d.items()}

    # Fallback: hard-assignment fractions (used when exposures aren't stored).
    counts: Dict[str, float] = {}
    total = 0
    for seg in genome.segments:
        if not seg.n_snvs or "hard_sig_choice" not in seg.snv_table.columns:
            continue
        vc = seg.snv_table["hard_sig_choice"].astype(str).value_counts()
        for sig, n in vc.items():
            if sig in ("nan", "None", ""):
                continue
            counts[sig] = counts.get(sig, 0.0) + float(n)
            total += int(n)
    if total <= 0:
        return {}
    return {s: c / total for s, c in counts.items()}


# ---------------------------------------------------------------------------
# Active signatures
# ---------------------------------------------------------------------------

def active_signatures(genome: Genome, thresh: float = 0.15,
                      expo: Optional[Dict[str, float]] = None) -> List[dict]:
    """Grouped signatures whose sample share exceeds *thresh*, always incl. clock.

    Returns a list of specs, each::

        {"name": "SBS17", "members": ["SBS17a", "SBS17b"], "share": 0.69}

    Share is the summed exposure estimate over the group's members. "clock"
    (SBS1+SBS5) is always included regardless of its share.
    """
    if expo is None:
        expo = sample_exposure_estimate(genome)

    # Bucket every observed ungrouped signature into its display group.
    groups: Dict[str, set] = {}
    for sig in expo:
        if sig in CLOCK_SIGNATURES:
            gname = "clock"
        elif sig in _NAMED_GROUPS["APOBEC"]:
            gname = "APOBEC"
        else:
            gname = _base_signature(sig)
        groups.setdefault(gname, set()).add(sig)

    specs = []
    for gname, members in groups.items():
        share = float(sum(expo.get(m, 0.0) for m in members))
        if gname == "clock" or share > thresh:
            specs.append(dict(
                name=gname,
                members=sorted(group_members(genome, gname)),
                share=share,
            ))

    # Guarantee clock is present even if no SBS1/SBS5 SNVs were hard-assigned.
    if not any(s["name"] == "clock" for s in specs):
        specs.append(dict(name="clock",
                          members=sorted(group_members(genome, "clock")),
                          share=float(sum(expo.get(m, 0.0) for m in CLOCK_SIGNATURES))))

    specs.sort(key=lambda s: (s["name"] != "clock", -s["share"]))
    return specs


# ---------------------------------------------------------------------------
# Per-SNV soft posterior weights
# ---------------------------------------------------------------------------

def signature_mut_w(genome: Genome, members: List[str],
                    expo: Optional[Dict[str, float]] = None
                    ) -> Dict[str, np.ndarray]:
    """Per-SNV soft posterior weights for signature group *members*.

    For each SNV with emission C[s][ctx] and exposure estimate e_hat[s]::

        w = sum_{s in members} C[s] * e_hat[s]   /   sum_{all s'} C[s'] * e_hat[s']

    Returns ``{seg_id -> np.ndarray}`` of length ``n_snvs`` (in snv_table order).
    """
    if expo is None:
        expo = sample_exposure_estimate(genome)
    all_cols = signature_columns(genome)
    members = [m for m in members if m in all_cols]

    out: Dict[str, np.ndarray] = {}
    for seg in genome.segments:
        n = seg.n_snvs
        if n == 0:
            out[seg.seg_id] = np.zeros(0, float)
            continue
        t = seg.snv_table
        denom = np.zeros(n, float)
        for s in all_cols:
            e = expo.get(s, 0.0)
            if e <= 0:
                continue
            denom += pd.to_numeric(t[s], errors="coerce").fillna(0.0).to_numpy(float) * e
        numer = np.zeros(n, float)
        for s in members:
            e = expo.get(s, 0.0)
            if e <= 0:
                continue
            numer += pd.to_numeric(t[s], errors="coerce").fillna(0.0).to_numpy(float) * e
        w = np.where(denom > 0, numer / denom, 0.0)
        out[seg.seg_id] = w
    return out


# ---------------------------------------------------------------------------
# Re-timing
# ---------------------------------------------------------------------------

def _retimed_genome(genome: Genome, members: List[str], env,
                    expo: Optional[Dict[str, float]] = None) -> Genome:
    """Deep-copy *genome*, install signature mut_w, re-run EM + timing.

    The copy is fully self-contained: the input genome is never mutated. EM is
    re-run with ``bootstrap_B=1`` so ``N_counts_boot`` holds a single (point)
    replicate per segment — the polytope solutions still span the route's free
    variables, only the SNV-resampling spread is collapsed.
    """
    if expo is None:
        expo = sample_exposure_estimate(genome)
    g = copy.deepcopy(genome)
    w_map = signature_mut_w(g, members, expo)
    for seg in g.segments:
        if seg.n_snvs:
            seg.snv_table = seg.snv_table.copy()
            seg.snv_table["mut_w"] = w_map.get(seg.seg_id, np.zeros(seg.n_snvs))
    g.calculate_multiplicities(bootstrap_B=1)
    for seg in g.segments:
        _segment_time_segment(seg, env=env)
    return g


def retimed_genome(genome: Genome, members: List[str], env,
                   expo: Optional[Dict[str, float]] = None) -> Genome:
    """Public alias for :func:`_retimed_genome`.

    Returns a fully re-timed ``Genome`` — structurally identical to a normally timed one, so it can
    be handed straight to ``tau.aggregate.export.build_all_tables`` to emit the SAME table set under
    a signature weighting.
    """
    return _retimed_genome(genome, members, env, expo)


def retime_under_signature(genome: Genome, members: List[str], env,
                           expo: Optional[Dict[str, float]] = None) -> dict:
    """Re-time every segment under a signature group's weighting.

    Returns a payload the viewer can consume directly::

        {
          "members": [...],
          "exposure": {sig: e_hat, ...},
          "segments": {
             seg_id: {
                "key": best route key,
                "event_times": [cumulative event times = cumsum(order_t(t))[:-1]],
                "mat": [[...], ...],        # full polytope-solution timing matrix
                                            # extract_mat(retimed_seg, key, 0),
                                            # shape n_x x K, ~5 sig figs
                "N_counts": [...],          # signature-weighted EM counts
                "dist_rel": float,
                "total_sum": float,         # calc_total_sum(N) = D / total_cn
                "n_snv": int,
                "cn_state": "major_minor",
             }, ...
          }
        }

    Only segments that were successfully timed (a best route with at least one
    polytope solution) appear in ``segments``.
    """
    g = _retimed_genome(genome, members, env, expo)
    if expo is None:
        expo = sample_exposure_estimate(genome)

    seg_out: Dict[str, dict] = {}
    for seg in g.segments:
        tr = getattr(seg, "timing_result", None)
        if not tr:
            continue
        key = pick_best_key(seg)
        if key is None:
            continue
        mat = extract_mat(seg, key, boot_id=0)
        if mat is None or mat.size == 0:
            continue
        # mean polytope solution -> cumulative event times
        t_mean = mat.mean(axis=0)
        s = t_mean.sum()
        t_mean = t_mean / s if s > 0 else t_mean
        event_times = np.cumsum(t_mean)[:-1].tolist()
        # full polytope-solution timing matrix (n_x x K), rounded to ~5 sig
        # figs, so the viewer can rebuild the SAME stacked-band geometry as the
        # clock overlay (not just flat rectangles from scalar event times).
        mat_round = np.round(np.asarray(mat, float), 5)
        mat_list = [[float(v) for v in row] for row in mat_round]

        N = np.asarray(getattr(seg, "N_counts", np.zeros(0)), float)
        D = float(sum((k + 1) * N[k] for k in range(len(N))))
        total_sum = D / seg.total_cn if seg.total_cn > 0 else 0.0
        qc0 = tr[key].get("qc", [{}])
        dist_rel = float(qc0[0].get("dist_rel", np.nan)) if qc0 else float("nan")

        seg_out[seg.seg_id] = dict(
            key=key,
            event_times=event_times,
            mat=mat_list,
            N_counts=N.tolist(),
            dist_rel=dist_rel,
            total_sum=float(total_sum),
            n_snv=int(seg.n_snvs),
            cn_state=f"{seg.major_cn}_{seg.minor_cn}",
        )

    return dict(members=list(members), exposure=expo, segments=seg_out)


# ---------------------------------------------------------------------------
# Activity ratio (per-interval signature force vs clock)
# ---------------------------------------------------------------------------

_WMIN = 0.01  # raw interval-width floor (matches sig_overview_ratio.py)


def _D(N: np.ndarray) -> float:
    """Doubling budget D = sum_k (k+1) * N_counts[k]."""
    N = np.asarray(N, float)
    return float(sum((k + 1) * N[k] for k in range(len(N))))


def activity_ratio(genome: Genome, members: List[str], env,
                   expo: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Per-(segment, interval) unnormalized signature/clock activity ratio.

    For each segment timed under BOTH the signature and the clock with the SAME
    best route key, and each gain interval i::

        ratio_i = (ws_i / wc_i) * (D_sig / D_clock)

    where ``ws_i``/``wc_i`` are the mean raw interval widths under signature/clock
    for that route (mean over polytope solutions,
    ``extract_mat(seg, key, boot_id=0).mean(axis=0)``), floored at WMIN, and
    ``D = sum_k (k+1) N_counts[k]``. ``abund = D_sig/D_clock``. This is the
    METRIC='unnorm' metric from dev/pcawg/signatures/sig_overview_ratio.py.

    ``clock_time`` is the CLOCK cumulative event time at interval boundary i.

    Returns a DataFrame with columns:
    ``seg_id, cn_state, interval, clock_time, ws, wc, abund, ratio``.
    """
    if expo is None:
        expo = sample_exposure_estimate(genome)

    gc = _retimed_genome(genome, group_members(genome, "clock"), env, expo)
    gs = _retimed_genome(genome, members, env, expo)
    seg_s = {s.seg_id: s for s in gs.segments}

    rows = []
    for cseg in gc.segments:
        if not getattr(cseg, "timing_result", None):
            continue
        sseg = seg_s.get(cseg.seg_id)
        if sseg is None or not getattr(sseg, "timing_result", None):
            continue
        key = pick_best_key(cseg)
        if key is None or key not in sseg.timing_result:
            continue
        mc = extract_mat(cseg, key, boot_id=0)
        ms = extract_mat(sseg, key, boot_id=0)
        if mc is None or ms is None or mc.size == 0 or ms.size == 0:
            continue
        if mc.shape[1] != ms.shape[1]:
            continue
        nc = np.asarray(getattr(cseg, "N_counts", np.zeros(0)), float)
        ns = np.asarray(getattr(sseg, "N_counts", np.zeros(0)), float)
        if _D(nc) <= 0 or _D(ns) <= 0:
            continue
        abund = _D(ns) / _D(nc)

        wc_raw = mc.mean(axis=0)
        ws_raw = ms.mean(axis=0)
        # clock cumulative event times (interval boundaries)
        clock_cum = np.cumsum(wc_raw / wc_raw.sum() if wc_raw.sum() > 0 else wc_raw)
        K = len(wc_raw)
        for i in range(K):
            ratio = (max(ws_raw[i], _WMIN) / max(wc_raw[i], _WMIN)) * abund
            rows.append(dict(
                seg_id=cseg.seg_id,
                cn_state=f"{cseg.major_cn}_{cseg.minor_cn}",
                interval=i,
                clock_time=float(clock_cum[i]),
                ws=float(ws_raw[i]),
                wc=float(wc_raw[i]),
                abund=float(abund),
                ratio=float(ratio),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Viewer signature-timing cache  (<sample>.sig_timing.json.gz payload)
# ---------------------------------------------------------------------------
# Aggregate-curve gates (see memory signature-activity-timing): an interval enters
# the cohort curve only if BOTH polytope spreads < SPREAD_CUT and the clock
# t-interval (denominator) ct > MIN_CLOCK_T. _WMIN is the divide-by-0 floor.
SPREAD_CUT = 0.01    # both clock & sig per-interval polytope spread below this = determined
MIN_CLOCK_T = 0.01   # clock t-interval denominator must exceed this (= _WMIN floor)


def clock_table(genome: Genome) -> Dict[str, dict]:
    """Per-segment stored clock timing: best route key, mean normalized t-widths,
    cumulative event times, full polytope matrix, D, eff_N, cn_state."""
    out: Dict[str, dict] = {}
    for seg in genome.segments:
        if not getattr(seg, "timing_result", None):
            continue
        k = pick_best_key(seg)
        if k is None:
            continue
        m = extract_mat(seg, k, boot_id=0)
        if m is None or m.size == 0:
            continue
        w = m.mean(0)
        s = w.sum()
        w = w / s if s > 0 else w
        out[seg.seg_id] = dict(
            key=k, widths=w.tolist(),
            event_times=np.cumsum(w)[:-1].tolist(),
            mat=[[round(float(v), 5) for v in row] for row in m],
            D=_D(getattr(seg, "N_counts", np.zeros(0))),
            eff_N=float(np.sum(getattr(seg, "N_counts", np.zeros(0)))),
            cn_state=f"{seg.major_cn}_{seg.minor_cn}",
        )
    return out


def exposures_from_file(path, sample) -> Dict[str, float]:
    """Load one sample's signature exposures from a fitter's output (e.g. MuSiCal), normalised.

    Accepts a `sample` or `aliquot_id` key column plus one column per `SBS*`. Use this to make the
    FILE authoritative for the signature step: ``genome.exposures`` only holds whatever was baked in
    at preprocessing time, and is absent entirely from genomes timed before that was stored — in which
    case the estimate silently falls back to hard-assignment fractions.
    """
    from tau.preprocessing import _load_exposures
    v = _load_exposures(path, sample)
    return {str(k): float(x) for k, x in v.items() if float(x) > 0}


def build_sig_payload(genome: Genome, env, thresh: float = 0.15,
                      verbose: bool = True, expo: Optional[Dict[str, float]] = None) -> dict:
    """Build the full per-signature timing payload for one genome.

    Detects active signatures (clock + exposure share > ``thresh``), re-times each
    non-clock signature once, and assembles the cache the viewer embeds:
    ``segment_times``, ``segment_mats`` (full polytope geometry per signature), the
    gated/ESS-weighted aggregate ``activity`` rows (each interval carries its
    clock-time span ``[t0, t1]`` so the curve is drawn as a spread, not a point),
    the per-segment ``seg_activity`` blocks (all intervals, ``ok`` determinacy flag),
    and the genome-wide ``genome_abund`` base rate. Returns the payload dict; the
    caller writes it to ``<sample>.sig_timing.json.gz``.
    """
    g = genome
    sample = getattr(g, "sample", None) or getattr(g, "name", None) or "sample"
    expo = expo or sample_exposure_estimate(g)
    specs = active_signatures(g, thresh, expo)
    if verbose:
        print(f"{sample}: active signatures "
              f"{[(s['name'], round(s['share'], 2)) for s in specs]}", flush=True)
    clk = clock_table(g)
    payload = dict(
        sample=sample, thresh=thresh, spread_cut=SPREAD_CUT, min_clock_t=MIN_CLOCK_T,
        signatures=[dict(name=s["name"], share=s["share"], members=s["members"])
                    for s in specs],
        segment_times={"clock": {sid: v["event_times"] for sid, v in clk.items()}},
        segment_mats={"clock": {sid: v["mat"] for sid, v in clk.items()}},
        activity={}, seg_activity={}, genome_abund={},
    )
    for spec in specs:
        if spec["name"] == "clock":
            continue
        t0w = time.time()
        re_segs = retime_under_signature(g, spec["members"], env, expo)["segments"]
        payload["segment_times"][spec["name"]] = {
            sid: v["event_times"] for sid, v in re_segs.items()}
        payload["segment_mats"][spec["name"]] = {
            sid: v["mat"] for sid, v in re_segs.items()}
        rows: List[dict] = []
        seg_block: Dict[str, dict] = {}
        n_total = 0
        Ds_tot = 0.0; Dc_tot = 0.0; n_common = 0   # genome-wide base-rate accumulators
        for sid, sv in re_segs.items():
            if sid not in clk:
                continue
            Ds_tot += _D(sv["N_counts"]); Dc_tot += clk[sid]["D"]; n_common += 1
            if clk[sid]["key"] != sv["key"]:
                continue
            cmat = np.asarray(clk[sid]["mat"], float)
            smat = np.asarray(sv["mat"], float)
            if cmat.ndim != 2 or smat.ndim != 2 or cmat.shape[1] != smat.shape[1]:
                continue
            ct = cmat.mean(0); ct = ct / ct.sum() if ct.sum() > 0 else ct
            sigt = smat.mean(0); sigt = sigt / sigt.sum() if sigt.sum() > 0 else sigt
            cspread = cmat.max(0) - cmat.min(0)
            sspread = smat.max(0) - smat.min(0)
            abund = _D(sv["N_counts"]) / clk[sid]["D"] if clk[sid]["D"] > 0 else np.nan
            effN_c = float(clk[sid].get("eff_N", 0.0))
            effN_s = float(np.sum(np.asarray(sv["N_counts"], float)))
            cclock = np.cumsum(ct)
            seg_rows: List[dict] = []
            for i in range(len(ct)):
                n_total += 1
                ratio = (max(sigt[i], _WMIN) / max(ct[i], _WMIN)) * abund
                ok = bool(cspread[i] < SPREAD_CUT and sspread[i] < SPREAD_CUT)
                # interval's clock-time span [t0, t1]; ratio is a constant density
                # over it, so both views draw it as a LINE across the span.
                t1 = float(min(cclock[i], 1.0))
                t0 = float(min(cclock[i - 1], 1.0)) if i > 0 else 0.0
                seg_rows.append(dict(
                    interval=i, t0=round(t0, 5), t1=round(t1, 5),
                    clock_time=round(t1, 5), ratio=round(float(ratio), 5),
                    ct=round(float(ct[i]), 5), st=round(float(sigt[i]), 5), ok=ok))
                if not ok or ct[i] <= MIN_CLOCK_T:
                    continue
                c_i = ct[i] * effN_c
                s_i = sigt[i] * effN_s
                ess = (c_i * s_i / (c_i + s_i)) if (c_i + s_i) > 0 else 0.0
                rows.append(dict(seg_id=sid, cn_state=clk[sid]["cn_state"], interval=i,
                                 t0=t0, t1=t1, clock_time=t1, ratio=float(ratio),
                                 ess=float(ess), clock_spread=float(cspread[i]),
                                 sig_spread=float(sspread[i])))
            # cn_state on the block, not only on the aggregate rows: a segment with no interval
            # that passes the gates would otherwise carry no CN state anywhere in the payload.
            seg_block[sid] = dict(abund=round(float(abund), 5),
                                  cn_state=clk[sid]["cn_state"], rows=seg_rows)
        payload["activity"][spec["name"]] = rows
        payload["seg_activity"][spec["name"]] = seg_block
        payload["genome_abund"][spec["name"]] = dict(
            D_sig=round(Ds_tot, 2), D_clock=round(Dc_tot, 2),
            ratio=round(Ds_tot / Dc_tot, 4) if Dc_tot > 0 else None, n_seg=n_common)
        if verbose:
            print(f"  {spec['name']}: retime {time.time()-t0w:.0f}s, "
                  f"{len(re_segs)} segs, {len(rows)}/{n_total} intervals kept", flush=True)
    return payload


# ---------------------------------------------------------------------------
# Flat tables  (what a user reads; the JSON payload above is for the viewer)
# ---------------------------------------------------------------------------
def payload_to_frames(payload: dict):
    """Split a :func:`build_sig_payload` payload into two tidy DataFrames.

    ``activity`` — one row per (signature, segment, gain interval)::

        sample signature seg_id cn_state interval t0 t1 clock_time ct st ratio ess ok

    ``ratio`` > 1 means the signature deposited mutations FASTER than the clock over
    that interval, < 1 slower. ``t0``/``t1`` bound the interval in clock time, so the
    ratio is a density across a span, not a value at a point. ``ok`` marks intervals
    where BOTH polytope spreads are below ``spread_cut`` — an interval with ``ok``
    False is not determined enough to read on its own; the aggregate curve uses only
    ``ok`` rows with a clock denominator above ``min_clock_t``, and ``ess`` is that
    row's weight in it.

    ``summary`` — one row per signature::

        sample signature share n_seg D_sig D_clock genome_ratio n_intervals n_ok

    ``genome_ratio`` is the genome-wide abundance of the signature relative to the
    clock; a per-interval ``ratio`` is only interesting relative to it, since a
    signature twice as abundant everywhere scores 2 everywhere.
    """
    sample = payload.get("sample", "sample")
    share = {s["name"]: s.get("share") for s in payload.get("signatures", [])}
    arows, srows = [], []
    for sig, blocks in (payload.get("seg_activity") or {}).items():
        n_ok = 0
        for sid, blk in blocks.items():
            for r in blk.get("rows", []):
                arows.append(dict(sample=sample, signature=sig, seg_id=sid,
                                  cn_state=blk.get("cn_state"), **r))
                n_ok += bool(r.get("ok"))
        ab = (payload.get("genome_abund") or {}).get(sig, {})
        srows.append(dict(sample=sample, signature=sig, share=share.get(sig),
                          n_seg=ab.get("n_seg"), D_sig=ab.get("D_sig"),
                          D_clock=ab.get("D_clock"), genome_ratio=ab.get("ratio"),
                          n_intervals=sum(len(b.get("rows", [])) for b in blocks.values()),
                          n_ok=n_ok))
    act = pd.DataFrame(arows)
    if len(act):
        # cn_state and ess live on the aggregate rows, not the per-segment blocks — join them back.
        # ess is only defined for rows that entered the aggregate, so it is NaN wherever ok is False.
        agg = {(sig, r["seg_id"], r["interval"]): r
               for sig, rows in (payload.get("activity") or {}).items() for r in rows}
        keys = list(zip(act.signature, act.seg_id, act.interval))
        act["ess"] = [agg[k]["ess"] if k in agg else np.nan for k in keys]
        # caches written before cn_state moved onto the block carry it only on aggregate rows
        if act.cn_state.isna().any():
            cn = {r["seg_id"]: r["cn_state"] for r in agg.values()}
            act["cn_state"] = act.cn_state.fillna(act.seg_id.map(cn))
        act = act[["sample", "signature", "seg_id", "cn_state", "interval", "t0", "t1",
                   "clock_time", "ct", "st", "ratio", "ess", "ok"]]
    return act, pd.DataFrame(srows)


def write_sig_tables(payload: dict, out_prefix) -> list:
    """Write ``<prefix>.sig_activity.tsv`` and ``<prefix>.sig_summary.tsv``. Returns the paths.

    ``out_prefix`` is a path stem, e.g. ``/out/dir/SAMPLE`` — not a filename.
    """
    from pathlib import Path
    act, summ = payload_to_frames(payload)
    paths = [Path(f"{out_prefix}.sig_activity.tsv"), Path(f"{out_prefix}.sig_summary.tsv")]
    act.to_csv(paths[0], sep="\t", index=False)
    summ.to_csv(paths[1], sep="\t", index=False)
    return paths


# ---------------------------------------------------------------------------
# Pooled-CLUSTER interval table  (the input tau.sig_states expects)
# ---------------------------------------------------------------------------
# Ported from dev/pcawg/signatures/make_clustertiming_retimed.py. That script needed a separate
# per-signature run of the whole cohort on disk (output_persig_<SIG>/) to get the signature-weighted
# genome; here `retimed_genome` produces it in-process from the clock genome, so one `tau run` is
# enough. The design is otherwise unchanged, and it is the supervisor's:
#
#   1. cluster the CLOCK genome's segments by multiplicity (per CN state)
#   2. merge each cluster's members into one pseudo-segment and RE-TIME it -> the cluster's clock times
#   3. merge the SAME members from the signature-weighted genome and re-time -> the signature's times
#
# Using clock-defined clusters for both is what makes clock and signature directly comparable.

def _interval_stats(G2d):
    """(n_draws, n_events) cumulative event times -> (normalised interval vector, per-interval range).

    The point estimate is the median over polytope solutions, renormalised to sum 1; the range is
    max-min of each normalised interval across solutions, i.e. how much the polytope leaves open.
    """
    nd = G2d.shape[0]
    B = np.column_stack([np.zeros(nd), np.clip(G2d, 0, 1), np.ones(nd)])
    Tm = np.diff(B, axis=1)
    med = np.median(Tm, axis=0)
    tot = med.sum()
    if tot > 0:
        med = med / tot
    return med, (np.nanmax(Tm, axis=0) - np.nanmin(Tm, axis=0))


def _cluster_event_times(genome, seg_cluster_df, env):
    """Pool `genome`'s segments into the given clusters, re-time, and return the polytope per cluster.

    -> {merged_seg_id: (cn_state, n_segments, G2d, ts)} where G2d is (n_draws, n_events) — ALL
    solutions, so the caller can take a median and a range — and `ts = D / total_cn` is the constant
    that un-normalises a t back to a raw mutation count.
    """
    from tau.clustering import make_clustered_genome
    cg = make_clustered_genome(genome, seg_cluster_df)
    cg.segments = [s for s in cg.segments if str(getattr(s, "seg_id", "")).startswith("merged_")]
    if not cg.segments:
        return {}
    cg.calculate_multiplicities(bootstrap_B=1, random_state=0)   # EM on the pooled, mut_w-weighted SNVs
    cg.time_segments(env)
    out = {}
    for s in cg.segments:
        if not getattr(s, "timing_result", None):
            continue
        best = pick_best_key(s)
        if not best or best not in s.timing_result:
            continue
        ev = [np.cumsum(order_t(d["t"]))[:-1] for d in s.timing_result[best]["draws"]
              if int(d.get("boot_id", 0)) == 0 and d.get("t")]
        if not ev:
            continue
        L = min(len(e) for e in ev)
        G2d = np.array([e[:L] for e in ev])
        nc = np.asarray(s.N_counts, dtype=float)
        tot = float(s.major_cn + s.minor_cn)
        ts = float(np.sum((np.arange(len(nc)) + 1) * nc) / tot) if tot > 0 and nc.sum() > 0 else np.nan
        out[s.seg_id] = (f"{s.major_cn}_{s.minor_cn}",
                         int(getattr(s, "num_segs", len(getattr(s, "seg_ids", [])))), G2d, ts)
    return out


def cluster_interval_table(genome, env, seg_cluster_df=None, *, sample="sample", tumor_type="NA",
                           thresh=0.15, verbose=True, expo=None) -> pd.DataFrame:
    """Per (pooled cluster, time interval) signature-vs-clock activity — WIDE by signature.

    This is the table :func:`tau.sig_states.fit_cluster_table` consumes: it carries
    ``clocklike_cumulative_start/end``, ``clocklike_raw_interval`` and, per active signature,
    ``<SIG>_ratio`` and ``<SIG>_raw_interval``.

    Ratios are RAW (un-normalised), so an interval's ratio reflects the signature's activity and not
    just the shape of its timing. Compare a ratio against that signature's exposure, carried on every
    row as ``<SIG>_exposure`` — a signature twice as abundant overall scores 2 everywhere.
    """
    from tau.clustering import cluster_segment_multiplicities
    if seg_cluster_df is None:
        seg_cluster_df = cluster_segment_multiplicities(genome)
    if seg_cluster_df is None or not len(seg_cluster_df) or (seg_cluster_df.cluster_label != -1).sum() == 0:
        return pd.DataFrame()

    expo = expo or sample_exposure_estimate(genome)
    specs = [s for s in active_signatures(genome, thresh, expo) if s["name"] != "clock"]
    clock = _cluster_event_times(genome, seg_cluster_df, env)
    if not clock:
        return pd.DataFrame()

    sig_results = {}
    for spec in specs:
        t0 = time.time()
        g_sig = _retimed_genome(genome, spec["members"], env, expo)
        sig_results[spec["name"]] = _cluster_event_times(g_sig, seg_cluster_df, env)
        if verbose:
            print(f"  {spec['name']}: cluster re-time {time.time()-t0:.0f}s, "
                  f"{len(sig_results[spec['name']])} clusters", flush=True)
    share = {s["name"]: s["share"] for s in specs}

    rows = []
    for mseg, (cn, nseg, G2d_c, ts_c) in clock.items():
        clabel = int(str(mseg).split("_")[-1])       # DBSCAN cluster index within the CN state
        tcn, tcs = _interval_stats(G2d_c)
        cum = np.r_[0.0, np.cumsum(tcn)]
        sig_iv = {}
        for sig, res in sig_results.items():
            # same event count => the two timings describe the same history and are alignable
            if mseg in res and res[mseg][2].shape[1] == G2d_c.shape[1]:
                tsn, tss = _interval_stats(res[mseg][2])
                sig_iv[sig] = (tsn, tss, res[mseg][3])
        for i in range(len(tcn)):
            cr = tcn[i] * ts_c
            row = dict(sample=sample, tumor_type=tumor_type, cn_state=cn, cluster_label=clabel,
                       cluster=mseg, n_segments=nseg, time_interval=i + 1,
                       clocklike_cumulative_start=cum[i], clocklike_cumulative_end=cum[i + 1],
                       clocklike_raw_interval=cr, clocklike_normalized_interval=tcn[i],
                       clocklike_range=tcs[i])
            for sig, (tsn, tss, ts_s) in sig_iv.items():
                row[f"{sig}_raw_interval"] = tsn[i] * ts_s
                row[f"{sig}_normalized_interval"] = tsn[i]
                row[f"{sig}_ratio"] = (tsn[i] * ts_s) / cr if cr > 0 else np.nan
                row[f"{sig}_range"] = tss[i]
            for sig in sig_results:
                row[f"{sig}_exposure"] = share.get(sig, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)
