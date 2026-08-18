"""tau.aggregate.export — genome object → per-sample composite tables (the EMIT layer).

Lifted verbatim (numeric behaviour) from paper/batch_scripts/backfill_chunk.py: reshape a timed
`tau.core.Genome` into the per-sample tau_* tables (segments, routes, multiplicities, gains + the
per-sample aggregate row). Cheap reshape, NO re-timing.

TWO-LEVEL output (folded in from paper/batch_scripts/emit_sample_v2.py, the verified POC):
  • INDIVIDUAL level (raw genome): tau_samples / tau_segments (+ merged_id, event_id) /
    tau_multiplicities / tau_routes / tau_gains (+ per-gain event_id) / tau_events.
  • MERGED level (clustered genome): tau_merged_segments (= segments minus coords, + n_segments +
    event_id) / tau_merged_multiplicities / tau_merged_routes / tau_merged_gains (minus coords, +
    event_id) / tau_merged_events.

Used two ways:
  • `tau run` (fresh) → `build_all_tables(raw_genome, sample, ..., seg_cluster_df=, ctb_raw=,
    clustered_genome=)` with the already-computed objects so nothing expensive is recomputed
    (cli.py should call this — see build_all_tables docstring — then write_all_tables).
  • `tau export-tables <genome.pkl.gz>` (backfill) → `export_tables_from_pkl(pkl, out_dir)` loads the
    sibling {sample}.genome.pkl.gz + {sample}.clustered_genome.pkl.gz + {sample}.segment_clusters.tsv,
    runs cluster_times_bottomup for events, then writes the full two-level table set.

Tau outputs are POLYTOPE SOLUTIONS (equally-valid points in an underdetermined space) — the per-gain
`draw_id` rows are those solutions, NOT posterior draws/samples.

Schemas: paper/data_tables/SCHEMA.md. Column/dtype source of truth: tau.aggregate.schema. tau_gains
(and tau_merged_gains) ship as gzipped TSV (pyarrow not in env; parquet is a one-line drop-in later).
"""
from __future__ import annotations
import gzip
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# per-sample output filenames: f"{sample}.tau_{name}.tsv" (gains gz-compressed)
TABLE_FILENAMES = {
    "samples": "tau_samples.tsv",
    "segments": "tau_segments.tsv",
    "routes": "tau_routes.tsv",
    "multiplicities": "tau_multiplicities.tsv",
    "gains": "tau_gains.tsv.gz",
    "events": "tau_events.tsv",
    "merged_segments": "tau_merged_segments.tsv",
    "merged_multiplicities": "tau_merged_multiplicities.tsv",
    "merged_routes": "tau_merged_routes.tsv",
    "merged_gains": "tau_merged_gains.tsv.gz",
    "merged_events": "tau_merged_events.tsv",
}

_MERGED_SEG_DROP = ["chrom", "start_bp", "end_bp", "seg_len_bp"]   # coords meaningless for a pooled unit
_MERGED_GAIN_DROP = ["chrom", "start_bp", "end_bp"]


# ── helpers (identical to backfill_chunk — do NOT change numeric behaviour) ──────────────
def order_t(tdict):
    """t is {t1:.., t2:..} (sympy-keyed); return values ordered by numeric suffix."""
    def idx(k):
        d = "".join(ch for ch in str(k) if ch.isdigit())
        return int(d) if d else 0
    return [float(tdict[k]) for k in sorted(tdict, key=idx)]


def best_route(seg):
    """The route Tau actually deploys — delegates to :func:`tau.utils.pick_best_key`.

    This used to reimplement the rule as "min dist_rel, first wins on ties", which disagreed with
    the deployed rule (min dist_rel, then MAX inside_margin, then canonical order) on 0.57% of PCAWG
    segments — 1,016 of 179,396, being 16.6% of the 3.4% that have a tie. `is_best_route` in
    tau_routes / tau_gains therefore named a different route than the pipeline itself would pick.
    Delegating means the two cannot drift again.

    Note pick_best_key also requires the route to have at least one polytope solution, so a route
    with QC but no draws can no longer be flagged best.
    """
    from tau.utils import pick_best_key
    return pick_best_key(seg)


def _sample_rows(g, sample, seg2evt, purity=None):
    """Per-sample reshape (was backfill_chunk.sample_rows). Returns row lists + the aggregate dict."""
    seg2evt = seg2evt or {}
    seg_rows, route_rows, mult_rows, gain_rows = [], [], [], []
    n_snvs = 0.0; ess_list = []; cn_weighted = 0.0; total_len = 0
    for seg in g.segments:
        chrom = str(getattr(seg, "chrom", "")).replace("chr", "")
        s0, s1 = int(getattr(seg, "start", -1)), int(getattr(seg, "end", -1))
        seg_id = getattr(seg, "seg_id", None) or f"{chrom}:{s0}-{s1}"
        M, m = int(getattr(seg, "major_cn", 0)), int(getattr(seg, "minor_cn", 0))
        nc = getattr(seg, "N_counts", None)
        ess = float(np.sum(nc)) if nc is not None else 0.0
        tr = getattr(seg, "timing_result", None) or {}
        is_timed = bool(tr)
        disc = getattr(seg, "discard_info", None)
        reason = (disc.get("reason") if isinstance(disc, dict) else None) if not is_timed else None
        seg_rows.append(dict(sample=sample, seg_id=seg_id, chrom=chrom, start_bp=s0, end_bp=s1,
                             seg_len_bp=max(0, s1 - s0), major_cn=M, minor_cn=m, total_cn=M + m,
                             ess=round(ess, 3), event_id=seg2evt.get(seg_id, pd.NA),
                             is_timed=is_timed, discard_reason=reason))
        # sample-level accumulators
        if nc is not None:
            n_snvs += float(np.sum(nc))
        if is_timed:
            ess_list.append(ess)
        if s1 > s0:
            cn_weighted += (M + m) * (s1 - s0); total_len += (s1 - s0)
        # multiplicities
        if nc is not None and np.sum(nc) > 0:
            tot = float(np.sum(nc))
            for k in range(len(nc)):
                mult_rows.append(dict(sample=sample, seg_id=seg_id, multiplicity=k + 1,
                                      n_count=round(float(nc[k]), 4), pi=round(float(nc[k]) / tot, 5)))
        # routes + gains
        bkey = best_route(seg)
        for key, rtr in tr.items():
            draws = rtr.get("draws", []); meta = rtr.get("meta", {}); qc = rtr.get("qc", [{}])
            n_ev = None
            for d in draws:
                tv = order_t(d.get("t", {}))
                if len(tv) < 2:
                    continue
                cum = np.cumsum(tv)[:-1]            # event (gain) times
                n_ev = len(cum)
                for gi, gt in enumerate(cum, start=1):
                    gain_rows.append(dict(sample=sample, seg_id=seg_id, route_key=key,
                                          is_best_route=(key == bkey), chrom=chrom, start_bp=s0, end_bp=s1,
                                          major_cn=M, minor_cn=m, gain_index=gi, draw_id=int(d.get("draw_id", 0)),
                                          gain_time_frac=round(float(gt), 5), n_draws=len(draws)))
            route_rows.append(dict(sample=sample, seg_id=seg_id, route_key=key,
                                   n_events=(n_ev if n_ev is not None else 0), n_draws=len(draws),
                                   free_var_count=int(meta.get("free_var_count", 0)),
                                   dist_rel=round(float(qc[0].get("dist_rel", np.nan)), 5) if qc else np.nan,
                                   inside_margin=round(float(qc[0].get("inside_margin", np.nan)), 5) if qc else np.nan,
                                   is_best_route=(key == bkey)))
    ploidy = round(cn_weighted / total_len, 4) if total_len else np.nan
    if purity is None:
        purity = float(getattr(g.segments[0], "purity", np.nan)) if g.segments else np.nan
    agg = dict(sample=sample, purity=float(purity) if purity is not None else np.nan,
               n_snvs=int(round(n_snvs)), n_segments=len(g.segments),
               n_timed_segments=len(ess_list), mean_ess=round(float(np.mean(ess_list)), 3) if ess_list else np.nan,
               ploidy_seg=ploidy)
    return seg_rows, route_rows, mult_rows, gain_rows, agg


# ── top-level entry points ─────────────────────────────────────────────────────────────
def export_sample_tables(genome, sample, purity=None, seg2evt=None) -> dict[str, pd.DataFrame]:
    """Reshape one timed genome → per-sample tables.

    Returns {samples, segments, routes, multiplicities, gains} as DataFrames (whatever the genome
    supports). `seg2evt` maps seg_id → event_id (from the run's segment_clusters.tsv); when absent,
    tau_segments.event_id is NA. NO re-timing.
    """
    sr, rr, mr, gr, agg = _sample_rows(genome, sample, seg2evt, purity=purity)
    return {
        "samples": pd.DataFrame([agg]),
        "segments": pd.DataFrame(sr),
        "routes": pd.DataFrame(rr),
        "multiplicities": pd.DataFrame(mr),
        "gains": pd.DataFrame(gr),
    }


def write_sample_tables(tables: dict, out_dir: Path, sample: str) -> dict[str, Path]:
    """Write per-sample tables as f"{sample}.tau_{name}.tsv" (gains/*_gains → .tsv.gz). Returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, fname in TABLE_FILENAMES.items():
        df = tables.get(name)
        if df is None:
            continue
        path = out_dir / f"{sample}.{fname}"
        df.to_csv(path, sep="\t", index=False, compression=("gzip" if fname.endswith(".gz") else None))
        written[name] = path
    return written


write_all_tables = write_sample_tables   # symmetric alias for the full two-level table set


# ── two-level (individual + merged + events) emit — folded in from emit_sample_v2.py ─────
def _events_df(sample, tumor_type, cluster_summary_df) -> pd.DataFrame:
    """cluster_times_bottomup's cluster_summary_df → tau_events schema."""
    cols = ["sample", "tumor_type", "event_id", "event_time_frac", "classification",
            "genome_frac", "n_chroms", "n_segments", "chrom"]
    if cluster_summary_df is None or len(cluster_summary_df) == 0:
        return pd.DataFrame(columns=cols)
    d = cluster_summary_df.rename(columns={"cluster_id": "event_id", "time": "event_time_frac",
                                           "gf": "genome_frac"})
    d.insert(0, "tumor_type", tumor_type)
    d.insert(0, "sample", sample)
    return d[[c for c in cols if c in d.columns]]


def _seg_event_map(segment_cluster_ids):
    """seg_id → (per-gain event-id list, representative event-id [unique event else NA])."""
    per_gain, rep = {}, {}
    for sid, arr in (segment_cluster_ids or {}).items():
        a = [int(x) for x in np.atleast_1d(arr)]
        per_gain[sid] = a
        ev = set(x for x in a if x >= 0)
        rep[sid] = (ev.pop() if len(ev) == 1 else pd.NA)
    return per_gain, rep


def _merged_id_map(seg_cluster_df):
    """seg_id → merged unit id from the STORED segment_clusters.tsv. cluster_label>=0 → merged_{M}_{m}_{label};
    label==-1 (passthrough) → the seg's own id. Segments absent from the file are unmapped (→ NA on join)."""
    if seg_cluster_df is None or len(seg_cluster_df) == 0:
        return {}
    return {r.seg_id: (f"merged_{int(r.major_cn)}_{int(r.minor_cn)}_{int(r.cluster_label)}"
                       if int(r.cluster_label) >= 0 else r.seg_id)
            for r in seg_cluster_df.itertuples()}


def build_all_tables(raw_genome, sample, tumor_type="NA", *, seg_cluster_df=None,
                     ctb_raw=None, clustered_genome=None, purity=None) -> dict[str, pd.DataFrame]:
    """Reshape a timed genome + its clustering into the full TWO-LEVEL table set (individual + merged + events).

    This is the packaged version of emit_sample_v2.py. Pass the already-computed objects so nothing
    expensive is recomputed:
      • raw_genome       — the timed raw `tau.core.Genome`.
      • seg_cluster_df   — the stored segment_clusters.tsv (DataFrame; cols seg_id, major_cn, minor_cn,
                           cluster_label). Source of merged_id (NOT a re-run of cluster_segment_multiplicities,
                           which would diverge from the stored clustered genome).
      • ctb_raw          — the 4-tuple from `cluster_times_bottomup(raw_genome)`
                           (event_times, segment_cluster_ids, original_times, cluster_summary_df);
                           source of per-gain + representative event_id and tau_events. Computed here if None.
      • clustered_genome — the pooled genome; drives all tau_merged_* tables (omit → merged tables skipped).
      • purity           — override; else read off the genome.

    `tau run` (cli.py) calls this with the objects it already has in memory, then write_all_tables.
    For merged events it temporarily propagates each pooled unit's timing onto its constituent raw
    segments and re-runs cluster_times_bottomup (cheap), then RESTORES the original per-segment timing,
    so raw_genome is left exactly as it was found. NO re-timing of the polytope itself.
    """
    from tau.clustering import cluster_times_bottomup   # local: avoid import cost when only reshaping

    mult_id = _merged_id_map(seg_cluster_df)
    if ctb_raw is None:
        ctb_raw = cluster_times_bottomup(raw_genome)
    _, seg_evt_ids, _, csum = ctb_raw
    per_gain, rep_evt = _seg_event_map(seg_evt_ids)

    # ── INDIVIDUAL level ────────────────────────────────────────────────────────────────
    t = export_sample_tables(raw_genome, sample, purity=purity)
    seg = t["segments"]
    seg["merged_id"] = seg["seg_id"].map(mult_id).astype("string")
    seg["event_id"] = seg["seg_id"].map(rep_evt).astype("Int64")          # representative (unique) event
    gains = t["gains"]
    gains["event_id"] = [                                                 # PER-GAIN event (precise mapping)
        (per_gain[sid][gi - 1] if sid in per_gain and gi - 1 < len(per_gain[sid]) and per_gain[sid][gi - 1] >= 0
         else pd.NA)
        for sid, gi in zip(gains["seg_id"], gains["gain_index"])]
    gains["event_id"] = gains["event_id"].astype("Int64")

    # tumor_type / n_events are known only here (not inside the per-genome reshape), and the schema
    # declares both on the sample row. Inserted rather than threaded through _sample_rows so that
    # function stays a verbatim reshape.
    samples = t["samples"].copy()
    samples.insert(1, "tumor_type", tumor_type)
    samples["n_events"] = int(len(csum)) if csum is not None else 0

    tables = {
        "samples": samples, "segments": seg, "multiplicities": t["multiplicities"],
        "routes": t["routes"], "gains": gains,
        "events": _events_df(sample, tumor_type, csum),
    }
    if clustered_genome is None:
        return tables

    # ── MERGED level (same emitter on the clustered genome) ───────────────────────────────
    tc = export_sample_tables(clustered_genome, sample)   # BEFORE mutating raw_genome below
    mc = tc["segments"].drop(columns=[c for c in _MERGED_SEG_DROP if c in tc["segments"].columns]).copy()
    constituents = {s.seg_id: list(getattr(s, "seg_ids", [s.seg_id])) for s in clustered_genome.segments}
    mc["n_segments"] = [len(constituents.get(sid, [sid])) for sid in mc["seg_id"]]

    # POOLED EVENTS (POC propagation method): propagate each merged unit's pooled timing down onto its
    # constituent raw segments — which KEEP their real chrom — then run chrom-aware event clustering on
    # those. Clustering the merged genome directly fails (merged units span many chroms).
    merged_tr = {s.seg_id: getattr(s, "timing_result", None) for s in clustered_genome.segments}
    original_tr = {id(s): getattr(s, "timing_result", None) for s in raw_genome.segments}
    for s in raw_genome.segments:
        mid = mult_id.get(s.seg_id)
        if mid in merged_tr and merged_tr[mid]:
            s.timing_result = merged_tr[mid]
    try:
        _, seg_evt_pool, _, csum_pool = cluster_times_bottomup(raw_genome)
    finally:
        # RESTORE. The propagation above is a means to chrom-aware pooled clustering, not a result:
        # leaving it in place hands every later consumer of raw_genome (the overview figure, the
        # signature re-timing, the viewer) POOLED timing in place of per-segment timing.
        for s in raw_genome.segments:
            s.timing_result = original_tr[id(s)]
    per_gain_pool, rep_evt_pool = _seg_event_map(seg_evt_pool)

    def _unit_evt(sid):                                    # pooled unit event = modal over constituents
        evs = [int(rep_evt_pool[c]) for c in constituents.get(sid, [sid])
               if c in rep_evt_pool and pd.notna(rep_evt_pool[c])]
        return Counter(evs).most_common(1)[0][0] if evs else pd.NA
    mc["event_id"] = pd.array([_unit_evt(sid) for sid in mc["seg_id"]], dtype="Int64")

    # pooled gains: per-gain event via a representative constituent (all constituents share the timing)
    rc = {sid: next((c for c in constituents.get(sid, [sid]) if c in per_gain_pool), None)
          for sid in tc["gains"]["seg_id"].unique()}
    pg = tc["gains"].drop(columns=[c for c in _MERGED_GAIN_DROP if c in tc["gains"].columns]).copy()
    pg["event_id"] = [
        (per_gain_pool[rc[sid]][gi - 1] if rc.get(sid) and gi - 1 < len(per_gain_pool[rc[sid]])
         and per_gain_pool[rc[sid]][gi - 1] >= 0 else pd.NA)
        for sid, gi in zip(pg["seg_id"], pg["gain_index"])]
    pg["event_id"] = pg["event_id"].astype("Int64")

    tables.update({
        "merged_segments": mc, "merged_multiplicities": tc["multiplicities"],
        "merged_routes": tc["routes"], "merged_gains": pg,
        "merged_events": _events_df(sample, tumor_type, csum_pool),
    })
    return tables


def _load_seg2evt(pkl_path: Path):
    """seg_id → event_id from a sibling {sample}.segment_clusters.tsv (as in backfill_chunk)."""
    pkl_path = Path(pkl_path)
    sample = pkl_path.name.replace(".genome.pkl.gz", "")
    scf = pkl_path.parent / f"{sample}.segment_clusters.tsv"
    if not scf.exists():
        return {}
    try:
        sc = pd.read_csv(scf, sep="\t", usecols=lambda c: c in ("seg_id", "cluster_label"))
        return {r.seg_id: (int(r.cluster_label) if r.cluster_label >= 0 else pd.NA)
                for r in sc.itertuples()}
    except Exception:
        return {}


def _load_pkl(path):
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def export_tables_from_pkl(pkl_path, out_dir, sample=None, purity=None,
                           tumor_type=None) -> dict[str, Path]:
    """CLI entry (`tau export-tables`). Backfill the full TWO-LEVEL table set from an existing pkl bundle.

    Loads the sibling {sample}.genome.pkl.gz + {sample}.clustered_genome.pkl.gz + {sample}.segment_clusters.tsv,
    runs cluster_times_bottomup for events, and writes tau_* + tau_merged_* + tau_events/tau_merged_events.
    NO re-timing (the polytope is loaded, only clustering is re-derived). This is exactly the emit_sample_v2.py
    backfill path.
    """
    pkl_path = Path(pkl_path)
    if sample is None:
        sample = pkl_path.name.replace(".genome.pkl.gz", "")
    sdir = pkl_path.parent
    if tumor_type is None:                      # tau_outputs layout is <tumor_type>/<sample>/<files>
        tumor_type = sdir.parent.name or "NA"

    graw = _load_pkl(pkl_path)
    gcl_path = sdir / f"{sample}.clustered_genome.pkl.gz"
    clustered = _load_pkl(gcl_path) if gcl_path.exists() else None
    scf = sdir / f"{sample}.segment_clusters.tsv"
    seg_cluster_df = pd.read_csv(scf, sep="\t") if scf.exists() else None

    tables = build_all_tables(graw, sample, tumor_type, seg_cluster_df=seg_cluster_df,
                              clustered_genome=clustered, purity=purity)
    return write_all_tables(tables, out_dir, sample)
