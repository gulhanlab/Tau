"""PROOF-OF-CONCEPT emit for the proposed NEW output structure (one sample).

Produces the full symmetric table set for a sample, into:
    paper/data_tables/per_sample/{sample}/{sample}.tau_*.tsv[.gz]

Two levels, same emitter (tau.aggregate.export.export_sample_tables), run on:
  • the RAW genome        -> tau_segments / _multiplicities / _routes / _gains / _events   (individual)
  • the CLUSTERED genome  -> tau_merged_segments / _multiplicities / _routes / _gains / _events (pooled)

New vs the current emit:
  • tau_segments gains  merged_id  (= merged_{M}_{m}_{label}, joins the pooled unit's seg_id)
  • event_id is taken from cluster_times_bottomup's OWN seg->event map (per gain), NOT segment_clusters.tsv
  • tau_events is emitted (from cluster_summary_df); pooled events attempted on the clustered genome
Usage:  python paper/batch_scripts/emit_sample_v2.py <sample_id>
"""
import sys, gzip, pickle
from pathlib import Path
import numpy as np, pandas as pd

R = next(p for p in Path(__file__).resolve().parents if (p / "paper").is_dir())
sys.path.insert(0, str(R))
from tau.aggregate.export import export_sample_tables
from tau.clustering import cluster_segment_multiplicities, cluster_times_bottomup

TAU_OUT = R / "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
OUT_ROOT = R / "paper/data_tables/per_sample"


def _load(pkl):
    with gzip.open(pkl, "rb") as f:
        return pickle.load(f)


def _events_df(sample, tt, cluster_summary_df):
    """cluster_summary_df -> tau_events schema."""
    if cluster_summary_df is None or len(cluster_summary_df) == 0:
        return pd.DataFrame(columns=["sample", "tumor_type", "event_id", "event_time_frac",
                                     "classification", "genome_frac", "n_chroms", "n_segments", "chrom"])
    d = cluster_summary_df.rename(columns={"cluster_id": "event_id", "time": "event_time_frac",
                                           "gf": "genome_frac"})
    d.insert(0, "tumor_type", tt); d.insert(0, "sample", sample)
    keep = [c for c in ["sample", "tumor_type", "event_id", "event_time_frac", "classification",
                        "genome_frac", "n_chroms", "n_segments", "chrom"] if c in d.columns]
    return d[keep]


def _seg_event_map(segment_cluster_ids):
    """seg_id -> (per-gain event-id array). Also a per-segment representative (unique event or NA)."""
    per_gain, rep = {}, {}
    for sid, arr in (segment_cluster_ids or {}).items():
        a = [int(x) for x in np.atleast_1d(arr)]
        per_gain[sid] = a
        ev = set(x for x in a if x >= 0)
        rep[sid] = (ev.pop() if len(ev) == 1 else pd.NA)
    return per_gain, rep


def emit(sample):
    sdir = next(iter(TAU_OUT.glob(f"*/{sample}*")))   # resolve short-or-full id to the sample dir
    sample = sdir.name                                # full uuid
    tt = sdir.parent.name
    graw = _load(sdir / f"{sample}.genome.pkl.gz")
    out = OUT_ROOT / sample; out.mkdir(parents=True, exist_ok=True)
    report = {}

    # Prefer a freshly RE-CLUSTERED genome in the bundle (recluster_sample.py — includes major-7 merges,
    # split-loader timing) over the stale dev/ artifact; fall back to dev/ if not re-clustered.
    gcl_path = out / f"{sample}.clustered_genome.pkl.gz"
    sc_path = out / f"{sample}.segment_clusters.tsv"
    if not gcl_path.exists():
        gcl_path = sdir / f"{sample}.clustered_genome.pkl.gz"
    if not sc_path.exists():
        sc_path = sdir / f"{sample}.segment_clusters.tsv"

    # ── merged_id from the segment_clusters.tsv that built the clustered genome.
    #    cluster>=0 -> merged unit; cluster==-1 -> passthrough (segment IS its own unit).
    sc = pd.read_csv(sc_path, sep="\t")
    mult_id = {r.seg_id: (f"merged_{int(r.major_cn)}_{int(r.minor_cn)}_{int(r.cluster_label)}"
                          if int(r.cluster_label) >= 0 else r.seg_id) for r in sc.itertuples()}
    # events (chrom-aware -> RAW only, canonical) + per-gain seg->event map, from cluster_times_bottomup
    _, seg_evt_ids, _, csum = cluster_times_bottomup(graw)
    per_gain, rep_evt = _seg_event_map(seg_evt_ids)

    # ── INDIVIDUAL level ────────────────────────────────────────────────────────────
    t = export_sample_tables(graw, sample)
    seg = t["segments"]
    seg["merged_id"] = seg["seg_id"].map(mult_id).astype("string")
    seg["event_id"] = seg["seg_id"].map(rep_evt).astype("Int64")     # representative (unique) event
    gains = t["gains"]
    gains["event_id"] = [                                            # PER-GAIN event (the precise mapping)
        (per_gain[sid][gi - 1] if sid in per_gain and gi - 1 < len(per_gain[sid]) and per_gain[sid][gi - 1] >= 0
         else pd.NA)
        for sid, gi in zip(gains["seg_id"], gains["gain_index"])]
    gains["event_id"] = gains["event_id"].astype("Int64")
    events = _events_df(sample, tt, csum)

    _w(out / f"{sample}.tau_samples.tsv", t["samples"])
    _w(out / f"{sample}.tau_segments.tsv", seg)
    _w(out / f"{sample}.tau_multiplicities.tsv", t["multiplicities"])
    _w(out / f"{sample}.tau_routes.tsv", t["routes"])
    _w(out / f"{sample}.tau_gains.tsv.gz", gains, gz=True)
    _w(out / f"{sample}.tau_events.tsv", events)
    report["individual"] = dict(segments=len(seg), routes=len(t["routes"]),
                                multiplicities=len(t["multiplicities"]), gains=len(gains),
                                events=len(events),
                                seg_with_merged_id=int(seg["merged_id"].notna().sum()),
                                gains_with_event=int(gains["event_id"].notna().sum()))

    # ── POOLED level (same emitter on the clustered genome) ──────────────────────────
    from collections import Counter
    if gcl_path.exists():
        gcl = _load(gcl_path)
        tc = export_sample_tables(gcl, sample)
        # a merged unit spans many loci -> physical coords are meaningless; drop them (recover via
        # tau_segments.merged_id). Keep CN state + pooled ess + n_segments.
        POOL_DROP = ["chrom", "start_bp", "end_bp", "seg_len_bp"]
        mc = tc["segments"].drop(columns=[c for c in POOL_DROP if c in tc["segments"].columns]).copy()
        constituents = {s.seg_id: list(getattr(s, "seg_ids", [s.seg_id])) for s in gcl.segments}
        mc["n_segments"] = [len(constituents.get(sid, [sid])) for sid in mc["seg_id"]]

        # POOLED EVENTS (user's method): propagate each merged unit's (pooled, cleaner) timing down to its
        # constituent raw segments — which KEEP their real chrom — then run the chrom-aware event clustering
        # on those. Directly clustering the merged genome fails (merged units span many chroms).
        merged_tr = {s.seg_id: getattr(s, "timing_result", None) for s in gcl.segments}
        for s in graw.segments:                                  # graw already emitted; safe to overwrite
            mid = mult_id.get(s.seg_id)
            if mid in merged_tr and merged_tr[mid]:
                s.timing_result = merged_tr[mid]
        _, seg_evt_pool, _, csum_pool = cluster_times_bottomup(graw)
        pooled_events = _events_df(sample, tt, csum_pool)
        per_gain_pool, rep_evt_pool = _seg_event_map(seg_evt_pool)

        def _unit_evt(sid):                                      # pooled unit event = modal over constituents
            evs = [int(rep_evt_pool[c]) for c in constituents.get(sid, [sid])
                   if c in rep_evt_pool and pd.notna(rep_evt_pool[c])]
            return Counter(evs).most_common(1)[0][0] if evs else pd.NA
        mc["event_id"] = pd.array([_unit_evt(sid) for sid in mc["seg_id"]], dtype="Int64")
        # pooled gains: per-gain event via a representative constituent (all constituents share the timing)
        rc = {sid: next((c for c in constituents.get(sid, [sid]) if c in per_gain_pool), None)
              for sid in tc["gains"]["seg_id"].unique()}
        pg = tc["gains"].drop(columns=[c for c in ["chrom", "start_bp", "end_bp"] if c in tc["gains"].columns]).copy()
        pg["event_id"] = [
            (per_gain_pool[rc[sid]][gi - 1] if rc.get(sid) and gi - 1 < len(per_gain_pool[rc[sid]])
             and per_gain_pool[rc[sid]][gi - 1] >= 0 else pd.NA)
            for sid, gi in zip(pg["seg_id"], pg["gain_index"])]
        pg["event_id"] = pg["event_id"].astype("Int64")

        _w(out / f"{sample}.tau_merged_segments.tsv", mc)
        _w(out / f"{sample}.tau_merged_multiplicities.tsv", tc["multiplicities"])
        _w(out / f"{sample}.tau_merged_routes.tsv", tc["routes"])
        _w(out / f"{sample}.tau_merged_gains.tsv.gz", pg, gz=True)
        _w(out / f"{sample}.tau_merged_events.tsv", pooled_events)
        report["pooled"] = dict(merged_segments=len(mc), routes=len(tc["routes"]),
                                multiplicities=len(tc["multiplicities"]), gains=len(pg),
                                events=len(pooled_events), merged_with_event=int(mc["event_id"].notna().sum()))


    # ── JOIN CHECKS ─────────────────────────────────────────────────────────────────
    evset = set(events["event_id"])
    pev = set(pooled_events["event_id"]) if gcl_path.exists() else set()
    pooled_ids = set(mc["seg_id"]) if gcl_path.exists() else set()
    mc_ids = set(seg["merged_id"].dropna())
    report["joins"] = dict(
        seg_with_merged_id=f"{int(seg['merged_id'].notna().sum())}/{len(seg)} segs "
                              f"({int(seg['is_timed'].sum())} timed)",
        merged_id_in_pooled=f"{len(mc_ids & pooled_ids)}/{len(mc_ids)} unique ids matched",
        indiv_gain_event_in_tau_events=f"{len(set(gains['event_id'].dropna()) & evset)}/"
                                       f"{gains['event_id'].nunique()} matched",
        pooled_unit_event_in_pooled_events=(f"{len(set(mc['event_id'].dropna()) & pev)}/{mc['event_id'].nunique()} matched"
                                            if gcl_path.exists() else "n/a"))
    return out, report


def _w(path, df, gz=False):
    df.to_csv(path, sep="\t", index=False, compression=("gzip" if gz else None))


if __name__ == "__main__":
    samp = sys.argv[1] if len(sys.argv) > 1 else "1e2dcbcc-771c-43c5-8c8d-e0eb77cb3494"
    out, rep = emit(samp)
    import json
    print(f"emitted -> {out}")
    print(json.dumps(rep, indent=2, default=str))
