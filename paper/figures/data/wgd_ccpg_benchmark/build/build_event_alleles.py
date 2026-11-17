"""Allelic composition of every detected WGD / ccPG event in PCAWG -> paper/figures/data/wgd_ccpg_benchmark/event_alleles.tsv.

The question is whether an event gained one allele or both. That is a property of the GAINS, not of the
final copy-number state: a (4,2) segment whose event contributed only a major-allele gain is monoallelic
at that event, and its final state cannot distinguish it from one gained on both alleles at once.

Each gain's allele comes from the route topology (tau/data/route_topology.json.gz), where `col_index` is
the 1-based gain index — allele-resolved, not inferred from minor_cn.

Why containment, and not a single representative solution
---------------------------------------------------------
Determined routes are exactly those with minor_cn <= 1 (verified: 866/867 minor_cn=1 segments are
determined, 578/633 minor_cn=2 are not). A segment with no second allele to gain can never be biallelic,
so biallelic gains are only ever visible in UNDERDETERMINED segments — there is no conservative
determined-only version of this measurement to fall back on.

Collapsing those segments to a median solution destroys the signal, and does so artefactually: the
median of gain 1 and the median of gain 2 are not a solution the polytope actually contains, and they
are pulled apart even when a coincident solution exists. Measured that way, 95% of WGD events look
monoallelic, which is not a finding but a summary artefact.

So a segment is scored biallelic at an event if ONE OF ITS SOLUTIONS places a major-allele gain and a
minor-allele gain both within TOL of the event time — the same containment logic the paper uses for
special-case solutions. Each draw is one polytope point, so this is evaluated per draw and then OR-ed.
For determined segments there is a single draw and the test reduces to reading it off.

This is permissive by construction for wide polytopes, so `frac_capable` (the share of contributing
segments with minor_cn >= 2, i.e. that have a minor allele to gain at all) is emitted alongside: an
event whose segments are all minor_cn <= 1 is monoallelic trivially rather than informatively.

Membership is by time proximity, NOT `tau_segments.event_id` — that column carries the multiplicity
cluster label and does not join tau_events. Proximity is also how the event clustering defines
membership, so this reproduces it rather than trusting a stored key.

Usage:  python paper/figures/data/wgd_ccpg_benchmark/build/build_event_alleles.py [tol]
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
OUT = cache_dir_of(__file__); OUT.mkdir(parents=True, exist_ok=True)
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
MIN_ESS = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0


def _allele_map():
    """(route_key, gain_index) -> 'major' | 'minor'."""
    top = json.load(gzip.open(R / "tau/data/route_topology.json.gz", "rt"))
    return {(k, ev["col_index"]): ev["allele"] for k, v in top.items() for ev in v["events"]}


def main():
    print(f"tol = {TOL}  min_ess = {MIN_ESS}")
    g = pd.read_csv(DATA / "tau_gains.tsv.gz", sep="\t",
                    usecols=["sample", "seg_id", "route_key", "is_best_route",
                             "gain_index", "draw_id", "gain_time_frac", "minor_cn"])
    g = g[g.is_best_route.astype(bool)].drop(columns=["is_best_route"])
    am = _allele_map()
    g["is_minor"] = np.array([am.get((k, i)) for k, i in zip(g.route_key, g.gain_index)]) == "minor"
    print(f"  {len(g):,} best-route gain draws, "
          f"{g.drop_duplicates(['sample', 'seg_id']).shape[0]:,} segments")

    seg = pd.read_csv(DATA / "tau_segments.tsv", sep="\t",
                      usecols=["sample", "seg_id", "seg_len_bp", "ess", "is_timed"])
    # the sample's whole timed genome — the denominator for bi_frac_timed below. It is a property of
    # the SAMPLE, not of the event, so it is computed once here rather than per event.
    timed = (seg[seg.is_timed.astype(bool)].groupby("sample").seg_len_bp.sum())
    g = g.merge(seg.drop(columns=["is_timed"]), on=["sample", "seg_id"], how="left")

    ev = pd.read_csv(DATA / "tau_events.tsv", sep="\t")
    ev = ev[ev.classification.isin(["WGD", "PGD"])].copy()
    print(f"  {len(ev):,} WGD/ccPG events")

    by_sample = {s: d for s, d in g.groupby("sample", sort=False)}
    rows = []
    for e in ev.itertuples():
        d = by_sample.get(e.sample)
        if d is None:
            continue
        near = d[((d.gain_time_frac - e.event_time_frac).abs() <= TOL) & (d.ess >= MIN_ESS)]
        if near.empty:
            continue
        # per (segment, solution): did THIS solution put both alleles at the event?
        per_draw = near.groupby(["seg_id", "draw_id"]).is_minor.agg(["any", "all"])
        bi_draw = per_draw["any"] & ~per_draw["all"]          # at least one minor AND one major
        bi = bi_draw.groupby("seg_id").any()                   # any solution suffices
        info = near.groupby("seg_id").agg(seg_len=("seg_len_bp", "first"),
                                          minor_cn=("minor_cn", "first"))
        info["bi"] = bi.reindex(info.index).fillna(False)
        L = info.seg_len.fillna(0).to_numpy(dtype=float)
        m = info.bi.to_numpy(dtype=bool)
        cap = (info.minor_cn.fillna(0) >= 2).to_numpy(dtype=bool)   # had a minor allele to gain
        rows.append(dict(
            sample=e.sample, tumor_type=e.tumor_type, event_id=e.event_id,
            classification="ccPG" if e.classification == "PGD" else "WGD",
            event_time_frac=e.event_time_frac, genome_frac=e.genome_frac,
            n_chroms=e.n_chroms, n_seg=int(len(info)),
            mono_frac_len=float(1 - L[m].sum() / L.sum()) if L.sum() > 0 else np.nan,
            mono_frac_seg=float(1 - m.mean()),
            frac_capable=float(L[cap].sum() / L.sum()) if L.sum() > 0 else np.nan,
            # biallelic share OF THE CAPABLE LENGTH — computed directly, length-weighted on both
            # sides. Dividing mono_frac_len by a segment-COUNT capable fraction would mix weightings.
            bi_frac_capable=float(L[m & cap].sum() / L[cap].sum()) if L[cap].sum() > 0 else np.nan,
            n_capable=int(cap.sum()),
            # ...and the same biallelic length against the sample's whole TIMED genome. This asks a
            # different question from bi_frac_capable: not "of what this event gained, how much was
            # biallelic" but "how much of the genome this event made biallelic". An event covering
            # little genome scores low here however biallelic its own segments were, which is the
            # point — it puts a small ccPG and a genome-wide WGD on one absolute scale.
            bi_len_bp=float(L[m].sum()),
            timed_len_bp=float(timed.get(e.sample, np.nan)),
            bi_frac_timed=(float(L[m].sum() / timed[e.sample])
                           if e.sample in timed.index and timed[e.sample] > 0 else np.nan)))
    t = pd.DataFrame(rows)
    p = OUT / "event_alleles.tsv"
    t.to_csv(p, sep="\t", index=False)
    print(f"\n{len(t):,} events with contributing gains")
    print(t.groupby("classification")[["mono_frac_len", "mono_frac_seg", "frac_capable",
                                       "bi_frac_capable", "bi_frac_timed"]].median().round(3).to_string())
    print(f"-> {p}")


if __name__ == "__main__":
    main()
