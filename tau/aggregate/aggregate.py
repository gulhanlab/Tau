"""tau.aggregate.aggregate — per-sample tables → cohort tables (the AGGREGATE layer).

Lifted from paper/batch_scripts/merge_backfill.py: concatenate all per-sample `{sample}.tau_{name}.tsv[.gz]`
under a staging dir into cohort tables. Plain union (same schema). Empty-chunk-safe gains read.
"""
from __future__ import annotations
import glob
from pathlib import Path

import pandas as pd


def _find(sample_tables_dir: Path, patt: str) -> list[str]:
    """Files matching `patt` directly in the directory OR anywhere beneath it.

    So `tau aggregate` can be pointed at a single sample's output directory, or at the parent of many
    of them, or at a tree nested by tumour type — without the caller having to know or say which.
    """
    d = Path(sample_tables_dir)
    return sorted(set(glob.glob(str(d / patt))
                      + glob.glob(str(d / "**" / patt), recursive=True)))


def _concat(sample_tables_dir: Path, patt: str) -> pd.DataFrame:
    fs = _find(sample_tables_dir, patt)
    frames = [pd.read_csv(f, sep="\t") for f in fs if Path(f).stat().st_size > 5]
    return pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()


def _read_gz(f):
    try:
        return pd.read_csv(f, sep="\t", low_memory=False, dtype={"chrom": str})
    except (pd.errors.EmptyDataError, EOFError):
        return None


def aggregate_run(sample_tables_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Concatenate per-sample tables under `sample_tables_dir` → cohort tau_* tables in `out_dir`.

    Reads `*.tau_{segments,routes,multiplicities,samples}.tsv` + `*.tau_gains.tsv.gz`. If `out_dir`
    already holds a tau_samples.tsv (e.g. the cheap base build), it is ENRICHED with the per-sample
    aggregates; otherwise the aggregates are written directly. Returns {table: path}.
    """
    sample_tables_dir, out_dir = Path(sample_tables_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    seg = _concat(sample_tables_dir, "*.tau_segments.tsv")
    if len(seg):
        seg = seg.sort_values(["sample", "chrom", "start_bp"])
    out["tau_segments"] = out_dir / "tau_segments.tsv"
    seg.to_csv(out["tau_segments"], sep="\t", index=False)

    rt = _concat(sample_tables_dir, "*.tau_routes.tsv")
    if len(rt):
        rt = rt.sort_values(["sample", "seg_id", "route_key"])
    out["tau_routes"] = out_dir / "tau_routes.tsv"
    rt.to_csv(out["tau_routes"], sep="\t", index=False)

    mu = _concat(sample_tables_dir, "*.tau_multiplicities.tsv")
    if len(mu):
        mu = mu.sort_values(["sample", "seg_id", "multiplicity"])
    out["tau_multiplicities"] = out_dir / "tau_multiplicities.tsv"
    mu.to_csv(out["tau_multiplicities"], sep="\t", index=False)
    print(f"tau_segments {len(seg)} | tau_routes {len(rt)} | tau_multiplicities {len(mu)}")

    # gains: concat gz-TSV per-sample tables → one gz-TSV + a head preview (skip empty/headerless)
    gfs = _find(sample_tables_dir, "*.tau_gains.tsv.gz")
    gframes = [g for g in (_read_gz(f) for f in gfs) if g is not None and len(g)]
    gains = pd.concat(gframes, ignore_index=True) if gframes else pd.DataFrame()
    if len(gains):
        gains = gains.sort_values(["sample", "seg_id", "route_key", "gain_index", "draw_id"])
    out["tau_gains"] = out_dir / "tau_gains.tsv.gz"
    gains.to_csv(out["tau_gains"], sep="\t", index=False, compression="gzip")
    gains.head(500).to_csv(out_dir / "tau_gains.head.tsv", sep="\t", index=False)
    print(f"tau_gains {len(gains)} rows -> tau_gains.tsv.gz (+ head)")

    # events + the MERGED (pooled) level — plain-union concat (same schema per file)
    ev = _concat(sample_tables_dir, "*.tau_events.tsv")
    if len(ev):
        ev = ev.sort_values(["sample", "event_id"])
    out["tau_events"] = out_dir / "tau_events.tsv"
    ev.to_csv(out["tau_events"], sep="\t", index=False)

    msg = _concat(sample_tables_dir, "*.tau_merged_segments.tsv")
    if len(msg):
        msg = msg.sort_values(["sample", "seg_id"])
    out["tau_merged_segments"] = out_dir / "tau_merged_segments.tsv"
    msg.to_csv(out["tau_merged_segments"], sep="\t", index=False)

    mmu = _concat(sample_tables_dir, "*.tau_merged_multiplicities.tsv")
    if len(mmu):
        mmu = mmu.sort_values(["sample", "seg_id", "multiplicity"])
    out["tau_merged_multiplicities"] = out_dir / "tau_merged_multiplicities.tsv"
    mmu.to_csv(out["tau_merged_multiplicities"], sep="\t", index=False)

    mrt = _concat(sample_tables_dir, "*.tau_merged_routes.tsv")
    if len(mrt):
        mrt = mrt.sort_values(["sample", "seg_id", "route_key"])
    out["tau_merged_routes"] = out_dir / "tau_merged_routes.tsv"
    mrt.to_csv(out["tau_merged_routes"], sep="\t", index=False)

    mev = _concat(sample_tables_dir, "*.tau_merged_events.tsv")
    if len(mev):
        mev = mev.sort_values(["sample", "event_id"])
    out["tau_merged_events"] = out_dir / "tau_merged_events.tsv"
    mev.to_csv(out["tau_merged_events"], sep="\t", index=False)
    print(f"tau_events {len(ev)} | tau_merged_segments {len(msg)} | tau_merged_routes {len(mrt)} | "
          f"tau_merged_multiplicities {len(mmu)} | tau_merged_events {len(mev)}")

    # merged_gains: gz-TSV (empty-chunk-safe like tau_gains)
    mgfs = _find(sample_tables_dir, "*.tau_merged_gains.tsv.gz")
    mgframes = [g for g in (_read_gz(f) for f in mgfs) if g is not None and len(g)]
    mgains = pd.concat(mgframes, ignore_index=True) if mgframes else pd.DataFrame()
    if len(mgains):
        mgains = mgains.sort_values(["sample", "seg_id", "route_key", "gain_index", "draw_id"])
    out["tau_merged_gains"] = out_dir / "tau_merged_gains.tsv.gz"
    mgains.to_csv(out["tau_merged_gains"], sep="\t", index=False, compression="gzip")
    print(f"tau_merged_gains {len(mgains)} rows -> tau_merged_gains.tsv.gz")

    # tau_samples: enrich an existing table if present, else write the aggregates directly
    agg = _concat(sample_tables_dir, "*.tau_samples.tsv")
    out["tau_samples"] = out_dir / "tau_samples.tsv"
    if out["tau_samples"].exists() and len(agg):
        a = agg.set_index("sample")
        s = pd.read_csv(out["tau_samples"], sep="\t")
        for col in ["purity", "n_snvs", "n_segments", "mean_ess"]:
            s[col] = s["sample"].map(a[col]) if col in a else s[col]
        s["ploidy"] = s["ploidy"].fillna(s["sample"].map(a["ploidy_seg"]) if "ploidy_seg" in a else s["ploidy"])
        for c in ["n_snvs", "n_segments", "n_timed_segments", "n_events"]:
            if c in s:
                s[c] = s[c].astype("Int64")
        s.to_csv(out["tau_samples"], sep="\t", index=False)
        print(f"tau_samples enriched for {a.shape[0]} samples")
    else:
        if len(agg):
            agg = agg.rename(columns={"ploidy_seg": "ploidy"}).sort_values("sample")
        agg.to_csv(out["tau_samples"], sep="\t", index=False)
        print(f"tau_samples {len(agg)} rows (standalone aggregate)")

    return out
