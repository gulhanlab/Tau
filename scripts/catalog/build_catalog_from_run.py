#!/usr/bin/env python
"""Assemble a browsable catalog from a completed `tau run` cohort.

The viewers already exist -- `tau run` writes `<sample>.overview_interactive.html` for
every sample -- so this only harvests metadata and lays out the directory. Nothing is
re-rendered and Sage is not needed.

Metadata comes from the cohort tables (`tau aggregate` output), so the catalog cannot
drift from the run it describes.

Usage:
    python build_catalog_from_run.py \
        --samples paper/pcawg_run/samples_v2 \
        --cohort  paper/pcawg_run/cohort_v2 \
        --out     paper/pcawg_run/catalog_v2 \
        [--link symlink|copy]

Then build the page (reuses the existing index builder):
    python scripts/catalog/build_catalog_index.py --out paper/pcawg_run/catalog_v2
"""
import argparse, json, os, shutil
from pathlib import Path
import pandas as pd

# The event class was written as "PGD" before the ccPG rename; accept both so this works
# on catalogs built from either vintage of run.
CCPG = ("ccPG", "PGD")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="per-sample run directory (tumour_type/sample/)")
    ap.add_argument("--cohort", required=True, help="cohort tables directory (tau aggregate output)")
    ap.add_argument("--out", required=True, help="catalog output directory")
    ap.add_argument("--link", choices=["symlink", "copy"], default="symlink",
                    help="symlink viewers (default, no extra disk) or copy them (self-contained)")
    args = ap.parse_args()

    samples_dir, cohort, out = Path(args.samples), Path(args.cohort), Path(args.out)
    (out / "_meta").mkdir(parents=True, exist_ok=True)

    smp = pd.read_csv(cohort / "tau_samples.tsv", sep="\t")
    evt = pd.read_csv(cohort / "tau_events.tsv", sep="\t")
    ev_by_sample = {s: d for s, d in evt.groupby("sample")}

    rows, fails = [], []
    for r in smp.itertuples(index=False):
        sid, tt = str(r.sample), str(r.tumor_type)
        src = samples_dir / tt / sid / f"{sid}.overview_interactive.html"
        if not src.exists():
            fails.append({"uuid": sid, "tumor_type": tt, "error": "no viewer"})
            continue

        dst_rel = Path("viewers") / tt / f"{sid}.overview_interactive.html"
        dst = out / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.link == "symlink":
            dst.symlink_to(os.path.relpath(src.resolve(), dst.parent))
        else:
            shutil.copy2(src, dst)

        e = ev_by_sample.get(sid)
        wgd = e[e.classification == "WGD"] if e is not None else None
        ccpg = e[e.classification.isin(CCPG)] if e is not None else None
        nz = lambda v: None if pd.isna(v) else v

        meta = dict(
            uuid=sid, tumor_type=tt,
            n_events=int(len(e)) if e is not None else 0,
            n_wgd=int(len(wgd)) if wgd is not None else 0,
            n_pgd=int(len(ccpg)) if ccpg is not None else 0,   # key kept for the existing index
            wgd=bool(wgd is not None and len(wgd)),
            pgd=bool(ccpg is not None and len(ccpg)),
            wgd_times=[round(float(t), 3) for t in wgd.event_time_frac] if wgd is not None and len(wgd) else [],
            pgd_times=[round(float(t), 3) for t in ccpg.event_time_frac] if ccpg is not None and len(ccpg) else [],
            ploidy=round(float(r.ploidy), 3) if nz(r.ploidy) is not None else None,
            purity=round(float(r.purity), 4) if nz(r.purity) is not None else None,
            n_mutations=int(r.n_snvs) if nz(r.n_snvs) is not None else None,
            n_segs_timed=int(r.n_timed_segments) if nz(r.n_timed_segments) is not None else None,
            html=str(dst_rel),
            size_mb=round(src.stat().st_size / 1e6, 1),
        )
        rows.append(meta)

    with open(out / "_meta" / "shard_000.json", "w") as f:
        json.dump({"rows": rows, "fails": fails}, f)
    print(f"{len(rows)} samples catalogued, {len(fails)} without a viewer")
    print(f"  viewers {args.link}ed into {out}/viewers/")
    print(f"  metadata -> {out}/_meta/shard_000.json")
    print(f"\nnext: python scripts/catalog/build_catalog_index.py --out {out}")


if __name__ == "__main__":
    main()
