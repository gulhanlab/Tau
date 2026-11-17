"""Merge backfill_chunks/* -> the cohort composite tables in paper/data_tables/.
Small tables concat -> TSV; gains concat -> tau_gains.tsv.gz + tau_gains.head.tsv; enrich tau_samples with
the per-sample pkl aggregates. Run after the backfill array finishes.
"""
import glob
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[2]
CH = R / "paper/batch_scripts/backfill_chunks"
DT = R / "paper/data_tables"


def _concat(patt):
    fs = sorted(glob.glob(str(CH / patt)))
    frames = [pd.read_csv(f, sep="\t") for f in fs if Path(f).stat().st_size > 5]
    return pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()


def main():
    seg = _concat("chunk_*.segments.tsv").sort_values(["sample", "chrom", "start_bp"])
    seg.to_csv(DT / "tau_segments.tsv", sep="\t", index=False)
    rt = _concat("chunk_*.routes.tsv").sort_values(["sample", "seg_id", "route_key"])
    rt.to_csv(DT / "tau_routes.tsv", sep="\t", index=False)
    mu = _concat("chunk_*.multiplicities.tsv").sort_values(["sample", "seg_id", "multiplicity"])
    mu.to_csv(DT / "tau_multiplicities.tsv", sep="\t", index=False)
    print(f"tau_segments {len(seg)} | tau_routes {len(rt)} | tau_multiplicities {len(mu)}")

    # gains: concat gz-TSV chunks -> one gz-TSV + a head preview (skip empty/headerless chunks)
    def _read_gz(f):
        try:
            return pd.read_csv(f, sep="\t", low_memory=False, dtype={"chrom": str})
        except (pd.errors.EmptyDataError, EOFError):
            return None
    gfs = sorted(glob.glob(str(CH / "chunk_*.gains.tsv.gz")))
    gframes = [g for g in (_read_gz(f) for f in gfs) if g is not None and len(g)]
    gains = pd.concat(gframes, ignore_index=True)
    gains = gains.sort_values(["sample", "seg_id", "route_key", "gain_index", "draw_id"])
    gains.to_csv(DT / "tau_gains.tsv.gz", sep="\t", index=False, compression="gzip")
    gains.head(500).to_csv(DT / "tau_gains.head.tsv", sep="\t", index=False)
    print(f"tau_gains {len(gains)} rows -> tau_gains.tsv.gz (+ head)")

    # enrich tau_samples with pkl aggregates (fill the NA columns from the cheap build)
    agg = _concat("chunk_*.sampleagg.tsv")
    s = pd.read_csv(DT / "tau_samples.tsv", sep="\t")
    a = agg.set_index("sample")
    for col in ["purity", "n_snvs", "n_segments", "mean_ess"]:
        s[col] = s["sample"].map(a[col]) if col in a else s[col]
    # prefer pkl-derived ploidy where the cheap ploidy was missing
    s["ploidy"] = s["ploidy"].fillna(s["sample"].map(a["ploidy_seg"]) if "ploidy_seg" in a else s["ploidy"])
    for c in ["n_snvs", "n_segments", "n_timed_segments", "n_events"]:
        s[c] = s[c].astype("Int64")
    s.to_csv(DT / "tau_samples.tsv", sep="\t", index=False)
    print(f"tau_samples enriched: purity/n_snvs/n_segments/mean_ess filled for {a.shape[0]} samples")


if __name__ == "__main__":
    main()
