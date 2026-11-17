"""STOPGAP (review package): build tau_samples.tsv + tau_events.tsv in the new schema by reshaping the
EXISTING cohort tables — no pkl backfill. Columns needing the genome objects (purity, n_snvs, n_segments,
mean_ess, donor_id) are left NA and filled later by the real tau/aggregate builder. Gives real,
navigable base tables now. Run: python paper/data_tables/build_cheap_base_tables.py
"""
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[2]
FW = R / "dev/pcawg/FigWGDPGD"
OUT = R / "paper/data_tables"

# ── tau_events ── from events_allpcawg (+ strict class from the annotated catalogue) ───
ev = pd.read_csv(FW / "events_allpcawg.tsv", sep="\t")
cat = pd.read_csv(R / "dev/pcawg/misc_data_and_plots/event_catalogue_allpcawg_annotated.tsv", sep="\t")
# match strict class on (sample, nearest event time)
cat_s = cat[["sample", "event_time", "event_class_v2"]].copy()
events = ev.rename(columns={"cluster_id": "event_id", "time": "event_time_frac",
                            "gf": "genome_frac"})[
    ["sample", "tumor_type", "event_id", "event_time_frac", "classification",
     "genome_frac", "n_chroms", "n_segments", "chrom"]].copy()
# nearest-time merge per sample for class_strict
def _strict(row):
    c = cat_s[cat_s["sample"] == row["sample"]]
    if len(c) == 0:
        return pd.NA
    i = (c["event_time_frac"] if False else (c["event_time"] - row["event_time_frac"]).abs()).idxmin()
    return c.loc[i, "event_class_v2"]
events["class_strict"] = events.apply(_strict, axis=1)
events = events[["sample", "tumor_type", "event_id", "event_time_frac", "classification",
                 "class_strict", "genome_frac", "n_chroms", "n_segments", "chrom"]]
events = events.sort_values(["sample", "event_time_frac"])
events.to_csv(OUT / "tau_events.tsv", sep="\t", index=False)

# ── tau_samples ── from samples_allpcawg + timeable_denom + aggregated events ───────────
samp = pd.read_csv(FW / "samples_allpcawg.tsv", sep="\t")
den = pd.read_csv(FW / "timeable_denom.tsv", sep="\t")[["sample", "ploidy", "n_timed", "denom"]]
wgd = ev[ev.classification == "WGD"].groupby("sample")["time"].min().rename("wgd_time_frac")
n_ev = ev.groupby("sample")["cluster_id"].nunique().rename("n_events")
has_w = ev[ev.classification == "WGD"].groupby("sample").size().gt(0).rename("has_wgd")
has_p = ev[ev.classification == "PGD"].groupby("sample").size().gt(0).rename("has_pgd")
s = (samp.merge(den, on="sample", how="left")
     .merge(wgd, on="sample", how="left").merge(n_ev, on="sample", how="left")
     .merge(has_w, on="sample", how="left").merge(has_p, on="sample", how="left"))
s["has_wgd"] = s["has_wgd"].fillna(False); s["has_pgd"] = s["has_pgd"].fillna(False)
s["n_events"] = s["n_events"].fillna(0).astype("Int64")
s["n_timed_segments"] = s["n_timed"].astype("Int64")
HG19_AUTOSOME_BP = 2_881_033_286   # denom is haploid-autosome bp; frac = denom / this
s["timeable_genome_frac"] = (s["denom"] / HG19_AUTOSOME_BP).round(4)
# schema columns NOT derivable without pkls -> NA (filled by the real tau/aggregate builder)
for c in ["donor_id", "is_representative_aliquot", "purity", "n_snvs", "n_segments", "mean_ess"]:
    s[c] = pd.NA
s["n_snvs"] = s["n_snvs"].astype("Int64"); s["n_segments"] = s["n_segments"].astype("Int64")
s = s[["sample", "donor_id", "is_representative_aliquot", "tumor_type", "purity", "ploidy",
       "n_snvs", "n_segments", "n_timed_segments", "n_events", "mean_ess",
       "has_wgd", "has_pgd", "wgd_time_frac", "timeable_genome_frac"]].sort_values("sample")
s.to_csv(OUT / "tau_samples.tsv", sep="\t", index=False)

print(f"tau_events.tsv:  {len(events)} rows, {events['sample'].nunique()} samples")
print(f"tau_samples.tsv: {len(s)} rows; has_wgd {int(s.has_wgd.sum())}, has_pgd {int(s.has_pgd.sum())}")
print("NA (need pkl backfill):", [c for c in s.columns if s[c].isna().all()])
