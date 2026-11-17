#!/usr/bin/env python3
"""Per-donor de-duplication for the multi-sample (multi-aliquot) PCAWG donors.

14 donors (concentrated in Breast + Prostate) contribute 2-25 aliquots each. The aliquots vary in
purity -> the low-purity ones return near-diploid CN (ploidy ~2, few timed segments) and unstable WGD
timing, so one tumour's aliquots smear across the whole ploidy/timing axis (the source of the prostate
"three clusters" in loss and the spurious bimodal WGD/PGD timing). Sample IDs are aliquot UUIDs; the
8-char prefix is the donor (random collision in a 16^8 space is negligible).

Representative = the HIGHEST-INFORMATION aliquot per donor = max n_timed (most gains timed ~= highest
purity / most reliable WGD call + timing). Deterministic tie-break by sample id.

Usage:
    from donor_dedup import representative_samples, dedup_df
    reps = representative_samples()                 # set of one sample id per donor
    df = dedup_df(df)                               # filter a dataframe to representatives
"""
import os
from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("TAU_REPO", "."))
TDN = BASE/"dev/pcawg/FigWGDPGD/timeable_denom.tsv"


ROSTER = BASE/"dev/pcawg/FigWGDPGD/samples_allpcawg.tsv"


def donor_id(sample: str) -> str:
    return str(sample)[:8]


def pelican_donors() -> set:
    """PELICAN rapid-autopsy metastatic-prostate donors (Gundem 2015, Nature) = multi-aliquot
    Prost.AdenoCA donors. These are MULTIPLE METASTASES from one patient (a metastatic cascade, ADT-
    treated), not treatment-naive primary tumours — excluded from cohort analyses. Returns donor prefixes.
    """
    ros = pd.read_csv(ROSTER, sep="\t")
    ros = ros[ros["tumor_type"] == "Prost.AdenoCA"].copy()
    ros["donor"] = ros["sample"].map(donor_id)
    sz = ros.groupby("donor").size()
    return set(sz[sz > 1].index)


def representative_samples(quality_col: str = "n_timed", drop_pelican: bool = True) -> set:
    """One sample id per donor: the max-quality (max n_timed) aliquot, tie-broken by sample id.
    drop_pelican=True also removes the PELICAN metastatic-prostate donors entirely."""
    d = pd.read_csv(TDN, sep="\t")
    if quality_col not in d.columns:
        quality_col = "ploidy"
    d = d.copy()
    d["donor"] = d["sample"].map(donor_id)
    d["_q"] = d[quality_col].fillna(-1)
    d = d.sort_values(["_q", "sample"], ascending=[False, True])
    reps = set(d.groupby("donor", sort=False).first()["sample"])
    if drop_pelican:
        pel = pelican_donors()
        reps = {s for s in reps if donor_id(s) not in pel}
    return reps


def dedup_df(df: pd.DataFrame, sample_col: str = "sample") -> pd.DataFrame:
    """Filter df to one representative aliquot per donor AND drop PELICAN metastatic-prostate donors."""
    reps = representative_samples()
    return df[df[sample_col].isin(reps)].copy()


if __name__ == "__main__":
    d = pd.read_csv(TDN, sep="\t")
    d["donor"] = d["sample"].map(donor_id)
    pel = pelican_donors()
    reps = representative_samples()
    reps_keep_pel = representative_samples(drop_pelican=False)
    sz = d.groupby("donor").size()
    print(f"{len(d)} samples -> {d.donor.nunique()} donors")
    print(f"multi-aliquot donors collapsed: {(sz > 1).sum()} donors, {sz[sz > 1].sum() - (sz > 1).sum()} "
          f"aliquots dropped")
    print(f"PELICAN metastatic-prostate donors EXCLUDED: {len(pel)}  ({', '.join(sorted(pel))})")
    print(f"final representative set: {len(reps)} (vs {len(reps_keep_pel)} if PELICAN kept)")
    # show what the representative looks like for the big multi-sample donors
    for don in sz[sz > 2].sort_values(ascending=False).index[:8]:
        g = d[d.donor == don]
        r = g[g["sample"].isin(reps)]
        tag = f"rep ploidy {r.iloc[0].ploidy:.1f}, n_timed {int(r.iloc[0].n_timed)}" if len(r) else "EXCLUDED (PELICAN)"
        print(f"  {don}: {len(g):2d} aliquots (ploidy {g.ploidy.min():.1f}-{g.ploidy.max():.1f}, "
              f"n_timed {int(g.n_timed.min())}-{int(g.n_timed.max())}) -> {tag}")
