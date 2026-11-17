"""Observed SBS96 spectrum and per-signature weights for one sample -> paper/figures/data/signature/spectrum_<id>.tsv.

Everything here is measured, not modelled. The genome object already carries what is needed:

  sbs96          each mutation's trinucleotide channel, computed once in tau.preprocessing from the
                 reference FASTA — so no reference is needed downstream
  SBS1..SBS100   the COSMIC likelihood s_j(context) for that mutation's channel (NOT a posterior:
                 the columns are unnormalised and cover signatures with zero exposure in the sample)

The weight Tau actually applies is then

    w_j(c) = sum_{j in group} e_j s_j(c)  /  sum_{all k} e_k s_k(c)

with e the sample's exposures. Verified against the stored mut_w: reconstructing the clock weight this
way reproduces it to 3e-16, which confirms both the formula and that the stored columns are likelihoods.

Exposures come from the exposures file because runs predating `Genome.exposures` do not carry them on
the pickle; newer runs will, and this falls back to the genome when present.

The spectrum is of the SNVs TAU USED — those on timed segments, after its filtering — not the whole
genome MuSiCal was fitted to. For this sample that is 48,438 against MuSiCal's 46,994. That is the
right denominator for a figure about Tau's weighting, but it is not the same population.

Usage:  python paper/figures/data/signature/build/build_spectrum.py [sample_prefix]
"""
import glob
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
OUT = cache_dir_of(__file__); OUT.mkdir(parents=True, exist_ok=True)
EXPO = REF / "MuSiCal_assignment_PCAWG.txt"
TAU_OUT = TAU_OUTPUTS
GROUPS = [("SBS1", ["SBS1"]), ("SBS5", ["SBS5"]),
          ("SBS17 (a+b)", ["SBS17a", "SBS17b"]), ("APOBEC (SBS2+13)", ["SBS2", "SBS13"]),
          ("SBS4", ["SBS4"]), ("SBS7 (a-d)", ["SBS7a", "SBS7b", "SBS7c", "SBS7d"]),
          ("SBS3", ["SBS3"])]
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "49dec0c2"


def main():
    p = glob.glob(str(TAU_OUT / "*" / f"{SAMPLE}*" / f"{SAMPLE}*.genome.pkl.gz"))
    if not p:
        sys.exit(f"no genome pkl for {SAMPLE}")
    g = pickle.load(gzip.open(p[0], "rb"))
    t = pd.concat([s.snv_table for s in g.segments
                   if getattr(s, "snv_table", None) is not None and len(s.snv_table)],
                  ignore_index=True)
    if "sbs96" not in t.columns:
        sys.exit("this genome predates the sbs96 column; re-run preprocessing with --ref_fasta")

    exp = getattr(g, "exposures", None)
    if not exp:
        e = pd.read_csv(EXPO, sep="\t")
        row = e[e.aliquot_id.astype(str).str.startswith(SAMPLE)].iloc[0]
        exp = {c: float(row[c]) for c in e.columns if c.startswith("SBS") and float(row[c]) > 0}
        src = "exposures file"
    else:
        src = "genome.exposures"
    sig = [c for c in exp if c in t.columns and exp[c] > 0]
    E = np.array([exp[c] for c in sig])
    X = t[sig].to_numpy(float)                       # s_j(context) per SNV
    den = (X * E).sum(1)

    obs = t.sbs96.value_counts()
    rows = []
    for lab, members in GROUPS:
        mem = [c for c in members if c in sig]
        if not mem:
            continue
        idx = [sig.index(c) for c in mem]
        w = np.divide((X[:, idx] * E[idx]).sum(1), den, out=np.zeros_like(den), where=den > 0)
        s = pd.Series(w).groupby(t.sbs96).sum()      # weighted mutations per channel
        rows.append(pd.DataFrame({"channel": s.index, "group": lab, "weighted": s.to_numpy()}))
    w = pd.concat(rows, ignore_index=True)
    w["observed"] = w.channel.map(obs).astype(float)
    w["weight"] = w.weighted / w.observed.replace(0, np.nan)
    w.insert(0, "sample", str(p[0]).split("/")[-1].split(".")[0])
    w.to_csv(OUT / f"spectrum_{SAMPLE}.tsv", sep="\t", index=False)

    print(f"{SAMPLE}: {len(t):,} SNVs, {t.sbs96.nunique()} channels, exposures from {src}")
    print(w.groupby("group").weighted.sum().round(0).to_string())
    print(f"total weighted over the {len(sig)} active signatures should equal n SNVs: "
          f"{(np.divide((X*E).sum(1), den, out=np.zeros_like(den), where=den>0)).sum():,.0f}")
    print(f"-> {OUT/f'spectrum_{SAMPLE}.tsv'}")


if __name__ == "__main__":
    main()
