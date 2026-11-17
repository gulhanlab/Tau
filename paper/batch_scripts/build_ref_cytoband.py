#!/usr/bin/env python3
"""Build ref_hg19_cytoband.tsv (one row per chromosome ARM) from the shipped hg19 centromere + chrom-size
files, so arm-level figures read one consistent source instead of hardcoded HG19/CEN dicts.
Columns: chrom, arm{p,q}, start_bp, end_bp, arm_len_bp, centromere_bp, chrom_size_bp.
"""
import sys
from pathlib import Path
import pandas as pd

R = next(p for p in Path(__file__).resolve().parents if (p / "paper").is_dir())
CEN = R / "tau/data/hg19.centromeres.tsv"
SIZES = R / "tau/data/hg19.chrom.sizes"
OUT = R / "paper/data_tables/ref/ref_hg19_cytoband.tsv"


def main():
    cen = pd.read_csv(CEN, sep="\t", dtype={"chrom": str})
    sz = pd.read_csv(SIZES, sep="\t", header=None, names=["chrom", "size"], dtype={"chrom": str})
    sz["chrom"] = sz["chrom"].str.replace("chr", "", regex=False)
    size = dict(zip(sz.chrom, sz["size"]))
    rows = []
    for r in cen.itertuples():
        c = str(r.chrom).replace("chr", "")
        cs = int(size.get(c, r.q_start))
        rows.append(dict(chrom=c, arm="p", start_bp=1, end_bp=int(r.p_end),
                         arm_len_bp=int(r.p_end), centromere_bp=int(r.centromere), chrom_size_bp=cs))
        rows.append(dict(chrom=c, arm="q", start_bp=int(r.q_start), end_bp=cs,
                         arm_len_bp=cs - int(r.q_start), centromere_bp=int(r.centromere), chrom_size_bp=cs))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"ref_hg19_cytoband: {len(df)} arm rows ({df.chrom.nunique()} chroms) -> {OUT.name}")
    print(df.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
