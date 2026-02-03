#!/usr/bin/env python3
"""
tau/preprocessing.py - VCF/CNV parsing and mutation preprocessing.

Loads SNVs from VCF, extracts SBS96 context, and applies COSMIC signature weights.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
import pysam
from scipy.stats import beta as beta_dist

EPS = 1e-12


def _fetch_sbs96(df: pd.DataFrame, fasta_path: str, flank: int = 1) -> pd.Series:
    fa = pysam.FastaFile(fasta_path)
    def revcomp(s: str) -> str:
        rc = str.maketrans("ACGTacgt", "TGCAtgca")
        return s.translate(rc)[::-1]
    out = []
    for c, p, r, a in zip(df["chrom"].astype(str), df["pos"].astype(int),
                          df["ref"].str.upper(), df["alt"].str.upper()):
        L = max(1, p - flank); R = p + flank
        left  = fa.fetch(c, L-1, p-1)
        right = fa.fetch(c, p, R)
        if len(left)  < flank: left  = "N"*(flank-len(left)) + left
        if len(right) < flank: right = right + "N"*(flank-len(right))
        fl, fr = left[-flank:], right[:flank]
        if r in ("A","G"): fl, fr, r, a = revcomp(fr), revcomp(fl), revcomp(r), revcomp(a)
        out.append(f"{fl}[{r}>{a}]{fr}")
    fa.close()
    return pd.Series(out, index=df.index, name="sbs96")


def _load_cosmic(cosmic_csv: str | Path) -> pd.DataFrame:
    C = pd.read_csv(cosmic_csv)
    C = C.rename(columns={"Type":"sbs96"}).set_index("sbs96")
    sig_cols = [c for c in C.columns if str(c).startswith("SBS")]
    return C[sig_cols].copy()


def _load_exposures(exposures_tsv: str | Path, sample: str) -> pd.Series:
    E = pd.read_csv(exposures_tsv, sep="\t")
    key = "sample" if "sample" in E.columns else ("aliquot_id" if "aliquot_id" in E.columns else None)
    row = E.loc[E[key] == sample]
    if row.empty: raise ValueError(f"Sample {sample} not found in exposures.")
    s = row.iloc[0]
    sig_cols = [c for c in s.index if str(c).startswith("SBS")]
    v = s[sig_cols].astype(float)
    if v.sum() <= 0: v[:] = 1.0
    return v / v.sum()


def attach_signature_weights(
    snv: pd.DataFrame,
    *,
    ref_fasta: str | Path | None,
    cosmic_csv: str | Path | None,
    exposures: pd.Series | None,
    signatures: list[str] | None,
    weight_mode: str = "soft",
) -> pd.DataFrame:
    if ref_fasta and "sbs96" not in snv.columns:
        snv = snv.copy()
        snv["sbs96"] = _fetch_sbs96(snv, ref_fasta)

    if not (cosmic_csv and exposures is not None and signatures is not None):
        out = snv.copy(); out["mut_w"] = 1.0; return out

    C = _load_cosmic(cosmic_csv)
    avail = [c for c in C.columns if c in exposures.index]
    if not avail:
        out = snv.copy(); out["mut_w"] = 1.0; return out

    X = snv.merge(C[avail], left_on="sbs96", right_index=True, how="left").fillna(0.0)
    W = np.vstack([X[s].to_numpy() * float(exposures[s]) for s in avail]).T  # (N, |avail|)
    winners_ix = np.argmax(W, axis=1)
    winners = np.array(avail, dtype=object)[winners_ix]
    out = X.copy()
    out["hard_sig_choice"] = winners

    if weight_mode == "hard":
        chosen = set(s for s in signatures if s in avail)
        out["mut_w"] = (np.isin(winners, list(chosen))).astype(float)
    else:
        use = [s for s in signatures if s in avail]
        out["mut_w"] = np.sum(np.vstack([X[s].to_numpy() * float(exposures[s]) for s in use]).T, axis=1) if use else 0.0
    return out


def flag_lowVAF_subclonal(
    df: pd.DataFrame,
    *,
    purity: float,
    min_alt: int = 3,
    min_vaf: float | None = None,
    alpha: float = 0.01,
) -> pd.Series:
    """Flag mutations with VAF too low to be clonal given copy number state."""
    nalt = pd.to_numeric(df.get("nalt", np.nan), errors="coerce").to_numpy()
    nref = pd.to_numeric(df.get("nref", np.nan), errors="coerce").to_numpy()
    n    = (np.nan_to_num(nalt, nan=-1) + np.nan_to_num(nref, nan=-1)).astype(int)

    maj  = pd.to_numeric(df.get("major_cn", np.nan), errors="coerce").to_numpy()
    mino = pd.to_numeric(df.get("minor_cn", np.nan), errors="coerce").to_numpy()
    tot  = np.nan_to_num(maj, nan=0).astype(int) + np.nan_to_num(mino, nan=0).astype(int)

    min_vaf = 0.0 if (min_vaf is None) else float(min_vaf)
    kmin = np.maximum(min_alt, np.ceil(min_vaf * n)).astype(int)
    detected = (n > 0) & (np.nan_to_num(maj, nan=0) > 0) & (nalt >= kmin)

    a = np.maximum(nalt, 0) + 1.0
    b = np.maximum(nref, 0) + 1.0
    with np.errstate(invalid="ignore"):
        p_u = beta_dist.ppf(1.0 - float(alpha), a, b)
    p_u = np.clip(p_u, 0.0, 1.0)

    denom = purity * tot + (1.0 - purity) * 2.0
    denom = np.where(denom <= 0, np.nan, denom)
    ccf_u_max = (p_u * denom) / max(purity, EPS)
    is_subclonal = detected & (ccf_u_max < (1.0 - 1e-12))
    is_subclonal = np.where(np.isnan(ccf_u_max), False, is_subclonal)
    return pd.Series(is_subclonal.astype(bool), index=df.index, name="tau_subclonal_lowVAF")


def read_vcf_counts(vcf_path: str | Path) -> pd.DataFrame:
    """Read SNV counts from a VCF file."""
    rows = []
    vcf = pysam.VariantFile(str(vcf_path))
    for rec in vcf:
        chrom = rec.contig.replace("chr","")
        if rec.filter.keys():  # drop non-PASS
            continue
        info = rec.info
        nalt = float(info.get("t_alt_count", -1)) if "t_alt_count" in info else -1
        nref = float(info.get("t_ref_count", -1)) if "t_ref_count" in info else -1
        if nalt < 0 or nref < 0:
            continue
        rows.append({"chrom": chrom, "pos": int(rec.pos),
                     "ref": rec.ref, "alt": rec.alts[0],
                     "nalt": nalt, "nref": nref})
    return pd.DataFrame(rows)


def map_snvs_to_cnv(snv: pd.DataFrame, cnv_tsv: str | Path) -> pd.DataFrame:
    """Map SNVs to copy number segments."""
    cnv = pd.read_csv(cnv_tsv, sep="\t").rename(columns={"chromosome":"chrom"})
    cnv["chrom"] = cnv["chrom"].astype(str).str.replace("^chr","", regex=True)
    ivx = pd.IntervalIndex.from_arrays(cnv["start"], cnv["end"], closed="both")
    parts = []
    for chrom, sub in snv.groupby("chrom", sort=False):
        mask = cnv["chrom"].values == chrom
        if not mask.any():
            continue
        iv_chrom = ivx[mask]; abs_rows = np.flatnonzero(mask)
        loc = iv_chrom.get_indexer(sub["pos"])
        mapped = sub.assign(seg_ix = abs_rows[loc])
        parts.append(mapped[mapped["seg_ix"] != -1])
    if not parts:
        return pd.DataFrame(columns=list(snv.columns)+["start","end","major_cn","minor_cn","segment_id"])
    snv_map = pd.concat(parts, ignore_index=True).merge(
        cnv.reset_index(drop=True),
        left_on="seg_ix", right_index=True, how="left", suffixes=("", "_cn")
    )
    snv_map["segment_id"] = (snv_map.chrom + ":" +
                             snv_map.start.astype(int).astype(str) + "-" +
                             snv_map.end.astype(int).astype(str))
    return snv_map


def preprocess_sample(
    sample: str,
    vcf_path: str | Path,
    cnv_tsv: str | Path,
    *,
    purity: float,
    ref_fasta: str | Path | None = None,
    cosmic_csv: str | Path | None = None,
    exposures_tsv: str | Path | None = None,
    apply_subclonal_filter: bool = True,
    detect_min_alt: int = 3,
    detect_min_vaf: float | None = None,
    subclonal_alpha: float = 0.01,
    signatures: list | None = None,
    subclonal_list: str | None = None,
    mode: str = 'soft'
) -> pd.DataFrame:
    """
    Main preprocessing function: VCF + CNV -> per-mutation table ready for Tau.

    Parameters
    ----------
    sample : str
        Sample identifier
    vcf_path : str or Path
        Path to VCF file with SNV calls
    cnv_tsv : str or Path
        Path to copy number segment file (TSV)
    purity : float
        Tumor purity estimate
    ref_fasta : str or Path, optional
        Reference FASTA for SBS96 context extraction
    cosmic_csv : str or Path, optional
        COSMIC signature definitions
    exposures_tsv : str or Path, optional
        Per-sample signature exposures
    apply_subclonal_filter : bool
        Whether to flag low-VAF subclonal mutations
    detect_min_alt : int
        Minimum alt reads for detection
    detect_min_vaf : float, optional
        Minimum VAF for detection
    subclonal_alpha : float
        Significance level for subclonal flagging
    signatures : list, optional
        List of signature names to use for weighting
    subclonal_list : str, optional
        Path to file with mutation IDs to flag as subclonal
    mode : str
        'soft' or 'hard' weighting mode

    Returns
    -------
    pd.DataFrame
        Preprocessed mutation table
    """
    snv = read_vcf_counts(vcf_path)
    if snv.empty: return pd.DataFrame()

    snv_map = map_snvs_to_cnv(snv, cnv_tsv)
    if snv_map.empty: return pd.DataFrame()

    # add mut_w (or all 1s if no signatures requested)
    sigs = signatures
    exposures = _load_exposures(exposures_tsv, sample) if (exposures_tsv and cosmic_csv) else None
    snv_map = attach_signature_weights(
        snv_map, ref_fasta=ref_fasta, cosmic_csv=cosmic_csv,
        exposures=exposures, signatures=sigs, weight_mode=mode
    )

    snv_map['mutation_id'] = snv_map['chrom'] + ':' + snv_map['pos'].astype(str) + ':' + snv_map['ref'] + '/' + snv_map['alt']
    snv_map['vaf'] = snv_map['nalt'] / (snv_map['nalt'] + snv_map['nref'])

    # optional: flag low-CCF subclonal (in-built simple method)
    if apply_subclonal_filter:
        snv_map["subclonal"] = flag_lowVAF_subclonal(
            snv_map, purity=float(purity),
            min_alt=detect_min_alt, min_vaf=detect_min_vaf, alpha=float(subclonal_alpha)
        )
    else:
        snv_map["subclonal"] = False

    # if a subclonal list is provided (from an external caller), flag these for downstream processing
    if subclonal_list:
        subclonal_set = set()
        with open(subclonal_list, 'r') as f:
            for line in f:
                subclonal_set.add(line.strip())
        snv_map["subclonal"] = snv_map["mutation_id"].isin(subclonal_set)

    # final column order (stable, easy to read)
    cols = ["chrom","pos","ref","alt","nalt","nref",
            "start","end","major_cn","minor_cn","segment_id","mut_w","subclonal"]
    extra = [c for c in snv_map.columns if c not in cols]
    return snv_map[cols + extra].copy()


def main():
    ap = argparse.ArgumentParser(description="Tau preprocessing: VCF + CNV ⇒ per-mutation table.")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--cnv", required=True)
    ap.add_argument("--purity", type=float, required=True)
    ap.add_argument("--ref-fasta", default=None)
    ap.add_argument("--cosmic-csv", default=None)
    ap.add_argument("--exposures", default=None)
    ap.add_argument("--no-subclonal-filter", action="store_true")
    ap.add_argument("--detect-min-alt", type=int, default=3)
    ap.add_argument("--detect-min-vaf", type=float, default=None)
    ap.add_argument("--subclonal-alpha", type=float, default=0.01)
    ap.add_argument("--signatures", default="ALL", help="comma-separated list of signatures you wish to run Tau on")
    ap.add_argument("--mode", default='soft')
    ap.add_argument("--subclonal_list", default=None)
    ap.add_argument("--out", required=True)  # .tsv or .parquet
    args = ap.parse_args()

    sigs = args.signatures.split(',')

    df = preprocess_sample(
        sample=args.sample,
        vcf_path=args.vcf,
        cnv_tsv=args.cnv,
        purity=args.purity,
        ref_fasta=args.ref_fasta,
        cosmic_csv=args.cosmic_csv,
        exposures_tsv=args.exposures,
        apply_subclonal_filter=not args.no_subclonal_filter,
        detect_min_alt=args.detect_min_alt,
        detect_min_vaf=args.detect_min_vaf,
        subclonal_alpha=args.subclonal_alpha,
        signatures=sigs,
        mode=args.mode,
        subclonal_list=args.subclonal_list
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, sep="\t", index=False)


if __name__ == "__main__":
    main()
