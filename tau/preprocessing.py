#!/usr/bin/env python3
"""
tau/preprocessing.py - VCF/CNV parsing and mutation preprocessing.

Loads SNVs from VCF, extracts SBS96 context, and applies COSMIC signature weights.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import gzip
import warnings
import os
import numpy as np
import pandas as pd
import pysam
from scipy.stats import beta as beta_dist

EPS = 1e-12

# Standard clock-like COSMIC signatures for use with signatures= parameter.
# SBS1  — age-related deamination (C>T at CpG); present in all tissues.
# SBS5  — replication-clock signature; flat spectrum, universal.
# Together these are the recommended default for Tau clock-weighted timing.
CLOCK_SIGNATURES: list[str] = ["SBS1", "SBS5"]


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


EXPOSURE_SAMPLE_ALIASES = (
    "sample", "aliquot_id", "samplename", "sample_name", "sample_id", "sampleid",
    "id", "name", "tumor", "tumour", "donor_id", "icgc_donor_id",
)


def _load_exposures(exposures_tsv: str | Path, sample: str) -> pd.Series:
    """Per-sample signature exposures. Delimiter is sniffed and the sample-ID column
    is matched case-insensitively; signature columns are any starting with "SBS"."""
    E = _read_table_any(exposures_tsv)
    norm = {}
    for c in E.columns:
        norm.setdefault(_norm_col(c), c)
    key = next((norm[a] for a in EXPOSURE_SAMPLE_ALIASES if a in norm), None)
    if key is None:
        raise ValueError(
            f"exposures table {exposures_tsv} has no sample-ID column.\n"
            f"  found columns: {list(E.columns)[:12]}\n"
            f"  accepted: {', '.join(EXPOSURE_SAMPLE_ALIASES[:6])}, ..."
        )
    row = E.loc[E[key].astype(str).str.strip() == str(sample).strip()]
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

    # Signature weighting needs a trinucleotide context per mutation. Without a reference
    # to derive it from, and without one supplied in the SNV table, degrade to equal weight
    # rather than failing — the timing itself does not depend on signatures.
    if "sbs96" not in snv.columns:
        warnings.warn(
            "signature weighting requested but no trinucleotide context is available "
            "(pass --ref_fasta, or include an sbs96/context column in the SNV table); "
            "falling back to equal-weight mutations.",
            RuntimeWarning, stacklevel=2,
        )
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
        if use:
            numer = np.sum(np.vstack([X[s].to_numpy() * float(exposures[s]) for s in use]).T, axis=1)
            # Normalize by all available signatures so mut_w = P(clock | context, exposures)
            denom = W.sum(axis=1)
            out["mut_w"] = np.where(denom > 0, numer / denom, 0.0)
        else:
            out["mut_w"] = 0.0
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
    """Read SNV counts from a VCF file.

    Supports two formats:
    - PCAWG-style: alt/ref counts in INFO fields t_alt_count / t_ref_count
    - PURPLE-style: alt/ref counts in FORMAT field AD (ref,alt) for the tumor
      sample (last sample column); FILTER=PASS is allowed.
    Only SNVs (single-base ref and alt) are retained.
    """
    rows = []
    vcf = pysam.VariantFile(str(vcf_path))
    for rec in vcf:
        # SNVs only
        if len(rec.ref) != 1 or not rec.alts or len(rec.alts[0]) != 1:
            continue
        chrom = rec.contig.replace("chr", "")
        # Keep records that are un-filtered OR have only PASS filter
        filters = set(rec.filter.keys())
        if filters and filters != {"PASS"}:
            continue
        info = rec.info
        nalt = float(info.get("t_alt_count", -1)) if "t_alt_count" in info else -1
        nref = float(info.get("t_ref_count", -1)) if "t_ref_count" in info else -1
        # Fallback: read from FORMAT:AD in tumor sample (PURPLE / HMF format)
        if nalt < 0 or nref < 0:
            samp_names = list(rec.samples.keys())
            # Tumor is typically the last sample; try reversed order
            for sname in reversed(samp_names):
                ad = rec.samples[sname].get("AD")
                if ad and len(ad) >= 2 and ad[0] is not None and ad[1] is not None:
                    nref, nalt = float(ad[0]), float(ad[1])
                    break
        if nalt < 0 or nref < 0:
            continue
        rows.append({"chrom": chrom, "pos": int(rec.pos),
                     "ref": rec.ref, "alt": rec.alts[0],
                     "nalt": nalt, "nref": nref})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tolerant tabular input
# ---------------------------------------------------------------------------
# Tau's own names come first; the rest are what other callers/tools emit, so a table
# from GRITIC, PURPLE, a MAF export or a hand-written TSV works without editing.
SNV_COLUMN_ALIASES = {
    "chrom": ("chrom", "chromosome", "chr", "#chrom", "contig", "seqnames"),
    "pos":   ("pos", "position", "start", "start_position", "posn"),
    "ref":   ("ref", "reference", "ref_allele", "reference_allele", "ref_base"),
    "alt":   ("alt", "alternate", "alt_allele", "alternate_allele", "tumor_seq_allele2",
              "var_allele", "alt_base"),
    "nalt":  ("nalt", "alt_count", "tumor_alt_count", "t_alt_count", "alt_counts",
              "altreadcount", "alt_reads", "ad_alt", "var_count"),
    "nref":  ("nref", "ref_count", "tumor_ref_count", "t_ref_count", "ref_counts",
              "refreadcount", "ref_reads", "ad_ref"),
}

CNV_COLUMN_ALIASES = {
    "chrom":    ("chrom", "chromosome", "chr", "#chrom", "contig", "seqnames"),
    "start":    ("start", "segment_start", "seg_start", "startpos", "start_position", "begin"),
    "end":      ("end", "segment_end", "seg_end", "endpos", "end_position", "stop"),
    "major_cn": ("major_cn", "majorallelecopynumber", "major", "major_copy_number",
                 "nmajor", "cn_major", "a1"),
    "minor_cn": ("minor_cn", "minorallelecopynumber", "minor", "minor_copy_number",
                 "nminor", "cn_minor", "a2"),
}


def _norm_col(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_").lstrip("#")


def resolve_columns(df, aliases, required, label, path):
    """Map a DataFrame's columns onto Tau's canonical names via `aliases`.

    Returns {canonical: actual}. Raises ValueError naming the missing field, the
    columns that were found and the names accepted — the format requirement is only
    discoverable at the point of failure, so it is stated there.
    """
    norm = {}
    for c in df.columns:
        norm.setdefault(_norm_col(c), c)
    out = {}
    for canon, names in aliases.items():
        for n in names:
            if n in norm:
                out[canon] = norm[n]
                break
    missing = [c for c in required if c not in out]
    if missing:
        raise ValueError(
            f"{label} {path} is missing required column(s): {', '.join(missing)}\n"
            f"  found columns: {list(df.columns)}\n"
            + "\n".join(f"  accepted for {m}: {', '.join(aliases[m])}" for m in missing)
        )
    return out


def _read_table_any(path):
    """Read a delimited table, sniffing tab / comma / whitespace."""
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    if df.shape[1] == 1:      # sniffing fell back to one column -- retry on whitespace
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
    return df


def looks_like_vcf(path):
    """True if *path* is a VCF, by extension or by its first line."""
    name = str(path).lower()
    if name.endswith((".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")):
        return True
    try:
        opener = gzip.open if name.endswith((".gz", ".bgz")) else open
        with opener(path, "rt", errors="ignore") as f:
            for line in f:
                if line.strip():
                    return line.startswith("##fileformat=VCF")
    except Exception:
        return False
    return False


def read_snv_table(path) -> pd.DataFrame:
    """Read SNV counts from a plain delimited table.

    Requires chromosome, position and the two read counts. ref/alt are optional —
    they are only needed for signature weighting (trinucleotide context) and for
    mutation IDs; absent, they are filled with "N" and signature weighting will not
    produce meaningful contexts.
    """
    df = _read_table_any(path)
    cols = resolve_columns(df, SNV_COLUMN_ALIASES, ("chrom", "pos", "nalt", "nref"),
                           "SNV table", path)
    out = pd.DataFrame({
        "chrom": df[cols["chrom"]].astype(str).str.replace("^chr", "", regex=True).str.strip(),
        "pos": pd.to_numeric(df[cols["pos"]], errors="coerce"),
        "ref": df[cols["ref"]].astype(str).str.strip() if "ref" in cols else "N",
        "alt": df[cols["alt"]].astype(str).str.strip() if "alt" in cols else "N",
        "nalt": pd.to_numeric(df[cols["nalt"]], errors="coerce"),
        "nref": pd.to_numeric(df[cols["nref"]], errors="coerce"),
    }).dropna(subset=["pos", "nalt", "nref"])
    out["pos"] = out["pos"].astype(int)
    # An already-computed trinucleotide context makes --ref_fasta unnecessary.
    ctx = next((c for c in df.columns
                if _norm_col(c) in ("sbs96", "context", "trinucleotide", "tri_context",
                                    "mutation_type", "sbs_96")), None)
    if ctx is not None:
        out["sbs96"] = df.loc[out.index, ctx].astype(str).str.strip()
    # SNVs only, matching the VCF reader's behaviour
    if "ref" in cols and "alt" in cols:
        out = out[(out["ref"].str.len() == 1) & (out["alt"].str.len() == 1)]
    if out.empty:
        raise ValueError(
            f"SNV table {path} yielded no usable rows (after dropping non-numeric "
            f"positions/counts and non-SNV rows)."
        )
    return out.reset_index(drop=True)


def read_snvs(path) -> pd.DataFrame:
    """Read SNV counts from either a VCF or a delimited table (auto-detected)."""
    if looks_like_vcf(path):
        snv = read_vcf_counts(path)
        if snv.empty:
            raise ValueError(
                f"VCF {path} yielded no usable SNVs. Tau reads read counts from either the "
                f"INFO fields t_alt_count/t_ref_count (PCAWG-style) or the FORMAT field AD of "
                f"the tumour sample (PURPLE-style); records with neither are skipped. "
                f"If your VCF carries counts elsewhere, pass a plain SNV table instead "
                f"(columns: chromosome, position, ref_count, alt_count)."
            )
        return snv
    return read_snv_table(path)


def map_snvs_to_cnv(snv: pd.DataFrame, cnv_tsv: str | Path) -> pd.DataFrame:
    """Map SNVs to copy number segments.

    Handles both PCAWG-style (major_cn/minor_cn integer columns) and
    PURPLE-style (majorAlleleCopyNumber/minorAlleleCopyNumber float columns).
    Float CN values are rounded to the nearest integer.
    """
    raw = _read_table_any(cnv_tsv)
    cols = resolve_columns(raw, CNV_COLUMN_ALIASES,
                           ("chrom", "start", "end", "major_cn", "minor_cn"),
                           "copy-number table", cnv_tsv)
    cnv = raw.rename(columns={v: k for k, v in cols.items()})
    cnv["chrom"] = cnv["chrom"].astype(str).str.replace("^chr","", regex=True).str.strip()
    # Round float CN values (e.g. from PURPLE) to nearest integer; drop rows with NaN CN
    cn_cols = [c for c in ("major_cn", "minor_cn") if c in cnv.columns]
    for col in cn_cols:
        if cnv[col].dtype != int:
            cnv[col] = cnv[col].round().astype("Int64")
    cnv = cnv.dropna(subset=cn_cols)
    for col in cn_cols:
        cnv[col] = cnv[col].astype(int)
    ivx = pd.IntervalIndex.from_arrays(cnv["start"], cnv["end"], closed="left")
    parts = []
    for chrom, sub in snv.groupby("chrom", sort=False):
        mask = cnv["chrom"].values == chrom
        if not mask.any():
            continue
        iv_chrom = ivx[mask]; abs_rows = np.flatnonzero(mask)
        loc = iv_chrom.get_indexer(sub["pos"])
        # get_indexer returns -1 where a position falls in no segment (a gap in the CN
        # profile, or a region dropped for missing CN). Mask BEFORE indexing: abs_rows[-1]
        # would silently resolve to the last segment on the chromosome, quietly moving
        # those SNVs onto a real segment and corrupting its multiplicity distribution.
        hit = loc >= 0
        if not hit.any():
            continue
        parts.append(sub[hit].assign(seg_ix=abs_rows[loc[hit]]))
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
        Run the built-in low-CCF test (:func:`flag_lowVAF_subclonal`). Flagged mutations get zero
        weight in the EM. Combines by UNION with ``subclonal_list`` — the two are independent.
    detect_min_alt : int
        Minimum alt reads for detection
    detect_min_vaf : float, optional
        Minimum VAF for detection
    subclonal_alpha : float
        Significance level for subclonal flagging
    signatures : list[str], optional
        Clock-like COSMIC signatures to weight mutations by.  Mutations whose
        SBS96 context is most consistent with these signatures receive higher
        weight in the EM and timing steps.

        Recommended: ``["SBS1", "SBS5"]`` (import as ``CLOCK_SIGNATURES``).

        - ``mode="soft"`` (default) — each mutation is weighted by
          P(clock signature | SBS96 context, exposures) ∈ [0, 1].
        - ``mode="hard"``  — binary: weight 1 if most-likely signature is a
          clock signature, 0 otherwise (more aggressive filtering).

        Defaults to None (all mutations weighted equally, equivalent to
        ignoring signature information).
    subclonal_list : str, optional
        Path to a file of mutation IDs to flag as subclonal, one ``{chrom}:{pos}:{ref}/{alt}`` per
        line (the format ``dev/pyclone/write_subclonal_labels.py`` writes). Use this for an external
        caller's assignment — PyClone-VI, DPClust, etc. ADDITIVE with ``apply_subclonal_filter``:
        a mutation is excluded if either source flags it.
    mode : str
        ``"soft"`` (default) or ``"hard"`` — see ``signatures`` above.

    Returns
    -------
    pd.DataFrame
        Preprocessed mutation table
    """
    snv = read_snvs(vcf_path)
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

    # Subclonal flagging — the two sources are INDEPENDENT and COMBINE (union). A mutation is
    # excluded from the EM if either the built-in test or an external caller says it is subclonal,
    # so a PyClone/DPClust list can be layered on top of the simple filter rather than replacing it.
    snv_map["subclonal"] = False
    if apply_subclonal_filter:
        snv_map["subclonal"] = flag_lowVAF_subclonal(
            snv_map, purity=float(purity),
            min_alt=detect_min_alt, min_vaf=detect_min_vaf, alpha=float(subclonal_alpha)
        )
    if subclonal_list:
        with open(subclonal_list, 'r') as f:
            subclonal_set = {line.strip() for line in f if line.strip()}
        snv_map["subclonal"] = (snv_map["subclonal"].fillna(False)
                                | snv_map["mutation_id"].isin(subclonal_set))

    # final column order (stable, easy to read)
    cols = ["chrom","pos","ref","alt","nalt","nref",
            "start","end","major_cn","minor_cn","segment_id","mut_w","subclonal"]
    extra = [c for c in snv_map.columns if c not in cols]
    out = snv_map[cols + extra].copy()
    # Carry the sample's signature exposures so Genome.create can store them on
    # the genome (used later by per-signature re-timing).
    if exposures is not None:
        out.attrs["exposures"] = {str(k): float(v) for k, v in exposures.items()}
    return out


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
