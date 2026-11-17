"""Compile the signature-vs-signature interval ratio table (segment + cluster level) into one tidy table.

Source: dev/pcawg/signatures/timing_tables/sig_intervals_range_<TT>_{segment,cluster}.tsv — see the
README there. Those are the real deliverable already, but they are WIDE: the signature name is baked
into the column names (`SBS17_ratio`, `APOBEC_raw_interval`, …). Because each tumour type carries a
different signature set, the per-tumour-type files cannot be concatenated, and a ratio between two
NON-clock signatures cannot be expressed at all.

This melts them to long — one row per (unit x interval x signature) — so that:
  * every tumour type and both levels stack into a single table (`level` = segment | cluster);
  * the clock columns ride along on every row, so `ratio_vs_clock` is precomputed for the common case;
  * ANY pair of signatures is a self-join on (level, sample, unit_id, time_interval):
        num.raw_interval / den.raw_interval
    which is the same definition as the stored `<SIG>_ratio` (raw, un-normalised, so the interval
    ratio reflects activity and not just shape). `--num/--den` writes that pairwise table directly.

Determinacy is per INTERVAL, not per segment (early gains of a high-CN state are often exact while
later ones are loose), so nothing is dropped here: `well_determined` flags rows where BOTH the clock
and the signature polytope range are under --range-cut, and downstream filters on it.

    python build_sig_ratio_tables.py                          # all tumour types -> tau_sig_ratio.tsv.gz
    python build_sig_ratio_tables.py --num SBS17 --den APOBEC  # + an explicit pairwise table
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R))
from paper.config import DATA

SRC = R / "dev/pcawg/signatures/timing_tables"
CLOCK = "clocklike"
RANGE_CUT = 0.05          # README: keep intervals where BOTH ranges are small; ~73% seg / ~60% cluster in Eso
SUFFIXES = ["raw_interval", "normalized_interval", "ratio", "range", "exposure"]

DESCRIPTORS = ["sample", "tumor_type", "segment_id", "chrom", "cn_state", "cluster_label", "route",
               "determined", "mutation_count", "cluster", "n_segments", "time_interval",
               "clocklike_cumulative_start", "clocklike_cumulative_end"]
OUT_COLS = ["sample", "tumor_type", "level", "unit_id", "chrom", "cn_state", "cluster_label",
            "route", "determined", "mutation_count", "n_segments", "time_interval",
            "clock_cum_start", "clock_cum_end", "signature", "raw_interval", "norm_interval",
            "range", "exposure", "clock_raw_interval", "clock_norm_interval", "clock_range",
            "ratio_vs_clock", "well_determined"]


def _safe_ratio(num, den):
    """num/den, but UNDEFINED (NaN) where den <= 0 — matches the `cr > 0 else np.nan` guard in
    make_interval_range_segment.py. A zero-width clock interval carries no activity information, so
    the ratio there is not infinite, it is unknown."""
    import numpy as np
    den = pd.to_numeric(den, errors="coerce")
    return pd.to_numeric(num, errors="coerce").where(den > 0) / den.where(den > 0)


def _signatures(cols):
    """Signature names present, from the `<SIG>_ratio` columns (the clock has no ratio column)."""
    return sorted({m.group(1) for c in cols if (m := re.fullmatch(r"(.+)_ratio", c))})


def _melt(path, level, range_cut):
    d = pd.read_csv(path, sep="\t")
    sigs = _signatures(d.columns)
    if not sigs:
        return None, []
    keep = [c for c in DESCRIPTORS if c in d.columns]
    out = []
    for sig in sigs:
        block = d[keep].copy()
        block["level"] = level
        block["unit_id"] = d["segment_id"] if level == "segment" else d["cluster"]
        block["signature"] = sig
        for suf, dest in [("raw_interval", "raw_interval"), ("normalized_interval", "norm_interval"),
                          ("range", "range"), ("exposure", "exposure")]:
            block[dest] = d.get(f"{sig}_{suf}")
        for suf, dest in [("raw_interval", "clock_raw_interval"),
                          ("normalized_interval", "clock_norm_interval"), ("range", "clock_range")]:
            block[dest] = d.get(f"{CLOCK}_{suf}")
        # prefer the stored ratio: the source guards `cr > 0 else NaN`, so a zero-width clock interval
        # leaves the ratio UNDEFINED rather than infinite (271/16,107 rows in Eso segment)
        block["ratio_vs_clock"] = (d[f"{sig}_ratio"] if f"{sig}_ratio" in d.columns
                                   else _safe_ratio(block["raw_interval"], block["clock_raw_interval"]))
        block["well_determined"] = (block["range"] < range_cut) & (block["clock_range"] < range_cut)
        block = block.rename(columns={"clocklike_cumulative_start": "clock_cum_start",
                                      "clocklike_cumulative_end": "clock_cum_end"})
        out.append(block)
    return pd.concat(out, ignore_index=True), sigs


def build(range_cut=RANGE_CUT, verbose=True):
    frames = []
    for level in ("segment", "cluster"):
        for f in sorted(SRC.glob(f"sig_intervals_range_*_{level}.tsv")):
            tt = re.fullmatch(rf"sig_intervals_range_(.+)_{level}\.tsv", f.name).group(1)
            df, sigs = _melt(f, level, range_cut)
            if df is None:
                continue
            frames.append(df)
            if verbose:
                print(f"  {tt:20s} {level:8s} {len(df):7,} rows  signatures={sigs}", flush=True)
    if not frames:
        raise SystemExit(f"no interval-range tables found in {SRC}")
    out = pd.concat(frames, ignore_index=True)
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[OUT_COLS]


def pairwise(long_df, num, den):
    """Ratio between ANY two signatures: self-join on the interval, num.raw / den.raw."""
    keys = ["level", "sample", "tumor_type", "unit_id", "cn_state", "time_interval",
            "clock_cum_start", "clock_cum_end"]
    def side(sig, tag):
        if sig == CLOCK:
            s = (long_df.drop_duplicates(keys)[keys + ["clock_raw_interval", "clock_range"]]
                 .rename(columns={"clock_raw_interval": f"raw_{tag}", "clock_range": f"range_{tag}"}))
            s[f"sig_{tag}"] = CLOCK
            return s
        s = long_df[long_df.signature == sig][keys + ["raw_interval", "range", "signature"]]
        return s.rename(columns={"raw_interval": f"raw_{tag}", "range": f"range_{tag}",
                                 "signature": f"sig_{tag}"})
    m = side(num, "num").merge(side(den, "den"), on=keys, how="inner")
    m["ratio"] = _safe_ratio(m["raw_num"], m["raw_den"])
    m["well_determined"] = (m["range_num"] < RANGE_CUT) & (m["range_den"] < RANGE_CUT)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range-cut", type=float, default=RANGE_CUT)
    ap.add_argument("--num", default=None, help="numerator signature for an explicit pairwise table")
    ap.add_argument("--den", default=CLOCK, help=f"denominator signature (default {CLOCK})")
    ap.add_argument("--out", default=str(DATA / "tau_sig_ratio.tsv.gz"))
    a = ap.parse_args()

    print("building long signature-ratio table:")
    long_df = build(a.range_cut)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(a.out, sep="\t", index=False, compression="gzip")
    print(f"\n-> {a.out}  {len(long_df):,} rows")
    print(long_df.groupby(["level", "signature"]).agg(
        rows=("ratio_vs_clock", "size"), samples=("sample", "nunique"),
        well_det=("well_determined", "mean")).round(3).to_string())

    if a.num:
        p = pairwise(long_df, a.num, a.den)
        out2 = Path(a.out).with_name(f"tau_sig_ratio_{a.num}_vs_{a.den}.tsv.gz")
        p.to_csv(out2, sep="\t", index=False, compression="gzip")
        print(f"\n-> {out2}  {len(p):,} rows  "
              f"(median ratio {p.loc[p.well_determined, 'ratio'].median():.3f} on well-determined)")


if __name__ == "__main__":
    main()
