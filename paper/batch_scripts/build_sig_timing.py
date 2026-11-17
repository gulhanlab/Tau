"""Build the tau_sig_timing base table from the existing per-signature cumulative caches.

Each dev/pcawg/signatures/sig_cumulative_cache_<SIG>.tsv holds, per (sample, segment, gain
event = time_point), the cumulative signature-vs-clock activity for one signature:
  G_clock / G_sig  = cumulative activity (mutation share) of the clock / this signature up to the event
  ts_clock / ts_sig = total-sum baselines used to normalise (expected activity if flat over molecular time)
  ess              = effective sample size (weighted SNV count) on the segment
  frac             = fractional activity of the signature on the segment

This just stacks them into one tidy table with a `signature` column — no re-timing. Signatures
already computed (SBS17, SBS18, SBS7, SBS4, SBS3, SBS2+13) come for free; new signatures need a
`tau run --signatures` rerun.  Output: paper/data_tables/tau_sig_timing.tsv.gz
"""
import sys
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R))
from paper.config import DATA

SIG_DIR = R / "dev/pcawg/signatures"
OUT = DATA / "tau_sig_timing.tsv.gz"

# filename suffix -> signature label. `ts` is the SBS17 cache (different column names).
FILES = {
    "SBS18": "SBS18", "SBS7": "SBS7", "SBS4": "SBS4", "SBS3": "SBS3",
    "SBS2_SBS13": "SBS2+SBS13", "ts": "SBS17",
}
# keys renamed to match the other base tables: segment_id -> seg_id, time_point -> gain_index
COLS = ["sample", "tumor_type", "seg_id", "signature", "gain_index",
        "ess", "G_clock", "G_sig", "ts_clock", "ts_sig", "frac"]


def _load(suffix, sig):
    f = SIG_DIR / f"sig_cumulative_cache_{suffix}.tsv"
    if not f.exists():
        print(f"  [skip] {f.name} missing"); return None
    d = pd.read_csv(f, sep="\t")
    if "G_sbs17" in d.columns:        # the SBS17 (`ts`) cache uses *_sbs17 names
        d = d.rename(columns={"G_sbs17": "G_sig", "ts_sbs17": "ts_sig"})
    d = d.rename(columns={"segment_id": "seg_id", "time_point": "gain_index"})
    d["signature"] = sig
    if "frac" not in d.columns:
        d["frac"] = pd.NA
    return d.reindex(columns=COLS)


def main():
    parts = [d for suffix, sig in FILES.items() if (d := _load(suffix, sig)) is not None]
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["signature", "sample", "seg_id", "gain_index"]).reset_index(drop=True)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"tau_sig_timing: {len(out):,} rows  x  {out.signature.nunique()} signatures  "
          f"x  {out['sample'].nunique():,} samples  ->  {OUT.name}")
    print(out.groupby("signature").size().to_string())


if __name__ == "__main__":
    main()
