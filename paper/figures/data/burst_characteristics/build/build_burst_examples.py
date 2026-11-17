"""Exemplar high-CN segments for the burst/gradual gallery -> paper/figures/data/burst_characteristics/burst_examples.tsv.

Picks segments to illustrate each tier of the burst taxonomy and ports their full polytope (every
solution's gain times) into the figure cache, so the panel needs no dev/ reach-through and no pickles.

Selection is deliberate about what an exemplar has to be. Each tier takes the best-powered segments
(highest mutation count), because a burst read off a low-count segment is not evidence of anything,
subject to two diversity rules: one tumour type per tier, and at most two segments of the same
copy-number state per tier. Without the second rule the gradual tier comes back as four (3,2)s — the
commonest high-CN state — and the row would illustrate a copy-number state rather than a timing
pattern. Beyond that the choice is by mutation count alone; no eyeballing for the tightest or prettiest
trajectory, which would make the gallery a picture of the selection rather than of the data.

Tiers, in the paper's order (first matching rule wins, see CLAUDE.md):
  complete_burst    every solution puts all gains in a narrow window
  incomplete_burst  every solution bursts a subset, with a real gap to the rest
  burst_compatible  at least one solution admits a burst — permissive, NOT a confident burst
  gradual           no solution admits one

Usage:  python paper/figures/data/burst_characteristics/build/build_burst_examples.py [n_per_tier]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
TAXO = ANALYSIS / "pcawg/oncogenes/figures_genome_highcn_catalogue/wgd_burst_taxonomy_polytope.tsv"
TIMING = ANALYSIS / "pcawg/misc_data_and_plots/genome_highcn_timing_cache.tsv"
OUT = cache_dir_of(__file__); OUT.mkdir(parents=True, exist_ok=True)
TIERS = ["complete_burst", "incomplete_burst", "burst_compatible", "gradual"]
SEG = ["sample", "chrom", "start", "end"]
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MIN_EVENTS = 3          # a burst is defined on 3+ gains, so 2-gain segments cannot illustrate one
MAX_PER_STATE = 2       # per tier
EXCLUDE_TYPES = {"Unknown"}   # not a tissue — cannot carry a tumour-type label in the gallery


def main():
    tx = pd.read_csv(TAXO, sep="\t", dtype={"chrom": str})
    tm = pd.read_csv(TIMING, sep="\t", dtype={"chrom": str})
    mut = tm.groupby(SEG).mutation_count.first().rename("mut")
    tx = tx.merge(mut, on=SEG, how="left")
    tx = tx[(tx.n_events >= MIN_EVENTS) & tx.mut.notna()
            & ~tx.tumor_type.isin(EXCLUDE_TYPES)]

    picks = []
    for tier in TIERS:
        cand = tx[tx.category == tier].sort_values("mut", ascending=False)
        seen_t, seen_s, keep = set(), {}, []
        for r in cand.itertuples():
            st = (r.major_cn, r.minor_cn)
            if r.tumor_type in seen_t or seen_s.get(st, 0) >= MAX_PER_STATE:
                continue
            seen_t.add(r.tumor_type); seen_s[st] = seen_s.get(st, 0) + 1
            keep.append(r.Index)
            if len(keep) == N_PER:
                break
        s = cand.loc[keep]
        print(f"{tier:18s} {len(s)} picked  "
              f"(mut {int(s.mut.min())}-{int(s.mut.max())}, "
              f"{', '.join(f'{r.tumor_type} ({r.major_cn},{r.minor_cn})' for r in s.itertuples())})")
        picks.append(s)
    p = pd.concat(picks, ignore_index=True)

    d = tm.merge(p[SEG + ["category", "max_B", "has_wgd", "wgd_time", "D"]], on=SEG, how="inner")
    d.to_csv(OUT / "burst_examples.tsv", sep="\t", index=False)
    print(f"\n{len(p)} exemplars, {len(d):,} solution rows -> {OUT/'burst_examples.tsv'}")


if __name__ == "__main__":
    main()
