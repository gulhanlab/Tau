"""Fig. 3 cache: stage the one DERIVED input that isn't fully reconstructable from the Tier-1 tables.

Panel E (PGD arm programmes) needs, per PGD event, the recovered autosome-ARM set + the event time.
That arm-level membership is a derived reshape (segment arm assignment) not carried in tau_events, so we
copy the existing dev table `dev/pcawg/FigWGDPGD/pgd_arm_membership.tsv` into figures/data/<label>/ and the
panel reads it there (treated like a Tier-2 fig cache). All other panels (a-d) read the composite tables
directly via the `_wgdpgd_common` shim and need no cache.

Run: python paper/figures/data/punctuated/build.py
"""
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401

SRC = ANALYSIS / "pcawg/FigWGDPGD/pgd_arm_membership.tsv"
OUT = Path(__file__).resolve().parent


def main():
    dst = OUT / "pgd_arm_membership.tsv"
    shutil.copyfile(SRC, dst)
    n = sum(1 for _ in open(dst)) - 1
    print(f"copied pgd_arm_membership.tsv -> {dst}  ({n} PGD events)")


if __name__ == "__main__":
    main()
