#!/usr/bin/env python3
"""The data bundle: every data file the panel scripts read (listed in DATA_BUNDLE.tsv).

    python paper/data_bundle.py check              list the files and whether each is present
    python paper/data_bundle.py pack bundle.tar    write the bundle (paths relative to the repository root)

DATA_BUNDLE.tsv was produced by running every panel script under a file-open audit, so it is the exact
set of inputs of `figures/run_all.py`; nothing else is needed to regenerate the panels.
"""
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIST = Path(__file__).resolve().parent / "DATA_BUNDLE.tsv"


def entries():
    for line in LIST.read_text().splitlines()[1:]:
        path, size = line.split("\t")[:2]
        yield path, int(size)


def main(argv):
    if len(argv) < 2 or argv[1] not in ("check", "pack") or (argv[1] == "pack" and len(argv) < 3):
        sys.exit(__doc__)
    rows = list(entries())
    missing = [p for p, _ in rows if not (REPO / p).exists()]
    if argv[1] == "check":
        for p, size in rows:
            print(f"{'ok     ' if (REPO / p).exists() else 'MISSING'} {size/1e6:9.2f} MB  {p}")
        print(f"\n{len(rows)} files, {sum(s for _, s in rows)/1e6:.0f} MB; {len(missing)} missing")
        sys.exit(1 if missing else 0)
    if missing:
        sys.exit("cannot pack, missing:\n  " + "\n  ".join(missing))
    with tarfile.open(argv[2], "w") as tar:
        for p, _ in rows:
            tar.add(REPO / p, arcname=p)          # follows nothing: the listed files are regular files
    print(f"wrote {argv[2]} ({len(rows)} files)")


if __name__ == "__main__":
    main(sys.argv)
