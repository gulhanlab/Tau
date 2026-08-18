#!/usr/bin/env python3
"""
Split the monolithic 7_5_solutions_updated.sobj into per-CN-state .sobj files
stored in tau/data/solutions/<major>_<minor>.sobj.

Run once:
    PATH=".pixi/envs/default/bin:$PATH" python tau/data/split_solutions.py
"""
import os, sys
from pathlib import Path

os.environ["PATH"] = str(Path(__file__).parent.parent.parent /
                         ".pixi/envs/default/bin") + ":" + os.environ.get("PATH", "")

from sage.all import load as sage_load, save as sage_save

SRC  = Path(__file__).parent / "7_5_solutions_updated.sobj"
DEST = Path(__file__).parent / "solutions"
DEST.mkdir(exist_ok=True)

print(f"Loading {SRC} …")
SOL = sage_load(str(SRC))
print(f"  {len(SOL):,} route keys")

# Group by CN state  (key format: "major_minor.variant")
from collections import defaultdict
by_state = defaultdict(dict)
for k, v in SOL.items():
    state = str(k).rsplit(".", 1)[0]   # "5_1" from "5_1.2"
    by_state[state][k] = v

print(f"  {len(by_state):,} distinct CN states")

for state, routes in sorted(by_state.items()):
    out = DEST / f"{state}.sobj"
    if out.exists():
        continue
    sage_save(routes, str(out))
    print(f"  saved {state:12s}  ({len(routes)} routes)  → {out.name}")

print("Done.")
