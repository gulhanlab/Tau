#!/usr/bin/env python3
"""Run every panel script under main/ and supp/ (each in its own process, from a neutral cwd) and report
which succeed. The slow object-based ones (examples a/b, signatures a) can be skipped with --fast.
--check runs nothing: it lists every manuscript panel in manifest.py and whether its file exists.
--alt also runs the alt_*.py alternatives, which are not manuscript panels.
Usage: python paper/figures/run_all.py [--fast] [--alt] [--only <substring>] | --check"""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

if "--check" in sys.argv:
    sys.path.insert(0, str(HERE))
    import manifest
    missing = 0
    for group, figs in (("Fig.", manifest.MAIN), ("Supp. Fig.", manifest.SUPP)):
        for fig in figs:
            for letter, src, note in fig["panels"]:
                if src is None:
                    print(f"  --      {group} {fig['number']}{letter:4s} (no script: {note})"); continue
                have = [e for e in ("svg", "png", "pdf") if (HERE / f"{src}.{e}").exists()]
                missing += not have
                print(f"  {'ok ' if have else 'MISSING'} {group} {fig['number']}{letter:4s} {src}  [{', '.join(have)}]")
    print(f"\n{missing} panel(s) missing")
    sys.exit(1 if missing else 0)
SLOW = {"main/examples/a_example1.py", "main/examples/b_example2.py",
        "main/signatures/a_sbs17_overview_trees.py"}
fast = "--fast" in sys.argv
lint = "--lint" in sys.argv          # print text overlaps found at save time (via _style's savefig hook)
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
alt = "--alt" in sys.argv               # also run the alt_*.py alternatives (not manuscript panels)
scripts = sorted(p for p in list(HERE.glob("main/*/*.py")) + list(HERE.glob("supp/*/*.py"))
                 if not p.name.startswith("_") and p.name != "ASSEMBLY.py"
                 and (alt or not p.name.startswith("alt_")))
ok, bad = [], []
for p in scripts:
    rel = str(p.relative_to(HERE))
    if (fast and rel in SLOW) or (only and only not in rel):
        continue
    t0 = time.time()
    import os
    env = dict(os.environ, TAU_FIG_LINT="1") if lint else None
    r = subprocess.run([sys.executable, str(p)], cwd="/", capture_output=True, text=True, env=env)
    dt = time.time() - t0
    (ok if r.returncode == 0 else bad).append(rel)
    print(f"{'ok ' if r.returncode == 0 else 'FAIL'}  {dt:6.1f}s  {rel}", flush=True)
    if r.returncode:
        print("      " + (r.stderr.strip().splitlines() or ["?"])[-1][:200])
    if lint:
        for line in r.stdout.splitlines():
            if "[lint]" in line:
                print("    " + line.strip())
print(f"\n{len(ok)} ok, {len(bad)} failed" + (": " + ", ".join(bad) if bad else ""))
