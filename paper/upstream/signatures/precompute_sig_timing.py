#!/usr/bin/env python
"""Precompute the per-signature timing cache for the interactive viewer.

Thin CLI wrapper around ``tau.sig_timing.build_sig_payload`` (the canonical builder
now lives in the package so ``tau run --include-signatures`` and the viewer scripts
share one implementation). For a sample's genome.pkl.gz: detect active signatures
(clock + exposure share > thresh), re-time each non-clock signature once, and write
``<sample>.sig_timing.json.gz`` next to the genome.

Usage: precompute_sig_timing.py <genome.pkl.gz> [thresh=0.15]
"""
import sys, gzip, json, pickle

from tau import timing
import tau.cli as cli
# re-export so existing callers (`from precompute_sig_timing import build_sig_payload`)
# and the constants keep working.
from tau.sig_timing import build_sig_payload, clock_table, SPREAD_CUT, MIN_CLOCK_T  # noqa: F401


def load(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def main():
    gpath = sys.argv[1]
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    sample = gpath.split("/")[-1].replace(".genome.pkl.gz", "")
    g = load(gpath)
    env = timing.load_routes(cli.DEFAULT_ROUTES_SAGE, cli.DEFAULT_MATRICES_H5)
    payload = build_sig_payload(g, env, thresh)
    payload["sample"] = sample
    outp = gpath.replace(".genome.pkl.gz", ".sig_timing.json.gz")
    with gzip.open(outp, "wt") as f:
        json.dump(payload, f)
    print(f"WROTE {outp}", flush=True)


if __name__ == "__main__":
    main()
