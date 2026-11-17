#!/usr/bin/env python3
"""Per-segment loss attributed to EACH doubling, for samples with any number of WGDs.

Why this exists
---------------
cohort_loss_compute.py handles double-WGD samples by taking W1 = first and W2 = LAST WGD time, so in a
sample with three doublings the middle one is silently dropped and its loss is folded into the other
two. That is fine for a two-way split but makes a per-doubling stratification impossible. This script
generalises the same recursion to k doublings.

The recursion, unchanged in kind
--------------------------------
Events are tagged to the nearest WGD within ASSIGN_R, else "focal". Then, walking the doublings in time
order and carrying the copy budget forward:

    cb_1 = 2 + (focal gains before W_1)
    L_1  = max(0, cb_1 - d_1)                       d_k = events tagged to doubling k
    cb_k = cb_{k-1} + d_{k-1} + (focal gains between W_{k-1} and W_k)
    L_k  = max(0, cb_k - d_k)

cb_k is doubling k's loss OPPORTUNITY: the copies present when it fired, all of which it should have
duplicated. L_k is how many of them are unaccounted for. Both are emitted per segment so the ratio can
be formed with matching numerator and denominator at any level of aggregation.

Consecutive doublings must be at least MIN_SEP apart in molecular time; below that they are not
separable and the sample is skipped rather than split on noise.

Usage:
    python dev/pcawg/loss_catalog/compute_multi_wgd_loss.py --types Ovary.AdenoCA
    python dev/pcawg/loss_catalog/compute_multi_wgd_loss.py            # every WGD sample
"""
import os
import argparse
import glob
import gzip
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(os.environ.get("TAU_REPO", "."))
sys.path.insert(0, str(BASE / "dev/pcawg/FigWGDPGD"))
sys.path.insert(0, str(BASE))
from wgd_split_filter import corrected_wgd_multiplicity          # noqa: E402
from donor_dedup import representative_samples                    # noqa: E402
from tau.trees import _pick_draw_for_tree                         # noqa: E402

ROOT = BASE / "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
OUT = BASE / "dev/pcawg/loss_catalog"
ASSIGN_R = 0.20        # a gain within this of a WGD time belongs to it; matches the other loss scripts
MIN_SEP = 0.15         # consecutive doublings closer than this are not separable
CHROMS = [str(i) for i in range(1, 23)]


def losses_multi(ev_times, Ws):
    """-> (L, cb) per doubling, as lists of length len(Ws)."""
    ev = sorted(float(x) for x in ev_times)
    tags = []
    for t in ev:
        d = [abs(t - w) for w in Ws]
        k = int(np.argmin(d))
        tags.append(k if d[k] <= ASSIGN_R else -1)          # -1 = focal
    L, CB = [], []
    cb = 2.0
    for k, w in enumerate(Ws):
        lo = Ws[k - 1] if k else -np.inf
        cb += sum(1 for t, g in zip(ev, tags) if g == -1 and lo < t < w)
        d = tags.count(k)
        loss = max(0.0, cb - d)
        L.append(loss)
        CB.append(cb)
        # PARSIMONIOUS carry-forward: the copies this doubling made, PLUS the ones that survived it —
        # which is (cb - loss) = d whenever anything was lost. The lineages just charged as lost are
        # gone, so they must not sit in the next doubling's budget and be charged a second time.
        # Old rule `cb += d` kept them, so e.g. a (3,0) booked its absent minor allele at every
        # doubling instead of once. It also forced cb_k = cb_{k-1} + d_{k-1}, the identity that made
        # the per-doubling split L1 vs L2 unidentifiable.
        cb = (cb - loss) + d                                # == 2*d when loss > 0
    return L, CB


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    ev = pd.read_csv(BASE / "dev/pcawg/FigWGDPGD/events_allpcawg.tsv", sep="\t")
    corr = corrected_wgd_multiplicity(ev)
    reps = representative_samples()
    tt = ev.drop_duplicates("sample").set_index("sample")["tumor_type"]
    wgd = {s: sorted(float(x) for x in g.time.values)
           for s, g in ev[ev.classification == "WGD"].groupby("sample")}
    gmap = {Path(f).name.replace(".genome.pkl.gz", ""): f
            for f in glob.glob(str(ROOT / "*" / "*" / "*.genome.pkl.gz"))}

    todo = [s for s, k in corr.items() if k >= 1 and s in reps and s in gmap and s in wgd]
    if a.types:
        todo = [s for s in todo if tt.get(s) in a.types]
    todo = sorted(todo)
    print(f"{len(todo)} WGD samples" + (f" in {a.types}" if a.types else ""), flush=True)

    rows, skipped = [], 0
    for n, s in enumerate(todo, 1):
        Ws = wgd[s]
        if len(Ws) > 1 and min(np.diff(Ws)) < MIN_SEP:
            skipped += 1
            continue
        try:
            with gzip.open(gmap[s], "rb") as fh:
                g = pickle.load(fh)
        except Exception:
            skipped += 1
            continue
        for seg in g.segments:
            if not getattr(seg, "timing_result", None) or seg.N_counts is None:
                continue
            if str(getattr(seg, "chrom", "")) not in CHROMS:
                continue
            et, _ = _pick_draw_for_tree(seg, Ws)
            if et is None or len(et) == 0:
                continue
            L, CB = losses_multi(et, Ws)
            base = dict(sample=s, tumor_type=tt.get(s, "?"), n_wgd=len(Ws),
                        chrom=str(seg.chrom), start=int(seg.start), end=int(seg.end),
                        length=int(seg.end) - int(seg.start))
            for k, (lk, ck) in enumerate(zip(L, CB), start=1):
                rows.append({**base, "wgd_index": k, "W": Ws[k - 1], "L": lk, "cb": ck})
        del g
        if n % 25 == 0:
            print(f"  …{n}/{len(todo)}, {len(rows):,} rows", flush=True)

    d = pd.DataFrame(rows)
    out = Path(a.out) if a.out else OUT / "multi_wgd_loss_segments.tsv"
    d.to_csv(out, sep="\t", index=False)
    print(f"\n{len(d):,} (segment, doubling) rows from {d['sample'].nunique()} samples "
          f"({skipped} skipped: doublings closer than {MIN_SEP} or unreadable)")
    print(d.groupby("wgd_index").agg(samples=("sample", "nunique"), segs=("length", "size"),
                                     mean_L=("L", "mean"), mean_cb=("cb", "mean")).round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
