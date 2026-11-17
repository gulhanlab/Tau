"""Enumerate WGD-consistent forms ("doubling blocks") of every amplification route — structurally.

A gain event splits one lineage carrying `parent_mult` final copies into two carrying `split_a` and
`split_b`. A whole-genome doubling duplicates EVERY lineage present at that instant, on BOTH alleles,
so in a route it appears as a run of events that

  1. is CONSECUTIVE — the events are simultaneous, so no other event can fall between them;
  2. splits each lineage present before the run exactly once, and none of the lineages the run itself
     creates (a child cannot re-split in the same instant);
  3. doubles each allele's LINEAGE count — major a -> 2a, minor b -> 2b.

Two things this deliberately does NOT require:

  * Balanced splits. `split_a`/`split_b` are FINAL copy counts, not lineage counts: a lineage that is
    duplicated and whose daughter later gains again ends up splitting 3 -> 1+2. (3,2) is exactly this
    — a doubling followed by one more major gain — so demanding split_a == split_b would wrongly
    exclude it.
  * That both alleles participate. An allele at copy number <= 1 cannot show a doubling without a
    subsequent loss, which the gain-only route model does not represent. (4,0) is a doubling of (2,0);
    (4,1) can carry a WGD in the major allele with a minor copy lost afterwards. So an allele with
    total copy number <= 1 is exempt, and the block is judged on the alleles that could show it.

The output is a purely structural property of the route — it never consults an inferred WGD time, which
is the whole point: constrain SIMULTANEITY and let the time fall out of the data.

Usage:  python dev/pcawg/doubling_blocks.py
"""
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

R = Path(os.environ.get("TAU_REPO", ".")); sys.path.insert(0, str(R))
from paper.data_tables.tables import load_route_topology

MAX_LOSS = 1      # only INFERRED loss is charged; loss 2 measured as mostly wall artefacts


def _events(df_route):
    return [(r.allele, int(r.parent_mult), int(r.split_a), int(r.split_b))
            for r in df_route.sort_values("event_index").itertuples()]


def doubling_blocks(events, major_cn, minor_cn, max_loss=None):
    """All consecutive runs that could be a whole-genome doubling, with the loss each one implies.

    A doubling duplicates every lineage present at that instant. If a daughter leaves no surviving
    copies, the route contains NO split for its parent — the lineage simply continues — so a doubling
    followed by loss appears as a run that splits only SOME of the lineages present. The un-split ones
    are the implied losses:

        implied_loss = (a - major lineages split) + (b - minor lineages split)

    implied_loss == 0 is a doubling visible in full. Larger values are doublings partly erased by
    later loss, which the gain-only route cannot otherwise express — e.g. (3,2) admits

        loss 0 : (1,1) -> WGD -> (2,2) -> +1 major -> (3,2)
        loss 1 : (1,1) -> +1 major -> (2,1) -> WGD -> (4,2) -> -1 major -> (3,2)

    Both are valid histories; only the first keeps every doubled lineage. Ranking by implied_loss is
    the parsimony ordering — it does NOT rule the others out, and nothing in a segment's own data
    distinguishes them, so treat loss > 0 as an assumption rather than a finding.

    Returns [(start, length, implied_loss), ...] with start 1-based.
    """
    blocks = []
    n = len(events)
    for i in range(n):
        state = {"major": Counter({major_cn: 1}) if major_cn > 0 else Counter(),
                 "minor": Counter({minor_cn: 1}) if minor_cn > 0 else Counter()}
        ok = True
        for (al, p, a, b) in events[:i]:
            if state[al][p] == 0:
                ok = False; break
            state[al][p] -= 1; state[al][a] += 1; state[al][b] += 1
        if not ok:
            continue
        have = {"major": sum(state["major"].values()), "minor": sum(state["minor"].values())}
        # minor_cn == 0 is an OBSERVED loss: the allele is gone before the doubling and contributes
        # no lineages, so its absence is data rather than an assumption and is not charged.
        if minor_cn == 0:
            have["minor"] = 0
        avail = {al: Counter(c) for al, c in state.items()}
        used = {"major": 0, "minor": 0}
        for k in range(1, n - i + 1):                    # grow the run one event at a time
            al, p, a, b = events[i + k - 1]
            if avail[al][p] == 0:                        # would re-split a lineage the run just made
                break
            avail[al][p] -= 1; used[al] += 1
            if used["major"] > have["major"] or used["minor"] > have["minor"]:
                break
            if k < 2:                                    # a 1-event run pins no interval -> no constraint
                continue
            loss = (have["major"] - used["major"]) + (have["minor"] - used["minor"])
            if max_loss is None or loss <= max_loss:
                blocks.append((i + 1, k, loss))
    return blocks


def main():
    d = load_route_topology()
    rows = []
    for key, sub in d.groupby("route_key"):
        maj, mino = int(sub.major_cn.iloc[0]), int(sub.minor_cn.iloc[0])
        ev = _events(sub)
        bl = doubling_blocks(ev, maj, mino, max_loss=MAX_LOSS)
        rows.append(dict(route_key=key, major=maj, minor=mino, n_events=len(ev),
                         n_blocks=len(bl), min_loss=(min((b[2] for b in bl), default=-1)),
                         blocks=";".join(f"{s}+{L}/{ls}" for s, L, ls in bl)))
    t = pd.DataFrame(rows)
    out = R / "dev/pcawg/doubling_blocks.tsv"
    t.to_csv(out, sep="\t", index=False)

    print(f"routes: {len(t):,}   admitting >=1 doubling block: {(t.n_blocks > 0).sum():,} "
          f"({(t.n_blocks > 0).mean():.1%})\n")
    st = (t.groupby(["major", "minor"])
            .agg(routes=("n_blocks", "size"), with_block=("n_blocks", lambda x: (x > 0).sum()))
            .reset_index())
    st["frac"] = (st.with_block / st.routes).round(3)
    piv = st.pivot(index="minor", columns="major", values="frac")
    print("fraction of routes admitting a doubling block, by CN state:")
    print(piv.fillna(-1).round(2).replace(-1, "").to_string())
    print(f"\n-> {out}")
    for k in ["2_2.1", "4_2.1", "4_2.4", "4_4.1", "3_1.1", "6_3.1"]:
        r = t[t.route_key == k]
        if len(r):
            print(f"   {k:8s} blocks={r.blocks.iloc[0] or '(none)'}")


if __name__ == "__main__":
    main()
