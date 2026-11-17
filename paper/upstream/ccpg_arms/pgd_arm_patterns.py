#!/usr/bin/env python3
"""PGD chromosome-ARM composition + recurrent arm patterns (new classification).

Same as pgd_chrom_patterns.py but at chromosome-ARM (p/q) resolution: each gained segment is
labelled by arm via tau.clustering.assign_chrom_arm (centromere-spanning segments count as both
arms). Recurrent exact arm-sets, recurrent arm-pairs, per-arm involvement → plots.

    PATH=".pixi/envs/default/bin:$PATH" .pixi/envs/default/bin/python \
        dev/pcawg/FigWGDPGD/pgd_arm_patterns.py
"""
import os
import glob, warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd

warnings.simplefilter("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
import sys
sys.path.insert(0, os.environ.get("TAU_REPO", "."))
from tau.clustering import assign_chrom_arm

BASE = Path(os.environ.get("TAU_REPO", "."))
DATA = BASE / "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg"
PLOTS = BASE / "notes_site/docs/plots"
AUTO = [str(i) for i in range(1, 23)]
MATCH_TOL = 0.15


def classify(pct, n):
    if pct >= 40 or n >= 10:
        return "WGD"
    return "PGD" if n >= 2 else "chrom_specific"


def arms_of(chrom, start, end):
    a = assign_chrom_arm(str(chrom), int(start), int(end))
    if a == "p":
        return [f"{chrom}p"]
    if a == "q":
        return [f"{chrom}q"]
    if a == "centromere":          # spans centromere → both arms
        return [f"{chrom}p", f"{chrom}q"]
    return []                       # missing / non-autosome


def extract():
    files = glob.glob(str(DATA / "*" / "*" / "*.time_clusters.tsv"))
    print(f"{len(files)} samples", flush=True)
    out = []
    for i, tcf in enumerate(files):
        if i % 300 == 0:
            print(f"  {i}/{len(files)}", flush=True)
        p = Path(tcf); sample, tt = p.parent.name, p.parent.parent.name
        try:
            tc = pd.read_csv(tcf, sep="\t")
        except Exception:
            continue
        if tc.empty or "n_chroms" not in tc.columns:
            continue
        pct = tc["pct_of_theoretical_genome"] if "pct_of_theoretical_genome" in tc else tc["gf"] * 100
        tc = tc.assign(nc=[classify(float(a), int(b)) for a, b in zip(pct, tc["n_chroms"])])
        ev = tc[tc.nc.isin(["WGD", "PGD"])]
        pgd = ev[ev.nc == "PGD"]
        if pgd.empty:
            continue
        stf = p.parent / f"{sample}.segment_times.tsv"
        if not stf.exists():
            continue
        try:
            st = pd.read_csv(stf, sep="\t",
                             usecols=["bootstrap_id", "chrom", "start", "end", "segment_id",
                                      "time_point", "time_fraction"])
        except Exception:
            continue
        st = st[st.bootstrap_id == 0]
        g = (st.groupby(["segment_id", "chrom", "start", "end", "time_point"])["time_fraction"]
             .median().reset_index())
        et = ev["time"].to_numpy(float); eid = ev["cluster_id"].to_numpy()
        members = {c: set() for c in ev["cluster_id"]}
        for _, r in g.iterrows():
            ch = str(r["chrom"]).replace("chr", "")
            if ch not in AUTO:
                continue
            d = np.abs(et - float(r["time_fraction"])); j = int(np.argmin(d))
            if d[j] <= MATCH_TOL:
                for arm in arms_of(ch, r["start"], r["end"]):
                    members[eid[j]].add(arm)
        for _, r in pgd.iterrows():
            arms = sorted(members.get(r["cluster_id"], set()),
                          key=lambda x: (int(x[:-1]), x[-1]))
            if len(arms) >= 2:
                out.append(dict(sample=sample, tumor_type=tt, time=float(r["time"]),
                                n_reported=int(r["n_chroms"]), arms=tuple(arms)))
    return pd.DataFrame(out)


def main():
    df = extract()
    df.to_csv(BASE / "dev/pcawg/FigWGDPGD/pgd_arm_membership.tsv", sep="\t", index=False)
    n = len(df)
    print(f"\n{n} PGD events with >=2 recovered arms (median {df.arms.apply(len).median():.0f} arms)")

    inv = Counter()
    for a in df.arms:
        inv.update(a)
    print("\nTop arms by involvement:")
    for a, c in inv.most_common(16):
        print(f"  {c:4d} ({c/n:.0%})  {a}")

    sc = Counter(df.arms)
    print("\nTop recurring EXACT arm-sets:")
    for s, c in sc.most_common(12):
        if c < 2:
            break
        print(f"  {c:2d}  {'+'.join(s)}   {dict(df[df.arms==s].tumor_type.value_counts())}")

    pc = Counter()
    for a in df.arms:
        for x, y in combinations(a, 2):
            pc[(x, y)] += 1
    print("\nTop recurring ARM pairs:")
    for (x, y), c in pc.most_common(18):
        print(f"  {c:3d}  {x}+{y}")

    # --- plots: per-arm involvement (ordered p then q) + top recurring arm-sets ---
    order = [f"{c}{a}" for c in AUTO for a in "pq"]
    order = [a for a in order if inv.get(a, 0) > 0]
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.bar(order, [inv[a] / n * 100 for a in order], color="#cd9f2c")
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_ylabel("% of PGD events"); ax.set_title(f"Per-arm involvement in PGDs (n={n})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(PLOTS / "pgd_arm_involvement.png", dpi=160, bbox_inches="tight")

    LAB = {"CNS.GBM": "CNS GBM", "CNS.Medullo": "Medulloblastoma", "Liver.HCC": "Liver",
           "Kidney.RCC": "Kidney", "Panc.AdenoCA": "Pancreatic", "Breast.AdenoCA": "Breast",
           "Lymph.BNHL": "Lymphoma", "Head.SCC": "Head/neck", "Stomach.AdenoCA": "Stomach",
           "ColoRect.AdenoCA": "Colorectal", "Eso.AdenoCA": "Esophageal", "Ovary.AdenoCA": "Ovarian"}
    top = [t for t, _ in df.tumor_type.value_counts().items() if t in LAB][:12]
    cols = cm.tab20(np.linspace(0, 1, len(top))); tcol = {t: cols[i] for i, t in enumerate(top)}
    items = [(s, c) for s, c in sc.most_common(30) if c >= 2]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.42 * len(items))))
    for yi, (s, c) in enumerate(items[::-1]):
        sub = df[df.arms == s]; left = 0
        for t, cnt in sub.tumor_type.value_counts().items():
            ax.barh(yi, cnt, left=left, color=tcol.get(t, (0.8, 0.8, 0.8, 1)), edgecolor="white", lw=0.4)
            left += cnt
    ax.set_yticks(range(len(items))); ax.set_yticklabels(["+".join(s) for s, _ in items[::-1]], fontsize=7)
    ax.set_xlabel("# PGD events"); ax.set_title("Recurring chromosome-ARM sets in PGDs (≥2 events)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(color=tcol[t], label=LAB[t]) for t in top], fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(PLOTS / "pgd_arm_sets.png", dpi=160, bbox_inches="tight")
    print("\nsaved arm plots")


if __name__ == "__main__":
    main()
