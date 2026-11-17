"""Emit the FULL standard Tau table set for every active non-clock signature in a sample.

A signature run is not a special kind of output: `tau.sig_timing.retimed_genome` returns a genuine
timed `Genome` (mut_w swapped for the signature's per-SNV posterior, EM and timing re-run on the
untouched multiplicity likelihoods), so the ordinary emitter reshapes it into exactly the same 11
tables as the clock run. Every table therefore carries the same columns plus a `signature` column,
and the clock and signature tables concatenate directly.

Two deliberate asymmetries, both so that signatures stay comparable:

  * MULTIPLICITY-CLUSTER MEMBERSHIP IS PINNED TO THE CLOCK RUN. `seg_cluster_df` is the clock run's
    stored segment_clusters.tsv, so `merged_4_2_0` pools the identical segments under every
    signature. Re-clustering per signature would move the unit boundaries and there would be
    nothing to compare. This is also what makes sparse signatures timeable at all: pooled ESS runs
    in the hundreds-to-thousands where a raw segment often fails effN_cut.
  * TIME CLUSTERING IS *NOT* PINNED. `ctb_raw` is left to recompute, so tau_events answers "when
    was WGD according to this signature's clock" rather than restating the SBS1+SBS5 answer.

Usage:
    python emit_signature_tables.py <sample_dir> [<sample_dir> ...] [--thresh 0.10] [--out DIR]

Writes {sample}.{SIG}.tau_*.tsv[.gz] beside the pkls (or into --out).
"""
import argparse
import gzip
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R))
from tau import timing
from tau.aggregate.export import build_all_tables, TABLE_FILENAMES
from tau.sig_timing import active_signatures, sample_exposure_estimate, retimed_genome

SOL = R / "tau/data/7_5_solutions_updated.sobj"
MAT = R / "tau/data/matrices_7_7.h5"
THRESH = 0.10          # a signature group is emitted when its exposure share exceeds this


def _load(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def emit_sample(sample_dir, env, thresh=THRESH, out_dir=None, verbose=True):
    sample_dir = Path(sample_dir)
    sample = sample_dir.name
    out_dir = Path(out_dir) if out_dir else sample_dir
    tumor_type = sample_dir.parent.name

    raw_pkl = sample_dir / f"{sample}.genome.pkl.gz"
    cl_pkl = sample_dir / f"{sample}.clustered_genome.pkl.gz"
    sc_tsv = sample_dir / f"{sample}.segment_clusters.tsv"
    if not raw_pkl.exists():
        return {"sample": sample, "status": "no_genome_pkl", "signatures": []}

    g = _load(raw_pkl)
    cg = _load(cl_pkl) if cl_pkl.exists() else None
    seg_cluster_df = pd.read_csv(sc_tsv, sep="\t") if sc_tsv.exists() else None

    expo = sample_exposure_estimate(g)
    specs = [s for s in active_signatures(g, thresh, expo) if s["name"] != "clock"]
    if verbose:
        print(f"{sample}: {[(s['name'], round(s['share'], 3)) for s in specs]}", flush=True)

    done = []
    for spec in specs:
        t0 = time.time()
        gs = retimed_genome(g, spec["members"], env, expo)
        cgs = retimed_genome(cg, spec["members"], env, expo) if cg is not None else None
        tables = build_all_tables(gs, sample, tumor_type,
                                  seg_cluster_df=seg_cluster_df,   # clock's pooling, held fixed
                                  clustered_genome=cgs)            # ctb_raw=None -> signature's own events
        for name, df in tables.items():
            if df is None or not len(df):
                continue
            df.insert(1, "signature", spec["name"])
            fname = TABLE_FILENAMES.get(name)
            if fname is None:
                continue
            path = Path(out_dir) / f"{sample}.{spec['name']}.{fname}"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, sep="\t", index=False,
                      compression=("gzip" if fname.endswith(".gz") else None))
        def _n(k):                                   # tables values are DataFrames or None
            v = tables.get(k)
            return 0 if v is None else int(len(v))
        done.append({"signature": spec["name"], "share": round(spec["share"], 4),
                     "secs": round(time.time() - t0, 1),
                     "n_seg": _n("segments"), "n_merged": _n("merged_segments")})
        if verbose:
            print(f"  {spec['name']:12s} share={spec['share']:.3f}  {done[-1]['secs']:5.1f}s  "
                  f"segments={done[-1]['n_seg']}  merged={done[-1]['n_merged']}", flush=True)
    return {"sample": sample, "status": "ok", "signatures": done}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dirs", nargs="+")
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    env = timing.load_routes(str(SOL), str(MAT))       # ~50 s, once per worker
    for d in a.sample_dirs:
        try:
            emit_sample(d, env, a.thresh, a.out)
        except Exception as e:                          # one bad sample must not kill the array task
            print(f"FAIL {Path(d).name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
