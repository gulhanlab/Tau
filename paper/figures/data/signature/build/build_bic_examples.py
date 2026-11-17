"""Worked stepwise-model fits -> paper/figures/data/signature/bic_examples_{fits,windows,trace}.tsv.

Three examples of what the BIC search selects: a sample with no supported transition, one with a single
transition, and one with two. Chosen from the well-powered fits (>=14 usable intervals) with the largest
spread between fitted states, so the panel shows the model working rather than a marginal case.

Three tables are written, because the panel needs all three layers:
  windows  the observed intervals — each a WINDOW over molecular time, not a point, which is why the
           model deconvolves rather than smooths
  fits     the fitted states with their standard errors
  trace    BIC along the greedy path, i.e. the criterion the search actually compares. The path is
           forward selection, so this is not an exhaustive scan over S — it is the sequence of models
           the algorithm considered, which is what "BIC selected S states" means here.

Usage:  python paper/figures/data/signature/build/build_bic_examples.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[5]   # repo root (build/<label>/data/figures/paper/<repo>)
from paper.config import DATA, REF, ANALYSIS, TAU_OUTPUTS, SIM_OUTPUTS, PER_SAMPLE as PER_SAMPLE_DIR  # noqa: E402,F401
from tau.sig_states import StateFitParams, fit_sample_signature, _build_B, _wls, _supported

TABLES = ANALYSIS / "pcawg/signatures/timing_tables"
OUT = cache_dir_of(__file__); OUT.mkdir(parents=True, exist_ok=True)
# The two Figure 2 tumours, plus one sample where BIC declines a transition — without that third
# column the panel would suggest the criterion always finds one.
EXAMPLES = [("Eso.AdenoCA", "0250e6f7", "SBS17", "Fig 2b Eso — declining"),
            ("Stomach.AdenoCA", "dc537fcf", "SBS17", "Fig 2a Stomach — rising"),
            ("Bladder.TCC", "973d0577", "APOBEC", "no transition supported")]


def _prep(clust, sample, sig, p):
    """Reproduce the fit's input filtering so the panel plots exactly what the model saw."""
    rcol, wcol = f"{sig}_ratio", f"{sig}_raw_interval"
    a = clust[clust["sample"] == sample]
    baseline = np.nansum(a[wcol].to_numpy(float)) / np.nansum(a.clocklike_raw_interval.to_numpy(float))
    span = a.clocklike_cumulative_end.to_numpy(float) - a.clocklike_cumulative_start.to_numpy(float)
    sub = a[(a.clocklike_raw_interval >= p.interval_min_clock) & (span > p.min_span)
            & (span <= p.max_span)]
    w, v = sub[wcol].to_numpy(float), sub[rcol].to_numpy(float)
    k = np.isfinite(w) & (w > 0) & np.isfinite(v) & (v > 0)
    return sub[k], w[k], v[k], baseline


def main():
    p = StateFitParams()
    win, fits, trace = [], [], []
    for tt, pre, sig, lab in EXAMPLES:
        clust = pd.read_csv(TABLES / f"sig_intervals_range_{tt}_cluster.tsv", sep="\t")
        sample = next(s for s in clust["sample"].unique() if s.startswith(pre))
        sub, w, v, baseline = _prep(clust, sample, sig, p)
        z = np.log2(v / baseline)
        S0 = sub.clocklike_cumulative_start.to_numpy(float)
        E0 = sub.clocklike_cumulative_end.to_numpy(float)
        win.append(pd.DataFrame(dict(example=lab, tumor_type=tt, sample=sample, signature=sig,
                                     lo=S0, hi=E0, z=z, w=w, baseline=baseline)))
        f = fit_sample_signature(clust, sample, sig, p)
        f.insert(0, "example", lab)
        fits.append(f)

        # BIC along the greedy path, recomputed with the same rules as the search
        n = len(z)
        bp = np.unique(np.concatenate([S0, E0])); dom_lo, dom_hi = bp.min(), bp.max()
        cand = bp[(bp > dom_lo) & (bp < dom_hi)]
        crit = lambda rss, S: n * np.log(rss / n) + p.state_penalty * S * np.log(n)
        sel = []
        B, _, _ = _build_B(S0, E0, [], dom_lo, dom_hi)
        trace.append(dict(example=lab, n_states=1, bic=crit(_wls(B, w, z)[1], 1), selected=False))
        while True:
            best_r, best_cp = np.inf, None
            for cp in cand:
                if cp in sel:
                    continue
                e = np.unique(np.concatenate([[dom_lo], sorted(sel + [cp]), [dom_hi]]))
                if not np.all(np.diff(e) >= p.min_state_width):
                    continue
                Bb, lo_e, hi_e = _build_B(S0, E0, sorted(sel + [cp]), dom_lo, dom_hi)
                ff = _wls(Bb, w, z)
                if ff is None or not _supported(ff[0], lo_e, hi_e, S0, E0, z, w, p):
                    continue
                if ff[1] < best_r:
                    best_r, best_cp = ff[1], cp
            if best_cp is None:
                break
            trace.append(dict(example=lab, n_states=len(sel) + 2, bic=crit(best_r, len(sel) + 2),
                              selected=False))
            if crit(best_r, len(sel) + 2) >= trace[-2]["bic"] - 1e-9:
                break
            sel = sorted(sel + [best_cp])
        sc = int(f.n_states.iloc[0])
        for r in trace:
            if r["example"] == lab and r["n_states"] == sc:
                r["selected"] = True
        print(f"{lab:<9} {tt:<14} {pre} {sig:<7} n={len(z):>3} -> {sc} state(s)")

    pd.concat(win, ignore_index=True).to_csv(OUT / "bic_examples_windows.tsv", sep="\t", index=False)
    pd.concat(fits, ignore_index=True).to_csv(OUT / "bic_examples_fits.tsv", sep="\t", index=False)
    pd.DataFrame(trace).to_csv(OUT / "bic_examples_trace.tsv", sep="\t", index=False)
    print(f"\n-> {OUT}/bic_examples_*.tsv")


if __name__ == "__main__":
    main()
