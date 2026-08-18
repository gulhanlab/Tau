"""Per-sample signature-activity state fitting — the stepwise-change model behind the paper's
"Tau fits a stepwise-change model to these relative rates and uses the Bayesian information criterion
to determine the number of activity transitions supported by each sample".

Python port of `estimate_cluster_timing.R` (D. Gulhan). The R remains the reference implementation;
this module is a line-for-line translation, and `tests/test_sig_states.py` checks it against the R's
own output rather than trusting the port.

The problem
-----------
A pooled route cluster gives, per time interval, the signature's mutation count and the clock's. Their
ratio is the signature's relative activity in that interval. Intervals from different clusters do not
share boundaries, though, so the per-interval ratios are not a time series — they are overlapping
windows over molecular time, each averaging whatever the true activity did across its own span.

The model therefore deconvolves rather than smooths. Molecular time is cut into S states with constant
log2 activity, and each observed interval is modelled as the overlap-weighted mean of the states it
spans:

    B[i, k] = |interval_i intersect state_k| / |interval_i|          (rows sum to 1)
    z_i     = log2(ratio_i / baseline)  ~  (B theta)_i,  weighted by that interval's signature count

theta is the fitted log2 fold-change per state, by weighted least squares.

Choosing the states
-------------------
Changepoints are added greedily from the observed interval boundaries, each step taking the one that
most reduces the weighted RSS, and stopping when BIC no longer improves:

    BIC = n log(RSS / n) + penalty * S * log(n)

Two guards keep the fit honest, and both matter more than the criterion:

  * `min_state_width` — no state narrower than 0.05 molecular time, so a changepoint cannot carve off
    a sliver to chase one interval.
  * `supported()` — a state's fitted level must lie inside the range of log-ratios actually observed
    in the intervals overlapping it, where "observed range" is trimmed to what at least
    `min_support_mut` mutations support. Without this the deconvolution will happily place a state at
    an extreme value that no single interval attests, because overlapping windows can imply it
    arithmetically. This is the constraint that stops the model inventing transitions.

Usage
-----
    from tau.sig_states import fit_cluster_table, fit_directory
    est = fit_directory("dev/pcawg/signatures/timing_tables")     # every tumour type x signature
    est = fit_cluster_table("…/sig_intervals_range_Eso.AdenoCA_cluster.tsv", signatures=["SBS17"])

or from the command line:

    python -m tau.sig_states <indir> -o cluster_timing_estimates.tsv
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["StateFitParams", "fit_sample_signature", "fit_cluster_table", "fit_directory"]


@dataclass(frozen=True)
class StateFitParams:
    """Defaults are the R script's; changing them changes the model, not just the reporting."""
    interval_min_clock: float = 5.0     # drop intervals with fewer clock mutations than this
    sample_min_clock: float = 20.0      # skip the sample entirely below this total
    min_span: float = 0.05              # drop intervals narrower than this (strict >)
    max_span: float = 0.75              # ...and wider than this (inclusive <=)
    min_state_width: float = 0.05       # no fitted state may be narrower
    min_segments: int = 5               # need this many usable intervals to fit at all
    state_penalty: float = 1.0          # multiplier on the BIC state penalty
    support_tol: float = 0.05           # slack when checking a state against observed log-ratios
    min_support_mut: float = 10.0       # mutations required to count as supporting a level


def _build_B(S0, E0, cps, dom_lo, dom_hi):
    """Overlap matrix: row i = how interval i's span divides among the states. Rows sum to 1."""
    edges = np.unique(np.concatenate([[dom_lo], np.asarray(cps, float), [dom_hi]]))
    lo, hi = edges[:-1], edges[1:]
    length = (E0 - S0)[:, None]
    ov = np.minimum(E0[:, None], hi[None, :]) - np.maximum(S0[:, None], lo[None, :])
    return np.clip(ov, 0.0, None) / length, lo, hi


def _wls(B, w, z):
    """Weighted least squares. Returns None where R's `qr(M)$rank < ncol(B)` would reject."""
    M = B.T @ (B * w[:, None])
    if np.linalg.matrix_rank(M, tol=1e-8 * max(1.0, float(np.abs(M).max()))) < B.shape[1]:
        return None
    try:
        theta = np.linalg.solve(M, B.T @ (w * z))
    except np.linalg.LinAlgError:
        return None
    r = z - B @ theta
    return theta, float(np.sum(w * r * r)), M


def _trimmed_range(zz, ww, min_support_mut):
    """(lo, hi) log-ratios that at least `min_support_mut` mutations support, from each end.

    Sorting descending and walking down until the cumulative weight reaches the threshold gives the
    highest level the data can actually vouch for; the ascending pass gives the lowest. Where the
    overlapping intervals carry less weight than the threshold in total, the untrimmed extreme is used
    — the same fallback as the R (`if (is.na(hi)) hi <- min(zz)`).
    """
    oh = np.argsort(-zz, kind="stable")
    c = np.cumsum(ww[oh]); i = np.flatnonzero(c >= min_support_mut)
    hi = zz[oh][i[0]] if i.size else zz.min()
    ol = np.argsort(zz, kind="stable")
    c = np.cumsum(ww[ol]); i = np.flatnonzero(c >= min_support_mut)
    lo = zz[ol][i[0]] if i.size else zz.max()
    return lo, hi


def _supported(theta, lo_e, hi_e, S0, E0, z, w, p):
    for k in range(len(lo_e)):
        ov = np.flatnonzero((S0 < hi_e[k]) & (E0 > lo_e[k]))
        if ov.size == 0:
            return False
        lo, hi = _trimmed_range(z[ov], w[ov], p.min_support_mut)
        if not (theta[k] <= hi + p.support_tol and theta[k] >= lo - p.support_tol):
            return False
    return True


def fit_sample_signature(clust: pd.DataFrame, sample: str, signature: str,
                         p: StateFitParams = StateFitParams()) -> pd.DataFrame | None:
    """Fit the state model for one (sample, signature). Returns one row per state, or None."""
    rcol, wcol = f"{signature}_ratio", f"{signature}_raw_interval"
    if rcol not in clust.columns or wcol not in clust.columns:
        return None
    allrows = clust[clust["sample"] == sample]
    if not len(allrows):
        return None
    total_clock = np.nansum(allrows.clocklike_raw_interval.to_numpy(float))
    if not total_clock > p.sample_min_clock:
        return None
    baseline = np.nansum(allrows[wcol].to_numpy(float)) / total_clock

    span = allrows.clocklike_cumulative_end.to_numpy(float) - \
        allrows.clocklike_cumulative_start.to_numpy(float)
    sub = allrows[(allrows.clocklike_raw_interval >= p.interval_min_clock)
                  & (span > p.min_span) & (span <= p.max_span)]
    w = sub[wcol].to_numpy(float)
    v = sub[rcol].to_numpy(float)
    keep = np.isfinite(w) & (w > 0) & np.isfinite(v) & (v > 0)
    sub, w, v = sub[keep], w[keep], v[keep]
    n = len(sub)
    if n < p.min_segments or not np.isfinite(baseline) or baseline <= 0:
        return None

    z = np.log2(v / baseline)
    S0 = sub.clocklike_cumulative_start.to_numpy(float)
    E0 = sub.clocklike_cumulative_end.to_numpy(float)
    bp = np.unique(np.concatenate([S0, E0]))
    dom_lo, dom_hi = bp.min(), bp.max()
    cand = bp[(bp > dom_lo) & (bp < dom_hi)]

    def crit(rss, S):
        return n * np.log(rss / n) + p.state_penalty * S * np.log(n)

    def width_ok(cps):
        e = np.unique(np.concatenate([[dom_lo], np.asarray(cps, float), [dom_hi]]))
        return bool(np.all(np.diff(e) >= p.min_state_width))

    B0, _, _ = _build_B(S0, E0, [], dom_lo, dom_hi)
    base = _wls(B0, w, z)
    if base is None:
        return None
    sel: list[float] = []
    cur = crit(base[1], 1)
    while True:
        best_r, best_cp = np.inf, None
        for cp in cand:
            if cp in sel or not width_ok(sorted(sel + [cp])):
                continue
            B, lo_e, hi_e = _build_B(S0, E0, sorted(sel + [cp]), dom_lo, dom_hi)
            f = _wls(B, w, z)
            if f is None or not _supported(f[0], lo_e, hi_e, S0, E0, z, w, p):
                continue
            if f[1] < best_r:
                best_r, best_cp = f[1], cp
        if best_cp is None or crit(best_r, len(sel) + 2) >= cur - 1e-9:
            break
        sel = sorted(sel + [best_cp])
        cur = crit(best_r, len(sel) + 1)

    B, lo_e, hi_e = _build_B(S0, E0, sel, dom_lo, dom_hi)
    theta, rss, M = _wls(B, w, z)
    S = len(lo_e)
    if n - S >= 1:
        se = np.sqrt(np.diag((rss / (n - S)) * np.linalg.inv(M)))
    else:
        se = np.full(S, np.nan)
    return pd.DataFrame({
        "tumor_type": allrows.tumor_type.iloc[0], "sample": sample, "signature": signature,
        "n_segments": n, "n_states": S, "state": np.arange(1, S + 1),
        "lo": lo_e, "hi": hi_e, "width": hi_e - lo_e, "baseline": baseline,
        "log2_fold": theta, "log2_fold_se": se,
        "log2_fold_lo": theta - se, "log2_fold_hi": theta + se,
        "ratio": baseline * 2.0 ** theta})


def fit_cluster_table(path_or_df, signatures=None, p: StateFitParams = StateFitParams()):
    """Fit every (sample, signature) in one `sig_intervals_range_<TT>_cluster.tsv`."""
    clust = pd.read_csv(path_or_df, sep="\t") if isinstance(path_or_df, (str, os.PathLike)) \
        else path_or_df
    if signatures is None:
        signatures = [c[:-len("_ratio")] for c in clust.columns if c.endswith("_ratio")]
    out = [r for sig in signatures for s in clust["sample"].unique()
           if (r := fit_sample_signature(clust, s, sig, p)) is not None]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fit_directory(indir, pattern="sig_intervals_range_*_cluster.tsv",
                  p: StateFitParams = StateFitParams()):
    """Fit every tumour-type table in a directory; returns the concatenated estimates."""
    out = []
    for f in sorted(glob.glob(os.path.join(str(indir), pattern))):
        r = fit_cluster_table(f, p=p)
        if len(r):
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("indir")
    ap.add_argument("-o", "--out", default="cluster_timing_estimates.tsv")
    ap.add_argument("--penalty", type=float, default=StateFitParams.state_penalty)
    a = ap.parse_args(argv)
    est = fit_directory(a.indir, p=StateFitParams(state_penalty=a.penalty))
    est.to_csv(a.out, sep="\t", index=False)
    fits = est.drop_duplicates(["tumor_type", "sample", "signature"])
    print(f"{len(est)} states across {len(fits)} fits, "
          f"{est.tumor_type.nunique()} tumour types, {est.signature.nunique()} signatures")
    for sg, g in fits.groupby("signature"):
        print(f"  {sg:<7}: {len(g)} fits, median states {g.n_states.median():g}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
