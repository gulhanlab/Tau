"""Fig 4e10 — same "time to WGD" question as e9, but fit with a WEIBULL distribution instead of an
exponential.

MOTIVATION: e9 showed the single-exponential model is a poor description of "time to WGD, given a WGD
occurs" -- with zero censoring in that dataset (every included sample DID have the event), the empirical
KM curve stays flat near 1.0 through molecular time ~0.3-0.4 before dropping steeply, the opposite of a
constant-hazard (memoryless) exponential's immediate decay. A Weibull hazard is proportional to t^(k-1):
k>1 gives a RISING hazard (low early risk, increasing later) that can actually reproduce that "delayed
onset, then catch-up" shape; k=1 recovers the exponential exactly, so this is a strict generalisation, not
a different family of claim.

Same "3 markers per row" structure as e9:
  - fitted MEDIAN, WGD-only samples (teal circle/diamond + 95% CI, now from the Weibull fit)
  - raw empirical median, WGD-only samples (grey x, no CI) -- identical to e9's, distribution-agnostic
  - fitted MEDIAN, ALL samples / full cohort (purple square + 95% CI, Weibull fit with proper censoring
    for the never-WGD samples)
A Weibull's median is `lam * ln(2)^(1/k)` (reduces to `ln(2)/rate` when k=1, matching e7/e8/e9's formula) --
same quantity, same forest-plot axis, just computed from a different (here, better-fitting) shape.

No closed-form CI exists for a Weibull's median under censoring (unlike the exponential's exact
chi-squared interval), so CIs here are PERCENTILE BOOTSTRAP (resample samples with replacement, refit,
take the 2.5/97.5 percentiles of the resampled medians) -- the standard, distribution-agnostic approach.

Run:
    PATH=".pixi/envs/default/bin:$PATH" .pixi/envs/default/bin/python e10_weibull_wgd_timing_model.py
Writes e10_weibull_wgd_timing_model.{png,svg} to ./output/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_arm_patterns_segment import LAB, MM, TXT, OUT, _all_types, plt   # noqa: E402
from e7_wgd_recurrence_model import _mle, _nll_single, _km, _wgd_times_by_sample, _per_sample_table   # noqa: E402
from e8_first_wgd_timing_model import _first_wgd_table   # noqa: E402
from e9_wgd_only_timing_model import _wgd_only_table, WGD_COL, ALL_COL   # noqa: E402

MIN_EVENTS = 3
TRANS_COL = "#2e7d32"   # "all WGD gaps pooled" fit (0->1, 1->2, 2->3... as iid draws from ONE renewal
                        # process) -- a genuinely different model from the other three series, gets its
                        # own colour rather than reusing WGD/grey/black.


def _all_transitions():
    """Every WGD+ sample's FULL gap sequence, one row per GAP (not per sample): (0->w1), (w1->w2),
    (w2->w3)... for however many WGD times it has, PLUS a final right-censored gap from its last observed
    WGD to molecular time 1 (no further WGD seen there). This is the dataset for the model the user asked
    for: "uses all WGDs in a sequence... considering the time between all consecutive ones as well as the
    first one" -- i.e. treats 0->1, 1->2, 2->3 as iid draws from ONE shared distribution (a renewal-process
    assumption), rather than e10's other three series which each look at only ONE specific transition.
    Note this does NOT include WGD-negative samples (no "0->never" row) -- consistent with e9/e10's
    WGD-only framing, not e8's full-cohort one; every WGD+ sample contributes >=1 event row (its first
    WGD) plus exactly one trailing censored row."""
    by_samp, tt = _wgd_times_by_sample()
    types_ok = set(_all_types())
    rows = []
    for s, times in by_samp.items():
        t = tt.get(s)
        if t not in types_ok:
            continue
        prev = 0.0
        for wi in times:
            rows.append(dict(tumor_type=t, duration=wi - prev, event=1))
            prev = wi
        rows.append(dict(tumor_type=t, duration=1.0 - prev, event=0))
    return pd.DataFrame(rows)
N_BOOT = 100   # per-type bootstrap reps (pooled uses more, see below) -- kept modest since each bootstrap
               # replicate warm-starts from the point estimate (single L-BFGS-B call, not a multi-restart
               # search), so accuracy per replicate doesn't depend on restart count the way the point
               # estimate itself does.


def _weibull_nll(params, t_event, t_cens):
    log_k, log_lam = params
    k, lam = np.exp(log_k), np.exp(log_lam)
    ll = 0.0
    if len(t_event):
        z = np.clip(t_event, 1e-9, None) / lam
        ll += np.sum(np.log(k) - np.log(lam) + (k - 1) * np.log(z) - z ** k)
    if len(t_cens):
        zc = t_cens / lam
        ll += -np.sum(zc ** k)
    return -ll


def _fit_weibull(t_event, t_cens, x0=None):
    """Weibull MLE (shape k, scale lam) with right-censoring. Returns rate-model-compatible dict with
    `median` = lam*ln(2)^(1/k) so it slots into the same forest-plot plumbing as the exponential fits.
    If `x0` (a (log_k, log_lam) pair) is given, does a SINGLE warm-started fit from there instead of the
    default multi-restart search -- used by the bootstrap loop, where each replicate's true optimum is
    close to the already-known point estimate, so a fresh multi-restart search per replicate would be
    almost pure wasted compute (this is what made the first, unoptimised version take >5 minutes)."""
    t_all = np.concatenate([t_event, t_cens])
    if len(t_all) == 0:
        return dict(k=np.nan, lam=np.nan, median=np.nan, nll=np.nan)
    m = max(t_all.mean(), 1e-3)
    bounds = [(-3.0, 3.0), (-8.0, 3.0)]
    if x0 is not None:
        res = minimize(_weibull_nll, np.asarray(x0), args=(t_event, t_cens), method="L-BFGS-B", bounds=bounds)
        if not res.success:
            return dict(k=np.nan, lam=np.nan, median=np.nan, nll=np.nan)
        k, lam = np.exp(res.x[0]), np.exp(res.x[1])
        return dict(k=k, lam=lam, median=lam * np.log(2) ** (1.0 / k), nll=res.fun)
    best = None
    for k0 in (0.8, 1.0, 1.5, 2.5):
        x0i = np.array([np.log(k0), np.log(m)])
        res = minimize(_weibull_nll, x0i, args=(t_event, t_cens), method="L-BFGS-B", bounds=bounds)
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return dict(k=np.nan, lam=np.nan, median=np.nan, nll=np.nan)
    k, lam = np.exp(best.x[0]), np.exp(best.x[1])
    return dict(k=k, lam=lam, median=lam * np.log(2) ** (1.0 / k), nll=best.fun)


def _fit_weibull_bootstrap(t_event, t_cens, point_fit, n_boot=N_BOOT, seed=0):
    """Percentile bootstrap CI on the median, warm-started at the POINT ESTIMATE's own (k, lam) for every
    replicate (see `_fit_weibull`'s docstring) -- this is what keeps runtime tractable across ~50
    type/dataset combinations."""
    rng = np.random.default_rng(seed)
    n_e, n_c = len(t_event), len(t_cens)
    n = n_e + n_c
    if n == 0 or not np.isfinite(point_fit["k"]):
        return np.nan, np.nan
    x0 = np.array([np.log(point_fit["k"]), np.log(point_fit["lam"])])
    tt = np.concatenate([t_event, t_cens])
    ev = np.concatenate([np.ones(n_e), np.zeros(n_c)])
    meds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        te, tc = tt[idx][ev[idx] == 1], tt[idx][ev[idx] == 0]
        if len(te) == 0:
            continue
        try:
            f = _fit_weibull(te, tc, x0=x0)
            if np.isfinite(f["median"]):
                meds.append(f["median"])
        except Exception:
            continue
    if len(meds) < 20:
        return np.nan, np.nan
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _fit_table_weibull(df, time_col="time", event_col="event", seed0=0):
    """Weibull analogue of e8's `_fit_table`: per-tumor_type + pooled Weibull MLE fit (point + bootstrap
    CI on the median), zero-filled over `_all_types()` exactly like the exponential version."""
    all_t = _all_types()
    rows = []
    for i, t in enumerate(all_t):
        sub = df[df.tumor_type == t]
        te = sub.loc[sub[event_col] == 1, time_col].to_numpy()
        tc = sub.loc[sub[event_col] == 0, time_col].to_numpy()
        f = _fit_weibull(te, tc)
        lo, hi = (_fit_weibull_bootstrap(te, tc, f, seed=seed0 + i + 1) if len(te) >= MIN_EVENTS
                  else (np.nan, np.nan))
        rows.append(dict(tumor_type=t, n_events=len(te), n_samples=len(te) + len(tc),
                         k=f["k"], lam=f["lam"], half_time=f["median"], lo=lo, hi=hi))
    fits = pd.DataFrame(rows)
    t_event_all = df.loc[df[event_col] == 1, time_col].to_numpy()
    t_cens_all = df.loc[df[event_col] == 0, time_col].to_numpy()
    f_pooled = _fit_weibull(t_event_all, t_cens_all)
    lo_p, hi_p = _fit_weibull_bootstrap(t_event_all, t_cens_all, f_pooled, n_boot=300, seed=999999)
    pooled = dict(k=f_pooled["k"], lam=f_pooled["lam"], half_time=f_pooled["median"], lo=lo_p, hi=hi_p,
                 nll=f_pooled["nll"])
    return fits, pooled, len(t_event_all)


CACHE = OUT.parent / "cache_tables"; CACHE.mkdir(exist_ok=True)
CACHE_FILE = CACHE / "e10_weibull_results_v2.pkl"   # v2: adds the pooled-transitions series + 1->2 raw median


def compute(force=False):
    """All the expensive bits (per-type + pooled Weibull MLE + bootstrap CI, for the WGD-only,
    all-samples/full-cohort, AND all-transitions-pooled datasets) -- cached to disk as ONE dict so
    `make_horizontal()` (or any other re-plot) never needs to re-run the bootstrap. Returns a dict:
      fits                 -- one row per tumor_type, columns:
                               half_time/lo/hi            = WGD-only (first WGD, 0->1), Weibull fit
                               all_half_time/all_lo/all_hi = ALL samples / full cohort, Weibull fit
                               trans_half_time/trans_lo/trans_hi = ALL transitions pooled (0->1,1->2,2->3...)
                               raw_median                 = empirical median of first-WGD times (0->1)
                               raw_median_12               = empirical median of 1st->2nd WGD gaps (where
                                                              that type has >=1 sample with a 2nd WGD)
      pooled, pooled_n, pooled_raw_median, pooled_all      -- as before
      pooled_trans, pooled_raw_median_12                   -- the two new pooled quantities
    """
    if not force and CACHE_FILE.exists():
        return pd.read_pickle(CACHE_FILE)

    df = _wgd_only_table()
    fits, pooled, pooled_n = _fit_table_weibull(df, seed0=0)

    raw = df.groupby("tumor_type")["time"].median().rename("raw_median").reset_index()
    fits = fits.merge(raw, on="tumor_type", how="left")
    pooled_raw_median = float(df["time"].median())

    df_all = _first_wgd_table()
    fits_all, pooled_all, _ = _fit_table_weibull(df_all, seed0=1_000_000)
    fits = fits.merge(fits_all[["tumor_type", "half_time", "lo", "hi"]].rename(
        columns={"half_time": "all_half_time", "lo": "all_lo", "hi": "all_hi"}), on="tumor_type", how="left")

    # NEW: pool 0->1, 1->2, 2->3... as iid draws from ONE shared Weibull (renewal-process assumption) --
    # reuses `_fit_table_weibull` unchanged, just pointed at the per-GAP (not per-sample) dataset.
    df_trans = _all_transitions()
    fits_trans, pooled_trans, _ = _fit_table_weibull(df_trans, time_col="duration", event_col="event",
                                                     seed0=3_000_000)
    fits = fits.merge(fits_trans[["tumor_type", "half_time", "lo", "hi"]].rename(
        columns={"half_time": "trans_half_time", "lo": "trans_lo", "hi": "trans_hi"}), on="tumor_type",
        how="left")

    # NEW: raw (uncensored, no model) median of the 1st->2nd WGD gap, among samples that actually had a
    # 2nd WGD -- the direct analogue of `raw_median` (0->1) for the NEXT transition, reusing e7's own
    # per-sample table (t2/event2) rather than recomputing it.
    df_e7 = _per_sample_table()
    raw12 = (df_e7[df_e7.event2 == 1].groupby("tumor_type")["t2"].median()
            .rename("raw_median_12").reset_index())
    fits = fits.merge(raw12, on="tumor_type", how="left")
    pooled_raw_median_12 = float(df_e7.loc[df_e7.event2 == 1, "t2"].median())

    result = dict(fits=fits, pooled=pooled, pooled_n=pooled_n, pooled_raw_median=pooled_raw_median,
                 pooled_all=pooled_all, pooled_trans=pooled_trans,
                 pooled_raw_median_12=pooled_raw_median_12)
    pd.to_pickle(result, CACHE_FILE)
    print(f"  [e10] cached fit tables -> {CACHE_FILE}")
    return result


def make(save=True, stem="e10_weibull_wgd_timing_model", force_recompute=False):
    R = compute(force=force_recompute)
    fits, pooled, pooled_n = R["fits"], R["pooled"], R["pooled_n"]
    pooled_raw_median, pooled_all = R["pooled_raw_median"], R["pooled_all"]
    pooled_trans, pooled_raw_median_12 = R["pooled_trans"], R["pooled_raw_median_12"]
    df = _wgd_only_table()   # only needed here for Panel B's KM curve + AIC comparison, not refit

    shown = fits[fits.n_events >= MIN_EVENTS].sort_values("half_time").reset_index(drop=True)
    hidden = fits[fits.n_events < MIN_EVENTS].sort_values("n_events", ascending=False).reset_index(drop=True)

    n_rows = len(shown) + len(hidden) + 2
    fig_h = 0.30 * n_rows + 1.6
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(190 * MM, fig_h), gridspec_kw={"width_ratios": [1.3, 1]})

    XMAX = 2.0

    def _errbar(ax, y, pt, lo, hi, fmt, color, ms):
        # Unlike e7/e8/e9's exponential MLE (where rate=0 -> half_time=inf is a real, meaningful "never
        # happens" result worth plotting-and-annotating), `_fit_weibull` returns NaN specifically when
        # there's NO DATA AT ALL for that type/series (e.g. Lymph.CLL has zero WGD ever, so the
        # all-transitions-pooled series has literally no rows for it) -- draw nothing rather than pin a
        # misleading marker at the axis edge.
        if not np.isfinite(pt):
            return
        pt_c = min(pt, XMAX * 0.985)
        lo_c = min(max(lo, 0.0) if np.isfinite(lo) else 0.0, pt_c)
        hi_c = max(min(hi, XMAX) if np.isfinite(hi) else XMAX, pt_c)
        if not np.isfinite(lo) and not np.isfinite(hi):
            ax.plot([pt_c], [y], marker=fmt, color=color, ms=ms, zorder=3)
            return
        ax.errorbar([pt_c], [y], xerr=[[pt_c - lo_c], [hi_c - pt_c]], fmt=fmt, color=color, ms=ms,
                   capsize=2, lw=1.1 if fmt == "D" else 0.9, zorder=3)
        over_pt = np.isfinite(pt) and pt > XMAX
        over_hi = np.isfinite(hi) and hi > XMAX
        if over_pt or over_hi:
            hi_txt = f"{hi:.1f}" if np.isfinite(hi) else "?"
            pt_txt = f"pt={pt:.1f}, " if over_pt else ""
            ax.text(XMAX * 0.99, y, f" ⟶ {pt_txt}CI to {hi_txt}", fontsize=4.2, color=color,
                   va="center", ha="left", clip_on=False)

    def _naive_marker(ax, y, med, marker="x", color="#888"):
        if np.isfinite(med):
            ax.plot([min(med, XMAX * 0.985)], [y], marker=marker, color=color, ms=4.5, mew=1.1, zorder=4)

    y = 0; yticks = []; ylabels = []
    _errbar(axA, y, pooled["half_time"], pooled["lo"], pooled["hi"], "D", "#111", 5)
    _naive_marker(axA, y, pooled_raw_median)
    _errbar(axA, y, pooled_all["half_time"], pooled_all["lo"], pooled_all["hi"], "s", ALL_COL, 4)
    _errbar(axA, y, pooled_trans["half_time"], pooled_trans["lo"], pooled_trans["hi"], "^", TRANS_COL, 4.5)
    _naive_marker(axA, y, pooled_raw_median_12, marker="+", color="#888")
    yticks.append(y); ylabels.append(f"ALL TYPES POOLED (n={pooled_n})")
    y += 1.4
    axA.axhline(y - 0.7, color="#ccc", lw=0.6)

    for _, r in shown.iterrows():
        _errbar(axA, y, r["half_time"], r["lo"], r["hi"], "o", WGD_COL, 4)
        _naive_marker(axA, y, r["raw_median"])
        _errbar(axA, y, r["all_half_time"], r["all_lo"], r["all_hi"], "s", ALL_COL, 3.5)
        _errbar(axA, y, r["trans_half_time"], r["trans_lo"], r["trans_hi"], "^", TRANS_COL, 4)
        _naive_marker(axA, y, r["raw_median_12"], marker="+", color="#888")
        yticks.append(y); ylabels.append(f"{LAB.get(r['tumor_type'], r['tumor_type'])} (n={r['n_events']})")
        y += 1

    if len(hidden):
        y += 0.4
        axA.axhline(y - 0.7, color="#eee", lw=0.6)
        for _, r in hidden.iterrows():
            yticks.append(y)
            ylabels.append(f"{LAB.get(r['tumor_type'], r['tumor_type'])} (n={r['n_events']}, too few)")
            y += 1

    axA.set_yticks(yticks); axA.set_yticklabels(ylabels, fontsize=5.3, color=TXT)
    axA.set_ylim(y - 0.4, -0.8); axA.tick_params(length=0)
    axA.set_xlim(0, XMAX)
    axA.set_xlabel("Median time (molecular-time units)", fontsize=6.6, color=TXT)
    axA.set_title("Time-to-WGD, WGD+ samples only vs. all samples vs. all transitions pooled\n"
                 "(Weibull model, 95% bootstrap CI)", fontsize=7.2, loc="left", color=TXT)
    axA.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    axA.legend(handles=[
        Line2D([], [], marker="o", ls="", color=WGD_COL, ms=4, label="fitted, first WGD only (0→1)"),
        Line2D([], [], marker="x", ls="", color="#888", ms=4.5, mew=1.1, label="raw median, first WGD (0→1)"),
        Line2D([], [], marker="s", ls="", color=ALL_COL, ms=4, label="fitted, first WGD, ALL samples (full cohort)"),
        Line2D([], [], marker="^", ls="", color=TRANS_COL, ms=4.5, label="fitted, ALL gaps pooled (0→1,1→2,2→3...)"),
        Line2D([], [], marker="+", ls="", color="#888", ms=4.5, mew=1.1, label="raw median, 2nd WGD gap (1→2)"),
    ], fontsize=4.2, frameon=False, loc="upper right")

    t_event = df["time"].to_numpy(); t_cens = np.array([])
    times, surv = _km(df["time"].to_numpy(), df["event"].to_numpy())
    axB.step(times, surv, where="post", color="#111", lw=1.2, label="empirical survival (observed)")
    tt = np.linspace(1e-4, 1, 200)
    k, lam = pooled["k"], pooled["lam"]
    axB.plot(tt, np.exp(-(tt / lam) ** k), color=WGD_COL, lw=1.3, ls="--",
            label=f"Weibull fit (k={k:.2f}, median={pooled['half_time']:.2f})")

    # AIC: exponential (1 param) vs Weibull (2 params), same WGD-only uncensored likelihood.
    rate_exp = _mle(len(t_event), t_event.sum())["rate"]
    nll_exp = _nll_single(rate_exp, t_event, t_cens)
    aic_exp = 2 * 1 + 2 * nll_exp
    f_pooled_wgdonly = _fit_weibull(t_event, t_cens)   # same data as `pooled`, refit here just to get its nll
    aic_weib = 2 * 2 + 2 * f_pooled_wgdonly["nll"]
    axB.set_xlabel("Molecular time of first WGD", fontsize=7, color=TXT)
    axB.set_ylabel("P(no WGD yet), among WGD+ samples", fontsize=7, color=TXT)
    axB.set_title("Model check: time to WGD, WGD+ samples only\n(no censoring)", fontsize=7.2, loc="left",
                 color=TXT)
    axB.legend(fontsize=4.8, frameon=False, loc="lower left")
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.text(0.98, 0.98, f"{pooled_n} WGD+ samples, uncensored\n"
            f"AIC: exponential={aic_exp:.1f}  Weibull={aic_weib:.1f}\n"
            f"(k={k:.2f} -- k=1 is exponential; k>1 = rising hazard)", transform=axB.transAxes,
            fontsize=5.3, color=TXT, ha="right", va="top")

    fig.tight_layout()
    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    print(f"  pooled (WGD+ only, Weibull): k={k:.3f}, lam={lam:.3f}, "
         f"median={pooled['half_time']:.3f} [{pooled['lo']:.3f}, {pooled['hi']:.3f}]")
    print(f"  pooled (ALL samples, full cohort, Weibull): k={pooled_all['k']:.3f}, "
         f"median={pooled_all['half_time']:.3f} [{pooled_all['lo']:.3f}, {pooled_all['hi']:.3f}] "
         f"-- {pooled_all['half_time']/pooled['half_time']:.2f}x slower than WGD-only estimate")
    print(f"  pooled (ALL transitions 0->1,1->2,2->3 pooled, Weibull): k={pooled_trans['k']:.3f}, "
         f"median={pooled_trans['half_time']:.3f} [{pooled_trans['lo']:.3f}, {pooled_trans['hi']:.3f}]")
    print(f"  raw median 1st->2nd WGD gap (uncensored, recurred samples only): {pooled_raw_median_12:.3f}")
    print(f"  AIC comparison (WGD-only, n={pooled_n}): exponential={aic_exp:.1f}  Weibull={aic_weib:.1f} "
         f"-> {'Weibull fits meaningfully better' if aic_exp - aic_weib > 2 else 'no clear improvement'}")
    print(f"  {len(shown)} types with >= {MIN_EVENTS} events shown with a fit; "
         f"{len(hidden)} types too sparse")
    return fig


def make_horizontal(save=True, stem="e10_weibull_horizontal", force_recompute=False, types_order=None,
                    show_wgd_only=True, fig_height=3.6):
    """Panel-A-only (the 3/5-marker forest plot from `make()`), ROTATED to horizontal: tumour types along
    the x-axis (45-degree labels), time-to-WGD on the y-axis. Loads from the on-disk cache by default (see
    `compute()`) -- no bootstrap re-run needed just to re-lay-out the same fitted numbers.
    Colour convention (as requested): WGD-only fitted median in Joseph's own WGD colour (`WGD_COL`,
    matches `paper/figures/main/punctuated/panel_scripts/_wgdpgd_common.py`'s WGD="#0b6979"); all-samples/full-cohort
    fitted median in neutral GREY (it's a comparison baseline, not a WGD-specific estimate); the raw/naive
    median markers in solid BLACK (data sanity-checks, not model fits); the all-transitions-pooled fit in
    green (a genuinely different model, gets its own colour).

    types_order: optional explicit list of internal tumor_type codes, in the exact order to display --
        overrides the default (types with >= MIN_EVENTS, sorted by half_time). Types not present in the
        cache are silently skipped (a warning is printed with exactly which ones and why).
    show_wgd_only: if False, drops the "fitted, first WGD only (0->1)" series (teal circle) entirely --
        e.g. when that series would just be visual clutter alongside the pooled-transitions fit.
    fig_height: figure height in inches (default 3.6; pass e.g. 1.8 for a compact "half height" version)."""
    R = compute(force=force_recompute)
    fits, pooled, pooled_n = R["fits"], R["pooled"], R["pooled_n"]
    pooled_raw_median, pooled_all = R["pooled_raw_median"], R["pooled_all"]
    pooled_trans, pooled_raw_median_12 = R["pooled_trans"], R["pooled_raw_median_12"]

    if types_order is not None:
        cache_types = set(fits["tumor_type"])
        missing = [t for t in types_order if t not in cache_types]
        if missing:
            print(f"  [e10 horizontal] {len(missing)} requested type(s) not in the analysis cohort "
                 f"(excluded upstream via EXCLUDE_TYPES, or a name mismatch) -- kept as an EMPTY "
                 f"row (label only, no markers) at their requested position: {missing}")
        # reindex (not filter): keeps EVERY requested type as its own row, in the EXACT given order --
        # types absent from `fits` just come back as an all-NaN row, which `_err`/the raw-median plotting
        # already skip drawing anything for (see their "if not np.isfinite(pt): return" checks).
        shown = fits.set_index("tumor_type").reindex(types_order).reset_index()
    else:
        shown = fits[fits.n_events >= MIN_EVENTS].sort_values("half_time").reset_index(drop=True)

    XMAX = 2.0
    GREY, BLACK = "#8c8c8c", "#111111"

    labels = ["ALL POOLED"] + [LAB.get(t, t) for t in shown["tumor_type"]]
    x = np.arange(len(labels))

    def _err(ax, xi, pt, lo, hi, color, marker, ms):
        # See `_errbar`'s comment in `make()`: NaN here means NO DATA (not "fit says infinity") -- skip.
        if not np.isfinite(pt):
            return
        pt_c = min(pt, XMAX * 0.985)
        if not np.isfinite(lo) and not np.isfinite(hi):
            ax.plot([xi], [pt_c], marker=marker, color=color, ms=ms, zorder=3)
            return
        lo_c = min(max(lo, 0.0) if np.isfinite(lo) else 0.0, pt_c)
        hi_c = max(min(hi, XMAX) if np.isfinite(hi) else XMAX, pt_c)
        ax.errorbar([xi], [pt_c], yerr=[[pt_c - lo_c], [hi_c - pt_c]], fmt=marker, color=color, ms=ms,
                   capsize=2, lw=1.0, zorder=3)

    fig_w = 0.34 * len(labels) + 1.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_height))

    if show_wgd_only:
        _err(ax, 0, pooled["half_time"], pooled["lo"], pooled["hi"], WGD_COL, "D", 5)
    if np.isfinite(pooled_raw_median):
        ax.plot([0], [min(pooled_raw_median, XMAX * 0.985)], marker="x", color=BLACK, ms=5, mew=1.2, zorder=4)
    _err(ax, 0, pooled_all["half_time"], pooled_all["lo"], pooled_all["hi"], GREY, "D", 5)
    _err(ax, 0, pooled_trans["half_time"], pooled_trans["lo"], pooled_trans["hi"], TRANS_COL, "D", 5)
    if np.isfinite(pooled_raw_median_12):
        ax.plot([0], [min(pooled_raw_median_12, XMAX * 0.985)], marker="+", color=BLACK, ms=5.5, mew=1.3, zorder=4)

    for i, (_, r) in enumerate(shown.iterrows(), start=1):
        if show_wgd_only:
            _err(ax, i, r["half_time"], r["lo"], r["hi"], WGD_COL, "o", 4)
        if np.isfinite(r["raw_median"]):
            ax.plot([i], [min(r["raw_median"], XMAX * 0.985)], marker="x", color=BLACK, ms=4.5, mew=1.1, zorder=4)
        _err(ax, i, r["all_half_time"], r["all_lo"], r["all_hi"], GREY, "s", 3.5)
        _err(ax, i, r["trans_half_time"], r["trans_lo"], r["trans_hi"], TRANS_COL, "^", 4)
        if np.isfinite(r["raw_median_12"]):
            ax.plot([i], [min(r["raw_median_12"], XMAX * 0.985)], marker="+", color=BLACK, ms=4.5, mew=1.1, zorder=4)

    ax.axvline(0.5, color="#ccc", lw=0.7, zorder=0)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6, color=TXT)
    ax.set_ylim(0, XMAX)
    ax.set_ylabel("Median time (mol. time units)" if fig_height < 3.0 else
                 "Median time (molecular-time units)", fontsize=7 if fig_height >= 3.0 else 6, color=TXT)
    title = ("Time-to-WGD (Weibull model): all samples vs. all transitions pooled" if not show_wgd_only else
            "Time-to-WGD (Weibull model): first WGD vs. all samples vs. all transitions pooled")
    ax.set_title(title, fontsize=8, loc="left", color=TXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=6)
    from matplotlib.lines import Line2D
    handles = []
    if show_wgd_only:
        handles.append(Line2D([], [], marker="o", ls="", color=WGD_COL, ms=4, label="fitted, first WGD only (0→1)"))
    handles += [
        Line2D([], [], marker="x", ls="", color=BLACK, ms=4.5, mew=1.1, label="raw median, first WGD (0→1)"),
        Line2D([], [], marker="s", ls="", color=GREY, ms=4, label="fitted, first WGD, ALL samples (full cohort)"),
        Line2D([], [], marker="^", ls="", color=TRANS_COL, ms=4.5, label="fitted, ALL gaps pooled (0→1,1→2,2→3...)"),
        Line2D([], [], marker="+", ls="", color=BLACK, ms=4.5, mew=1.1, label="raw median, 2nd WGD gap (1→2)"),
    ]
    ax.legend(handles=handles, fontsize=4.6, frameon=False, loc="upper right")

    fig.tight_layout()
    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    return fig


def make_vertical(save=True, stem="e10_weibull_vertical", force_recompute=False, types_order=None,
                  show_wgd_only=True, fig_w=3.4, fig_h=None):
    """Vertical counterpart of `make_horizontal()`: tumour types as ROWS, top-to-bottom, time-to-WGD on
    the x-axis -- the same layout as `make()`'s Panel A, but as its own standalone figure, with support for
    an explicit `types_order` (kept in the EXACT order given, not re-sorted by half_time) and for dropping
    the "first WGD only" series via `show_wgd_only=False`. Loads from the on-disk cache (see `compute()`).
    fig_h: override the auto-computed height (0.30in/row + 1.2in) -- e.g. to shrink an already-tuned figure
    by a fixed factor without touching any fontsize (rows just get physically more compressed)."""
    R = compute(force=force_recompute)
    fits, pooled, pooled_n = R["fits"], R["pooled"], R["pooled_n"]
    pooled_raw_median, pooled_all = R["pooled_raw_median"], R["pooled_all"]
    pooled_trans, pooled_raw_median_12 = R["pooled_trans"], R["pooled_raw_median_12"]

    if types_order is not None:
        cache_types = set(fits["tumor_type"])
        missing = [t for t in types_order if t not in cache_types]
        if missing:
            print(f"  [e10 vertical] {len(missing)} requested type(s) not in the analysis cohort "
                 f"(excluded upstream via EXCLUDE_TYPES, or a name mismatch) -- kept as an EMPTY "
                 f"row (label only, no markers) at their requested position: {missing}")
        shown = fits.set_index("tumor_type").reindex(types_order).reset_index()
    else:
        shown = fits[fits.n_events >= MIN_EVENTS].sort_values("half_time").reset_index(drop=True)

    XMAX = 2.0
    GREY, BLACK = "#8c8c8c", "#111111"

    labels = ["ALL POOLED"] + [LAB.get(t, t) for t in shown["tumor_type"]]
    yt = np.arange(len(labels))

    def _err(ax, yi, pt, lo, hi, color, marker, ms):
        if not np.isfinite(pt):   # NaN = no data at all for this series/type -- see make()'s _errbar comment
            return
        pt_c = min(pt, XMAX * 0.985)
        if not np.isfinite(lo) and not np.isfinite(hi):
            ax.plot([pt_c], [yi], marker=marker, color=color, ms=ms, zorder=3)
            return
        lo_c = min(max(lo, 0.0) if np.isfinite(lo) else 0.0, pt_c)
        hi_c = max(min(hi, XMAX) if np.isfinite(hi) else XMAX, pt_c)
        ax.errorbar([pt_c], [yi], xerr=[[pt_c - lo_c], [hi_c - pt_c]], fmt=marker, color=color, ms=ms,
                   capsize=2, lw=1.0, zorder=3)

    if fig_h is None:
        fig_h = 0.30 * len(labels) + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if show_wgd_only:
        _err(ax, 0, pooled["half_time"], pooled["lo"], pooled["hi"], WGD_COL, "D", 5)
    if np.isfinite(pooled_raw_median):
        ax.plot([min(pooled_raw_median, XMAX * 0.985)], [0], marker="x", color=BLACK, ms=5, mew=1.2, zorder=4)
    _err(ax, 0, pooled_all["half_time"], pooled_all["lo"], pooled_all["hi"], GREY, "s", 5)
    _err(ax, 0, pooled_trans["half_time"], pooled_trans["lo"], pooled_trans["hi"], TRANS_COL, "^", 5)
    if np.isfinite(pooled_raw_median_12):
        ax.plot([min(pooled_raw_median_12, XMAX * 0.985)], [0], marker="+", color=BLACK, ms=5.5, mew=1.3, zorder=4)

    for i, (_, r) in enumerate(shown.iterrows(), start=1):
        if show_wgd_only:
            _err(ax, i, r["half_time"], r["lo"], r["hi"], WGD_COL, "o", 4)
        if np.isfinite(r["raw_median"]):
            ax.plot([min(r["raw_median"], XMAX * 0.985)], [i], marker="x", color=BLACK, ms=4.5, mew=1.1, zorder=4)
        _err(ax, i, r["all_half_time"], r["all_lo"], r["all_hi"], GREY, "s", 3.5)
        _err(ax, i, r["trans_half_time"], r["trans_lo"], r["trans_hi"], TRANS_COL, "^", 4)
        if np.isfinite(r["raw_median_12"]):
            ax.plot([min(r["raw_median_12"], XMAX * 0.985)], [i], marker="+", color=BLACK, ms=4.5, mew=1.1, zorder=4)

    ax.axhline(0.5, color="#ccc", lw=0.7, zorder=0)
    ax.set_ylim(len(labels) - 0.4, -0.6)   # pooled row at TOP, first requested type just below, etc.
    ax.set_yticks(yt); ax.set_yticklabels(labels, fontsize=6, color=TXT); ax.tick_params(length=0)
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("Median time (molecular-time units)", fontsize=7, color=TXT)
    title = ("Time-to-WGD (Weibull model): all samples vs. all transitions pooled" if not show_wgd_only else
            "Time-to-WGD (Weibull model): first WGD vs. all samples vs. all transitions pooled")
    ax.set_title(title, fontsize=8, loc="left", color=TXT, pad=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=6)
    from matplotlib.lines import Line2D
    handles = []
    if show_wgd_only:
        handles.append(Line2D([], [], marker="o", ls="", color=WGD_COL, ms=4, label="fitted, first WGD only (0→1)"))
    handles += [
        Line2D([], [], marker="x", ls="", color=BLACK, ms=4.5, mew=1.1, label="raw median, first WGD (0→1)"),
        Line2D([], [], marker="s", ls="", color=GREY, ms=4, label="fitted, first WGD, ALL samples (full cohort)"),
        Line2D([], [], marker="^", ls="", color=TRANS_COL, ms=4.5, label="fitted, ALL gaps pooled (0→1,1→2,2→3...)"),
        Line2D([], [], marker="+", ls="", color=BLACK, ms=4.5, mew=1.1, label="raw median, 2nd WGD gap (1→2)"),
    ]
    ax.legend(handles=handles, fontsize=4.4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
             ncol=2, columnspacing=1.0, handletextpad=0.4)

    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.65 / fig_h)   # reserve a fixed ~0.65in band above the axes for
                                                # title+legend, since tight_layout alone doesn't know
                                                # about the legend's bbox_to_anchor placement outside axes
    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    return fig


# User-specified display order (internal tumor_type codes, mapped from the requested display names via
# `LAB`). 4 of the originally-requested names -- CNS.Oligo, Myeloid.MPN, Thy.AdenoCA, CNS.PiloAstro -- are
# NOT in `_all_types()` at all (excluded upstream by e2's EXCLUDE_TYPES as ultra-rare/mostly-empty types),
# so they're omitted here rather than silently guessed at; `make_horizontal` also prints a warning listing
# exactly what's missing and why, in case that cache ever changes. SoftTissue.Leiomyo exists in the cohort
# but was not in the requested list, so it's intentionally left out of this specific ordering too.
SELECTED_ORDER = [
    "Eso.AdenoCA", "Bladder.TCC", "Lung.AdenoCA", "Lung.SCC", "ColoRect.AdenoCA", "Skin.Melanoma",
    "Ovary.AdenoCA", "Bone.Osteosarc", "Head.SCC", "Stomach.AdenoCA", "SoftTissue.Liposarc",
    "Breast.AdenoCA", "Cervix.SCC", "Panc.AdenoCA", "Uterus.AdenoCA", "Panc.Endocrine", "Liver.HCC",
    "Kidney.ChRCC", "Biliary.AdenoCA", "CNS.Medullo", "CNS.GBM", "Lymph.BNHL", "Kidney.RCC",
    "Prost.AdenoCA",
    # The remaining 4 are NOT in `_all_types()` (excluded upstream by e2's EXCLUDE_TYPES as ultra-rare/
    # mostly-empty types) -- kept here anyway, in the exact requested SEQUENCE POSITION, so the row is
    # still drawn (label only, no markers -- `compute()`'s `fits` table simply has no data for them,
    # `.reindex()` in make_vertical/make_horizontal turns that into an all-NaN row that `_err`/`_naive_
    # marker` already skip drawing anything for) rather than silently compressing the list.
    "CNS.Oligo", "Myeloid.MPN", "Lymph.CLL", "Thy.AdenoCA", "CNS.PiloAstro",
]


if __name__ == "__main__":
    make()
    make_horizontal()
    # 30 rows (1 pooled + 29 requested) at the default 0.30in/row + 1.2in formula -> fig_h=10.2, fig_w=3.4;
    # shrunk here to 1/3 height (3.4in) and 0.75x width (2.55in) per request, fonts unchanged (rows just
    # sit physically closer together).
    make_vertical(stem="e10_weibull_vertical_selected", types_order=SELECTED_ORDER, show_wgd_only=False,
                 fig_w=2.55, fig_h=3.4)
    print("e10_weibull_wgd_timing_model: PNG+SVG written")
