"""Fig 4e7 — modeling WGD RECURRENCE as a Poisson process: after a first WGD, does a second one arrive
at a roughly constant hazard rate (i.e. an exponentially-distributed waiting time), and if so what's the
implied HALF-TIME (median time to a second WGD) per tumor type?

MOTIVATION: samples with >=2 WGD events (double/triple/quadruple-WGD) often have the first WGD early and
subsequent ones spaced out afterward. This fits the simplest quantitative version of "regular-ish
recurrence" — a homogeneous Poisson process for genome-doubling (constant hazard = exponential waiting
time) — and reports each tumor type's half-time under that model.

DATA / CENSORING: every WGD-positive sample contributes a first WGD time w1 — the start of "at risk of a
further WGD". If the sample went on to have k total WGD times (w1 < ... < wk), the window after the LAST
observed one is followed by (1 - wk) of further molecular time with NO further WGD seen — i.e.
RIGHT-CENSORED at molecular time 1 (we only observe a tumor up to "now", never its full future
evolutionary trajectory). Single-WGD samples (k=1) are entirely censored: (1 - w1) of at-risk time, zero
events.

MLE for a constant-hazard (exponential) recurrence process, pooling ALL of a sample's transitions (valid
under the memoryless property: consecutive waiting times are iid Exponential(lambda) under this model),
telescopes neatly per sample: total exposure = every observed gap + the final censored gap
= (wk - w1) + (1 - wk) = 1 - w1, REGARDLESS of k. So, per tumor type:
    lambda_hat = (total additional WGDs beyond each sample's first) / sum(1 - w1)
    half_time  = ln(2) / lambda_hat                    (MOLECULAR-TIME units, 0-1 scale — NOT years/Mya)
— exactly the standard "total events / total person-time at risk" Poisson-rate MLE. A 95% CI on lambda
uses the exact Poisson-count interval (chi-squared relationship: 2*lambda*T ~ chi2(2n)), converted to
half-time (note half-time is DECREASING in lambda, so the CI flips when converting).

Diagnostics (does the data actually look exponential, i.e. is "regular intervals" really constant-hazard
or something tighter?): a pooled Kaplan-Meier curve of "time from 1st WGD to 2nd WGD, or censored at
1-w1" plotted against the fitted exponential survival curve, plus the raw observed w2-w1 gap histogram.

Run:
    PATH=".pixi/envs/default/bin:$PATH" .pixi/envs/default/bin/python e7_wgd_recurrence_model.py
Writes e7_wgd_recurrence_model.{png,svg} to ./output/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_arm_patterns_segment import LAB, MM, TXT, OUT, _events_deduped, _all_types, plt   # noqa: E402

MIN_EVENTS = 3   # minimum pooled "extra WGD" events required to report a per-type half-time (else "too few")
WGD_COL = "#0b6979"
MIX_COL = "#b23a1f"
SEG_COL = "#2e7d32"
EARLY_CUT, LATE_CUT = 0.3, 0.4   # head window [0, EARLY_CUT), tail window [LATE_CUT, 1] for the segmented fit


def _wgd_times_by_sample():
    ev = _events_deduped()
    wgd = ev[ev.classification == "WGD"].copy()
    wgd["time"] = wgd["time"].astype(float)
    by_samp = wgd.groupby("sample")["time"].apply(lambda s: sorted(s.tolist()))
    tt = wgd.drop_duplicates("sample").set_index("sample")["tumor_type"]
    return by_samp, tt


def _per_sample_table():
    """One row per WGD-positive sample: tumor_type, k (# WGD times), n_extra (=k-1, the MLE's event
    count), exposure (=1-w1, telescoped person-time at risk), and (t2, event2) for the simple
    first-transition Kaplan-Meier check (t2 = w2-w1 if k>=2 else 1-w1; event2 = 1 if k>=2 else 0)."""
    by_samp, tt = _wgd_times_by_sample()
    types_ok = set(_all_types())
    rows = []
    for s, times in by_samp.items():
        t = tt.get(s)
        if t not in types_ok:
            continue
        k = len(times); w1 = times[0]
        if k >= 2:
            t2, event2 = times[1] - w1, 1
        else:
            t2, event2 = 1.0 - w1, 0
        rows.append(dict(sample=s, tumor_type=t, k=k, n_extra=k - 1, exposure=1.0 - w1,
                         t2=t2, event2=event2))
    return pd.DataFrame(rows)


def _mle(n_events, total_exposure, alpha=0.05):
    """Poisson-rate MLE + exact CI (chi-squared relationship), converted to half-time = ln2/lambda."""
    if total_exposure <= 0:
        return dict(rate=np.nan, half_time=np.nan, lo=np.nan, hi=np.nan)
    rate = n_events / total_exposure
    if n_events == 0:
        lo_rate, hi_rate = 0.0, stats.chi2.ppf(1 - alpha / 2, 2) / (2 * total_exposure)
    else:
        lo_rate = stats.chi2.ppf(alpha / 2, 2 * n_events) / (2 * total_exposure)
        hi_rate = stats.chi2.ppf(1 - alpha / 2, 2 * (n_events + 1)) / (2 * total_exposure)
    half = np.log(2) / rate if rate > 0 else np.inf
    hi_half = np.log(2) / lo_rate if lo_rate > 0 else np.inf   # CI flips: half-time DECREASES with rate
    lo_half = np.log(2) / hi_rate if hi_rate > 0 else np.inf
    return dict(rate=rate, half_time=half, lo=lo_half, hi=hi_half)


def _nll_single(rate, t_event, t_cens):
    if rate <= 0:
        return np.inf
    return -(len(t_event) * np.log(rate) - rate * (t_event.sum() + t_cens.sum()))


def _nll_mixture(params, t_event, t_cens):
    """Two-LATENT-CLASS mixture: each sample is (unobserved) "fast" (rate l1, prob pi) or "slow" (rate
    l2, prob 1-pi) — NOT the same as a single population with hazard pi*l1+(1-pi)*l2. Survival is the
    pi-weighted AVERAGE of the two exponential survival curves: S(t) = pi*exp(-l1 t) + (1-pi)*exp(-l2 t)."""
    logit_pi, log_l1, log_l2 = params
    pi = 1.0 / (1.0 + np.exp(-logit_pi))
    l1, l2 = np.exp(log_l1), np.exp(log_l2)
    f_event = pi * l1 * np.exp(-l1 * t_event) + (1 - pi) * l2 * np.exp(-l2 * t_event)
    s_cens = pi * np.exp(-l1 * t_cens) + (1 - pi) * np.exp(-l2 * t_cens)
    ll = np.sum(np.log(np.clip(f_event, 1e-300, None))) + np.sum(np.log(np.clip(s_cens, 1e-300, None)))
    return -ll


def fit_mixture(t_event, t_cens, rate_guess, seed=0):
    """MLE for the 2-component mixture above via bounded numerical optimisation (L-BFGS-B, several
    restarts — fixed + random — to dodge local optima / the fast<->slow label-switching symmetry).
    log-rates are BOUNDED to [1e-6, 1e4] x rate_guess: an unconstrained fit can wander to a "rate ~ 0"
    boundary (a degenerate but technically-maximal "cure fraction that never recurs"), which is a
    legitimate MLE point but not a stable or meaningfully-reportable number (its own half-time blows up
    to ~1e13+) — bounding keeps the fit in a regime where "slow component" still means a real, if slow,
    rate rather than a numerical artifact of an unbounded optimizer.
    Returns pi, l1 (fast, l1>=l2 enforced), l2 (slow), and the achieved NLL (for AIC/BIC vs the single-rate
    model)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.log(rate_guess * 1e-4 + 1e-12), np.log(rate_guess * 1e4)
    bounds = [(-6.0, 6.0), (lo, hi), (lo, hi)]   # logit_pi, log_l1, log_l2
    starts = [(0.25, 6.0, 0.3), (0.15, 10.0, 0.5), (0.4, 4.0, 0.15), (0.3, 3.0, 1.0)]
    starts += [(rng.uniform(0.1, 0.6), rng.uniform(2, 15), rng.uniform(0.05, 1.5)) for _ in range(8)]
    best = None
    for pi0, m1, m2 in starts:
        x0 = np.array([np.log(pi0 / (1 - pi0)), np.clip(np.log(rate_guess * m1), lo, hi),
                       np.clip(np.log(rate_guess * m2), lo, hi)])
        res = minimize(_nll_mixture, x0, args=(t_event, t_cens), method="L-BFGS-B", bounds=bounds)
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    logit_pi, log_l1, log_l2 = best.x
    pi = 1.0 / (1.0 + np.exp(-logit_pi)); l1, l2 = np.exp(log_l1), np.exp(log_l2)
    if l1 < l2:
        l1, l2, pi = l2, l1, 1 - pi
    return dict(pi=pi, l1=l1, l2=l2, nll=best.fun)


def _km(durations, events):
    """Simple Kaplan-Meier product-limit estimator. Returns (times, survival) step-function arrays."""
    d = np.asarray(durations); e = np.asarray(events)
    uniq_times = np.unique(d[e == 1])
    surv = 1.0
    times_out, surv_out = [0.0], [1.0]
    for ut in uniq_times:
        n_risk = int(np.sum(d >= ut))
        n_event = int(np.sum((d == ut) & (e == 1)))
        if n_risk > 0:
            surv *= (1 - n_event / n_risk)
        times_out.append(float(ut)); surv_out.append(surv)
    return np.array(times_out), np.array(surv_out)


def _surv_at(times, surv, t):
    """Right-continuous step-function lookup: the KM survival value in effect at time t (i.e. just
    before the next drop), used to place censoring tick marks ON the step curve itself."""
    idx = np.searchsorted(times, t, side="right") - 1
    return surv[np.clip(idx, 0, len(surv) - 1)]


def _rate_window(t_event, t_cens, tlo, thi):
    """Piecewise-exponential (constant-hazard) rate MLE restricted to the interval [tlo, thi), the
    standard "person-time in a window" construction: any sample whose observed time (event or censoring)
    is <= tlo was no longer at risk during the window at all (excluded); everyone else is at risk from
    tlo onward and contributes exposure = min(t, thi) - tlo, with an event counted only if it's an actual
    WGD-recurrence event AND falls inside the window (t <= thi) -- otherwise it's treated as censored at
    thi for this window's purposes. This lets us fit a separate rate to the early, steep part of the
    curve and a separate rate to the late, shallow tail, directly on person-time (rather than eyeballing
    a slope), instead of the single global-MLE mixture fit."""
    t_all = np.concatenate([t_event, t_cens])
    e_all = np.concatenate([np.ones(len(t_event)), np.zeros(len(t_cens))])
    m = t_all > tlo
    t_all, e_all = t_all[m], e_all[m]
    exposure = np.minimum(t_all, thi) - tlo
    is_event = (e_all == 1) & (t_all <= thi)
    n_events = int(is_event.sum())
    n_censored_contrib = int(m.sum() - n_events)
    fit = _mle(n_events, float(exposure.sum()))
    return fit, n_events, n_censored_contrib, float(exposure.sum())


def _nll_pi_only(logit_pi, l1, l2, t_event, t_cens):
    pi = 1.0 / (1.0 + np.exp(-logit_pi))
    f_event = pi * l1 * np.exp(-l1 * t_event) + (1 - pi) * l2 * np.exp(-l2 * t_event)
    s_cens = pi * np.exp(-l1 * t_cens) + (1 - pi) * np.exp(-l2 * t_cens)
    ll = np.sum(np.log(np.clip(f_event, 1e-300, None))) + np.sum(np.log(np.clip(s_cens, 1e-300, None)))
    return -ll


def fit_segmented_mixture(t_event, t_cens, l1, l2):
    """Combine the two INDEPENDENTLY-fit early/late rates (l1, l2, from `_rate_window`) into one mixture
    survival curve S(t) = pi*exp(-l1 t) + (1-pi)*exp(-l2 t), solving for the single free parameter pi (the
    "right proportion") by 1-D MLE against the FULL dataset -- l1 and l2 are held fixed at their
    segment-fit values, only the mixing weight is optimised. This is deliberately NOT the same as the
    fully-jointly-optimised 3-parameter mixture above: it's a two-stage estimate (rates fixed from
    disjoint early/late windows, then a proportion fit on top), closer to what the user asked for
    ("fit the tail and the head separately, then combine with the right proportions") than a single
    global optimiser that's free to trade the two rates against each other."""
    from scipy.optimize import minimize_scalar
    lo, hi = min(l1, l2), max(l1, l2)
    res = minimize_scalar(lambda lp: _nll_pi_only(lp, hi, lo, t_event, t_cens),
                          bounds=(-10, 10), method="bounded")
    pi = 1.0 / (1.0 + np.exp(-res.x))
    return dict(pi=pi, l1=hi, l2=lo, nll=res.fun)


def make(save=True, stem="e7_wgd_recurrence_model"):
    df = _per_sample_table()

    agg = (df.groupby("tumor_type").agg(n_wgd_pos=("k", "size"), n_events=("n_extra", "sum"),
                                        exposure=("exposure", "sum")).reset_index())
    all_t = pd.DataFrame({"tumor_type": _all_types()})
    agg = all_t.merge(agg, on="tumor_type", how="left").fillna(
        {"n_wgd_pos": 0, "n_events": 0, "exposure": 0.0})

    fits = []
    for _, r in agg.iterrows():
        m = _mle(r["n_events"], r["exposure"])
        fits.append(dict(tumor_type=r["tumor_type"], n_wgd_pos=int(r["n_wgd_pos"]),
                         n_events=int(r["n_events"]), exposure=float(r["exposure"]), **m))
    fits = pd.DataFrame(fits)
    pooled = _mle(df["n_extra"].sum(), df["exposure"].sum())
    pooled_n = int(df["n_extra"].sum())

    # Make the censoring treatment explicit rather than leaving it implicit in the exposure formula:
    # single-WGD samples (k=1) contribute ZERO events but their full (1-w1) remaining molecular time as
    # right-censored exposure — e.g. a lone WGD at time 0.6 contributes 0.4 of "at risk, nothing observed
    # yet" time to the denominator. They are the MAJORITY of the exposure, not a minor correction.
    n_censored = int((df["k"] == 1).sum()); n_multi = int((df["k"] >= 2).sum())
    exp_censored = float(df.loc[df["k"] == 1, "exposure"].sum())
    exp_multi = float(df.loc[df["k"] >= 2, "exposure"].sum())

    # Naive comparator: the raw median OBSERVED first-to-second gap (t2), using ONLY samples that actually
    # recurred (event2==1) -- no censoring adjustment at all, i.e. exactly what you'd get by eyeballing the
    # multi-WGD samples alone. Shown alongside the properly-censored MLE half-time so the size of the
    # survivorship-bias gap between the two is visible directly on the plot, per type and pooled, instead
    # of requiring an ad-hoc calculation to notice it (as Ovary.AdenoCA's ~0.28 naive vs ~1.6 fitted showed).
    naive = (df[df.event2 == 1].groupby("tumor_type")["t2"].agg(naive_median="median", naive_n="size")
            .reset_index())
    fits = fits.merge(naive, on="tumor_type", how="left")
    pooled_naive_median = float(df.loc[df.event2 == 1, "t2"].median())
    pooled_naive_n = int((df.event2 == 1).sum())

    shown = fits[fits.n_events >= MIN_EVENTS].sort_values("half_time").reset_index(drop=True)
    hidden = fits[fits.n_events < MIN_EVENTS].sort_values(["n_events", "n_wgd_pos"], ascending=False
                                                          ).reset_index(drop=True)

    n_rows = len(shown) + len(hidden) + 2
    fig_h = 0.30 * n_rows + 1.6
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(190 * MM, fig_h), gridspec_kw={"width_ratios": [1.3, 1]})

    # Point estimates / CI upper bounds routinely blow out to 10-16+ molecular-time units for
    # sparse types (e.g. n=3 events) — plausible given the data, but a linear axis that accommodates the
    # single worst-case CI makes every other row unreadable. Cap the visible axis at XMAX and mark any
    # point/CI that extends past it with an arrow + its real value instead of silently cutting it off.
    XMAX = 2.0

    def _errbar(ax, y, pt, lo, hi, fmt, color, ms):
        pt_c = min(pt, XMAX * 0.985) if np.isfinite(pt) else XMAX * 0.985
        lo_c = min(max(lo, 0.0) if np.isfinite(lo) else 0.0, pt_c)
        hi_c = max(min(hi, XMAX) if np.isfinite(hi) else XMAX, pt_c)
        ax.errorbar([pt_c], [y], xerr=[[pt_c - lo_c], [hi_c - pt_c]], fmt=fmt, color=color, ms=ms,
                   capsize=2, lw=1.1 if fmt == "D" else 0.9, zorder=3)
        over_pt = np.isfinite(pt) and pt > XMAX
        over_hi = (not np.isfinite(hi)) or hi > XMAX
        if over_pt or over_hi:
            hi_txt = "inf" if not np.isfinite(hi) else f"{hi:.1f}"
            pt_txt = f"pt={pt:.1f}, " if over_pt else ""
            ax.text(XMAX * 0.99, y, f" ⟶ {pt_txt}CI to {hi_txt}", fontsize=4.4, color=color,
                   va="center", ha="left", clip_on=False)

    def _naive_marker(ax, y, med, n):
        if not (np.isfinite(med) and n >= 2):
            return
        ax.plot([min(med, XMAX * 0.985)], [y], marker="x", color="#888", ms=4.5, mew=1.1, zorder=4)

    y = 0; yticks = []; ylabels = []
    _errbar(axA, y, pooled["half_time"], pooled["lo"], pooled["hi"], "D", "#111", 5)
    _naive_marker(axA, y, pooled_naive_median, pooled_naive_n)
    yticks.append(y); ylabels.append(f"ALL TYPES POOLED (n={pooled_n})")
    y += 1.4
    axA.axhline(y - 0.7, color="#ccc", lw=0.6)

    for _, r in shown.iterrows():
        lo = r["lo"] if np.isfinite(r["lo"]) else 0.0
        hi = r["hi"] if np.isfinite(r["hi"]) else np.inf
        _errbar(axA, y, r["half_time"], lo, hi, "o", WGD_COL, 4)
        _naive_marker(axA, y, r["naive_median"], r["naive_n"] if np.isfinite(r["naive_n"]) else 0)
        yticks.append(y); ylabels.append(f"{LAB.get(r['tumor_type'], r['tumor_type'])} (n={r['n_events']})")
        y += 1

    if len(hidden):
        y += 0.4
        axA.axhline(y - 0.7, color="#eee", lw=0.6)
        for _, r in hidden.iterrows():
            yticks.append(y)
            ylabels.append(f"{LAB.get(r['tumor_type'], r['tumor_type'])} "
                           f"(n={r['n_events']}, {r['n_wgd_pos']} WGD+, too few)")
            y += 1

    axA.set_yticks(yticks); axA.set_yticklabels(ylabels, fontsize=5.3, color=TXT)
    axA.set_ylim(y - 0.4, -0.8); axA.tick_params(length=0)
    axA.axvline(1.0, color="#bbb", lw=0.6, ls="--")
    axA.set_xlim(0, XMAX)
    axA.set_xlabel("Half-time to next WGD (molecular-time units)", fontsize=7, color=TXT)
    axA.set_title("WGD recurrence half-time by tumour type\n(Poisson/exponential model, 95% CI)",
                 fontsize=7.2, loc="left", color=TXT)
    axA.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    axA.legend(handles=[Line2D([], [], marker="o", ls="", color=WGD_COL, ms=4,
                              label="fitted half-time (censored MLE)"),
                        Line2D([], [], marker="x", ls="", color="#888", ms=4.5, mew=1.1,
                              label="naive median gap (recurred samples only, uncensored)")],
              fontsize=5.0, frameon=False, loc="upper right")

    # Two-component ("bimodal") mixture fit, POOLED data only — per-type samples are too sparse to
    # support 3 extra parameters reliably. Compared to the single-rate model via AIC AND BIC (both fit by
    # MLE on the SAME censored likelihood, just 1 vs 3 free parameters — BIC's ln(n) penalty is much
    # harsher than AIC's flat 2 once n is in the hundreds, so the two criteria can disagree).
    t_event = df.loc[df.event2 == 1, "t2"].to_numpy()
    t_cens = df.loc[df.event2 == 0, "t2"].to_numpy()
    n_obs = len(t_event) + len(t_cens)
    mix = fit_mixture(t_event, t_cens, pooled["rate"])
    nll_single = _nll_single(pooled["rate"], t_event, t_cens)
    aic_single, bic_single = 2 * 1 + 2 * nll_single, 1 * np.log(n_obs) + 2 * nll_single
    if mix is not None:
        aic_mixture = 2 * 3 + 2 * mix["nll"]; bic_mixture = 3 * np.log(n_obs) + 2 * mix["nll"]
    else:
        aic_mixture = bic_mixture = np.inf
    mixture_wins = mix is not None and aic_mixture < aic_single - 2   # >=2 AIC units = meaningfully better

    # Segmented two-rate fit: fit the STEEP head [0, EARLY_CUT) and the SHALLOW tail [LATE_CUT, 1]
    # separately (each its own person-time Poisson-rate MLE, properly left-truncating anyone whose
    # event/censoring already happened before the window starts -- see `_rate_window`), then combine the
    # two FIXED rates into one mixture curve by fitting only the mixing proportion pi against the full
    # dataset. This directly answers "does the head decay faster than the tail" without letting a single
    # global optimiser trade the two rates against each other.
    early_fit, n_early_ev, n_early_cens, exp_early = _rate_window(t_event, t_cens, 0.0, EARLY_CUT)
    late_fit, n_late_ev, n_late_cens, exp_late = _rate_window(t_event, t_cens, LATE_CUT, 1.0)
    seg = None
    if np.isfinite(early_fit["rate"]) and np.isfinite(late_fit["rate"]) and early_fit["rate"] > 0 and late_fit["rate"] > 0:
        seg = fit_segmented_mixture(t_event, t_cens, early_fit["rate"], late_fit["rate"])

    times, surv = _km(df["t2"].to_numpy(), df["event2"].to_numpy())
    axB.step(times, surv, where="post", color="#111", lw=1.2, label="Kaplan–Meier (observed)")
    # Censoring tick marks (the classic "+" survival-plot convention): one per single-WGD sample, placed
    # on the KM step curve at its own censoring time. Directly visualises that the (majority) censored
    # single-WGD samples ARE being used -- they're not silently dropped, they set where/how long the risk
    # set stays populated between event times.
    cens_t = t_cens
    cens_y = _surv_at(times, surv, cens_t)
    axB.plot(cens_t, cens_y, "+", color="#111", ms=4.5, mew=0.7, alpha=0.55, zorder=2,
            label=f"censored (single-WGD, n={len(cens_t)})")
    tt = np.linspace(0, 1, 200)
    axB.plot(tt, np.exp(-pooled["rate"] * tt), color=WGD_COL, lw=1.3, ls="--",
            label=f"single exponential (t½={pooled['half_time']:.2f}), AIC={aic_single:.1f}")
    if mix is not None:
        s_mix = mix["pi"] * np.exp(-mix["l1"] * tt) + (1 - mix["pi"]) * np.exp(-mix["l2"] * tt)
        t_half_fast = np.log(2) / mix["l1"]
        # The "slow" component's rate often gets pushed to (or near) its optimisation lower bound — i.e.
        # the MLE wants an effectively-zero-hazard "never recurs" fraction. Its half-time is then a
        # bound-dependent artifact, not a meaningful number, so report it qualitatively past a threshold
        # instead of a spuriously precise decimal.
        slow_desc = "t½≈never" if mix["l2"] * 1 < 0.02 else f"t½={np.log(2) / mix['l2']:.2f}"
        star = " *preferred*" if mixture_wins else ""
        axB.plot(tt, s_mix, color=MIX_COL, lw=1.3, ls=":",
                label=f"2-comp mixture{star} (fast t½={t_half_fast:.2f}, slow {slow_desc})")
    if seg is not None:
        s_seg = seg["pi"] * np.exp(-seg["l1"] * tt) + (1 - seg["pi"]) * np.exp(-seg["l2"] * tt)
        th_early, th_late = np.log(2) / seg["l1"], np.log(2) / seg["l2"]
        axB.plot(tt, s_seg, color=SEG_COL, lw=1.3, ls="-.",
                label=f"segmented fit (head t½={th_early:.2f}, tail t½={th_late:.2f}, "
                      f"{seg['pi']*100:.0f}% head-wt)")
    axB.set_xlabel("Time since 1st WGD (molecular-time units)", fontsize=7, color=TXT)
    axB.set_ylabel("P(no 2nd WGD observed yet)", fontsize=7, color=TXT)
    axB.set_title("Model check: pooled survival to 2nd WGD", fontsize=7.2, loc="left", color=TXT)
    axB.legend(fontsize=4.6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.22))
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.text(0.02, 0.04, f"{n_censored} single-WGD samples: censored\n(contribute {exp_censored:.0f} "
            f"exposure-units, 0 events)\n{n_multi} multi-WGD samples: {pooled_n} observed events\n"
            f"+ {exp_multi:.0f} exposure-units", transform=axB.transAxes, fontsize=5.3, color=TXT,
            ha="left", va="bottom")

    fig.tight_layout()
    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    print(f"  pooled: n_events={pooled_n}, rate={pooled['rate']:.3f}/unit, "
         f"half_time={pooled['half_time']:.3f} [{pooled['lo']:.3f}, {pooled['hi']:.3f}]")
    print(f"  pooled naive median gap (recurred samples only, n={pooled_naive_n}, uncensored): "
         f"{pooled_naive_median:.3f}")
    print(f"  exposure breakdown: {n_censored} single-WGD samples contribute {exp_censored:.1f} "
         f"censored exposure-units (no 2nd WGD observed); {n_multi} multi-WGD samples contribute "
         f"{exp_multi:.1f} exposure-units + all {pooled_n} observed events")
    print(f"  {len(shown)} types with >= {MIN_EVENTS} events shown with a fit; "
         f"{len(hidden)} types too sparse")
    if mix is not None:
        print(f"  pooled model comparison (n={n_obs}): single-exponential AIC={aic_single:.1f} "
             f"BIC={bic_single:.1f}  vs  2-component-mixture AIC={aic_mixture:.1f} BIC={bic_mixture:.1f}")
        print(f"    -> {'mixture preferred by AIC (delta>=2)' if mixture_wins else 'single exponential not clearly beaten by AIC'}; "
             f"{'mixture' if bic_mixture < bic_single else 'single'} preferred by BIC "
             f"(harsher penalty, more conservative)")
        slow_desc = "essentially never recurs" if mix["l2"] < 0.02 else f"t½={np.log(2)/mix['l2']:.3f}"
        print(f"    mixture: {mix['pi']*100:.0f}% fast component (t½={np.log(2)/mix['l1']:.3f}), "
             f"{(1-mix['pi'])*100:.0f}% slow component ({slow_desc})")
    print(f"  segmented fit: head [0,{EARLY_CUT}) rate={early_fit['rate']:.3f}/unit "
         f"(t½={early_fit['half_time']:.3f}, n_events={n_early_ev}, n_at_risk={n_early_ev + n_early_cens}, "
         f"exposure={exp_early:.1f})")
    print(f"                 tail [{LATE_CUT},1] rate={late_fit['rate']:.3f}/unit "
         f"(t½={late_fit['half_time']:.3f}, n_events={n_late_ev}, n_at_risk={n_late_ev + n_late_cens}, "
         f"exposure={exp_late:.1f})")
    if seg is not None:
        print(f"    combined (pi fit on full data, rates fixed): {seg['pi']*100:.0f}% head-rate / "
             f"{(1-seg['pi'])*100:.0f}% tail-rate, NLL={seg['nll']:.1f} "
             f"(vs single-exp NLL={nll_single:.1f}, joint-mixture NLL={mix['nll']:.1f})")
    return fig


if __name__ == "__main__":
    make()
    print("e7_wgd_recurrence_model: PNG+SVG written")
