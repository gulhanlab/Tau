"""Fig 4e9 — time to WGD, CONDITIONAL on the sample ever having one: restricted to the 705 WGD+ samples
only, using each one's own observed first-WGD time w1 as fully-observed (UNCENSORED) data -- no
"never-WGD" samples included at all.

MOTIVATION: e8 treats every WGD-negative sample as right-censored -- "hasn't happened yet, but might if we
followed it further" -- which assumes WGD-negative tumours are drawn from the SAME at-risk population as
WGD-positive ones, just observed to a shorter effective depth. That's the standard survival-analysis
assumption, but it may not hold biologically: a WGD-negative sample might instead belong to a
fundamentally different population that will never undergo genome doubling regardless of how long you
watch it (a "cure fraction", in survival-analysis language) -- e.g. because it never acquires whatever
driver event triggers doubling in the first place. If that's the right picture, treating those samples as
merely censored members of one shared hazard mis-specifies the model and (as seen in e8) can swing the
fitted half-time a great deal depending on how the "never" population is counted.

This script sidesteps that assumption entirely by only asking the question WITHIN the population that
demonstrably CAN and DOES undergo WGD: given a sample is going to have (at least) one WGD, when does it
happen? Every included sample's w1 is a real, fully-observed event time -- there is no censoring in this
subset by construction, so the fitted rate is simply n_events / sum(w1), no person-time bookkeeping needed
beyond that.

Each row of the forest plot now carries THREE comparison points, not one:
  - fitted median, WGD-only samples (teal circle/diamond + 95% CI)          -- this script's main estimate
  - raw empirical median, WGD-only samples (grey x, no CI)                  -- sanity check: does the
    exponential shape actually describe these observations, or is the fit being pulled around by it?
  - fitted median, ALL samples i.e. e8's full-cohort estimate (purple square + 95% CI) -- included
    specifically to show that excluding the "never-WGD" samples (this script) vs including them as
    censored (e8) makes WGD look like it happens substantially FASTER/EARLIER: restricting to WGD+ samples
    only removes all the long, event-free exposure time that drags the full-cohort rate down.

Same Poisson/exponential MLE + 2-component-mixture + Kaplan-Meier machinery as e7/e8 (imported directly).

Run:
    PATH=".pixi/envs/default/bin:$PATH" .pixi/envs/default/bin/python e9_wgd_only_timing_model.py
Writes e9_wgd_only_timing_model.{png,svg} to ./output/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_arm_patterns_segment import LAB, MM, TXT, OUT, _events_deduped, _all_types, plt   # noqa: E402
from e7_wgd_recurrence_model import (_mle, _nll_single, fit_mixture, _km, _surv_at,   # noqa: E402
                                     WGD_COL, MIX_COL)
from e8_first_wgd_timing_model import _first_wgd_table, _fit_table   # noqa: E402

MIN_EVENTS = 3   # minimum pooled WGD+ samples required to report a per-type half-time (else "too few")
ALL_COL = "#7b3294"   # "all samples, full cohort" comparator (e8's estimate), distinct from WGD_COL/MIX_COL


def _wgd_only_table():
    """One row per WGD+ sample: tumor_type, time = w1 (first WGD time), event = 1 always (no censoring --
    every sample here is, by construction, one that demonstrably underwent WGD)."""
    ev = _events_deduped()
    types_ok = set(_all_types())
    ev = ev[ev.tumor_type.isin(types_ok)].copy()
    tt = ev.drop_duplicates("sample").set_index("sample")["tumor_type"]
    wgd = ev[ev.classification == "WGD"].copy()
    wgd["time"] = wgd["time"].astype(float)
    w1 = wgd.groupby("sample")["time"].min()
    rows = [dict(sample=s, tumor_type=tt[s], time=float(t), event=1) for s, t in w1.items()]
    return pd.DataFrame(rows)


def make(save=True, stem="e9_wgd_only_timing_model"):
    df = _wgd_only_table()
    fits, pooled, pooled_n = _fit_table(df)

    raw = df.groupby("tumor_type")["time"].median().rename("raw_median").reset_index()
    fits = fits.merge(raw, on="tumor_type", how="left")
    pooled_raw_median = float(df["time"].median())

    df_all = _first_wgd_table()
    fits_all, pooled_all, pooled_all_n = _fit_table(df_all)
    fits = fits.merge(fits_all[["tumor_type", "half_time", "lo", "hi"]].rename(
        columns={"half_time": "all_half_time", "lo": "all_lo", "hi": "all_hi"}), on="tumor_type", how="left")

    shown = fits[fits.n_events >= MIN_EVENTS].sort_values("half_time").reset_index(drop=True)
    hidden = fits[fits.n_events < MIN_EVENTS].sort_values("n_events", ascending=False).reset_index(drop=True)

    n_rows = len(shown) + len(hidden) + 2
    fig_h = 0.30 * n_rows + 1.6
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(190 * MM, fig_h), gridspec_kw={"width_ratios": [1.3, 1]})

    # XMAX must accommodate the ALL-SAMPLES comparator too, which routinely exceeds 1 (its exposure
    # includes the long, event-free "never had a WGD" population) -- e.g. Prost.AdenoCA's all-samples
    # estimate is ~36. Same 2.0 cap + truncation-arrow convention as e7/e8.
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
            ax.text(XMAX * 0.99, y, f" ⟶ {pt_txt}CI to {hi_txt}", fontsize=4.2, color=color,
                   va="center", ha="left", clip_on=False)

    def _naive_marker(ax, y, med):
        if np.isfinite(med):
            ax.plot([min(med, XMAX * 0.985)], [y], marker="x", color="#888", ms=4.5, mew=1.1, zorder=4)

    y = 0; yticks = []; ylabels = []
    _errbar(axA, y, pooled["half_time"], pooled["lo"], pooled["hi"], "D", "#111", 5)
    _naive_marker(axA, y, pooled_raw_median)
    _errbar(axA, y, pooled_all["half_time"], pooled_all["lo"], pooled_all["hi"], "s", ALL_COL, 4)
    yticks.append(y); ylabels.append(f"ALL TYPES POOLED (n={pooled_n})")
    y += 1.4
    axA.axhline(y - 0.7, color="#ccc", lw=0.6)

    for _, r in shown.iterrows():
        lo = r["lo"] if np.isfinite(r["lo"]) else 0.0
        hi = r["hi"] if np.isfinite(r["hi"]) else np.inf
        _errbar(axA, y, r["half_time"], lo, hi, "o", WGD_COL, 4)
        _naive_marker(axA, y, r["raw_median"])
        alo = r["all_lo"] if np.isfinite(r["all_lo"]) else 0.0
        ahi = r["all_hi"] if np.isfinite(r["all_hi"]) else np.inf
        _errbar(axA, y, r["all_half_time"], alo, ahi, "s", ALL_COL, 3.5)
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
    axA.set_xlabel("Median/half-time to WGD (molecular-time units)", fontsize=6.6, color=TXT)
    axA.set_title("Time-to-WGD, WGD+ samples only vs. all samples\n(Poisson/exponential model, 95% CI)",
                 fontsize=7.2, loc="left", color=TXT)
    axA.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    axA.legend(handles=[
        Line2D([], [], marker="o", ls="", color=WGD_COL, ms=4, label="fitted, WGD+ samples only"),
        Line2D([], [], marker="x", ls="", color="#888", ms=4.5, mew=1.1, label="raw median, WGD+ samples only"),
        Line2D([], [], marker="s", ls="", color=ALL_COL, ms=4, label="fitted, ALL samples (full cohort, e8)"),
    ], fontsize=4.6, frameon=False, loc="upper right")

    t_event = df["time"].to_numpy()
    t_cens = np.array([])   # no censoring in this subset by construction
    n_obs = len(t_event)
    mix = fit_mixture(t_event, t_cens, pooled["rate"])
    nll_single = _nll_single(pooled["rate"], t_event, t_cens)
    aic_single, bic_single = 2 * 1 + 2 * nll_single, 1 * np.log(n_obs) + 2 * nll_single
    if mix is not None:
        aic_mixture = 2 * 3 + 2 * mix["nll"]; bic_mixture = 3 * np.log(n_obs) + 2 * mix["nll"]
    else:
        aic_mixture = bic_mixture = np.inf
    mixture_wins = mix is not None and aic_mixture < aic_single - 2

    times, surv = _km(df["time"].to_numpy(), df["event"].to_numpy())
    axB.step(times, surv, where="post", color="#111", lw=1.2, label="empirical survival (observed)")
    tt = np.linspace(0, 1, 200)
    axB.plot(tt, np.exp(-pooled["rate"] * tt), color=WGD_COL, lw=1.3, ls="--",
            label=f"single exponential (t½={pooled['half_time']:.2f})")
    if mix is not None:
        s_mix = mix["pi"] * np.exp(-mix["l1"] * tt) + (1 - mix["pi"]) * np.exp(-mix["l2"] * tt)
        t_half_fast = np.log(2) / mix["l1"]
        slow_desc = "t½≈never" if mix["l2"] * 1 < 0.02 else f"t½={np.log(2) / mix['l2']:.2f}"
        star = " *preferred*" if mixture_wins else ""
        axB.plot(tt, s_mix, color=MIX_COL, lw=1.3, ls=":",
                label=f"2-comp mixture{star} ({mix['pi']*100:.0f}% fast t½={t_half_fast:.2f}, "
                      f"{(1-mix['pi'])*100:.0f}% slow {slow_desc})")
    axB.set_xlabel("Molecular time of first WGD", fontsize=7, color=TXT)
    axB.set_ylabel("P(no WGD yet), among WGD+ samples", fontsize=7, color=TXT)
    axB.set_title("Model check: time to WGD, WGD+ samples only\n(no censoring)", fontsize=7.2, loc="left",
                 color=TXT)
    axB.legend(fontsize=4.8, frameon=False, loc="lower left")
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.text(0.98, 0.98, f"{pooled_n} WGD+ samples, ALL fully observed\n(no censoring: every included "
            f"sample DID have a WGD)\nmedian observed w1 = {df['time'].median():.3f}, "
            f"mean = {df['time'].mean():.3f}", transform=axB.transAxes, fontsize=5.3, color=TXT,
            ha="right", va="top")

    fig.tight_layout()
    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    print(f"  pooled (WGD+ only, uncensored): n={pooled_n}, rate={pooled['rate']:.3f}/unit, "
         f"half_time={pooled['half_time']:.3f} [{pooled['lo']:.3f}, {pooled['hi']:.3f}], "
         f"mean={pooled['half_time']/np.log(2):.3f}")
    print(f"  raw w1: median={df['time'].median():.3f}, mean={df['time'].mean():.3f}, "
         f"min={df['time'].min():.3f}, max={df['time'].max():.3f}")
    print(f"  pooled (ALL samples, full cohort, e8): n={pooled_all_n}/{len(df_all)}, "
         f"half_time={pooled_all['half_time']:.3f} [{pooled_all['lo']:.3f}, {pooled_all['hi']:.3f}] "
         f"-- {pooled_all['half_time']/pooled['half_time']:.2f}x slower than WGD-only estimate")
    print(f"  {len(shown)} types with >= {MIN_EVENTS} events shown with a fit; "
         f"{len(hidden)} types too sparse")
    if mix is not None:
        print(f"  pooled model comparison (n={n_obs}): single-exponential AIC={aic_single:.1f} "
             f"BIC={bic_single:.1f}  vs  2-component-mixture AIC={aic_mixture:.1f} BIC={bic_mixture:.1f}")
        print(f"    -> {'mixture preferred by AIC (delta>=2)' if mixture_wins else 'single exponential not clearly beaten by AIC'}; "
             f"{'mixture' if bic_mixture < bic_single else 'single'} preferred by BIC")
        slow_desc = "essentially never (within [0,1])" if mix["l2"] < 0.02 else f"t½={np.log(2)/mix['l2']:.3f}"
        print(f"    mixture: {mix['pi']*100:.0f}% fast component (t½={np.log(2)/mix['l1']:.3f}), "
             f"{(1-mix['pi'])*100:.0f}% slow component ({slow_desc})")
    return fig


if __name__ == "__main__":
    make()
    print("e9_wgd_only_timing_model: PNG+SVG written")
