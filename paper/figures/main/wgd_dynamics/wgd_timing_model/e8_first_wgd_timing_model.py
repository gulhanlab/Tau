"""Fig 4e8 — modeling the time to a FIRST WGD, as a distinct question from e7's recurrence model.

e7 asks: "given a sample already had one WGD, how fast (if ever) does a second one arrive?" — and
deliberately starts that risk clock at w1 (the first WGD's own time), NOT at molecular time 0, since
folding w1 itself into that same rate would require also including every sample that NEVER got a WGD at
all as a censored observation for THAT clock — otherwise you're conditioning on "already succeeded" and
then treating the conditional sample as if it were an unconditional draw, which inflates the rate (shrinks
the half-time) in the same way ignoring censoring inflated it upward elsewhere, just in the opposite
direction.

This script is that missing, PROPERLY-CENSORED first-WGD analysis: every sample in the cohort (not just
WGD+ ones) contributes either an observed first-WGD time w1 (event) or right-censoring at molecular time 1
(no WGD ever observed). Same Poisson/exponential MLE + 2-component-mixture machinery as e7 (imported
directly, since none of it is specific to "recurrence" vs. "first occurrence" — it's just "time to an
event with censoring").

DATA: 1327 samples across the retained tumour types (via `_all_types()`/`_events_deduped()`), of which 705
(53%) have >=1 WGD (event = their first WGD time w1) and 622 (47%) have none at all (censored at 1.0).

Run:
    PATH=".pixi/envs/default/bin:$PATH" .pixi/envs/default/bin/python e8_first_wgd_timing_model.py
Writes e8_first_wgd_timing_model.{png,svg} to ./output/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_arm_patterns_segment import LAB, MM, TXT, OUT, _events_deduped, _all_types, load_samples, plt   # noqa: E402
from e7_wgd_recurrence_model import (_mle, _nll_single, fit_mixture, _km, _surv_at,   # noqa: E402
                                     WGD_COL, MIX_COL)

MIN_EVENTS = 3   # minimum pooled WGD events required to report a per-type half-time (else "too few")


def _first_wgd_table():
    """One row per sample in the retained cohort (WGD+ AND WGD-negative): tumor_type, time (= w1 if the
    sample ever had a WGD, else 1.0 = right-censored at the full molecular-time window), event (1/0).

    The sample universe MUST come from `load_samples()` (every representative-aliquot sample in the
    retained tumour types), NOT from `_events_deduped()`'s sample list -- the events table only contains
    samples with >=1 recorded (WGD/PGD/chrom_specific) event, so deriving "no WGD" samples from "not in
    the events table" silently drops every sample with a genuinely flat/untimed genome (859 of 2186 here,
    ~39%) from the censored population entirely. That undercounts total exposure substantially and
    inflates the fitted rate (shrinks the half-time) -- the same shape of bias as the survivorship-bias
    issue discussed for the recurrence model, just introduced by an incomplete denominator instead of
    incomplete censoring logic. `has_wgd` in `load_samples()` agrees exactly with the events-derived WGD+
    set (verified: identical 705-sample set either way), so only the "no event at all" samples were ever
    at risk of being dropped."""
    ev = _events_deduped()
    types_ok = set(_all_types())
    wgd = ev[ev.classification == "WGD"].copy()
    wgd["time"] = wgd["time"].astype(float)
    w1 = wgd.groupby("sample")["time"].min()

    samp = load_samples()
    samp = samp[samp.is_representative_aliquot & samp.tumor_type.isin(types_ok)]
    rows = []
    for s, t in zip(samp["sample"], samp["tumor_type"]):
        if s in w1.index:
            rows.append(dict(sample=s, tumor_type=t, time=float(w1.loc[s]), event=1))
        else:
            rows.append(dict(sample=s, tumor_type=t, time=1.0, event=0))
    return pd.DataFrame(rows)


def _fit_table(df, time_col="time", event_col="event"):
    """Generic per-tumor_type + pooled Poisson/exponential MLE fit table from any (time, event) censored
    dataset with a `tumor_type` column -- shared by e8 (all-samples, full-cohort), e9 (WGD-only samples),
    and e10 (same datasets, Weibull fit instead) so the exact same aggregation logic (and `_all_types()`
    zero-fill for types with no observations) isn't duplicated three times."""
    agg = (df.groupby("tumor_type").agg(n_samples=(event_col, "size"), n_events=(event_col, "sum"),
                                        exposure=(time_col, "sum")).reset_index())
    all_t = pd.DataFrame({"tumor_type": _all_types()})
    agg = all_t.merge(agg, on="tumor_type", how="left").fillna(
        {"n_samples": 0, "n_events": 0, "exposure": 0.0})
    fits = []
    for _, r in agg.iterrows():
        m = _mle(r["n_events"], r["exposure"])
        fits.append(dict(tumor_type=r["tumor_type"], n_samples=int(r["n_samples"]),
                         n_events=int(r["n_events"]), exposure=float(r["exposure"]), **m))
    fits = pd.DataFrame(fits)
    pooled = _mle(df[event_col].sum(), df[time_col].sum())
    pooled_n = int(df[event_col].sum())
    return fits, pooled, pooled_n


def make(save=True, stem="e8_first_wgd_timing_model"):
    df = _first_wgd_table()
    fits, pooled, pooled_n = _fit_table(df)

    n_censored = int((df["event"] == 0).sum()); n_event = int((df["event"] == 1).sum())
    exp_censored = float(df.loc[df["event"] == 0, "time"].sum())
    exp_event = float(df.loc[df["event"] == 1, "time"].sum())

    shown = fits[fits.n_events >= MIN_EVENTS].sort_values("half_time").reset_index(drop=True)
    hidden = fits[fits.n_events < MIN_EVENTS].sort_values(["n_events", "n_samples"], ascending=False
                                                          ).reset_index(drop=True)

    n_rows = len(shown) + len(hidden) + 2
    fig_h = 0.30 * n_rows + 1.6
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(190 * MM, fig_h), gridspec_kw={"width_ratios": [1.3, 1]})

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

    y = 0; yticks = []; ylabels = []
    _errbar(axA, y, pooled["half_time"], pooled["lo"], pooled["hi"], "D", "#111", 5)
    yticks.append(y); ylabels.append(f"ALL TYPES POOLED (n={pooled_n}/{len(df)})")
    y += 1.4
    axA.axhline(y - 0.7, color="#ccc", lw=0.6)

    for _, r in shown.iterrows():
        lo = r["lo"] if np.isfinite(r["lo"]) else 0.0
        hi = r["hi"] if np.isfinite(r["hi"]) else np.inf
        _errbar(axA, y, r["half_time"], lo, hi, "o", WGD_COL, 4)
        yticks.append(y); ylabels.append(f"{LAB.get(r['tumor_type'], r['tumor_type'])} "
                                         f"(n={r['n_events']}/{r['n_samples']})")
        y += 1

    if len(hidden):
        y += 0.4
        axA.axhline(y - 0.7, color="#eee", lw=0.6)
        for _, r in hidden.iterrows():
            yticks.append(y)
            ylabels.append(f"{LAB.get(r['tumor_type'], r['tumor_type'])} "
                           f"(n={r['n_events']}/{r['n_samples']}, too few)")
            y += 1

    axA.set_yticks(yticks); axA.set_yticklabels(ylabels, fontsize=5.3, color=TXT)
    axA.set_ylim(y - 0.4, -0.8); axA.tick_params(length=0)
    axA.axvline(1.0, color="#bbb", lw=0.6, ls="--")
    axA.set_xlim(0, XMAX)
    axA.set_xlabel("Half-time to FIRST WGD (molecular-time units)", fontsize=7, color=TXT)
    axA.set_title("Time-to-first-WGD half-time by tumour type\n(Poisson/exponential model, 95% CI)",
                 fontsize=7.2, loc="left", color=TXT)
    axA.spines[["top", "right"]].set_visible(False)

    t_event = df.loc[df.event == 1, "time"].to_numpy()
    t_cens = df.loc[df.event == 0, "time"].to_numpy()
    n_obs = len(t_event) + len(t_cens)
    mix = fit_mixture(t_event, t_cens, pooled["rate"])
    nll_single = _nll_single(pooled["rate"], t_event, t_cens)
    aic_single, bic_single = 2 * 1 + 2 * nll_single, 1 * np.log(n_obs) + 2 * nll_single
    if mix is not None:
        aic_mixture = 2 * 3 + 2 * mix["nll"]; bic_mixture = 3 * np.log(n_obs) + 2 * mix["nll"]
    else:
        aic_mixture = bic_mixture = np.inf
    mixture_wins = mix is not None and aic_mixture < aic_single - 2

    times, surv = _km(df["time"].to_numpy(), df["event"].to_numpy())
    axB.step(times, surv, where="post", color="#111", lw=1.2, label="Kaplan–Meier (observed)")
    cens_y = _surv_at(times, surv, t_cens)
    axB.plot(t_cens, cens_y, "+", color="#111", ms=4.5, mew=0.7, alpha=0.4, zorder=2,
            label=f"censored (no WGD ever, n={len(t_cens)})")
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
    axB.set_xlabel("Molecular time", fontsize=7, color=TXT)
    axB.set_ylabel("P(no WGD observed yet)", fontsize=7, color=TXT)
    axB.set_title("Model check: pooled survival to FIRST WGD", fontsize=7.2, loc="left", color=TXT)
    axB.legend(fontsize=5.0, frameon=False, loc="upper right")
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.text(0.02, 0.04, f"{n_censored} samples: never observed a WGD\n(censored, contribute "
            f"{exp_censored:.0f} exposure-units, 0 events)\n{n_event} samples: WGD observed "
            f"({pooled_n} events)\n+ {exp_event:.0f} exposure-units", transform=axB.transAxes,
            fontsize=5.3, color=TXT, ha="left", va="bottom")

    fig.tight_layout()
    if save:
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        print(f"  saved {stem} -> {OUT}")
    print(f"  pooled: n_events={pooled_n}/{len(df)}, rate={pooled['rate']:.3f}/unit, "
         f"half_time={pooled['half_time']:.3f} [{pooled['lo']:.3f}, {pooled['hi']:.3f}], "
         f"mean={pooled['half_time']/np.log(2):.3f}")
    print(f"  exposure breakdown: {n_censored} samples never had a WGD (contribute {exp_censored:.1f} "
         f"censored exposure-units); {n_event} samples had a WGD (contribute {exp_event:.1f} "
         f"exposure-units + all {pooled_n} events)")
    print(f"  {len(shown)} types with >= {MIN_EVENTS} events shown with a fit; "
         f"{len(hidden)} types too sparse")
    if mix is not None:
        print(f"  pooled model comparison (n={n_obs}): single-exponential AIC={aic_single:.1f} "
             f"BIC={bic_single:.1f}  vs  2-component-mixture AIC={aic_mixture:.1f} BIC={bic_mixture:.1f}")
        print(f"    -> {'mixture preferred by AIC (delta>=2)' if mixture_wins else 'single exponential not clearly beaten by AIC'}; "
             f"{'mixture' if bic_mixture < bic_single else 'single'} preferred by BIC")
        slow_desc = "essentially never gets a WGD" if mix["l2"] < 0.02 else f"t½={np.log(2)/mix['l2']:.3f}"
        print(f"    mixture: {mix['pi']*100:.0f}% fast component (t½={np.log(2)/mix['l1']:.3f}), "
             f"{(1-mix['pi'])*100:.0f}% slow component ({slow_desc})")
    return fig


if __name__ == "__main__":
    make()
    print("e8_first_wgd_timing_model: PNG+SVG written")
