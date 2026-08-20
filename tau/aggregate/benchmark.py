"""tau.aggregate.benchmark — recovered clustering ⨝ simulation truth → bench_* tables.

Shippable cohort tool (`tau benchmark`): evaluate Tau against a simulated cohort with known truth. Sims run
through `tau run`/`export-tables`/`aggregate` exactly like real samples (same `tau_*` schema, recovered
timing) and are clustered like real samples; the only extra input is `{SIM}.truth.json` per sample. This
module parses truth and joins it to the recovered clustering, emitting two complementary event tables:

  bench_events.tsv        one row per CALLED event (Tau's clustering output + match flag). Drives
                          false-positives and matched-timing accuracy.
  bench_truth_events.tsv  one row per TRUE event (from truth.json) + whether Tau detected it, plus
                          event_size (affected genome fraction) and min_dist (to nearest other true event).
                          Drives RECALL / false-negatives — impossible from a calls-only table.

Mirrors the truth-matching in dev/benchmarking/plot_figure3_benchmarking.py exactly.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# hg19 autosome sizes — event_size denominator (matches the dev benchmark)
HG19_SIZES = {"1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260,
              "6": 171115067, "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747,
              "11": 135006516, "12": 133851895, "13": 115169878, "14": 107349540, "15": 102531392,
              "16": 90354753, "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
              "21": 48129895, "22": 51304566}
GENOME_SIZE = sum(HG19_SIZES.values())
MATCH_TOL = 0.15        # detection matching tolerance (molecular-time units) — dev value
MAX_TRUE_EVENTS = 3     # samples with >3 true WGD/PGD events are excluded (dev convention)


def sim_truth_rows(truth: dict) -> list[dict]:
    """Parse one `{SIM}.truth.json` → one row per TRUE WGD/PGD event, with event_size + min_dist.

    event_size = fraction of the (hg19-autosome) genome in segments whose cumulative gain time sits within
    0.05 of the event and that are amplified (major>=2) — i.e. how much genome the event visibly affects.
    min_dist   = molecular-time distance to the nearest OTHER true event in the sample (NaN if only one).
    """
    wgd_pgd = [e for e in truth.get("genome_events", []) if e["type"] in ("WGD", "ccPG", "PGD")]
    ev_times = [float(e["time"]) for e in wgd_pgd]
    n_true = len(wgd_pgd)

    seg_info = {}
    for ts in truth.get("segments", []):
        t_vals = ts.get("t_values", [])
        cum = list(np.cumsum(t_vals[:-1])) if len(t_vals) >= 2 else []
        try:
            _, coords = ts["seg_id"].split(":")
            s, e = coords.split("-")
            length = int(e) - int(s)
        except Exception:
            length = 0
        seg_info[ts["seg_id"]] = {"major": ts.get("major", 0), "cum": cum, "length": length}

    rows = []
    for i, ev in enumerate(wgd_pgd):
        t_event = float(ev["time"])
        det_len = sum(info["length"] for info in seg_info.values()
                      if info["major"] >= 2 and any(abs(c - t_event) < 0.05 for c in info["cum"]))
        other = [ev_times[j] for j in range(n_true) if j != i]
        rows.append({
            "true_type": ev["type"], "true_time_frac": t_event,
            "event_size_frac": det_len / GENOME_SIZE,
            "min_dist_frac": float(min(abs(t_event - ot) for ot in other)) if other else np.nan,
        })
    return rows


def build_bench_tables(sim_run_dir: Path, recovered_tsv: Path, out_dir: Path) -> dict[str, Path]:
    """Join the recovered clustering (`recovered_tsv` = the cohort clustering output over the sim run, with
    columns sample, event_time, classification, matched, true_time, true_type, timing_error) against each
    `{SIM}.truth.json` under `sim_run_dir` → bench_events.tsv (calls) + bench_truth_events.tsv (truth).
    Only samples that finished Tau (have a truth.json + clustering) and have <=3 true events are scored."""
    sim_run_dir, out_dir = Path(sim_run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_info = {}
    for sd in sorted(p for p in sim_run_dir.iterdir() if p.is_dir()):
        tr = sd / f"{sd.name}.truth.json"
        if not tr.exists() or not (sd / f"{sd.name}.time_clusters.tsv").exists():
            continue          # skip samples that never finished Tau (else their events count as all-missed)
        rows = sim_truth_rows(json.loads(tr.read_text()))
        if len(rows) <= MAX_TRUE_EVENTS:
            truth_info[sd.name] = rows
    valid = set(truth_info)

    # ── bench_events: Tau's calls, restricted to scored samples ──────────────────────────────
    calls = pd.read_csv(recovered_tsv, sep="\t")
    calls = calls[calls["sample"].isin(valid)].rename(columns={
        "event_time": "event_time_frac", "true_time": "true_time_frac",
        "timing_error": "timing_error_frac", "matched": "is_matched", "gf": "genome_frac"})
    calls_path = out_dir / "bench_events.tsv"
    calls.to_csv(calls_path, sep="\t", index=False)

    # ── bench_truth_events: every true event + whether Tau detected it ────────────────────────
    matched = calls[(calls["is_matched"] == True) & calls["classification"].isin(["WGD", "ccPG", "PGD"])].copy()
    matched["true_time_frac"] = pd.to_numeric(matched["true_time_frac"], errors="coerce")
    matched["event_time_frac"] = pd.to_numeric(matched["event_time_frac"], errors="coerce")
    trecs = []
    for sample, events in truth_info.items():
        sub = matched[matched["sample"] == sample]
        for ev in events:
            hit = sub[(sub["true_time_frac"] - ev["true_time_frac"]).abs() < 0.01]
            detected = bool(len(hit))
            rec_t = float(hit["event_time_frac"].iloc[0]) if detected else np.nan
            trecs.append({"sample": sample, **ev, "is_detected": detected,
                          "recovered_time_frac": rec_t,
                          "timing_error_frac": (rec_t - ev["true_time_frac"]) if detected else np.nan})
    truth_path = out_dir / "bench_truth_events.tsv"
    pd.DataFrame(trecs).to_csv(truth_path, sep="\t", index=False)

    return {"bench_events": calls_path, "bench_truth_events": truth_path}
