"""
Summary statistics for the allpcawg run (output_sbs1_soft_allpcawg).
Produces per-sample event table and per-tumor-type summary.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT_DIR   = Path("dev/pcawg/output_sbs1_soft_allpcawg")
META_FILE = Path(os.path.join(os.environ.get("TAU_PCAWG_INPUT_DIR", "."), "MuSiCal_assignment_PCAWG.txt"))
OUT_FIG   = Path("dev/pcawg/summary_allpcawg.png")
OUT_TSV   = Path("dev/pcawg/summary_allpcawg_per_sample.tsv")

# ------------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------------
meta = pd.read_csv(META_FILE, sep="\t", usecols=["aliquot_id", "tumor_type", "MMRD_POLE_category"])
meta = meta.rename(columns={"aliquot_id": "sample_id"})
print(f"Metadata: {len(meta)} samples, {meta['tumor_type'].nunique()} tumor types")

# ------------------------------------------------------------------
# Load all time_clusters.tsv files
# ------------------------------------------------------------------
rows = []
for sample_dir in sorted(OUT_DIR.iterdir()):
    sample_id = sample_dir.name
    tsv = sample_dir / f"{sample_id}.time_clusters.tsv"
    if not tsv.exists():
        continue
    df = pd.read_csv(tsv, sep="\t")
    # filter out Private cluster
    events = df[df["classification"] != "Private"]
    n_wgd = int((events["classification"] == "WGD").sum())
    n_pgd = int((events["classification"] == "PGD").sum())
    wgd_times = sorted(events.loc[events["classification"] == "WGD", "time"].tolist())
    pgd_times = sorted(events.loc[events["classification"] == "PGD", "time"].tolist())
    rows.append({
        "sample_id": sample_id,
        "n_wgd": n_wgd,
        "n_pgd": n_pgd,
        "n_events": n_wgd + n_pgd,
        "wgd_times": wgd_times,
        "pgd_times": pgd_times,
        "wgd_time_1": wgd_times[0] if wgd_times else np.nan,
        "wgd_time_2": wgd_times[1] if len(wgd_times) > 1 else np.nan,
        "pgd_time_1": pgd_times[0] if pgd_times else np.nan,
    })

samples = pd.DataFrame(rows)
samples = samples.merge(meta, on="sample_id", how="left")
print(f"Loaded {len(samples)} samples")
print(f"  Missing tumor type: {samples['tumor_type'].isna().sum()}")

# ------------------------------------------------------------------
# Overall stats
# ------------------------------------------------------------------
print("\n=== Overall event counts ===")
print(samples[["n_wgd", "n_pgd", "n_events"]].describe().round(3))

print("\n=== WGD count distribution ===")
print(samples["n_wgd"].value_counts().sort_index())

print("\n=== PGD count distribution ===")
print(samples["n_pgd"].value_counts().sort_index())

n_any_wgd = (samples["n_wgd"] >= 1).sum()
n_any_pgd = (samples["n_pgd"] >= 1).sum()
n_both    = ((samples["n_wgd"] >= 1) & (samples["n_pgd"] >= 1)).sum()
n_neither = ((samples["n_wgd"] == 0) & (samples["n_pgd"] == 0)).sum()
print(f"\nAny WGD:      {n_any_wgd} / {len(samples)} ({100*n_any_wgd/len(samples):.1f}%)")
print(f"Any PGD:      {n_any_pgd} / {len(samples)} ({100*n_any_pgd/len(samples):.1f}%)")
print(f"Both WGD+PGD: {n_both}    ({100*n_both/len(samples):.1f}%)")
print(f"Neither:      {n_neither} ({100*n_neither/len(samples):.1f}%)")

# ------------------------------------------------------------------
# Per-tumor-type summary
# ------------------------------------------------------------------
def pct(x): return 100 * x.mean()

tt = (samples.groupby("tumor_type")
      .agg(
          n_samples    = ("sample_id", "count"),
          pct_wgd      = ("n_wgd",  lambda x: 100*(x>=1).mean()),
          pct_pgd      = ("n_pgd",  lambda x: 100*(x>=1).mean()),
          pct_double_wgd = ("n_wgd", lambda x: 100*(x>=2).mean()),
          mean_wgd_time  = ("wgd_time_1", "mean"),
          median_wgd_time= ("wgd_time_1", "median"),
      )
      .reset_index()
      .sort_values("pct_wgd", ascending=False))

print("\n=== Per-tumor-type summary (sorted by WGD rate) ===")
print(tt.to_string(index=False))

# ------------------------------------------------------------------
# Save per-sample table
# ------------------------------------------------------------------
samples.drop(columns=["wgd_times", "pgd_times"]).to_csv(OUT_TSV, sep="\t", index=False)
print(f"\nPer-sample table saved: {OUT_TSV}")

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"PCAWG allpcawg run  (n={len(samples)} samples)", fontsize=14)

# 1. WGD rate by tumor type (top 20 by sample count)
ax = axes[0, 0]
top_tt = tt[tt["n_samples"] >= 10].sort_values("pct_wgd", ascending=True).tail(25)
ax.barh(top_tt["tumor_type"], top_tt["pct_wgd"], color="steelblue")
ax.set_xlabel("% samples with WGD")
ax.set_title("WGD rate by tumor type (n≥10)")
ax.axvline(100*n_any_wgd/len(samples), color="red", linestyle="--", linewidth=0.8, label="overall")
ax.legend(fontsize=8)

# 2. PGD rate by tumor type
ax = axes[0, 1]
top_tt2 = tt[tt["n_samples"] >= 10].sort_values("pct_pgd", ascending=True).tail(25)
ax.barh(top_tt2["tumor_type"], top_tt2["pct_pgd"], color="coral")
ax.set_xlabel("% samples with PGD")
ax.set_title("PGD rate by tumor type (n≥10)")
ax.axvline(100*n_any_pgd/len(samples), color="red", linestyle="--", linewidth=0.8, label="overall")
ax.legend(fontsize=8)

# 3. WGD timing distribution
ax = axes[0, 2]
wgd_t = samples["wgd_time_1"].dropna()
ax.hist(wgd_t, bins=40, color="steelblue", edgecolor="white", linewidth=0.3)
ax.set_xlabel("WGD timing (t)")
ax.set_ylabel("# samples")
ax.set_title(f"WGD timing distribution (n={len(wgd_t)})\nmean={wgd_t.mean():.3f}, median={wgd_t.median():.3f}")
ax.axvline(wgd_t.median(), color="red", linestyle="--")

# 4. PGD timing distribution
ax = axes[1, 0]
pgd_t = samples["pgd_time_1"].dropna()
ax.hist(pgd_t, bins=40, color="coral", edgecolor="white", linewidth=0.3)
ax.set_xlabel("PGD timing (t)")
ax.set_ylabel("# samples")
ax.set_title(f"PGD timing distribution (n={len(pgd_t)})\nmean={pgd_t.mean():.3f}, median={pgd_t.median():.3f}")
ax.axvline(pgd_t.median(), color="red", linestyle="--")

# 5. Event count distribution
ax = axes[1, 1]
vc = samples["n_events"].value_counts().sort_index()
ax.bar(vc.index.astype(str), vc.values, color="mediumpurple")
ax.set_xlabel("# events detected")
ax.set_ylabel("# samples")
ax.set_title("Event count distribution")
for i, (k, v) in enumerate(vc.items()):
    ax.text(i, v+5, f"{v}", ha="center", fontsize=8)

# 6. Double WGD rate by tumor type
ax = axes[1, 2]
top_dbl = tt[tt["n_samples"] >= 10].sort_values("pct_double_wgd", ascending=True).tail(20)
ax.barh(top_dbl["tumor_type"], top_dbl["pct_double_wgd"], color="navy")
ax.set_xlabel("% samples with ≥2 WGD")
ax.set_title("Double-WGD rate by tumor type (n≥10)")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Figure saved: {OUT_FIG}")
