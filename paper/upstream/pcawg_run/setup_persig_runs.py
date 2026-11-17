#!/usr/bin/env python3
"""
Generate per-signature Tau run configs and SLURM submission scripts.

For each non-SBS1/SBS5 COSMIC signature (or combined group) with enough
exposure in PCAWG:
  - Q1 is computed over non-zero samples of the signature SUM
  - Samples with sum > Q1 are included in that run
  - Component signatures are not run individually when part of a group

Canonical groups (always run together):
  SBS2+SBS13       — APOBEC cytosine deamination
  SBS17a+SBS17b    — 5-FU / oxidative damage
  SBS7a+7b+7c+7d   — UV damage

Run from repo root.
Outputs:
  dev/pcawg/configs_persig/config_{NAME}.yaml
  dev/pcawg/configs_persig/run_{NAME}.sh
  dev/pcawg/configs_persig/manifest.tsv
  dev/pcawg/submit_all_persig.sh
"""
import os
import textwrap
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

EXPOSURES    = os.path.join(os.environ.get("TAU_PCAWG_INPUT_DIR", "."), "MuSiCal_assignment_PCAWG.txt")
EXCLUDE_SIGS = {"SBS1", "SBS5"}   # clock sigs already covered by allpcawg run
MIN_NZ       = 20                  # min non-zero samples to include
OUT_DIR      = Path("dev/pcawg/configs_persig")

# Canonical combinations: run name → [component sigs]
# Components are excluded from solo runs.
COMBOS = {
    "SBS2_SBS13":    ["SBS2",  "SBS13"],
    "SBS17a_SBS17b": ["SBS17a", "SBS17b"],
    "SBS7":          ["SBS7a",  "SBS7b", "SBS7c", "SBS7d"],
}
COMBO_MEMBERS = {s for members in COMBOS.values() for s in members}

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(EXPOSURES, sep="\t")
print(f"Loaded {len(df)} samples, {sum(c.startswith('SBS') for c in df.columns)} SBS signatures")

all_sbs = [c for c in df.columns if c.startswith("SBS") and c not in EXCLUDE_SIGS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_job(sigs):
    """Return (q1, n_nonzero, n_pass, aliquots) for a list of signatures."""
    summed = df[[s for s in sigs if s in df.columns]].sum(axis=1)
    nz     = summed[summed > 0]
    if len(nz) < MIN_NZ:
        return None
    q1       = float(np.percentile(nz, 25))
    aliquots = df.loc[summed > q1, "aliquot_id"].tolist()
    return q1, len(nz), len(aliquots), aliquots


def run_script(name):
    base = os.environ.get("TAU_REPO", ".")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=tau_{name}",
        "#SBATCH --partition=park",
        "#SBATCH --account=park_contrib",
        "#SBATCH --mem=2G",
        "#SBATCH --time=72:00:00",
        f"#SBATCH --output={base}/dev/pcawg/snakemake_logs/tau_{name}_%j.out",
        f"#SBATCH --error={base}/dev/pcawg/snakemake_logs/tau_{name}_%j.err",
        "",
        f"mkdir -p {base}/dev/pcawg/snakemake_logs",
        f"cd {base}/dev/pcawg",
        "",
        'export PATH="$HOME/.pixi/bin:$PATH"',
        "",
        f"pixi run --manifest-path {base}/pyproject.toml \\",
        "    snakemake all \\",
        f"    --configfile configs_persig/config_{name}.yaml \\",
        "    --executor slurm \\",
        "    --default-resources \\",
        "        slurm_account=park_contrib \\",
        "        slurm_partition=park \\",
        "        runtime=720 \\",
        "    --jobs 100 \\",
        "    --rerun-triggers mtime \\",
        "    --retries 3 \\",
        "    --rerun-incomplete \\",
        "    --nolock",
        "",
    ]
    return "\n".join(lines)


def write_job(name, sigs, result):
    q1, n_nz, n_pass, aliquots = result
    cfg = {
        "outdir":        f"output_persig_{name}",
        "signatures":    sigs,
        "weight_mode":   "soft",
        "focus_samples": aliquots,
        "focus_tumors":  [],
    }
    cfg_path = OUT_DIR / f"config_{name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    sh_path = OUT_DIR / f"run_{name}.sh"
    sh_path.write_text(run_script(name))
    sh_path.chmod(0o755)

    return dict(name=name, sigs="+".join(sigs), q1=round(q1, 1),
                n_nonzero=n_nz, n_pass_q1=n_pass,
                config=str(cfg_path), run_script=str(sh_path))


# ── 1. Combined groups ────────────────────────────────────────────────────────
manifest_rows = []
print("\nCombined groups:")
for name, sigs in sorted(COMBOS.items()):
    result = make_job(sigs)
    if result is None:
        print(f"  SKIP {name}  (<{MIN_NZ} non-zero samples)")
        continue
    row = write_job(name, sigs, result)
    manifest_rows.append(row)
    print(f"  {name:22s}  [{', '.join(sigs)}]  q1={row['q1']:.0f}  n_pass={row['n_pass_q1']}")

# ── 2. Solo signatures (not in any combo) ────────────────────────────────────
print("\nSolo signatures:")
for sig in sorted(all_sbs):
    if sig in COMBO_MEMBERS:
        continue
    result = make_job([sig])
    if result is None:
        continue
    row = write_job(sig, [sig], result)
    manifest_rows.append(row)
    print(f"  {sig:22s}  q1={row['q1']:.0f}  n_pass={row['n_pass_q1']}")

# ── Write manifest ────────────────────────────────────────────────────────────
manifest = pd.DataFrame(manifest_rows).sort_values("n_pass_q1", ascending=False)
manifest.to_csv(OUT_DIR / "manifest.tsv", sep="\t", index=False)

total_runs = manifest["n_pass_q1"].sum()
n_jobs     = len(manifest)
print(f"\n{n_jobs} jobs → {total_runs:,} total sample-runs")
print(f"Configs + scripts written to: {OUT_DIR}/")

# ── Master submit script ──────────────────────────────────────────────────────
submit_lines = [
    "#!/bin/bash",
    "# Submit all per-signature Tau runs to SLURM",
    "# Run from repo root: bash dev/pcawg/submit_all_persig.sh",
    f"# {n_jobs} Snakemake coordinators  →  {total_runs:,} total sample-runs",
    "",
]
for _, row in manifest.iterrows():
    submit_lines.append(f"sbatch {row['run_script']}")
submit_text = "\n".join(submit_lines) + "\n"

for path in [Path("dev/pcawg/submit_all_persig.sh"), Path("submit_all_persig.sh")]:
    path.write_text(submit_text)
    path.chmod(0o755)

print(f"Master submit script: dev/pcawg/submit_all_persig.sh")
print(f"\nTo launch:  bash dev/pcawg/submit_all_persig.sh")
