#!/bin/bash
# Submit the benchmarking Snakemake pipeline to SLURM.
# Usage: sbatch runSnakemake.sh [config.yaml]
#
#SBATCH --job-name=tau_benchmark
#SBATCH --partition=park
#SBATCH --account=park_contrib
#SBATCH --mem=1G
#SBATCH --time=12:00:00
# run from the simulation working directory (sbatch --chdir=<dir>)
#SBATCH --output=snakemake_logs/%x_%j.out
#SBATCH --error=snakemake_logs/%x_%j.err

CONFIG=${1:-config.yaml}
echo "Using config: $CONFIG"

mkdir -p snakemake_logs

export PATH="$HOME/.pixi/bin:$PATH"

pixi run --manifest-path ${TAU_REPO:-.}/pyproject.toml snakemake all \
    --configfile "$CONFIG" \
    --executor slurm \
    --default-resources \
        slurm_account=park_contrib \
        slurm_partition=park \
        runtime=720 \
    --jobs 300 \
    --rerun-triggers mtime \
    --retries 3 \
    --rerun-incomplete
