#!/bin/bash
#SBATCH --job-name=snakemake_pipeline     # Job name
#SBATCH --partition=park                  # Partition name
#SBATCH --account=park_contrib            # Account name
#SBATCH --mem=1G                         # Total memory for the job
#SBATCH --time=7-00:00:00                 # Time limit (7 days)
#SBATCH --output=snakemake_logs/%x_%j.out # Standard output log file
#SBATCH --error=snakemake_logs/%x_%j.err  # Standard error log file

snakemake all --use-conda --executor slurm --default-resources slurm_account=park_contrib slurm_partition=park runtime=14400 --jobs 300 --rerun-triggers mtime --retries 3 --rerun-incomplete --dry-run
