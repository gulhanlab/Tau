#!/bin/sh
#SBATCH -p park
#SBATCH -A park_contrib
#SBATCH -t 10-00:00:00
#SBATCH --job-name=run_monopogen
#SBATCH --mem=500M
#SBATCH -e %j.err
#SBATCH -o %j.out

snakemake all --unlock --use-conda --executor slurm --default-resources slurm_account=park_contrib slurm_partition=park runtime=14400 --jobs 300 --rerun-triggers mtime --retries 3 
