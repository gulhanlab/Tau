#!/bin/bash
#SBATCH -p park
#SBATCH -A park_contrib
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH -J backfill
#SBATCH --array=1-271
#SBATCH -o backfill_logs/%A_%a.out
R=${TAU_REPO:-$(git rev-parse --show-toplevel)}
mkdir -p "$R/paper/batch_scripts/backfill_logs"
PATH="$R/.pixi/envs/default/bin:$PATH" python "$R/paper/batch_scripts/backfill_chunk.py" ${SLURM_ARRAY_TASK_ID} 10
