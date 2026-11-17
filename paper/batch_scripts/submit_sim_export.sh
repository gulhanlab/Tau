#!/bin/bash
#SBATCH -p park
#SBATCH -A park_contrib
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH -J sim_export
#SBATCH --array=1-242
#SBATCH -o sim_export_logs/%A_%a.out
R=${TAU_REPO:-$(git rev-parse --show-toplevel)}
mkdir -p "$R/paper/batch_scripts/sim_export_logs"
PATH="$R/.pixi/envs/default/bin:$PATH" python "$R/paper/batch_scripts/export_sim_chunk.py" ${SLURM_ARRAY_TASK_ID} 10
