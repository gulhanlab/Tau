#!/bin/bash
#SBATCH --job-name=analytical_conv
#SBATCH --partition=park
#SBATCH --account=park_contrib
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --array=0-32
#SBATCH --output=logs/analytical_conv_%A_%a.out
#SBATCH --error=logs/analytical_conv_%A_%a.err

set -euo pipefail
mkdir -p logs

PYTHON=/n/data1/hms/dbmi/park/jbrew/Tau/Tau_github/.pixi/envs/default/bin/python
export PATH="/n/data1/hms/dbmi/park/jbrew/Tau/Tau_github/.pixi/envs/default/bin:$PATH"
cd /n/data1/hms/dbmi/park/jbrew/Tau/Tau_github

# All 33 CN states in order (matches sorted glob of solutions/*.sobj)
STATES=(
  1_0 1_1
  2_0 2_1 2_2
  3_0 3_1 3_2 3_3
  4_0 4_1 4_2 4_3 4_4
  5_0 5_1 5_2 5_3 5_4 5_5
  6_0 6_1 6_2 6_3 6_4 6_5 6_6
  7_0 7_1 7_2 7_3 7_4 7_5
)

STATE=${STATES[$SLURM_ARRAY_TASK_ID]}
echo "Converting state: $STATE"

$PYTHON tau/data/convert_solutions_to_analytical.py --state $STATE

echo "Done: $STATE"
