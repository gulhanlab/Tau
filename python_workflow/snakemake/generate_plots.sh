#!/bin/bash
#SBATCH -p short                  # Partition (queue)
#SBATCH -A park                   # Account
#SBATCH -t 1:00:00                # Time limit (1 hour)
#SBATCH --mem=3G                # Memory per job
#SBATCH -e generate_plots_logs/%A_%a_generate_plots.err # Error file (job array-specific)
#SBATCH -o generate_plots_logs/%A_%a_generate_plots.out # Output file (job array-specific)
#SBATCH --array=1-109             # Job array: 109 tasks (25 files per task)

# Directory containing the files
INPUT_DIR="output/clonal_multiplicities"

# Get the list of files
FILES=($(ls $INPUT_DIR))

# Number of files per job array task
FILES_PER_TASK=25

# Calculate the start and end index for this task
START=$(( ($SLURM_ARRAY_TASK_ID - 1) * FILES_PER_TASK ))

#a=2
#START=$(( ($a - 1) * FILES_PER_TASK ))
END=$(( START + FILES_PER_TASK - 1 ))

# Ensure the end index does not exceed the number of files
if [ $END -ge ${#FILES[@]} ]; then
    END=$((${#FILES[@]} - 1))
fi

# Loop through the files assigned to this task
for ((i=START; i<=END; i++)); do
    FILE="${FILES[$i]}"
    echo "Processing file: $FILE"
    python evaluate_N_counts_SBS1.py "$INPUT_DIR/$FILE"
done
