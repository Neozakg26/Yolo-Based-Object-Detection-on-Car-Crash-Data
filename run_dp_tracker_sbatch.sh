#!/bin/bash
#SBATCH --job-name=tracker
#SBATCH --output=logs/tracker_%A_%a.out
#SBATCH --error=logs/tracker_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

BATCH_FILE=$(ls scene_batch_* | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")

echo "Processing batch: $BATCH_FILE"

while read SCENE; do
    echo "Running scene $SCENE"
    python3 -m execute_tracker --path "$SCENE"
done < "$BATCH_FILE"
