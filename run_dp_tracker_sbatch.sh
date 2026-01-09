#!/bin/bash
#SBATCH --job-name=tracker
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_errors.txt
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
