#!/bin/bash
#SBATCH --job-name=tracker
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_errors.txt
N=$(wc -l < scenes.txt)
#SBATCH --array=0-$(($N-1))%6 run_tracker.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

SCENE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" scenes.txt)

echo "Processing scene: $SCENE"

python -m execute_tracker --path "$SCENE"