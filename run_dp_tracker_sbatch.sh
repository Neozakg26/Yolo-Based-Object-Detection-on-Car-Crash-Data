#!/bin/bash
#SBATCH --job-name=tracker
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_errors.txt
#SBATCH --array=0-119%6
N=$(wc -l < scenes.txt)
#SBATCH --array=0-$(($N-1))%6 run_tracker.sbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

SCENE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" scenes.txt)

echo "Processing scene: $SCENE"

python -m execute_tracker --path "$SCENE"