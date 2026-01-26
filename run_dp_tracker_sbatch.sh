#!/bin/bash
#SBATCH --job-name=tracker
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_errors.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "SLURM TASK ID: $SLURM_ARRAY_TASK_ID"

# Convert number → 6-digit zero padded
SCENE_NUM=$(printf "%06d" $SLURM_ARRAY_TASK_ID)
SCENE="C_${SCENE_NUM}_"

echo "Processing scene: $SCENE"

python3 -m execute_tracker --path "$SCENE"
