#!/bin/bash
#SBATCH --job-name=training_yolo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/cluster_project_location/sbatch_results.txt
#SBATCH --error=/cluster_project_location/sbatch_errors.txt
#SBATCH --partition=bigbatch
sleep 30

torchrun --nproc_per_node=2 execute.py --job train
