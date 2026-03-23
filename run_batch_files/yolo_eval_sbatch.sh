#!/bin/bash
#SBATCH --job-name=evaluation_yolo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/cluster_project_location/sbatch_eval_results.txt
#SBATCH --error=/cluster_project_location/sbatch_eval_errors.txt
#SBATCH --partition=bigbatch
sleep 30

torchrun --nproc_per_node=1 execute.py --job val