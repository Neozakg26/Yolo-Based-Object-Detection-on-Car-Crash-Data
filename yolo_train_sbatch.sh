#!/bin/bash
#SBATCH --job-name=training_yolo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_errors.txt
#SBATCH --partition=bigbatch
sleep 30

torchrun --nproc_per_node=1 execute.py --job train
