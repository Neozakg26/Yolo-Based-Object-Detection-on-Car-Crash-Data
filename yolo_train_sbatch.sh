#!/bin/bash
#SBATCH--job-name=training_yolo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH--output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_results.txt
#SBATCH --error=logs/%x_%j.err
sleep 30

torchrun --nproc_per_node=2 execute_training.py