#!/bin/bash
#SBATCH --job-name=feat_extractor
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/global_feat_extractor_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/global_feat_extractor_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

DATASET="/datasets/nmaja/CrashBest/results"
MODEL_PATH="/datasets/nmaja/CrashBest/results/global_features_graph.pkl"

echo "Starting global Features Extractor"
python3 -m execute.global_features_extract --results_dir $DATASET --out_path $MODEL_PATH  --tau_max 2 --pc_alpha 0.01 --fdr_q 0.01 --min_effect 0.20 
echo "Finished Global Features Extractor"
wait  # wait for remaining jobs
echo "All scenes complete."
