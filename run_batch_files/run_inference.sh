#!/bin/bash
#SBATCH --job-name=glob_inference
#SBATCH --output=/cluster_project_location/global_train_dbn_results.txt
#SBATCH --error=/cluster_project_location/global_train_dbn_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "Starting global Train DBN"

python3 -m execute.infer_scene_with_global_model --model_path C:/Users/neokg/Coding_Projects/yolo-detector/cluster_results/results/global_risk_model.parquet --track_path  C:/Users/neokg/Coding_Projects/yolo-detector/cluster_results/results/001500_tracks.parquet  --env_path   C:/Users/neokg/Coding_Projects/yolo-detector/cluster_results/results/001500_env.parquet  --inference_method belief_propagation

echo "Finished Global Train DBN"