#!/bin/bash
#SBATCH --job-name=pred1
#SBATCH --output=/cluster_project_location/prediction_out_fold_01.txt
#SBATCH --error=/cluster_project_location/prediction_err_fold_01.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "Starting Accident Prediction"

DATASET="/cluster_db_location/CrashBest/new_results"
MODEL_PATH="/cluster_db_location/CrashBest/results/eval_plots_cv/fold_01/global_model_fold_01.parquet"
CLASSIFIER="global_model_fold_05.classifier.pkl"
SCENE_CSV="scene_index.csv"

python3 -m execute.predict_accident --results_dir $DATASET --model_path  $MODEL_PATH --scene_labels $SCENE_CSV --k 5 --thr 0.85

echo "Fnishing Accident Prediction"
