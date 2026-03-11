#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/prediction_out_fold_02.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/prediction_err_fold_02.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "Starting Accident Prediction"

DATASET="/datasets/nmaja/CrashBest/new_results"
MODEL_PATH="/datasets/nmaja/CrashBest/results/eval_plots_cv/fold_05/global_model_fold_05.parquet"
CLASSIFIER="global_model_fold_05.classifier.pkl"
SCENE_CSV="scene_index.csv"

python3 -m execute.predict_accident --results_dir $DATASET --model_path  $MODEL_PATH --scene_labels $SCENE_CSV --k 2 --thr 0.85

echo "Fnishing Accident Prediction"
