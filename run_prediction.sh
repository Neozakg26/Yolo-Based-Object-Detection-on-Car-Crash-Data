#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/global_prediction.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/global_prediction_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "Starting Accident Prediction"

python -m execute.predict_accident --results_dir C:/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/CCD_images/results --model_path  C:/Users/neokg/Coding_Projects/yolo-detector/cluster_results/results/global_risk_model.parquet --scene_labels C:/Users/neokg/Coding_Projects/yolo-detector/scene_index.csv

echo "Fnishing Accident Prediction"
