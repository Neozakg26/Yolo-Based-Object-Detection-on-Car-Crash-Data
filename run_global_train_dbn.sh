#!/bin/bash
#SBATCH --job-name=glob_train_dbn
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/global_train_dbn_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/global_train_dbn_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "Starting global Train DBN"

python3 -m execute.train_global_dbn --results_dir /datasets/nmaja/CrashBest/results --meta_csv /datasets/nmaja/CrashBest/results/Crash_Table.csv --features_path /datasets/nmaja/CrashBest/results/global_features_graph.pkl --out_model /datasets/nmaja/CrashBest/results/global_risk_model.parquet --train_classifiers

echo "Finished Global Train DBN"
