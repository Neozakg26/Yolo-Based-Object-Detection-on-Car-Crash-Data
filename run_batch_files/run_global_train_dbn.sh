#!/bin/bash
#SBATCH --job-name=glob_train_dbn
#SBATCH --output=/cluster_project_location/global_train_dbn_results.txt
#SBATCH --error=/cluster_project_location/global_train_dbn_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "Starting global Train DBN"

python3 -m execute.train_global_dbn --results_dir /cluster_db_location/CrashBest/results --meta_csv /cluster_db_location/CrashBest/Crash_Table.csv --features_path /cluster_db_location/CrashBest/results/global_features_graph.pkl --out_model /cluster_db_location/CrashBest/results/global_risk_model.parquet --train_classifiers --preacc_only

echo "Finished Global Train DBN"
