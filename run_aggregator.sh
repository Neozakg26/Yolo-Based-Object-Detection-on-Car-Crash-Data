#!/bin/bash
#SBATCH --job-name=agg_graphs
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/aggregate_out.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/aggregate_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "Starting Aggregate Graphs"

DATASET="/datasets/nmaja/CrashBest/new_results"
OUTPUT="/datasets/nmaja/CrashBest/results/agg_results"

python3 -m explainability.causal_graph_aggregator --results_dir $DATASET --output_dir  $OUTPUT 

echo "Fnishing Aggregate Graphs"
