#!/bin/bash
#SBATCH --job-name=agg_graphs
#SBATCH --output=/cluster_project_location/aggregate_out.txt
#SBATCH --error=/cluster_project_location/aggregate_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

echo "Starting Aggregate Graphs"

DATASET="/cluster_db_location/CrashBest/results"
OUTPUT="/cluster_db_location/CrashBest/results/agg_results"

python3 -m aggregate_causal_graphs --results_dir $DATASET --output_dir  $OUTPUT --min_scenes 70

echo "Fnishing Aggregate Graphs"
