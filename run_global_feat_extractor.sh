#!/bin/bash
#SBATCH --job-name=feat_extractor
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_feat_extr_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_feat_extr_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

MAX_JOBS=4   # how many scenes to run at once

run_scene () {
    echo "Starting global Features Extractor"
    python3 -m execute.global_features_extract --results_dir /datasets/nmaja/CrashBest/results --out_path /datasets/nmaja/CrashBest/results/global_features_graph.pkl  --tau_max 1 --pc_alpha 0.01 --fdr_q 0.01 --min_effect 0.20 
    echo "Finished Global Features Extractor"
}

export -f run_scene

# for i in $(seq 1 1500); do
#     run_scene $i &

#     # limit to 4 concurrent jobs
#     if (( $(jobs -r | wc -l) >= MAX_JOBS )); then
#         wait -n
#     fi
# done

wait  # wait for remaining jobs
echo "All scenes complete."
