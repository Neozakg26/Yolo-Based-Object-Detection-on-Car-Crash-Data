#!/bin/bash
#SBATCH --job-name=feat_extractor
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_feat_extr_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_feat_extr_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

MAX_JOBS=4   # how many scenes to run at once

run_scene () {
    SCENE_NUM=$(printf "%06d" $1)
    SCENE="C_${SCENE_NUM}_"
    echo "Starting $SCENE"
    python3 -m execute.extract_features --track_path "C:/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/CCD_images/results/${SCENE_NUM}tracks.parquet" --env_path "C:/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/CCD_images/results/${SCENE_NUM}env.parquet"
    echo "Finished $SCENE"
}

export -f run_scene

for i in $(seq 1 1500); do
    run_scene $i &

    # limit to 4 concurrent jobs
    if (( $(jobs -r | wc -l) >= MAX_JOBS )); then
        wait -n
    fi
done

wait  # wait for remaining jobs
echo "All scenes complete."
