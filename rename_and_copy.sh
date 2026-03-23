#!/bin/bash
#SBATCH --job-name=renamer
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/rename_copy_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/rename_copy_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

MAX_JOBS=4
SOURCE_DIR=/datasets/nmaja/bdd100k/images/seg_track_20/train
TARGET_DIR=/datasets/nmaja/CrashBest/normal_files

run_scene () {
    local SCENE_DIR="$1"
    local SCENE_NAME
    SCENE_NAME=$(basename "$SCENE_DIR")
    echo "Starting $SCENE_NAME"
    python3 -m accident_detect.bddk_converter --path "$SCENE_DIR" --output "$TARGET_DIR"
    echo "Finished $SCENE_NAME"
}

export -f run_scene
export TARGET_DIR

for SCENE_DIR in "$SOURCE_DIR"/*/; do
    [ -d "$SCENE_DIR" ] || continue
    run_scene "$SCENE_DIR" &

    # limit to 4 concurrent jobs
    if (( $(jobs -r | wc -l) >= MAX_JOBS )); then
        wait -n
    fi
done

wait  # wait for remaining jobs
echo "All scenes complete."
