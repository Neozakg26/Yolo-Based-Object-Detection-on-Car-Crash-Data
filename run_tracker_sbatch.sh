#!/bin/bash
#SBATCH --job-name=tracker
#SBATCH --output=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_results.txt
#SBATCH --error=/home-mscluster/nmaja/Yolo-Based-Object-Detection-on-Car-Crash-Data/sbatch_tracker_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigbatch

MAX_JOBS=4

CRASH_ROOT="/datasets/nmaja/CrashBest"
NORMAL_ROOT="/datasets/nmaja/CrashBest/normal_files"
SCENE_CSV="scene_labels.csv"

mkdir -p logs

run_scene () {
    SCENE_ID="$1"

    if [[ "$SCENE_ID" == C_* ]]; then
        SCENE_PATH="${CRASH_ROOT}/C_${SCENE_ID}_"
    else
        SCENE_PATH="${NORMAL_ROOT}/${SCENE_ID}_"
    fi

    MATCH_COUNT=$(ls ${SCENE_PATH}*.jpg 2>/dev/null | wc -l)

    if [ "$MATCH_COUNT" -eq 0 ]; then
        echo "[SKIP] No frames found for ${SCENE_ID} with prefix ${SCENE_PATH}"
        return 0
    fi

    echo "Starting ${SCENE_ID} using prefix ${SCENE_PATH} (${MATCH_COUNT} frames)"
    python3 -m execute.execute_tracker --path "$SCENE_PATH" \
        > "logs/${SCENE_ID}.out" 2> "logs/${SCENE_ID}.err"
    echo "Finished ${SCENE_ID}"
}

export -f run_scene
export CRASH_ROOT
export NORMAL_ROOT

tail -n +2 "$SCENE_CSV" | cut -d',' -f1 | while read -r SCENE_ID; do
    run_scene "$SCENE_ID" &

    if (( $(jobs -r | wc -l) >= MAX_JOBS )); then
        wait -n
    fi
done

wait
echo "All scenes complete."