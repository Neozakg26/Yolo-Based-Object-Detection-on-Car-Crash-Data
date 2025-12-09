$ #!/bin/bash

DEST_IMAGES="/c/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/test_images"
DEST_LABELS="/c/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/test_labels"

echo "Copying matching JSON files..."

for img in "$DEST_IMAGES"/*; do
    base=$(basename "$img")
    name="${base%.*}"
    json_file="${name}.json"

    if [[ -f "$json_file" ]]; then
        cp "$json_file" "$DEST_LABELS"
        echo "Copied $json_file"
    else
        echo "Missing JSON for: $name"
    fi
done
