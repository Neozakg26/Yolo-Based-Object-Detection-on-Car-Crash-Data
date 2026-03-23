#!/bin/bash
# Script to safely copy a folder to a destination
# Usage: ./copy_folder.sh /path/to/source /path/to/destination

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print usage
usage() {
    echo "Usage: $0 <source_folder> <destination_folder>"
    exit 1
}

# Validate arguments
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

SOURCE="$1"
DEST="$2"

# Check if source exists and is a directory
if [ ! -d "$SOURCE" ]; then
    echo "Error: Source folder '$SOURCE' does not exist or is not a directory."
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DEST" ]; then
    echo "Destination '$DEST' does not exist. Creating..."
    mkdir -p "$DEST"
fi

# Perform the copy with verbose output and preserve attributes
# -a : archive mode (preserves permissions, timestamps, symlinks, etc.)
# -v : verbose output
# -n : no overwrite of existing files (optional safety)
echo "Copying '$SOURCE' to '$DEST'..."
cp -avn "$SOURCE" "$DEST"

# Check if copy succeeded
if [ $? -eq 0 ]; then
    echo "✅ Copy completed successfully."
else
    echo "❌ Copy failed."
    exit 1
fi

