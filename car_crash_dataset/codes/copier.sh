##DEAD BASH FILE. Decided a simpler approach


# set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

# echo "creating training directory" 
# mkdir -p ../train

# echo "Copying Images from $1 to training_directory"
# cp -a $1 ../train

# echo "Copying annotations from $2 to training_directory"
# cp -a  $2 ../train


# echo "Verifying copy..."
# if [ ! -d "../train" ] || [ -z "$(ls -A ../train)" ]; then
#     echo "ERROR: Training directory is empty after copy"
#     exit 1
# fi

# echo "SUCCESS: Training directory prepared"
# exit 0