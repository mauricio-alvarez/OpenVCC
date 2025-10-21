#!/bin/bash

#----------------------------------------------------------------#
# Slurm Directives
#----------------------------------------------------------------#
#SBATCH -J vcc_tiny     # New job name
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o vcc_output_tiny%j.log
#SBATCH -e vcc_error_tiny%j.log

#----------------------------------------------------------------#
# Environment Setup
#----------------------------------------------------------------#
echo "Job started on $(hostname) at $(date)"
module load miniconda/3.0
conda activate VCC # Using 'source activate' is more robust for scripts
echo "Environment is ready."
echo "--------------------------------------------------------"

#----------------------------------------------------------------#
# --- NEW: Sanity Check for Dataset Path ---
#----------------------------------------------------------------#
echo "--- STARTING SANITY CHECKS ---"

# 1. Define the paths we are going to use.
#    THIS IS THE PATH WE ARE DEBUGGING. MAKE SURE IT IS CORRECT.
DATASET_PATH="$HOME/tesis/imagenet_val/imagenet-val"
TARGET_CLASS_NAME="sports_car"
TARGET_CLASS="n04285008"
FULL_TARGET_PATH="$DATASET_PATH/$TARGET_CLASS"

echo "Checking for dataset path: $DATASET_PATH"
echo "Checking for target class directory: $FULL_TARGET_PATH"
echo ""

# 2. Check if the target directory actually exists and is a directory.
if [ -d "$FULL_TARGET_PATH" ]; then
    echo "SUCCESS: The target directory was found!"
    echo "Listing a few files from it to confirm it's not empty:"
    ls "$FULL_TARGET_PATH" | head -n 5
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "FATAL ERROR: The target directory was NOT FOUND at the path:"
    echo "$FULL_TARGET_PATH"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
    echo "The script cannot continue. Please fix the DATASET_PATH variable."
    echo "Listing the contents of the parent directory ($DATASET_PATH) to help you debug:"
    ls -l "$DATASET_PATH" | head -n 10
    exit 1 # Stop the script here!
fi

echo "--- SANITY CHECKS PASSED ---"
echo "--------------------------------------------------------"

#----------------------------------------------------------------#
# Execute the Python Script (only runs if sanity checks pass)
#----------------------------------------------------------------#
# Set TORCH_HOME to use the pre-downloaded model
export TORCH_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HUB_OFFLINE=1
# Change to the project directory
cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

echo "Starting the VCC experiment..."

# Use the variables we defined and checked above
python run_vcc.py \
    --target_class "$TARGET_CLASS_NAME" \
    --model_to_run tiny_vit \
    --feature_names 2 5 8 11 \
    --imagenet_path "$DATASET_PATH"

echo "Python script finished."

#----------------------------------------------------------------#
# Cleanup
#----------------------------------------------------------------#
conda deactivate
module unload miniconda/3.0
echo "Job finished at $(date)"

