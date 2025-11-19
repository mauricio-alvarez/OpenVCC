#!/bin/bash

#----------------------------------------------------------------#
# Slurm Directives
#----------------------------------------------------------------#
#SBATCH -J vcc_visualize_all    # Job name for the visualization pipeline
#SBATCH -p gpu                  # Still need a GPU node for model loading
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH -o logs/visualize_%A_%a.log  # Save logs in a 'logs' directory
#SBATCH -e logs/visualize_%A_%a.err  # %A is job ID, %a is task ID

# --- Job Array: 4 layers x 2 submodules = 8 total jobs ---
#SBATCH --array=0-7

#----------------------------------------------------------------#
# Create Output Directories
#----------------------------------------------------------------#
echo "--- Setting up main output directories ---"
mkdir -p logs
mkdir -p final_connectomes
echo "--- Setup complete ---"

#----------------------------------------------------------------#
# Define Experiment Parameters for the Array
#----------------------------------------------------------------#
# Define all the layers and submodules you have already analyzed
LAYERS=(5 11 17 23 5 11 17 23)
SUBMODULES=(attn attn attn attn mlp mlp mlp mlp)

# Get the specific parameters for THIS job task from the arrays
LAYER_TO_RUN=${LAYERS[$SLURM_ARRAY_TASK_ID]}
SUBMODULE_TO_RUN=${SUBMODULES[$SLURM_ARRAY_TASK_ID]}

echo "--- Starting VCC Visualization Task ---"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Target Layer: $LAYER_TO_RUN"
echo "Target Submodule: $SUBMODULE_TO_RUN"
echo "----------------------------------------"

#----------------------------------------------------------------#
# Environment and Execution
#----------------------------------------------------------------#
module load miniconda/3.0
conda activate VCC_final

# Set cache variables for offline model loading (gen_vcc.py loads the model)
export TORCH_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HUB_OFFLINE=1

cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

# Define the working directory where the results are already saved
WORKING_DIR="outputs/4v_large_vit_sports_car_L${LAYER_TO_RUN}_${SUBMODULE_TO_RUN}"

# --- STEP 1: Check if the required 'cd.pkl' file exists ---
if [ -f "$WORKING_DIR/cd.pkl" ]; then
    echo "--- Found cd.pkl in $WORKING_DIR. Running gen_vcc.py ---"
    python gen_vcc.py --working_dir "$WORKING_DIR"
    
    # --- STEP 2: Organize the final plot ---
    if [ -f "$WORKING_DIR/vcc.png" ]; then
        FINAL_PLOT_DIR="final_connectomes"
        NEW_FILENAME="vcc_L${LAYER_TO_RUN}_${SUBMODULE_TO_RUN}.png"
        mv "$WORKING_DIR/vcc.png" "$FINAL_PLOT_DIR/$NEW_FILENAME"
        echo "--- Successfully generated and moved plot to $FINAL_PLOT_DIR/$NEW_FILENAME ---"
    else
        echo "--- WARNING: gen_vcc.py ran but vcc.png was not found in $WORKING_DIR. ---"
    fi
else
    echo "--- ERROR: cd.pkl not found in $WORKING_DIR. Skipping this task. ---"
fi

echo "--- Job Task $SLURM_ARRAY_TASK_ID Finished ---"
