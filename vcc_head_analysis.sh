#!/bin/bash

#----------------------------------------------------------------#
# Slurm Directives
#----------------------------------------------------------------#
#SBATCH -J vcc_submodule      # Job name
#SBATCH --partition=gpu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH -o logs/submodule_%A_%a.log  # Save logs in a 'logs' directory
#SBATCH -e logs/submodule_%A_%a.err  # %A is job ID, %a is task ID
#SBATCH --account=investigacion1

# We will run 8 experiments (4 layers x 2 submodules)
#SBATCH --array=0-7

#----------------------------------------------------------------#
# Create Log Directory
#----------------------------------------------------------------#
mkdir -p logs

#----------------------------------------------------------------#
# Define Experiment Parameters
#----------------------------------------------------------------#
# Create bash arrays to hold our parameters
LAYERS=(5 11 17 23 5 11 17 23)
SUBMODULES=(attn attn attn attn mlp mlp mlp mlp)

# Get the parameters for THIS specific job task
# SLURM_ARRAY_TASK_ID will be a number from 0 to 7
LAYER_TO_RUN=${LAYERS[$SLURM_ARRAY_TASK_ID]}
SUBMODULE_TO_RUN=${SUBMODULES[$SLURM_ARRAY_TASK_ID]}

echo "--- Starting VCC Submodule Analysis ---"
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

export TORCH_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

# Define a unique working directory for each job
WORKING_DIR="outputs/4v_large_vit_sports_car_L${LAYER_TO_RUN}_${SUBMODULE_TO_RUN}"

python run_vcc.py \
    --working_dir "$WORKING_DIR" \
    --target_class "sports_car" \
    --model_to_run "vit_large" \
    --feature_names "$LAYER_TO_RUN" \
    --target_layer "$LAYER_TO_RUN" \
    --target_submodule "$SUBMODULE_TO_RUN" \
    --imagenet_path "$HOME/tesis/imagenet_val/imagenet-val"

echo "--- Job Task $SLURM_ARRAY_TASK_ID Finished ---"
