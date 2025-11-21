#!/bin/bash
#SBATCH -J vcc_stylized_datagen
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o logs/stylized_datagen_%A.log
#SBATCH -e logs/stylized_datagen_%A.err

mkdir -p logs

# --- Define Experiment Parameters ---
# IMPORTANT: Replace these with your actual class names (folder names)
CLASSES=(elephant) 
CLASS_TO_RUN=${CLASSES[$SLURM_ARRAY_TASK_ID]}

# IMPORTANT: Specify the layer you want to analyze
LAYER_TO_RUN=11 

echo "--- Starting Data Generation for Stylized ImageNet ---"
echo "Model Path: YOUR_MODEL_PATH.pth, Class: $CLASS_TO_RUN, Layer: $LAYER_TO_RUN"

# --- Environment and Execution ---
module load miniconda/3.0
conda activate VCC_final

cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

# Define a unique working directory
WORKING_DIR="outputs/stylized_${CLASS_TO_RUN}_L${LAYER_TO_RUN}_16_2"

python run_vcc.py \
    --working_dir "$WORKING_DIR" \
    --target_class "$CLASS_TO_RUN" \
    --model_to_run "custom" \
    --model_path "/home/mauricio.alvarez/tesis/VCC/model_weights/tiny_stylized_22acc.pth" \
    --feature_names "$LAYER_TO_RUN" \
    --target_layer "$LAYER_TO_RUN" \
    --target_dataset "custom" \
    --imagenet_path "/home/mauricio.alvarez/tesis/archive/session-1" \
    --num_classes 16

echo "--- Job Task Finished ---"
