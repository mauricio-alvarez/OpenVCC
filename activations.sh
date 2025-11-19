#!/bin/bash
#SBATCH -J vcc_raw_activations
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH -o logs/raw_activations_%j.log
#SBATCH -e logs/raw_activations_%j.err

mkdir -p logs

# --- Define Experiment Parameters ---
# We run this once for each model. This example is for your custom model.
MODEL_NAME="custom"
MODEL_PATH="/home/mauricio.alvarez/tesis/VCC/model_weights/ViT-L_head_only_best_finetuned.pth"
LAYER_TO_RUN=23

echo "--- Starting Raw Activation Generation ---"
echo "Model: $MODEL_NAME, Layer: $LAYER_TO_RUN"

# --- Environment and Execution ---
module load miniconda/3.0
conda activate VCC_final

cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

python activations.py \
    --model_to_run "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --dataset_path "/home/mauricio.alvarez/tesis/archive/session-1" \
    --layer "$LAYER_TO_RUN" \
    --class_names car truck \
    --num_classes 16 \
    --batch_size 16

echo "--- Job Finished ---"
