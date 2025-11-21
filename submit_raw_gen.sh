#!/bin/bash
#SBATCH -J raw_activations
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o logs/raw_gen_%j.log
#SBATCH -e logs/raw_gen_%j.err

mkdir -p logs

# --- PARAMETERS ---
# Change these for your specific comparison
MODEL_NAME="shvit"
MODEL_PATH="/home/mauricio.alvarez/tesis/VCC/model_weights/shvit_stylized_38acc.pth"
DATASET_PATH="/home/mauricio.alvarez/tesis/archive/session-1"
LAYER="3"
CLASSES=("car" "elephant")
OUTPUT_DIR="analysis/raw_activations_L${LAYER}_car_elephant"

# --- EXECUTION ---
module load miniconda/3.0
conda activate VCC_final

export TORCH_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HUB_OFFLINE=1

cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

python generate_raw_activations.py \
    --model_to_run "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --layer "$LAYER" \
    --class_names "${CLASSES[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --num_classes 16 \
    --batch_size 16 # Safe batch size

echo "Data generation complete."
