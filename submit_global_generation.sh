#!/bin/bash
#SBATCH -J vcc_global_gen
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/global_gen_%j.log
#SBATCH -e logs/global_gen_%j.err

mkdir -p logs

# --- PARAMETERS ---
MODEL_TO_RUN="custom"
MODEL_PATH="/home/mauricio.alvarez/tesis/VCC/model_weights/tiny_stylized_22acc.pth"
DATASET_PATH="/home/mauricio.alvarez/tesis/archive/session-1"

LAYER=11 

OUTPUT_DIR="analysis/global_activations_${MODEL_TO_RUN}_tiny_L${LAYER}"

# --- ENVIRONMENT ---
module load miniconda/3.0
conda activate VCC_final
export TORCH_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HUB_OFFLINE=1

cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

# --- AUTO-DETECT CLASSES ---
CLASS_LIST=($(ls -d $DATASET_PATH/*/ | xargs -n 1 basename))

echo "--- Detected ${#CLASS_LIST[@]} classes ---"
echo "${CLASS_LIST[@]}"

echo "--- Starting Global Activation Generation ---"

python generate_raw_activations.py \
    --model_to_run "$MODEL_TO_RUN" \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --layer "$LAYER" \
    --output_dir "$OUTPUT_DIR" \
    --num_classes 16 \
    --batch_size 32 \
    --class_names "${CLASS_LIST[@]}"

echo "--- Job Finished ---"
