#!/bin/bash
#SBATCH -J pruning_models
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o logs/prunning_paper_%j.log
#SBATCH -e logs/prunning_paper_%j.err

module load miniconda/3.0
conda activate VCC_final

DATA_PATH="/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val"
WEIGHTS_DIR="/home/mauricio.alvarez/tesis/sanity-check/weights"
mkdir -p "$WEIGHTS_DIR"

# ================= CONFIGURATION =================
# Choose your model here
# MODEL_TYPE="small"
MODEL_TYPE="base"

if [ "$MODEL_TYPE" == "tiny" ]; then
    MODEL_NAME="vit_tiny_patch16_224"
    TOTAL_HEADS=36
    MIN_HEADS=12
    STEP_SIZE=6  # Decrease by 6 every time
elif [ "$MODEL_TYPE" == "small" ]; then
    MODEL_NAME="vit_small_patch16_224"
    TOTAL_HEADS=72
    MIN_HEADS=12
    STEP_SIZE=6  # Decrease by 6 (Doing 2 would take forever: 66 runs)
elif [ "$MODEL_TYPE" == "base" ]; then
    MODEL_NAME="vit_base_patch16_224"
    TOTAL_HEADS=144
    MIN_HEADS=12
    STEP_SIZE=6  # Decrease by 6 every time
fi
# =================================================

echo "Starting Pruning Sweep for $MODEL_NAME"
echo "From $TOTAL_HEADS down to $MIN_HEADS (Step: $STEP_SIZE)"

CURRENT_TARGET=$TOTAL_HEADS
PREV_CHECKPOINT=""

# Calculate first step down
CURRENT_TARGET=$((CURRENT_TARGET - STEP_SIZE))

while [ $CURRENT_TARGET -ge $MIN_HEADS ]; do
    echo "----------------------------------------------------"
    echo "Targeting: $CURRENT_TARGET Heads"
    OUTPUT_FILE="${WEIGHTS_DIR}/${MODEL_TYPE}_${CURRENT_TARGET}heads.pth"
    
    CMD="python paper_based_pruning_v2.py \
        --data_path \"$DATA_PATH\" \
        --model \"$MODEL_NAME\" \
        --epochs 20 \
        --target_heads $CURRENT_TARGET \
        --tolerance 2 \
        --patience 2 \
        --lambda_l0 0.5 \
        --output_path \"$OUTPUT_FILE\""
    
    if [ -n "$PREV_CHECKPOINT" ]; then
        CMD="$CMD --initial_checkpoint \"$PREV_CHECKPOINT\""
    fi
    
    eval $CMD
    
    if [ -f "$OUTPUT_FILE" ]; then
        PREV_CHECKPOINT="$OUTPUT_FILE"
    else
        echo "Error: Failed to create $OUTPUT_FILE. Stopping chain."
        break
    fi
        
    CURRENT_TARGET=$((CURRENT_TARGET - STEP_SIZE))
done

echo "Sweep Complete."