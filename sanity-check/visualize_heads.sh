#!/bin/bash

# --- Configuration ---
WEIGHTS_DIR="/home/mauricio.alvarez/tesis/sanity-check/weights" # Adjust if needed
OUTPUT_DIR="heads_plots"
VIZ_SCRIPT="visualize_heads.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Looking for weights in: $WEIGHTS_DIR"
echo "Outputting plots to: $OUTPUT_DIR"
echo "------------------------------------------------"

# Loop through all .pth files
for filepath in "$WEIGHTS_DIR"/*.pth; do
    
    # Extract filename (e.g., base_102heads.pth)
    filename=$(basename "$filepath")
    
    # Logic to map filename prefix to TIMM model name
    if [[ $filename == base* ]]; then
        MODEL_NAME="vit_base_patch16_224"
    elif [[ $filename == small* ]]; then
        MODEL_NAME="vit_small_patch16_224"
    elif [[ $filename == tiny* ]]; then
        MODEL_NAME="vit_tiny_patch16_224"
    else
        echo "⚠️  Skipping unknown file format: $filename"
        continue
    fi

    echo "Processing: $filename"
    echo "   -> Architecture: $MODEL_NAME"

    # Run the Python visualization script
    python "$VIZ_SCRIPT" \
        --model "$MODEL_NAME" \
        --checkpoint "$filepath"

    # The python script saves files as 'density_{filename}.png'
    # We move that file to our output directory
    generated_img="density_${filename%.pth}.png"
    
    if [ -f "$generated_img" ]; then
        mv "$generated_img" "$OUTPUT_DIR/"
        echo "   -> Saved to $OUTPUT_DIR/$generated_img"
    else
        echo "   -> Error: Image was not generated."
    fi
    echo ""
done

echo "Batch visualization complete."