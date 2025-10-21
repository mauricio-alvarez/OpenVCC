#!/bin/bash

#----------------------------------------------------------------#
#SBATCH -J vcc_huge_final     # Final job name
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o vcc_output_huge_%j.log
#SBATCH -e vcc_error_huge_%j.log

#----------------------------------------------------------------#
echo "Job started on $(hostname) at $(date)"
module load miniconda/3.0
conda activate VCC_final
echo "--- Querying NVIDIA driver and CUDA environment ---"
nvidia-smi
python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("Compiled with CUDA:", torch.version.cuda)
print("CUDA available at runtime:", torch.cuda.is_available())
print("Detected GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF
echo "--- End of CUDA test ---"

echo "Environment is ready."
export TORCH_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HUB_OFFLINE=1
#----------------------------------------------------------------#
# Change to the project directory
cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

echo "Starting the VCC experiment for huge_tiny with manual loading..."

python run_vcc.py \
    --working_dir "outputs/vit_space_bar" \
    --target_class "space_bar" \
    --model_to_run "vit_b" \
    --feature_names 2 5 8 10 \
    --imagenet_path "$HOME/tesis/imagenet_val/imagenet-val"

echo "Python script finished."

#----------------------------------------------------------------#
# Cleanup
#----------------------------------------------------------------#
conda deactivate
module unload miniconda/3.0
echo "Job finished at $(date)"

