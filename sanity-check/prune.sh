#!/bin/bash
#SBATCH -J pruning_models
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/prunning_%j.log
#SBATCH -e logs/prunning_%j.err


# --- ENVIRONMENT ---
module load miniconda/3.0
conda activate VCC_final

cd "/home/mauricio.alvarez/tesis/sanity-check" || exit 1

echo "--- Starting Head Pruning ---"
export TORCH_HOME="/home/mauricio.alvarez/.cache/torch"
python head_prunning.py \
  --model vit_base_patch16_224 \
  --data_dir /home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val \
  --batch_size 64  \
  --iterations 11

echo "--- Job Finished ---"
