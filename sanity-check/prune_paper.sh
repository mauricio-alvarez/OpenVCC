#!/bin/bash
#SBATCH -J pruning_models
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -o logs/prunning_paper_%j.log
#SBATCH -e logs/prunning_paper_%j.err


# --- ENVIRONMENT ---
module load miniconda/3.0
conda activate VCC_final

cd "/home/mauricio.alvarez/tesis/sanity-check" || exit 1

echo "--- Starting Head Pruning ---"
export TORCH_HOME="/home/mauricio.alvarez/.cache/torch"

python paper_based_pruning.py \
  --data_path "/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val" \
  --model "vit_tiny_patch16_224"\
  --output_path "pruned_vit_tiny.pth"\
  --epochs 10 \
  --lambda_l0 1.0 \
  --batch_size 64

  echo "--- Job Finished ---"