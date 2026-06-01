#!/bin/bash

#----------------------------------------------------------------#
# Slurm Directives
#----------------------------------------------------------------#
#SBATCH -J repair_stage1_umap
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --time=04:00:00
#SBATCH -o repair_stage1_umap_out_%j.log
#SBATCH -e repair_stage1_umap_err_%j.log

set -euo pipefail

echo "Stage-1 repair + 3D UMAP job started on $(hostname) at $(date)"

module load miniconda/3.0
conda activate VCC_final

PROJECT_DIR="/home/mauricio.alvarez/tesis/VCC"
cd "${PROJECT_DIR}"

export TORCH_HOME="${PROJECT_DIR}/model_cache"
export HF_HOME="${PROJECT_DIR}/model_cache"
export HF_HUB_OFFLINE=1

SHVIT_CKPT="${PROJECT_DIR}/model_weights/SHViT/shvit_s1.pth"
DHVIT_CKPT="${PROJECT_DIR}/88_shvit_s1_doublehead_1805_100epochs.pth"
OUT_DIR="${PROJECT_DIR}/analysis/repaired_stage1_umap"
REPAIRED_CKPT="${OUT_DIR}/dhvit_s1_stage1_repaired.pth"

IMAGENETR_ROOT="/home/mauricio.alvarez/tesis/archive/imagenet-r"
CLASS_FILE="${PROJECT_DIR}/imagenet_r_200_classes.txt"

mkdir -p "${OUT_DIR}"

python repair_stage1_and_umap3d.py \
  --shvit-checkpoint "${SHVIT_CKPT}" \
  --dhvit-checkpoint "${DHVIT_CKPT}" \
  --output-checkpoint "${REPAIRED_CKPT}" \
  --model-size s1 \
  --num-classes 1000 \
  --plot-umap \
  --dataset-root "${IMAGENETR_ROOT}" \
  --dataset-name imagenetr \
  --class-file "${CLASS_FILE}" \
  --activation-output-dir "${OUT_DIR}/activations_imagenetr" \
  --feature-key feat_mean \
  --umap-output "${OUT_DIR}/dhvit_s1_stage1_repaired_feat_mean_umap3d.html" \
  --max-images-per-class 50 \
  --max-umap-points 10000 \
  --batch-size 64 \
  --num-workers 8

echo "Stage-1 repair + 3D UMAP job finished at $(date)"

conda deactivate
module unload miniconda/3.0
