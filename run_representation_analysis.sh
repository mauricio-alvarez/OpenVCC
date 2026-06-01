#!/bin/bash

#----------------------------------------------------------------#
# Slurm Directives
#----------------------------------------------------------------#
#SBATCH -J repr_shvit_dhvit
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --time=08:00:00
#SBATCH -o repr_analysis_out_%j.log
#SBATCH -e repr_analysis_err_%j.log

set -euo pipefail

echo "Representation analysis job started on $(hostname) at $(date)"

module load miniconda/3.0
conda activate VCC_final

PROJECT_DIR="/home/mauricio.alvarez/tesis/VCC"
cd "${PROJECT_DIR}"

export TORCH_HOME="${PROJECT_DIR}/model_cache"
export HF_HOME="${PROJECT_DIR}/model_cache"
export HF_HUB_OFFLINE=1

#----------------------------------------------------------------#
# Configuration
#----------------------------------------------------------------#

SHVIT_CKPT="${PROJECT_DIR}/model_weights/SHViT/shvit_s1.pth"
DHVIT_CKPT="${PROJECT_DIR}/122_shvit_s1_doublehead_1805_100epochs.pth"

# ImageNet-200 should be a folder with ImageNet-R WNID class subfolders from ImageNet val.
# If you already ran test_imagenet_r(), this may exist as ./imagenet_val_for_imagenet_r.
IMAGENET200_ROOT="${PROJECT_DIR}/imagenet_val_for_imagenet_r"
IMAGENET_VAL_ROOT="/home/mauricio.alvarez/tesis/archive/imagenet-val/imagenet-val"
IMAGENETR_ROOT="/home/mauricio.alvarez/tesis/archive/imagenet-r"

CLASS_FILE="${PROJECT_DIR}/imagenet_r_16_classes.txt"
OUT_ROOT="${PROJECT_DIR}/analysis/representation"

BATCH_SIZE=64
MAX_IMAGES_PER_CLASS=50
NUM_WORKERS=8

echo "SHViT checkpoint: ${SHVIT_CKPT}"
echo "DHViT checkpoint: ${DHVIT_CKPT}"
echo "ImageNet-200 root: ${IMAGENET200_ROOT}"
echo "ImageNet val root: ${IMAGENET_VAL_ROOT}"
echo "ImageNet-R root: ${IMAGENETR_ROOT}"
echo "Class file: ${CLASS_FILE}"
echo "Output root: ${OUT_ROOT}"

echo "Ensuring ImageNet-200 symlinks exist at ${IMAGENET200_ROOT}"
mkdir -p "${IMAGENET200_ROOT}"
while read -r wnid; do
  if [ -z "${wnid}" ] || [[ "${wnid}" == \#* ]]; then
    continue
  fi
  if [ ! -e "${IMAGENET200_ROOT}/${wnid}" ]; then
    ln -s "${IMAGENET_VAL_ROOT}/${wnid}" "${IMAGENET200_ROOT}/${wnid}"
  fi
done < "${CLASS_FILE}"

#----------------------------------------------------------------#
# Step 1: Extract ImageNet-200 Representations
#----------------------------------------------------------------#

python representation_extract.py \
  --model-kind shvit \
  --model-size s1 \
  --checkpoint "${SHVIT_CKPT}" \
  --dataset-root "${IMAGENET200_ROOT}" \
  --dataset-name imagenet200 \
  --class-file "${CLASS_FILE}" \
  --output-dir "${OUT_ROOT}" \
  --max-images-per-class "${MAX_IMAGES_PER_CLASS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

python representation_extract.py \
  --model-kind dhvit \
  --model-size s1 \
  --checkpoint "${DHVIT_CKPT}" \
  --dataset-root "${IMAGENET200_ROOT}" \
  --dataset-name imagenet200 \
  --class-file "${CLASS_FILE}" \
  --output-dir "${OUT_ROOT}" \
  --max-images-per-class "${MAX_IMAGES_PER_CLASS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

#----------------------------------------------------------------#
# Step 2: Extract ImageNet-R Representations
#----------------------------------------------------------------#

python representation_extract.py \
  --model-kind shvit \
  --model-size s1 \
  --checkpoint "${SHVIT_CKPT}" \
  --dataset-root "${IMAGENETR_ROOT}" \
  --dataset-name imagenetr \
  --class-file "${CLASS_FILE}" \
  --output-dir "${OUT_ROOT}" \
  --max-images-per-class "${MAX_IMAGES_PER_CLASS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

python representation_extract.py \
  --model-kind dhvit \
  --model-size s1 \
  --checkpoint "${DHVIT_CKPT}" \
  --dataset-root "${IMAGENETR_ROOT}" \
  --dataset-name imagenetr \
  --class-file "${CLASS_FILE}" \
  --output-dir "${OUT_ROOT}" \
  --max-images-per-class "${MAX_IMAGES_PER_CLASS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

#----------------------------------------------------------------#
# Step 3: Analyze Saved Representations
#----------------------------------------------------------------#

python analyze_representations.py \
  --input-root "${OUT_ROOT}" \
  --source-dataset imagenet200 \
  --shift-dataset imagenetr \
  --output-dir "${OUT_ROOT}/results" \
  --max-metric-samples 8000 \
  --max-plot-samples 3000

echo "Representation analysis job finished at $(date)"

conda deactivate
module unload miniconda/3.0
