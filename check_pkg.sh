#!/bin/bash

#SBATCH -J env_check          # Job name
#SBATCH -p gpu                # The partition with GPUs
#SBATCH --gres=gpu:1          # Request a GPU to land on a real compute node
#SBATCH -o env_check_%j.log   # File to save the output

echo "--- Running on compute node: $(hostname) ---"
echo ""

echo "--- Loading modules and activating Conda environment ---"
module load miniconda/3.0
conda activate VCC_final
echo "Conda environment activated."
echo ""

echo "--- Checking conda list for scikit-image ---"
conda list | grep scikit-image
echo ""

echo "--- Attempting to import skimage directly with Python ---"
python -c "import skimage; print('>>> SUCCESS: skimage imported successfully!')"

echo "--- Job finished ---"
