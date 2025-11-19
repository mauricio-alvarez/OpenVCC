#!/bin/bash

#----------------------------------------------------------------#
# Slurm Directives
#----------------------------------------------------------------#
#SBATCH -J vcc_decompose      # Job name
#SBATCH -p gpu                # Partition name (assuming 'gpu')
#SBATCH -c 4                  # Number of CPU cores
#SBATCH --mem=32G             # Memory request (32G is safe for a 'base' model)
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH -o decompose_out_%j.log  # File for standard output
#SBATCH -e decompose_err_%j.log  # File for standard error

#----------------------------------------------------------------#
# Environment Setup
#----------------------------------------------------------------#
echo "Job started on $(hostname) at $(date)"
echo "Loading required modules..."

module load miniconda/3.0
conda activate shvit # <<< --- FIX: Using 'conda activate' which works on your system
echo "Environment is ready."

#----------------------------------------------------------------#
# Execute the Python Script
#----------------------------------------------------------------#
# Set cache variables and force offline mode to ensure timm loads from cache
export TORCH_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HOME="/home/mauricio.alvarez/tesis/VCC/model_cache"
export HF_HUB_OFFLINE=1

# Change to the project directory where your scripts are located
cd "/home/mauricio.alvarez/tesis/VCC" || exit 1

echo "Starting image decomposition..."

# Use the full path with $HOME instead of ~ for better reliability in scripts
python main.py 

echo "Python script finished."

#----------------------------------------------------------------#
# Cleanup
#----------------------------------------------------------------#
conda deactivate
module unload miniconda/3.0
echo "Job finished at $(date)"

