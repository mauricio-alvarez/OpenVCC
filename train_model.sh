#!/bin/bash

#----------------------------------------------------------------#
# Slurm Directives
#----------------------------------------------------------------#
#SBATCH -J vcc_train      # Job name
#SBATCH -p gpu                # Partition name (assuming 'gpu')
#SBATCH -c 16                  # Number of CPU cores
#SBATCH --mem=64G             # Memory request (32G is safe for a 'base' model)
#SBATCH --gres=gpu:1      # Request 1 RTX A6000 GPU (g002)
#SBATCH --nodelist=g001            # Target the g002 node specifically
#SBATCH --time=66:00:00
#SBATCH -o vcc_train_out_%j.log  # File for standard output
#SBATCH -e vcc_train_err_%j.log  # File for standard error

#----------------------------------------------------------------#
# Environment Setup
#----------------------------------------------------------------#
echo "Job started on $(hostname) at $(date)"
echo "Loading required modules..."

module load miniconda/3.0
conda activate VCC_final 
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

echo "Starting image training..."

# Use the full path with $HOME instead of ~ for better reliability in scripts
python main.py 

echo "Python script finished."

#----------------------------------------------------------------#
# Cleanup
#----------------------------------------------------------------#
conda deactivate
module unload miniconda/3.0
echo "Job finished at $(date)"

