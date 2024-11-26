#!/bin/bash
#SBATCH --job-name=train_job            # Job name
#SBATCH --output=train_output.log       # Standard output log file
#SBATCH --error=train_error.log         # Standard error log file
#SBATCH --ntasks=1                      # Number of tasks (typically 1 for a single script)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=16G                       # Memory per node (adjust as needed)
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec (8 hours)
#SBATCH --gres=gpu:a6000:1               # Request 1 GPU (adjust/remove if not needed)
#SBATCH --partition=compsci-gpu          # Specify the partition (check with your cluster for available partitions)
#SBATCH --output=runs/train_output_%j.log   # Standard output log file with Job ID
#SBATCH --error=runs/train_error_%j.log     # Standard error log file with Job ID

# Source bashrc to set up environment variables and paths
source ~/.bashrc

# load cuda
module load cuda/cuda-11.4

eval "$(micromamba shell hook --shell bash)"

# Activate your micromamba environment
micromamba activate ece661

lspci | grep -i nvidia

echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of CUDA Devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Torch Cuda Version: $(python -c 'import torch; print(torch.version.cuda)')"
nvidia-smi