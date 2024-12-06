#!/bin/bash
#SBATCH --job-name=cycleGAN             # Job name
#SBATCH --ntasks=1                      # Number of tasks (typically 1 for a single script)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=16G                       # Memory per node (adjust as needed)
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec (8 hours)
#SBATCH --gres=gpu:a5000:1               # Request 1 GPU (adjust/remove if not needed)
#SBATCH --partition=compsci-gpu          # Specify the partition (check with your cluster for available partitions)
#SBATCH --output=runs/train_output_%j.log   # Standard output log file with Job ID
#SBATCH --error=runs/train_error_%j.log     # Standard error log file with Job ID

# Source bashrc to set up environment variables and paths
source ~/.bashrc

# load cuda
module load cuda/cuda-11.4

# This
eval "$(micromamba shell hook --shell bash)"

# Activate your micromamba environment
micromamba activate ece661

# Change to the directory from which the job was submitted
cd "$SLURM_SUBMIT_DIR"

# Create necessary directories to avoid PermissionError
mkdir -p runs/checkpoints
mkdir -p runs/logs
mkdir -p logs

# wandb setup
export WANDB_API_KEY=2cffe09f8f0eb9f83180bfc498a98e1476803894

# Capture the TensorBoard process ID to terminate it later
TB_PID=$!

# Run your training script
python ./src/train.py

# After training completes, kill the TensorBoard process
kill $TB_PID
