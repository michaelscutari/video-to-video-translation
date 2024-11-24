#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --output=test_cuda_%j.log
#SBATCH --error=test_cuda_%j.err
#SBATCH --partition=compsci-gpu 
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Load modules
module load python/3.8
module load cuda/11.8
module load pytorch/1.12.1

# Run test script
srun python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))"