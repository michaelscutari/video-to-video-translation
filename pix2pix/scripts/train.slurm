#!/bin/bash
#SBATCH --job-name=train_job            # Job name
#SBATCH --output=train_output.log       # Output log file
#SBATCH --error=train_error.log         # Error log file
#SBATCH --ntasks=1                      # Number of tasks (typically 1 for a single script)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=16G                       # Memory per node (adjust as needed)
#SBATCH --time=08:00:00                 # Time limit hrs:min:sec (8 hours for overnight)
#SBATCH --gres=gpu:a6000:1                    # Request 1 GPU if needed (adjust/remove if not needed)
#SBATCH --partition=compsci-gpu            # Specify the partition (check with your cluster for available partitions)

# bashrc
source ~/.bashrc

# micromamba run
micromamba run -n ece661 python ./src/train.py
