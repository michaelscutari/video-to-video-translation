#!/bin/bash

# Initialize total available GPUs
total_available=0

# Fetch all nodes with A6000 GPUs in the compsci-gpu partition
sinfo -p compsci-gpu -o "%N %G" | grep "gpu:A5000" | while read -r node gres; do
    # Extract the number of A6000 GPUs on the node
    total_gpus=$(echo "$gres" | grep -oP 'gpu:A5000:\d+' | grep -oP '\d+')
    total_gpus=${total_gpus:-0}  # Default to 0 if not found

    # Get the number of A6000 GPUs currently in use on the node
    used_gpus=$(squeue --noheader --format="%b" -w "$node" | grep -o 'A5000' | wc -l)
    
    # Calculate available GPUs on this node
    free_gpus=$((total_gpus - used_gpus))
    
    # Add to the total available GPUs
    total_available=$((total_available + free_gpus))
done

# Output the total number of available A6000 GPUs
echo "Available NVIDIA A6000 GPUs on compsci-gpu cluster: $total_available"