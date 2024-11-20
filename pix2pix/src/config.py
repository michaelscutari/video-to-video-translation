import torch

class Config:
    # Training parameters
    num_epochs = 200
    batch_size = 8
    learning_rate = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    lambda_L1 = 100  # Weight for L1 loss component
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Paths
    train_input_dir = './data/train/input'
    train_target_dir = './data/train/output'
    val_input_dir = './data/val/input'
    val_target_dir = './data/val/output'
    # Checkpoint and output directories
    checkpoint_dir = './runs/checkpoints'
    sample_dir = './runs/samples'