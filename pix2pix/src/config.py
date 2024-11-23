import torch
from torchvision import transforms

class Config:
    # Training parameters
    num_epochs = 200
    batch_size = 16
    learning_rate = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    lambda_L1 = 50  # Weight for L1 loss component
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Paths
    train_input_dir = './data/train/input'
    train_target_dir = './data/train/output'
    val_input_dir = './data/val/input'
    val_target_dir = './data/val/output'
    # Checkpoint and output directories
    checkpoint_dir = './runs/run5/checkpoints'
    sample_dir = './runs/run5/samples'
    log_dir = './runs/run5/logs'

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])