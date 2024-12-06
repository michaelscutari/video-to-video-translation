import torch
from torchvision import transforms

class Config:
    # NAME OF RUN (CHANGE FOR EVERY RUN)
    run_name = 'cheddar'
    # Training parameters
    num_epochs = 200
    batch_size = 4
    learning_rate = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    lambda_L1 = 10  # Weight for L1 loss component
    # Paths
    train_input_dir = './data/sat2map/map'
    train_target_dir = './data/sat2map/sat'
    # Checkpoint and output directories
    checkpoint_dir = './runs/' + run_name + '/checkpoints'
    sample_dir = './runs/' + run_name + '/samples'
    log_dir = './runs/' + run_name + '/logs'

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])