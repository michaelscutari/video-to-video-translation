import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.utils as vutils  # For image logging
import wandb  # Import Weights & Biases
import time

# custom classes
from model import Generator, Discriminator
from dataset import CycleGANDataset
from config import Config

# Device
device = torch.device('cuda')

# Initialize Weights & Biases
wandb.init(
    project="cycleGAN-project",
    config={
        "learning_rate": Config.lr,
        "beta1": Config.b1,
        "beta2": Config.b2,
        "batch_size": Config.batch_size,
        "num_epochs": Config.num_epochs,
        #"lambda_L1": Config.lambda_L1,
    },
    name=f"red_delicious",
    save_code=False
)

config = wandb.config

# Create necessary directories
os.makedirs(Config.checkpoint_dir, exist_ok=True)


# Dataset and DataLoader
train_dataset = CycleGANDataset(
    X_dir=Config.train_x_dir,
    Y_dir=Config.train_y_dir,
    transform=Config.data_transforms
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True
)

# Initialize models
gen_X_to_Y = Generator().to(device)
gen_Y_to_X = Generator().to(device)

discr_X = Discriminator().to(device)
discr_Y = Discriminator().to(device)


# Denormalize!
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Initialize weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

initialize_weights(gen_X_to_Y)
initialize_weights(gen_Y_to_X)
initialize_weights(discr_X)
initialize_weights(discr_Y)

## LOSSES
criterion_identity = nn.L1Loss()
criterion_cycle = nn.L1Loss()
criterion_GAN = nn.BCEWithLogitsLoss()

##############################################################################################
## Optimizers, Learning Rate Schedulers
##############################################################################################
optimizer_gen_XY = optim.Adam(gen_X_to_Y.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optimizer_gen_YX = optim.Adam(gen_Y_to_X.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

optimizer_discr_X = optim.Adam(discr_X.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optimizer_discr_Y = optim.Adam(discr_Y.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

class LambdaLR():
    def __init__(self, num_epochs, offset, decay_start_epoch):
        assert ((num_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.num_epochs = num_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    def step(self, epoch):
        # learning rate linearly decays from 1 to 0 after the decay starts
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.num_epochs - self.decay_start_epoch)

lr_scheduler_gen_XY = optim.lr_scheduler.LambdaLR(optimizer_gen_XY, lr_lambda=LambdaLR(Config.num_epochs, Config.epoch, Config.decay_epoch).step)
lr_scheduler_gen_YX = optim.lr_scheduler.LambdaLR(optimizer_gen_YX, lr_lambda=LambdaLR(Config.num_epochs, Config.epoch, Config.decay_epoch).step)

lr_scheduler_discr_X = optim.lr_scheduler.LambdaLR(optimizer_discr_X, lr_lambda=LambdaLR(Config.num_epochs, Config.epoch, Config.decay_epoch).step)
lr_scheduler_discr_Y = optim.lr_scheduler.LambdaLR(optimizer_discr_Y, lr_lambda=LambdaLR(Config.num_epochs, Config.epoch, Config.decay_epoch).step)

# Replay buffer
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), "Buffer size must be greater than 0"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size: # if there's space in buffer, add element
                self.data.append(element)
                to_return.append(element)
            else:
                if torch.rand(1).item() > .5: # 50% chance
                    i = torch.randint(0, self.max_size, (1,)).item() # pick random element to replace
                    to_return.append(self.data[i].clone()) # return the element we're replacing
                    self.data[i] = element # fill in the spot
                else:  # other 50% of time
                    to_return.append(element)
        return torch.cat(to_return)
    
fake_X_buffer = ReplayBuffer()
fake_Y_buffer = ReplayBuffer()

# Select a fixed sample for consistent monitoring
fixed_sample = train_dataset[0]  # Change the index to select a different sample
fixed_X = fixed_sample['X'].unsqueeze(0).to(device)  # Add batch dimension
fixed_Y = fixed_sample['Y'].unsqueeze(0).to(device)  # Add batch dimension

input_images = denormalize(fixed_X.cpu())
target_images = denormalize(fixed_Y.cpu())

img_grid_input = vutils.make_grid(input_images, normalize=False)
img_grid_target = vutils.make_grid(target_images, normalize=False)

wandb.log({
            'X Images': [wandb.Image(img_grid_input, caption="X")],
            'Y Images': [wandb.Image(img_grid_target, caption="Y")],
        })

# DEBUG
print("Starting training...")

# timing
training_start_time = time.time()

# Training loop
for epoch in range(1, Config.num_epochs + 1):
    
    gen_X_to_Y.train()
    gen_Y_to_X.train()
    discr_X.train()
    discr_Y.train()

    # Timing

    epoch_start_time = time.time()
    epoch_batch_times = []

    for batch_idx, batch in enumerate(train_loader):
        batch_start_time = time.time()

        # Get input and target images
        real_x = batch['X'].to(device)
        real_y = batch['Y'].to(device)

        # Get the output shape from the discriminator
        with torch.no_grad():
            output_shape = discr_X(real_x).shape

        real_label = torch.ones(output_shape, device=device)
        fake_label = torch.zeros(output_shape, device=device)

        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_gen_XY.zero_grad()
        optimizer_gen_YX.zero_grad()

        # --------
        #  Identity Loss (g_xy(y) = y ?)
        # --------
        if Config.IDENTITY_LOSS_INCLUDED == True:
            allegedly_same_Y = gen_X_to_Y(real_y) # G(Y) should remain Y
            iden_loss_XY = criterion_identity(allegedly_same_Y, real_y)

            allegedly_same_X = gen_Y_to_X(real_x) # G(X) should remain X
            iden_loss_YX = criterion_identity(allegedly_same_X, real_x)
        else:
            iden_loss_XY = 0
            iden_loss_YX = 0

        # --------
        # Generator adversarial loss
        # --------

        # Generate images
        fake_y = gen_X_to_Y(real_x)
        fake_x = gen_Y_to_X(real_y)

        # Discriminator evaluates the fake images
        pred_fake_X = discr_X(fake_x)
        pred_fake_Y = discr_Y(fake_y)

        if Config.GAN_LOSS_INCLUDED:
            adv_loss_XY = criterion_GAN(pred_fake_Y, real_label)
            adv_loss_YX = criterion_GAN(pred_fake_X, real_label)
        else:
            adv_loss_XY = 0
            adv_loss_YX = 0

        # --------
        # Cycle loss
        # --------
        if Config.CYCLE_LOSS_INCLUDED:
            allegedly_reconstructed_x = gen_Y_to_X(fake_y)
            cycle_loss_X = criterion_cycle(real_x, allegedly_reconstructed_x)
            allegedly_reconstructed_y = gen_X_to_Y(fake_x)
            cycle_loss_Y = criterion_cycle(real_y, allegedly_reconstructed_y)

        # Corrected total generator loss
        tot_loss_XY = (Config.IDENTITY_WEIGHT * iden_loss_XY +
                    Config.GAN_WEIGHT * adv_loss_XY +
                    Config.CYCLE_WEIGHT * cycle_loss_X)

        tot_loss_YX = (Config.IDENTITY_WEIGHT * iden_loss_YX +
                    Config.GAN_WEIGHT * adv_loss_YX +
                    Config.CYCLE_WEIGHT * cycle_loss_Y)

        tot_loss_XY.backward()
        tot_loss_YX.backward()

        torch.nn.utils.clip_grad_norm_(gen_X_to_Y.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(gen_Y_to_X.parameters(), max_norm=1.0)

        optimizer_gen_XY.step()
        optimizer_gen_YX.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_discr_X.zero_grad()
        optimizer_discr_Y.zero_grad()

        # choose fake images to compare from replay buffer
        fake_x = fake_X_buffer.push_and_pop(fake_x)
        fake_y = fake_Y_buffer.push_and_pop(fake_y)

        # Real images
        pred_real_X = discr_X(real_x)
        discr_loss_real_X = criterion_GAN(pred_real_X, real_label)

        pred_real_Y = discr_Y(real_y)
        discr_loss_real_Y = criterion_GAN(pred_real_Y, real_label)

        # fake images
        pred_fake_X = discr_X(fake_x)
        discr_loss_fake_X = criterion_GAN(pred_fake_X, fake_label)

        pred_fake_Y = discr_Y(fake_y)
        discr_loss_fake_Y = criterion_GAN(pred_fake_Y, fake_label)

        # Total discriminator loss
        discr_loss_X = (discr_loss_fake_X + discr_loss_real_X) * 0.5
        discr_loss_Y = (discr_loss_fake_Y + discr_loss_real_Y) * 0.5
        
        discr_loss_X.backward()
        discr_loss_Y.backward()

        torch.nn.utils.clip_grad_norm_(discr_X.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(discr_Y.parameters(), max_norm=1.0)

        optimizer_discr_X.step()
        optimizer_discr_Y.step()

        # ---------------------
        #  Logging
        # ---------------------

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        epoch_batch_times.append(batch_time)

        batches_done = epoch * len(train_loader) + batch_idx  # total number of batches processed so far
        if batches_done % 500 == 0:
            # DEBUG
            print(f"Logging! Total batches: {batches_done}", flush=True)
            # Log scalar metrics
            wandb.log({
                'Loss/Identity_XY': iden_loss_XY.item(),
                'Loss/Identity_YX': iden_loss_YX.item(),
                'Loss/GAN_XY': adv_loss_XY.item(),
                'Loss/GAN_YX': adv_loss_YX.item(),
                'Loss/Cycle_X': cycle_loss_X.item(),
                'Loss/Cycle_Y': cycle_loss_Y.item(),

                'Loss/Tot_XY': tot_loss_XY.item(),
                'Loss/Tot_YX': tot_loss_YX.item(),

                'Loss/Discr_fake_X': discr_loss_fake_X.item(),
                'Loss/Discr_fake_Y': discr_loss_fake_Y.item(),
                'Loss/Discr_real_X': discr_loss_real_X.item(),
                'Loss/Discr_real_Y': discr_loss_real_Y.item(),

                'Loss/Discr_Tot_X': discr_loss_X.item(),
                'Loss/Discr_Tot_Y': discr_loss_Y.item(),

                'Learning Rate/Gen_XY': optimizer_gen_XY.param_groups[0]['lr'],
                'Learning Rate/Gen_YX': optimizer_gen_YX.param_groups[0]['lr'],

                'Learning Rate/Discr_X': optimizer_discr_X.param_groups[0]['lr'],
                'Learning Rate/Discr_Y': optimizer_discr_Y.param_groups[0]['lr'],
                'Epoch': epoch,
                'Batch': batch_idx,
                'Batch_time': batch_time
            })

    # ---------------------
    #  Logging
    # ---------------------

    # timing
    epoch_time = time.time() - epoch_start_time
    average_batch_time = sum(epoch_batch_times) / len(epoch_batch_times)

    # estimated time remaining
    remaining_epochs = Config.num_epochs - epoch
    estimated_remaining_time = remaining_epochs * epoch_time
    time_since_start = time.time() - training_start_time

    # Log epoch metrics and timing
    wandb.log({
        'Epoch': epoch,
        'Epoch_duration': epoch_time,
        'Average_batch_time': average_batch_time,
        'Estimated_remaining_time': estimated_remaining_time,
        'Time_since_start': time_since_start,
    })

    print(f"Epoch {epoch}/{Config.num_epochs} completed in {epoch_time:.2f}s. "
          f"Average batch time: {average_batch_time:.2f}s. "
          f"Estimated remaining time: {estimated_remaining_time/60:.2f} minutes."
          f"Time since start: {time_since_start/60:.2f} minutes", flush=True)

    # Log images
    with torch.no_grad():
        gen_X_to_Y.eval()
        fake_Y = gen_X_to_Y(fixed_X)

        gen_Y_to_X.eval()
        fake_X = gen_Y_to_X(fixed_Y)

        output_X = denormalize(fake_X.cpu())
        output_Y = denormalize(fake_Y.cpu())

        # Create image grids
        img_grid_fake_X = vutils.make_grid(output_X, normalize=False)
        img_grid_fake_Y = vutils.make_grid(output_Y, normalize=False)

        # Log images to Weights & Biases
        wandb.log({
            'Fake X': [wandb.Image(img_grid_fake_X, caption="Fake X")],
            'Fake Y': [wandb.Image(img_grid_fake_Y, caption="Fake Y")]
        })

        gen_X_to_Y.train()
        gen_Y_to_X.train()

    # Update learning rates
    lr_scheduler_discr_X.step()
    lr_scheduler_discr_Y.step()
    lr_scheduler_gen_XY.step()
    lr_scheduler_gen_YX.step()
    

    # Save model checkpoints every 10 epochs
    if (epoch + 1) % 10 == 0:
        gen_X_to_Y_path = os.path.join(Config.checkpoint_dir, f'gen_X_to_Y_epoch_{epoch+1}.pth')
        gen_Y_to_X_path = os.path.join(Config.checkpoint_dir, f'gen_Y_to_X_epoch_{epoch+1}.pth')
        discr_X_path = os.path.join(Config.checkpoint_dir, f'discr_X_epoch_{epoch+1}.pth')
        discr_Y_path = os.path.join(Config.checkpoint_dir, f'discr_Y_epoch_{epoch+1}.pth')
        torch.save(gen_X_to_Y.state_dict(), gen_X_to_Y_path)
        torch.save(gen_Y_to_X.state_dict(), gen_Y_to_X_path)
        torch.save(discr_X.state_dict(), discr_X_path)
        torch.save(discr_Y.state_dict(), discr_Y_path)

        # Log model checkpoints to Weights & Biases as artifacts
        wandb.save(gen_X_to_Y_path)
        wandb.save(gen_Y_to_X_path)
        wandb.save(discr_X_path)
        wandb.save(discr_Y_path)

# Finish the wandb run
wandb.finish()