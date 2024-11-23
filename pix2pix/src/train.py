import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
import torchvision.utils as vutils  # For image logging
# custom classes
from model import GeneratorUNet, DiscriminatorPatchGAN
from dataset import Pix2PixDataset
from config import Config


# Create necessary directories
os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.log_dir, exist_ok=True)

# tensorboard writer
writer = SummaryWriter(Config.log_dir)

# Dataset and DataLoader
train_dataset = Pix2PixDataset(
    input_dir=Config.train_input_dir,
    target_dir=Config.train_target_dir,
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
generator = GeneratorUNet().to(Config.device)
discriminator = DiscriminatorPatchGAN().to(Config.device)

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


initialize_weights(generator)
initialize_weights(discriminator)

# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss().to(Config.device)
criterion_L1 = nn.L1Loss().to(Config.device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=Config.learning_rate, betas=(Config.beta1, Config.beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=Config.learning_rate, betas=(Config.beta1, Config.beta2))

# Learning rate schedulers
lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1 - epoch / Config.num_epochs)
lr_scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 1 - epoch / Config.num_epochs)

# Select a fixed sample for consistent monitoring
fixed_sample = train_dataset[0]  # Change the index to select a different sample
fixed_input = fixed_sample['input'].unsqueeze(0).to(Config.device)  # Add batch dimension
fixed_target = fixed_sample['target'].unsqueeze(0).to(Config.device)  # Add batch dimension

# Training loop
for epoch in range(Config.num_epochs):
    generator.train()
    discriminator.train()
    
    for batch_idx, batch in enumerate(train_loader):
        # Get input and target images
        input_image = batch['input'].to(Config.device)
        target_image = batch['target'].to(Config.device)
        
        # Labels for real and fake images
        real_label = torch.ones((input_image.size(0), 1, 30, 30), device=Config.device)
        fake_label = torch.zeros((input_image.size(0), 1, 30, 30), device=Config.device)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        optimizer_G.zero_grad()
        
        # Generate images
        fake_image = generator(input_image)
        
        # Discriminator evaluates the fake images
        pred_fake = discriminator(input_image, fake_image)
        
        # Generator adversarial loss
        loss_GAN = criterion_GAN(pred_fake, real_label)
        
        # L1 loss
        loss_L1 = criterion_L1(fake_image, target_image) * Config.lambda_L1
        
        # Total generator loss
        loss_G = loss_GAN + loss_L1
        loss_G.backward()
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real images
        pred_real = discriminator(input_image, target_image)
        loss_D_real = criterion_GAN(pred_real, real_label)
        
        # Fake images (detach to avoid training generator on these gradients)
        pred_fake = discriminator(input_image, fake_image.detach())
        loss_D_fake = criterion_GAN(pred_fake, fake_label)
        
        # Total discriminator loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        #  Logging
        # ---------------------

        batches_done = epoch * len(train_loader) + batch_idx # total number of batches processed so far
        if batches_done % 100 == 0:
            # log scalar metrics
            writer.add_scalar('Loss/Generator_GAN', loss_GAN.item(), batches_done)
            writer.add_scalar('Loss/Generator_L1', loss_L1.item(), batches_done)
            writer.add_scalar('Loss/Generator_Total', loss_G.item(), batches_done)
            writer.add_scalar('Loss/Discriminator', loss_D.item(), batches_done)

            # log learning rates
            current_lr_G = optimizer_G.param_groups[0]['lr']
            current_lr_D = optimizer_D.param_groups[0]['lr']
            writer.add_scalar('Learning Rate/Generator', current_lr_G, batches_done)
            writer.add_scalar('Learning Rate/Discriminator', current_lr_D, batches_done)

            # log images
            with torch.no_grad():
                generator.eval()
                fake_sample = generator(fixed_input)
                
                # denormalize!
                def denormalize(tensor):
                    return tensor * 0.5 + 0.5
                
                input_images = denormalize(fixed_input.cpu())
                target_images = denormalize(fixed_target.cpu())
                output_images = denormalize(fake_sample.cpu())
                
                # Create image grids
                img_grid_input = vutils.make_grid(input_images, normalize=False)
                img_grid_target = vutils.make_grid(target_images, normalize=False)
                img_grid_fake = vutils.make_grid(output_images, normalize=False)
                
                # Log images to TensorBoard
                writer.add_image('Input Images', img_grid_input, batches_done)
                writer.add_image('Target Images', img_grid_target, batches_done)
                writer.add_image('Fake Images', img_grid_fake, batches_done)
                
                generator.train()

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    
    # Save model checkpoints every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(Config.checkpoint_dir, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(Config.checkpoint_dir, f'discriminator_epoch_{epoch+1}.pth'))