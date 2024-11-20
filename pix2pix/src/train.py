import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
# custom classes
from model import GeneratorUNet, DiscriminatorPatchGAN
from dataset import Pix2PixDataset
from config import Config

# Create necessary directories
os.makedirs(Config.checkpoint_dir, exist_ok=True)
os.makedirs(Config.sample_dir, exist_ok=True)

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset and DataLoader
train_dataset = Pix2PixDataset(
    input_dir=Config.train_input_dir,
    target_dir=Config.train_target_dir,
    transform=data_transforms
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
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
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
    
    loop = tqdm(train_loader, leave=True)
    for batch_idx, batch in enumerate(loop):
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
        
        # Update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{Config.num_epochs}]")
        loop.set_postfix(
            loss_G=loss_G.item(),
            loss_D=loss_D.item()
        )
    
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    
        # Save model checkpoints and sample images every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(Config.checkpoint_dir, f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(Config.checkpoint_dir, f'discriminator_epoch_{epoch+1}.pth'))
        
        # Save sample outputs from the fixed sample
        generator.eval()
        with torch.no_grad():
            fake_sample = generator(fixed_input)
            
            # Denormalize images (assuming normalization was mean=0.5, std=0.5)
            def denormalize(tensor):
                return tensor * 0.5 + 0.5
            
            input_images = denormalize(fixed_input.cpu())
            target_images = denormalize(fixed_target.cpu())
            output_images = denormalize(fake_sample.cpu())
            
            # Concatenate images horizontally (input, output, target)
            combined = torch.cat((input_images[0], output_images[0], target_images[0]), dim=2)  # Concatenate along width
            
            # Save the combined image
            save_path = os.path.join(Config.sample_dir, f'epoch_{epoch+1}_sample.png')
            transforms.ToPILImage()(combined).save(save_path)