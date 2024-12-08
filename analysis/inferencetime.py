import torch
from torchvision import transforms
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import sys
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################
#### Bring in the Model Architectures for each
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pix2pix.src.model import GeneratorUNet as GEN_pix2pix_UNet # goes X->Y or Y->X
from cycleGAN.src.model import Generator as GEN_cycle_UNet # goes X->Y or Y->X
from rerecycleGAN.generators import UNet as GEN_rerecycle_UNet # goes X->Y or Y->X
from rerecycleGAN.generators import ResNet as GEN_rerecycle_ResNet # goes X1->X2 or Y1->Y2

################################################
#### Bring in the Weights & Biases to fill the Architectures for each
run = wandb.init()

# HOW TO USE <------------------  <------------------  <------------------
# Download your desired model & put it into the "/analysis/models" folder.

# Load a PTH file from my /models folder into torch 
PTH_pix2pix_UNet = torch.load("analysis/models/generator_epoch_130.pth", map_location=torch.device('cpu'))
PTH_cycle_UNet = torch.load("analysis/models/gen_X_to_Y_epoch_200.pth", map_location=torch.device('cpu'))
# PTH_rerecycle_UNet = torch.load("models/")
# PTH_rerecycle_ResNet = torch.load("models/")

# load the best_model into an instance of GEN_pix2pix_UNet
model_pix2pix_UNet = GEN_pix2pix_UNet().to(device)
model_pix2pix_UNet.load_state_dict(PTH_pix2pix_UNet)
model_pix2pix_UNet.eval()

model_cycle_UNet = GEN_cycle_UNet().to(device)
model_cycle_UNet.load_state_dict(PTH_cycle_UNet)
model_cycle_UNet.eval()

# model_rerecycle_UNet = GEN_rerecycle_UNet().to(device)  # TODO: Uncomment when ReReCycleGAN is ready.
# model_rerecycle_UNet.load_state_dict(torch.load(PTH_rerecycle_UNet.name))
# model_rerecycle_UNet.eval()

# model_rerecycle_ResNet = GEN_rerecycle_ResNet().to(device)
# model_rerecycle_ResNet.load_state_dict(torch.load(PTH_rerecycle_ResNet.name))
# model_rerecycle_ResNet.eval()


################################################
#### Bring in the Image to test the Model

# Load and preprocess the image                   .
image_path = 'analysis/data/frame_0_real.png'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])
input_image = transform(image).unsqueeze(0).to(device)

################################################
#### Run inference
with torch.no_grad():
    output_pix2pix = model_pix2pix_UNet(input_image)
    output_cycle = model_cycle_UNet(input_image)
    # output_rerecycle_UNet = model_rerecycle_UNet(input_image)  # TODO: Uncomment when ReReCycleGAN is ready.
    # output_rerecycle_ResNet = model_rerecycle_ResNet(input_image)

################################################
#### Post-process and save the output image
output_pix2pix = output_pix2pix.cpu().squeeze(0)
output_pix2pix = output_pix2pix * 0.5 + 0.5  # De-normalize
output_pix2pix = transforms.ToPILImage()(output_pix2pix)
output_pix2pix.save('output_pix2pix.jpg')

output_cycle = output_cycle.cpu().squeeze(0)
output_cycle = output_cycle * 0.5 + 0.5  # De-normalize
output_cycle = transforms.ToPILImage()(output_cycle)
output_cycle.save('output_cycle.jpg')

# output_rerecycle_UNet = output_rerecycle_UNet.cpu().squeeze(0)  # TODO: Uncomment when ReReCycleGAN is ready.
# output_rerecycle_UNet = output_rerecycle_UNet * 0.5 + 0.5  # De-normalize
# output_rerecycle_UNet = transforms.ToPILImage()(output_rerecycle_UNet)
# output_rerecycle_UNet.save('output_rerecycle_UNet.jpg')

# output_rerecycle_ResNet = output_rerecycle_ResNet.cpu().squeeze(0)
# output_rerecycle_ResNet = output_rerecycle_ResNet * 0.5 + 0.5  # De-normalize
# output_rerecycle_ResNet = transforms.ToPILImage()(output_rerecycle_ResNet)
# output_rerecycle_ResNet.save('output_rerecycle_ResNet.jpg')

################################################
#### Display the output image
plt.figure(figsize=(12, 12))
plt.subplot(1, 5, 1)
plt.imshow(image)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(output_pix2pix)
plt.title('Output Image (pix2pix)')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(output_cycle)
plt.title('Output Image (cycleGAN)')
plt.axis('off')

# plt.subplot(1, 5, 4)                          #TODO: Uncomment when ReReCycleGAN is ready.
# plt.imshow(output_rerecycle_UNet)
# plt.title('Output Image (ReReCycleGAN UNet)')
# plt.axis('off')

# plt.subplot(1, 5, 5)
# plt.imshow(output_rerecycle_ResNet)
# plt.title('Output Image (ReReCycleGAN ResNet)')
# plt.axis('off')

plt.show()

