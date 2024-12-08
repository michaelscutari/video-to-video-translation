import torch
from torchvision import transforms
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import sys
import os

################################################
###### HOW TO USE!
################################################
# MODEL: Download your desired model & put it into the "/analysis/models" folder.
# VIDEO: Put your video into the "/scripts/videos" folder.
#        Run video_to_imgs.py with the right video path to extract the frames.
#        The frames will be saved in the "/analysis/data/YOUR_TITLE_HERE/" folder.

# Define Directories
input_dir = 'analysis/data/lion_vid_chase'
output_dir = 'analysis/output/lion_vid_chase'

pix_model_dir = 'analysis/models/generator_epoch_130.pth'
cycle_model_dir = 'analysis/models/gen_Y_to_X_epoch_150.pth'
rerecycle_UNet_model_dir = ''
rerecycle_ResNet_model_dir = ''

# which models to use?
yesPIX2PIX = False
yesCYCLE = True
yesRERECYCLE = False


# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


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

# Load a PTH file from my /models folder into torch 
if yesPIX2PIX:
    PTH_pix2pix_UNet = torch.load(pix_model_dir, map_location=torch.device('cpu'))
if yesCYCLE:
    PTH_cycle_UNet = torch.load(cycle_model_dir, map_location=torch.device('cpu'))
if yesRERECYCLE:
    PTH_rerecycle_UNet = torch.load("models/", map_location=torch.device('cpu'))
    PTH_rerecycle_ResNet = torch.load("models/", map_location=torch.device('cpu'))

# load the best_model into an instance of GEN_pix2pix_UNet
if yesPIX2PIX:
    model_pix2pix_UNet = GEN_pix2pix_UNet().to(device)
    model_pix2pix_UNet.load_state_dict(PTH_pix2pix_UNet)
    model_pix2pix_UNet.eval()

if yesCYCLE:
    model_cycle_UNet = GEN_cycle_UNet().to(device)
    model_cycle_UNet.load_state_dict(PTH_cycle_UNet)
    model_cycle_UNet.eval()

if yesRERECYCLE:
    model_rerecycle_UNet = GEN_rerecycle_UNet().to(device)  # TODO: Uncomment when ReReCycleGAN is ready.
    model_rerecycle_UNet.load_state_dict(torch.load(PTH_rerecycle_UNet.name))
    model_rerecycle_UNet.eval()

    model_rerecycle_ResNet = GEN_rerecycle_ResNet().to(device)
    model_rerecycle_ResNet.load_state_dict(torch.load(PTH_rerecycle_ResNet.name))
    model_rerecycle_ResNet.eval()


################################################
#### Bring in the Video 

# get the list of input image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# define the preprocessing transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])

# Run inference on each image
if yesPIX2PIX:  
    model_pix2pix_UNet.eval()   
if yesCYCLE:
    model_cycle_UNet.eval()
if yesRERECYCLE:
    model_rerecycle_UNet.eval()  # TODO: Uncomment when ReReCycleGAN is ready.
    model_rerecycle_ResNet.eval()


with torch.no_grad():
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)

        if yesPIX2PIX:
            output_pix2pix = model_pix2pix_UNet(input_image)
            output_pix2pix = output_pix2pix.cpu().squeeze(0)
            output_pix2pix = output_pix2pix * 0.5 + 0.5  # De-normalize
            output_pix2pix = transforms.ToPILImage()(output_pix2pix)
            output_pix2pix.save(os.path.join(output_dir, 'output_pix2pix_' + image_file))

        if yesCYCLE:
            output_cycle = model_cycle_UNet(input_image)
            output_cycle = output_cycle.cpu().squeeze(0)
            output_cycle = output_cycle * 0.5 + 0.5  # De-normalize
            output_cycle = transforms.ToPILImage()(output_cycle)
            output_cycle.save(os.path.join(output_dir, 'output_cycle_' + image_file))

        if yesRERECYCLE:
            output_rerecycle_UNet = model_rerecycle_UNet(input_image)
            output_rerecycle_UNet = output_rerecycle_UNet.cpu().squeeze(0)
            output_rerecycle_UNet = output_rerecycle_UNet * 0.5 + 0.5  # De-normalize
            output_rerecycle_UNet = transforms.ToPILImage()(output_rerecycle_UNet)

            output_rerecycle_ResNet = model_rerecycle_ResNet(input_image)
            output_rerecycle_ResNet = output_rerecycle_ResNet.cpu().squeeze(0)
            output_rerecycle_ResNet = output_rerecycle_ResNet * 0.5 + 0.5  # De-normalize
            output_rerecycle_ResNet = transforms.ToPILImage()(output_rerecycle_ResNet)
