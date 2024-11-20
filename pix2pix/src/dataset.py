import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# dataset function
class Pix2PixDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        
        # sort input and target to make sure it is paired
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))

        # check that there is an matching number of input/targets
        assert len(self.input_files) == len(self.target_files), "Mismatch in input and target files."

    def __len__(self):
        # number of pair
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # get image paths
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        # open images
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        # apply tranformation
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        # return directory with paired images
        return {"input": input_image, "target": target_image}