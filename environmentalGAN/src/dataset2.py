import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# dataset function
class CycleGANDataset(Dataset):
    def __init__(self, X_dir, Y_dir, transform=None):
        self.X_dir = X_dir
        self.Y_dir = Y_dir
        self.transform = transform
        
        # sort X and Y files to ensure consistency
        self.X_files = sorted(os.listdir(X_dir))
        self.Y_files = sorted(os.listdir(Y_dir))

        # check that there is at least one X and one Y file
        assert len(self.X_files) > 0 and len(self.Y_files) > 0, "Empty directories for X or Y images."

    def __len__(self):
        # return number of X images as the dataset length
        return len(self.X_files)
    
    def __getitem__(self, idx):
        # get X image path
        X_path = os.path.join(self.X_dir, self.X_files[idx])
        
        # select a random index for Y
        random_idx = random.randint(0, len(self.Y_files) - 1)
        Y_path = os.path.join(self.Y_dir, self.Y_files[random_idx])

        # open images
        X_image = Image.open(X_path).convert("RGB")
        Y_image = Image.open(Y_path).convert("RGB")

        # apply transformation
        if self.transform:
            X_image = self.transform(X_image)
            Y_image = self.transform(Y_image)

        # return dictionary with paired images
        return {"X": X_image, "Y": Y_image}