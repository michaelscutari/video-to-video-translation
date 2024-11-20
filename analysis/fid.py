import numpy as np
import torch
import torchvision.models as models
from torchvision.transforms import transforms
from PIL import Image
from scipy.linalg import sqrtm

# implementation of fid (frechet inception distance) score
def fid_score(real_samples, fake_samples):
    # check if metal is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # load inception v3 for feature extraction
    inception = models.inception_v3(weights='DEFAULT')
    inception.eval()
    inception.to(device)

    # remove the last classification layer
    inception.fc = torch.nn.Identity()

    # inception requires images = [B, 3, 299, 299]
    # and images normalized to inception stats
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    real_samples = torch.stack([transform(img) for img in real_samples]).to(device)
    fake_samples = torch.stack([transform(img) for img in fake_samples]).to(device)

    # extract features
    with torch.no_grad():
        real_features = inception(real_samples)
        fake_features = inception(fake_samples)

    # calculate mean and covariance statistics
    mu_real = torch.mean(real_features, dim=0)
    mu_fake = torch.mean(fake_features, dim=0)
    sigma_real = torch.cov(real_features.T)
    sigma_fake = torch.cov(fake_features.T)

    # calculate fid
    # fid = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2 * sqrt(sigma_real * sigma_fake))
    diff = mu_real - mu_fake
    covmean, _ = sqrtm((sigma_real @ sigma_fake).cpu().numpy(), disp=False)
    covmean = torch.from_numpy(covmean.real).to(device)
    fid = (diff @ diff + torch.trace(sigma_real + sigma_fake - 2 * covmean)).item()
    
    return fid

# example usage
image_real_out = [Image.open('data/frame_0_real.png')]
image_fake_out = [Image.open('data/frame_0_fake.png')]

fid = fid_score(image_real_out, image_fake_out)

print(f'FID: {fid}')




