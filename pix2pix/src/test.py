from PIL import Image
from torchvision import transforms

image_path = "../data/train/input/1_input.jpg"

image = Image.open(image_path)

to_tensor = transforms.ToTensor()

image = to_tensor(image)

# add batch dim
image = image.unsqueeze(0)

from model import GeneratorUNet

model = GeneratorUNet()

model.eval()
output_image = model(image).squeeze(0)

to_pil = transforms.ToPILImage()

output_image = to_pil(output_image)

output_image.save("out.png")