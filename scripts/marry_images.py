import os
from PIL import Image

# Define directories
input_dir = '/Users/peterbanyas/Desktop/ECE 661/Project 661/ece661-GAN-project/analysis/data/mufasa_lost/'
output_dir = '/Users/peterbanyas/Desktop/ECE 661/Project 661/ece661-GAN-project/analysis/output/mufasa_lost/'
merged_dir = '/Users/peterbanyas/Desktop/ECE 661/Project 661/ece661-GAN-project/analysis/merged/mufasa_lost/'

# Create merged directory if it doesn't exist
os.makedirs(merged_dir, exist_ok=True)

# Get list of files in input and output directories
input_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
output_files = sorted([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])

# Ensure the number of files match
print(f"{len(input_files)} vs {len(output_files)}")
assert len(input_files) == len(output_files), "Mismatch in number of files between input and output directories"

# Iterate through files and merge images
for input_file, output_file in zip(input_files, output_files):
    input_image = Image.open(os.path.join(input_dir, input_file))
    output_image = Image.open(os.path.join(output_dir, output_file))

    # Ensure images have the same height
    assert input_image.size[1] == output_image.size[1], "Mismatch in image heights"

    # Create a new image with width = sum of both images' widths and height = height of the images
    merged_image = Image.new('RGB', (input_image.size[0] + output_image.size[0], input_image.size[1]))

    # Paste the images side-by-side
    merged_image.paste(input_image, (0, 0))
    merged_image.paste(output_image, (input_image.size[0], 0))

    # Save the merged image
    merged_image.save(os.path.join(merged_dir, input_file))

print("Images merged successfully!")