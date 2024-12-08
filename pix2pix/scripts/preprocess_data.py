import os
import shutil
from PIL import Image

# original data path
# original_data_path = "/home/users/mas296/.cache/kagglehub/datasets/vikramtiwari/pix2pix-dataset/versions/2/cityscapes/cityscapes/"
# original_data_path = "../data/sat2map"
original_data_path = "../data/"
output_data_path = "../data/s_preprocessed"

# subdirectories
subdirectories = [""]

# loop
for subdir in subdirectories:
    # get full paths
    input_dir = os.path.join(original_data_path, subdir)
    output_dir = os.path.join(output_data_path, subdir)
    # make new dirs for input and output images
    input_img_dir = os.path.join(output_dir, "input")
    output_img_dir = os.path.join(output_dir, "output")
    # delete old preprocessing
    if os.path.exists(output_img_dir):
        shutil.rmtree(output_img_dir)
    # make new dirs
    os.makedirs(input_img_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    # process each image
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # load image
            filepath = os.path.join(input_dir, filename)
            image = Image.open(filepath)
            width, height = image.size
            # ensure even size for split
            if width % 2 != 0:
                raise ValueError(f"Image width is not even for {filename}")
            # split image
            left_half = image.crop((0, 0, width // 2, height))
            right_half = image.crop((width // 2, 0, width, height))
            # Save each half with a consistent naming pattern
            base_name = os.path.splitext(filename)[0]
            right_half.save(os.path.join(output_img_dir, f"{base_name}_output.jpg"))
            left_half.save(os.path.join(input_img_dir, f"{base_name}_input.jpg"))

print("Done preprocessing!")