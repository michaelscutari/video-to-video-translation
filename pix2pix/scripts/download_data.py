import kagglehub

# Download the latest version of the dataset to the specified path
dataset_path = kagglehub.dataset_download("vikramtiwari/pix2pix-dataset")

print("Path to dataset files:", dataset_path)
