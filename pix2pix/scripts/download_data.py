import kagglehub

# Download the latest version of the dataset to the specified path
dataset_path = kagglehub.dataset_download("balraj98/monet2photo")

print("Path to dataset files:", dataset_path)

# move /Users/peterbanyas/.cache/kagglehub/datasets/vikramtiwari/pix2pix-dataset/versions/2 to /Users/peterbanyas/Desktop/ECE 661/Project 661/ece661-GAN-project/data
shutil.move(dataset_path, '/Users/peterbanyas/Desktop/ECE 661/Project 661/ece661-GAN-project/data')