import wandb
import shutil
import os
run = wandb.init()
artifact = run.use_artifact('michael-scutari-duke-university/rerecycleGAN/model-fi966rkw:v29', type='model')
artifact_dir = artifact.download()
print(artifact_dir)

destination_dir = '/Users/peterbanyas/Desktop/ECE 661/Project 661/ece661-GAN-project/analysis/models/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for filename in os.listdir(artifact_dir):
    shutil.move(os.path.join(artifact_dir, filename), destination_dir)