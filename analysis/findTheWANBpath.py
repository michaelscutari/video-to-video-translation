# run this to list the paths to the runs
import wandb

api = wandb.Api()
runs = api.runs("michael-scutari-duke-university/pix2pix-map2sat")

for run in runs:
    print(run.path)