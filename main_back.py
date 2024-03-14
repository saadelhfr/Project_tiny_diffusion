import torch
import os
import numpy as np
from src.layers import MLP, Noise_Scheduler
import matplotlib.pyplot as plt
from src.utils.datasets import get_dataset
from src.utils.training import Trainer

import argparse


parser = argparse.ArgumentParser(
    description="create png plots a model on a specific digit from the mnist dataset"
)
parser.add_argument(
    "--digit",
    type=int,
    default=5,
    help="The MNIST digit to generate frames for (default: 5)",
)

args = parser.parse_args()

digit_tomake = args.digit

savingModel_path = f"./BestModels/best_model{digit_tomake}.pth"

device = torch.device("cuda")
dataset_name = "mnist"

Model = MLP(depth=10, size=256, hidden_dim=256, output_dim=2, device="cuda")

noise_scheduler_instance = Noise_Scheduler(beta_schedule="quadratic", device="cuda")

dataset_instance = get_dataset(dataset_name, n=100000, digit=digit_tomake)

dataset_loader = torch.utils.data.DataLoader(
    dataset_instance, batch_size=64, shuffle=True
)

trainer_instance = Trainer(
    Model,
    noise_scheduler_instance,
    torch.optim.Adam(Model.parameters(), lr=0.001),
    torch.nn.MSELoss(),
    device,
    save_path=savingModel_path,
)

losses, frames = trainer_instance.train(
    num_epochs=100, batch_size=64, gradient_clipthres=1, train_loader=dataset_loader
)

# prepare a directory to store the frames and the losses
# plot the losses and save the plot as a png


outdir = f"OutputsMnsit2_{digit_tomake}"
print("Saving images...")
imgdir = f"{outdir}/images"
os.makedirs(imgdir, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("Outputs/loss.png")

frames = np.stack(frames)
if dataset_name == "mnist":
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    s = 1
else:
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    s = 2
for i, frame in enumerate(frames):
    plt.figure(figsize=(10, 10))
    plt.scatter(frame[:, 0], frame[:, 1], s=s)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.savefig(f"{imgdir}/{i:04}.png")
    plt.close()

print("Saving loss as numpy array...")
np.save(f"{outdir}/loss.npy", np.array(losses))

print("Saving frames...")
np.save(f"{outdir}/frames.npy", frames)

# write unite test for the model
