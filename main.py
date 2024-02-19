import torch
import os
import numpy as np
from layers.model import MLP, Noise_Scheduer
import matplotlib.pyplot as plt
from utils.datasets import get_dataset
from utils.training import Trainer


Model = MLP(depth=5, size=128, hidden_dim=128, output_dim=2)

noise_scheduler_instance = Noise_Scheduer()

dataset_instance = get_dataset("moons")

dataset_loader = torch.utils.data.DataLoader(
    dataset_instance, batch_size=64, shuffle=True
)

trainer_instance = Trainer(
    Model,
    noise_scheduler_instance,
    torch.optim.Adam(Model.parameters(), lr=0.001),
    torch.nn.MSELoss(),
    "cpu",
)

losses, frames = trainer_instance.train(
    num_epochs=100, batch_size=64, gradient_clipthres=1, train_loader=dataset_loader
)

# prepare a directory to store the frames and the losses
# plot the losses and save the plot as a png


outdir = "Outputs"
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
xmin, xmax = -6, 6
ymin, ymax = -6, 6
for i, frame in enumerate(frames):
    plt.figure(figsize=(10, 10))
    plt.scatter(frame[:, 0], frame[:, 1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.savefig(f"{imgdir}/{i:04}.png")
    plt.close()

print("Saving loss as numpy array...")
np.save(f"{outdir}/loss.npy", np.array(losses))

print("Saving frames...")
np.save(f"{outdir}/frames.npy", frames)

# write unite test for the model
