import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import numpy as np
from layers.model import MLP, Noise_Scheduler
from layers.positional_embedding import Sinusoidal_embedding_trans
import matplotlib.pyplot as plt
from utils.training import Trainer
from utils.transforms import MNISTToFlattenedTransform
from utils.datasets import get_dataset
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
digit = args.digit


directory_outputs = f"./OutputsMnsit2_{digit}/"
frames_numpy = np.load(directory_outputs + "frames.npy")


print(frames_numpy.shape)

xmax = 1
xmin = 0

ymax = 1
ymin = 0


plt.figure(figsize=(20, 10))
for i in range(frames_numpy.shape[0]):

    data_np = frames_numpy[i]

    ax = plt.subplot(2, 6, i + 1)

    # Create scatter plot for the digit
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.scatter(data_np[:, 0], data_np[:, 1], s=0.5)
    ax.set_title(f"Epoch {i}")
    ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a png
plt.savefig(directory_outputs + "frames.png")

# now we change this into an image

# Assuming frames_numpy is an array of shape (num_frames, num_points, 2)
# where the last dimension holds (x, y) coordinates for each point

num_frames = frames_numpy.shape[0]
grid = np.zeros((num_frames, 28, 28))

# Scale points from [0, 1) to [0, 28)
scale_frames = np.floor(frames_numpy * 28).astype(int)

# Populate the grid with counts
for j in range(num_frames):
    for point in scale_frames[j]:
        x, y = point
        if 0 <= x < 28 and 0 <= y < 28:
            grid[j, y, x] += 1

# Normalize each frame individually to enhance contrast
normalized_grid = np.zeros_like(grid)
for j in range(num_frames):
    max_count = grid[j].max() if grid[j].max() > 0 else 1  # Avoid division by zero
    normalized_grid[j] = (grid[j] / max_count) * 255

# Visualization
plt.figure(figsize=(20, 10))
for i in range(num_frames):
    ax = plt.subplot(2, int(np.ceil(num_frames / 2)), i + 1)  # Adjust subplot layout
    ax.imshow(normalized_grid[i], cmap="gray", interpolation="nearest")
    ax.set_title(f"Epoch {i}")
    ax.axis("off")  # Simplified way to hide both x and y ticks

plt.tight_layout()
plt.savefig(directory_outputs + "frames_as_images.png")  # Adjust path as needed
