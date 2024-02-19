import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import numpy as np
from layers.model import MLP, Noise_Scheduer
from layers.positional_embedding import Sinusoidal_embedding_trans
import matplotlib.pyplot as plt
from utils.training import Trainer
from utils.transforms import MNISTToFlattenedTransform
from utils.datasets import get_dataset


plt.figure(figsize=(20, 10))
for i in range(0, 10):
    # Load the dataset for the digit 'i'
    data_set = get_dataset(name="mnist", n=10000, digit=i)

    # Convert the dataset to a numpy array
    data_np = data_set.tensors[0].detach().numpy()

    # Determine the subplot index (5 rows, 2 columns)
    ax = plt.subplot(2, 5, i + 1)

    # Create scatter plot for the digit
    ax.scatter(data_np[:, 0], data_np[:, 1], s=1)
    ax.set_title(f"Digit {i}")
    ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
