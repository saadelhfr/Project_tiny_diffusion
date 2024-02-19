import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import numpy as np
from layers.model import MLP
from layers.positional_embedding import Sinusoidal_embedding_trans
import matplotlib.pyplot as plt
from utils.training import Trainer
from utils.transforms import MNISTToFlattenedTransform


# Load MNIST dataset
Mnist_train = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

# Initialize an aggregate density map

plt.figure(figsize=(20, 10))
for i in range(10):

    digits = [image.squeeze() for image, label in Mnist_train if label == i]

    digits_torch = torch.stack(digits)  # Stack all the images into a single tensor
    density = digits_torch.sum(dim=0)  # Sum the images to get the density map
    torch.save(
        density, f"./Densities/density_map_{i}.pt"
    )  # Save the density map to a file
    #
    plt.subplot(2, 5, i + 1)
    plt.imshow(density, cmap="hot")
    plt.title(f"Density map for digit {i}")

plt.tight_layout()
plt.show()
