import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST dataset
Mnist_train = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

# Initialize an aggregate density map

plt.figure(figsize=(20, 10))
for i in range(10):

    digits = [image.squeeze() for image, label in Mnist_train if label == i]
    digits_torch  = digits[0]
    # cutoff any values less than (200/255) to 0
    digits_torch[digits_torch < (250/255)] = 0
    density = digits_torch  # Sum the images to get the density map
    #digits_torch = torch.stack(digits)  # Stack all the images into a single tensor
    #density = digits_torch.sum(dim=0)  # Sum the images to get the density map
    torch.save(
        density, f"./Densities/density_map_{i}.pt"
    )  # Save the density map to a file
    #
    print(density.shape)
    plt.subplot(2, 5, i + 1)
    plt.imshow(density, cmap="hot")
    plt.title(f"Density map for digit {i}")

plt.tight_layout()
plt.show()
