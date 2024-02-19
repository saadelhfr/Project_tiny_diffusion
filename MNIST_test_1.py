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

# download the MNIST :

coords_transform = MNISTToFlattenedTransform()

transform_pipeline = transforms.Compose(
    [coords_transform]
)  # this will transform the image into a sparse representation


Mnist_train = datasets.MNIST(
    root="./data_transformed", train=True, transform=transform_pipeline, download=True
)
Mnist_test = datasets.MNIST(
    root="./data_transformed", train=False, transform=transform_pipeline, download=True
)

# tak the first image  :
Model = MLP(depth=5, size=128, hidden_dim=128, output_dim=1, input_size=1)
noise_scheduler_instance = Noise_Scheduer()

dataset = Mnist_train[0][0]  # take the first image


dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

trainer_instance = Trainer(
    Model,
    noise_scheduler_instance,
    torch.optim.Adam(Model.parameters(), lr=0.001),
    torch.nn.MSELoss(),
    "cpu",
)

losses, frames = trainer_instance.train(
    num_epochs=200, batch_size=64, gradient_clipthres=1, train_loader=dataset_loader
)


# transfor the frames into images and store them  :


output_dir = "OutputsMnist"
image_dir = f"{output_dir}/images"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
# make plot of the losses
losees = np.array(losses)

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(f"{output_dir}/loss.png")

# save the npy file of losses
np.save(f"{output_dir}/loss.npy", losses)

frames = np.stack(frames)

for i, frame in enumerate(frames):
    plt.figure(figsize=(10, 10))
    plt.imshow(frame.reshape(28, 28), cmap="gray")
    plt.savefig(f"{output_dir}/images/{i:04}.png")
    plt.close()


np.save(f"{output_dir}/frames.npy", frames)

original_image = dataset.reshape(28, 28)
plt.figure(figsize=(10, 10))
plt.imshow(original_image, cmap="gray")
plt.savefig(f"{output_dir}/original_image.png")
