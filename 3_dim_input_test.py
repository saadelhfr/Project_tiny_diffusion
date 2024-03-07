import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from layers.model import *
from utils.transforms import ImageToNormalizedCoordinatesTransform
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from layers.seq2seq import Seq2Seq
from utils.training import *

# Variable intialisation
DEPTH = 3
SIZE = 256
HIDDEN_DIM = 256
OUTPUT_DIM = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 3
PATH = "Seq2Seq.pth"
# download the MNIST :
transform = transforms.Compose(
    [ToTensor(), ImageToNormalizedCoordinatesTransform(keep_non_zero_only=False)]
)

Mnist_train = datasets.MNIST(
    root="./data_sequence_all", train=True, download=True, transform=transform
)

MNIST_loader = DataLoader(Mnist_train, batch_size=64, shuffle=True)


Model = Seq2Seq(
    depth=DEPTH,
    size=SIZE,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    input_size=INPUT_SIZE,
    device=DEVICE,
)

noise_scheduler_instance = Noise_Scheduler(beta_schedule="sinusoidal", device=DEVICE)

for batch, _ in MNIST_loader:
    print(batch)
    break

trainer_instance = Trainer(
    Model,
    noise_scheduler_instance,
    torch.optim.Adam(Model.parameters(), lr=0.01),
    nn.MSELoss(),
    DEVICE,
    save_path=PATH,
)

print(len(MNIST_loader))
losses, frames = trainer_instance.train_seq2seq(
    num_epochs=100,
    batch_size=64,
    gradient_clipthres=1.0,
    train_loader=MNIST_loader,
)


print(losses)
print(frames)

# save this stuff

frames_np = np.stack(frames)
losses_np = np.array(losses)

np.save("frames.npy", frames_np)
np.save("losses.npy", losses_np)
