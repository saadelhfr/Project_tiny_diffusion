from torch.utils.data import TensorDataset
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
from utils.datasets import get_dataset
from utils.training import *

torch.manual_seed(64)
# Variable intialisation
DEPTH = 1
SIZE = 256
HIDDEN_DIM = 256
OUTPUT_DIM = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 2

PATH = "Seq2Seq.pth"


dataset_moons = get_dataset("moons", 50000)
dataset_circle = get_dataset("circle", 50000)
points_perseries = 500
nbr_series = (50000 // points_perseries) * 2
dataset1 = []
dataset2 = []

for i in range(nbr_series):
    start = i * points_perseries
    end = (i + 1) * points_perseries
    sliced_data1 = dataset_circle[start:end][0]
    sliced_data2 = dataset_moons[start:end][0]
    if sliced_data2.shape[0] == 0:
        pass
    else:
        dataset2.append(sliced_data2)
    if sliced_data1.shape[0] == 0:
        pass
    else:
        dataset1.append(sliced_data1)

print(len(dataset1))
print(len(dataset2))
dataset_final = []
dataset_final.extend(dataset2)
dataset_final.extend(dataset1)
print(len(dataset_final))
for element in dataset_final:
    print(element.shape)
tensor_data_final = TensorDataset(torch.stack(dataset_final))

data_loader = DataLoader(tensor_data_final, batch_size=64, shuffle=True)
print(len(data_loader))

Model = Seq2Seq(
    depth=DEPTH,
    size=SIZE,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    input_size=INPUT_SIZE,
    device=DEVICE,
)

noise_scheduler_instance = Noise_Scheduler(beta_schedule="linear", device=DEVICE)

for batch in data_loader:
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

print(len(data_loader))
losses, frames = trainer_instance.train_seq2seq(
    num_epochs=100,
    batch_size=64,
    gradient_clipthres=10.0,
    train_loader=data_loader,
)


print(losses)
print(frames)

# save this stuff

frames_np = np.stack(frames)
losses_np = np.array(losses)

np.save("framesbiglr2linearbeta.npy", frames_np)
np.save("lossesbiglr2lienarbeta.npy", losses_np)
