import torch
from torchvision import datasets, transforms
import os
import numpy as np
from src.layers import MLP, Noise_Scheduler
import matplotlib.pyplot as plt
from src.utils.training import Trainer
from src.utils.transforms import MNISTToFlattenedTransform

savingModel_path = "Outputs/BestModels/best3_model.pth"

device = torch.device("cuda")
dataset_name = "mnist"


coords_transform = MNISTToFlattenedTransform()
transform_pipeline = transforms.Compose(
    [coords_transform]
)  # this will transform the image into a sparse representation


Mnist_train = datasets.MNIST(
    root="./data_MNIST_transformed", transform=coords_transform, train=True, download=True
)
Mnist_test = datasets.MNIST(
    root="./data_MNIST_transformed", transform=coords_transform, train=False, download=True
)

imagesList = [Mnist_train[i][0].squeeze() for i in range(len(Mnist_train))]
shape = imagesList[0].shape[0]
images = torch.cat(imagesList, dim=0).unsqueeze(1)
# print("Mnist_train", Mnist_train[0])
DataLoaderInstance = torch.utils.data.DataLoader(
    images, batch_size=shape, shuffle=False
)
print(images)

noise_scheduler_instance = Noise_Scheduler(device=device)
Model = MLP(
    depth=7, size=128, hidden_dim=128, output_dim=1, input_size=1, device=device
)

noise_scheduler_instance = Noise_Scheduler(device=device)


trainer_instance = Trainer(
    Model,
    noise_scheduler_instance,
    torch.optim.Adam(Model.parameters(), lr=0.001),
    torch.nn.MSELoss(),
    device,
    save_path=savingModel_path,
)

losses, frames = trainer_instance.train(
    num_epochs=100,
    batch_size=shape,
    gradient_clipthres=1,
    train_loader=DataLoaderInstance,
)

# prepare a directory to store the frames and the losses
# plot the losses and save the plot as a png


outdir = "OutputsMnsit3"
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
