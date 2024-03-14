import torch
import os
import numpy as np
from src.layers import MLP, Noise_Scheduler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse

directory_outputs = "./Bigger_Model_Tests/"
os.makedirs(directory_outputs, exist_ok=True)

parser = argparse.ArgumentParser(
    description="create png plots a model on a specific digit from the mnist dataset"
)
parser.add_argument(
    "--digit",
    type=int,
    default=5,
    help="The MNIST digit to generate frames for (default: 5)",
)
parser.add_argument(
    "--nbr_images",
    type=int,
    default=5,
    help="The number of images to generate (default: 5)",
)

parser.add_argument(
    "--nbr_samples",
    type=int,
    default=10,
    help="The nbr of samples to generate for each image ",
)

args = parser.parse_args()
nbr_samples = args.nbr_samples
nbr_images = args.nbr_images
digit_tomake = args.digit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_weight_dict = torch.load(
    f"./BestModels/best_model{digit_tomake}.pth", map_location=device
)
model = MLP(depth=10, size=256, hidden_dim=256, output_dim=2, device=device)
model.load_state_dict(model_weight_dict)
model.to(device)


noise_scheduler_instance = Noise_Scheduler(beta_schedule="quadratic", device=device)

model.eval()

frames = []

for i in range(nbr_images):
    sample = torch.randn(nbr_samples, 2).to(device)
    timesteps = list(range(len(noise_scheduler_instance)))[::-1]
    for _, t in enumerate(tqdm(timesteps, desc=f"Image {i}")):
        t = torch.from_numpy(np.repeat(t, nbr_samples)).long().to(device)
        with torch.no_grad():
            residual = model(sample, t)
        sample = noise_scheduler_instance.step(residual, t[0], sample)

    frames.append(sample.cpu().detach().numpy())


frames_numpy = np.stack(frames)
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
    ax.set_title(f"Image of the best model {i}")
    ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a png
plt.savefig(directory_outputs + f"frames{digit_tomake}_{nbr_samples}.png")

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
    ax.set_title(f"image of best model {i}")
    ax.axis("off")  # Simplified way to hide both x and y ticks

plt.tight_layout()
plt.savefig(
    directory_outputs + f"frames_as_images{digit_tomake}_{nbr_samples}.png"
)  # Adjust path as needed
