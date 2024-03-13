import os
import numpy as np
import pandas as pd
import torch
from config import PROJECT_ROOT
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    y = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
    df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "Data/static/DatasaurusDozen.tsv"),
        sep="\t",
    )
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x / 54 - 1) * 4
    y = (y / 48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def sample_continuous_subpixel(
    n_samples: int = 1000,
    path_to_density=os.path.join(PROJECT_ROOT, "Data/Densities/density_map_0.pt"),
):
    density_map = torch.load(path_to_density)
    # Flatten and normalize the density map to create a PDF
    pdf = density_map.flatten() / torch.sum(density_map)

    # Sample indices from this PDF
    sampled_indices = torch.multinomial(pdf, n_samples, replacement=True)

    # Convert indices to 2D coordinates
    y_indices = sampled_indices // density_map.size(1)
    x_indices = sampled_indices % density_map.size(1)

    # Adjust to sample within each pixel's area
    # Generate random offsets within each pixel's area
    y_offsets = torch.randn_like(y_indices.float()) * 0.9
    x_offsets = torch.randn_like(x_indices.float()) * 0.9

    # Combine indices with offsets and normalize to [0, 1] range
    x_continuous = (x_indices.float() + x_offsets) / density_map.size(1)
    y_continuous = (y_indices.float() + y_offsets) / density_map.size(0)

    return TensorDataset(torch.stack((x_continuous, -y_continuous), dim=1))


def get_dataset(name, n=8000, **kwargs):
    # get the digit if it is mnist
    path = kwargs.get("path", 5)
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    elif name == "mnist":
        return sample_continuous_subpixel(n_samples=n, path_to_density=path)
    else:
        raise ValueError(f"Unknown dataset: {name}")
