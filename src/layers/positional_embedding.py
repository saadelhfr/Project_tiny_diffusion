from typing import Union
import torch
import torch.nn as nn
import numpy as np


class Sinusoidal_embedding_trans(nn.Module):
    def __init__(self, size: int, scale: float, device="cpu"):
        super(Sinusoidal_embedding_trans, self).__init__()
        self.size = size
        self.scale = scale
        self.device = torch.device(device)  # Store device information

    def forward(self, x: Union[torch.Tensor, np.ndarray]):
        if isinstance(x, np.ndarray):
            x = torch.tensor(
                x, device=self.device
            )  # Ensure tensor is on the correct device
        else:
            x = x.to(self.device)
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.tensor([10000.0], device=self.device)) / (half_size - 1)
        emb = torch.exp(torch.arange(half_size, device=self.device).float() * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def __len__(self):
        return self.size


class Basic_embedding(nn.Module):
    def __init__(self, device="cpu"):
        super(Basic_embedding, self).__init__()
        self.device = torch.device(device)  # Store device information

    def forward(self, x: Union[torch.Tensor, np.ndarray]):
        if isinstance(x, np.ndarray):
            x = torch.tensor(
                x, device=self.device
            )  # Convert and move to the correct device
        else:
            x = x.to(self.device)
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class Positional_embedding(nn.Module):
    def __init__(self, size: int, type_: str, device="cpu", **kwargs):
        super(Positional_embedding, self).__init__()
        self.device = torch.device(device)  # Store device information
        if type_ == "sinusoidal":
            self.embedding = Sinusoidal_embedding_trans(
                size, device=self.device, **kwargs
            )
        elif type_ == "basic":
            self.embedding = Basic_embedding(device=self.device)
        else:
            raise ValueError("Invalid type_")

    def forward(self, x: Union[torch.Tensor, np.ndarray]):
        return self.embedding(x)

    def __len__(self):
        return len(self.embedding)
