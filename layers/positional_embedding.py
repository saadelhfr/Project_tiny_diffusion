from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np


class Sinusoidal_embedding_trans(nn.Module):
    """This is an implemenation of the sinusoidal encoding as described in the paper attention is all you need"""

    def __init__(self, size: int, scale: float):
        super(Sinusoidal_embedding_trans, self).__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: Union[torch.tensor, np.array]):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(torch.arange(half_size).float() * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def __len__(self):
        return self.size


class Basic_embedding(nn.Module):
    def __init__(self):
        super(Basic_embedding, self).__init__()

    def forward(self, x: Union[torch.tensor, np.array]):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class Positional_embedding(nn.Module):
    def __init__(self, size: int, type_: str, **kwargs):
        super(Positional_embedding, self).__init__()
        if type_ == "sinusoidal":
            self.embedding = Sinusoidal_embedding_trans(size, **kwargs)
        elif type_ == "basic":
            self.embedding = Basic_embedding()
        else:
            raise ValueError("Invalid type_")

    def forward(self, x: Union[torch.tensor, np.array]):
        return self.embedding(x)

    def __len__(self):
        return len(self.embedding)
