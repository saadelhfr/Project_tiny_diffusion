from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
from layers.model import *
from utils.transforms import ImageToNormalizedCoordinatesTransform
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils.datasets import get_dataset
from utils.training import *
import torch.nn as nn
import torch
import torch.nn.functional as F


class RollingAttention(nn.Module):
    def __init__(self, input_dim, output_dim, max_length=250, device="cpu"):
        super(RollingAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.device = device

        # Initialize layers for the attention mechanism
        self.query_layer = nn.Linear(input_dim, output_dim).to(self.device)
        self.key_layer = nn.Linear(input_dim, output_dim).to(self.device)
        self.value_layer = nn.Linear(input_dim, output_dim).to(self.device)

        self.bn_query = nn.BatchNorm1d(output_dim).to(self.device)
        self.bn_key = nn.BatchNorm1d(output_dim).to(self.device)
        self.bn_value = nn.BatchNorm1d(output_dim).to(self.device)
        # History of vectors (initialized as an empty list)
        self.history = (
            torch.zeros(0, self.input_dim).requires_grad_(False).to(self.device)
        )
        self.train_history = torch.zeros(
            0, self.input_dim, device=device, requires_grad=False
        )
        self.eval_history = torch.zeros(
            0, self.input_dim, device=device, requires_grad=False
        )

    def forward(self, x):
        # x is the current vector of shape [batch_size, input_dim]

        # Update history
        x = x.to(self.device)
        current_history = self.train_history if self.training else self.eval_history
        updated_history = torch.cat((current_history, x), dim=0)[-self.max_length :]
        if self.training:
            self.train_history = updated_history
        else:
            self.eval_history = updated_history

            # Calculate query, key, value
        Q = self.query_layer(x)
        Q = self.bn_query(Q)  # Apply BN

        K = self.key_layer(updated_history)
        K = self.bn_key(K)  # Apply BN

        V = self.value_layer(updated_history)
        V = self.bn_value(V)  # Apply BN

        # Scale dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_dim**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute context vector
        context = torch.matmul(attention_weights, V)
        context = torch.nn.functional.tanh(context)

        return context

    def reset_history(self):
        self.train_history = torch.zeros(
            0, self.input_dim, device=self.device, requires_grad=False
        )
        self.eval_history = torch.zeros(
            0, self.input_dim, device=self.device, requires_grad=False
        )

    def context_dim(self):
        # Return the dimension of the context vector
        return self.output_dim


class RollingAttention2(nn.Module):
    def __init__(self, input_dim, output_dim, size=256, max_length=250, device="cpu"):
        super(RollingAttention2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.size = size
        self.device = device

        # Initialize layers for the attention mechanism
        self.query_layer = nn.Linear(size * 2, output_dim).to(self.device)
        self.key_layer = nn.Linear(input_dim, output_dim).to(self.device)
        self.value_layer = nn.Linear(input_dim, output_dim).to(self.device)

        self.bn_query = nn.BatchNorm1d(output_dim).to(self.device)
        self.bn_key = nn.BatchNorm1d(output_dim).to(self.device)
        self.bn_value = nn.BatchNorm1d(output_dim).to(self.device)
        # History of vectors (initialized as an empty list)
        self.positional_embeder_1 = Positional_embedding(
            size=size, type_="sinusoidal", scale=25.0, device=self.device
        )
        self.positional_embeder_2 = Positional_embedding(
            size=size, type_="sinusoidal", scale=25.0, device=self.device
        )
        self.history = (
            torch.zeros(0, self.input_dim).requires_grad_(False).to(self.device)
        )
        self.train_history = torch.zeros(
            0, self.input_dim, device=device, requires_grad=False
        )
        self.eval_history = torch.zeros(
            0, self.input_dim, device=device, requires_grad=False
        )

    def update_history(self, x):

        pos_embed_1 = self.positional_embeder_1(x[:, 0])
        pos_embed_2 = self.positional_embeder_2(x[:, 1])

        x = torch.cat([pos_embed_1, pos_embed_2], dim=-1)
        current_history = self.train_history if self.training else self.eval_history

        updated_history = torch.cat((current_history, x), dim=0)[-self.max_length :]
        if self.training:
            self.train_history = updated_history
        else:
            self.eval_history = updated_history

    def forward(self, x):
        # x is the current vector of shape [batch_size, input_dim]

        # Update history
        x = x.to(self.device)
        current_history = self.train_history if self.training else self.eval_history

        # Calculate query, key, value
        Q = self.query_layer(x)
        Q = self.bn_query(Q)  # Apply BN

        K = self.key_layer(current_history)
        K = self.bn_key(K)  # Apply BN

        V = self.value_layer(current_history)
        V = self.bn_value(V)  # Apply BN

        # Scale dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.input_dim**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute context vector
        context = torch.matmul(attention_weights, V)
        context = torch.nn.functional.tanh(context)

        return context

    def reset_history(self):
        if self.training:
            self.train_history = torch.zeros(
                0, self.input_dim, device=self.device, requires_grad=False
            )
        else:
            self.eval_history = torch.zeros(
                0, self.input_dim, device=self.device, requires_grad=False
            )

    def context_dim(self):
        # Return the dimension of the context vector
        return self.output_dim


class MLP_Rolling_attention(nn.Module):
    def __init__(
        self,
        depth: int,
        size: int,
        hidden_dim: int,
        output_dim: int,
        attention_module: nn.Module,
        pos_emb: str = "sinusoidal",
        input_size: int = 2,
        device="cpu",
    ):
        super(MLP_Rolling_attention, self).__init__()
        self.device = torch.device(device)
        self.input_size = input_size
        self.depth = depth
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention_shape = attention_module.context_dim()
        self.time_embedding = Positional_embedding(
            size=size, type_=pos_emb, scale=1.0, device=self.device
        )
        self.positional_embeder_1 = Positional_embedding(
            size=size, type_=pos_emb, scale=25.0, device=self.device
        )

        if self.input_size == 2:
            self.positional_embeder_2 = Positional_embedding(
                size=size, type_=pos_emb, scale=25.0, device=self.device
            )
        if self.input_size == 3:
            self.positional_embeder_3 = Positional_embedding(
                size=size, type_=pos_emb, scale=25.0, device=self.device
            )

        if self.input_size == 1:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + self.attention_shape
            )
        elif self.input_size == 2:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + len(self.positional_embeder_2)
                + self.attention_shape
            )
        else:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + len(self.positional_embeder_2)
                + len(self.positional_embeder_3)
                + self.attention_shape
            )

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )

        for i in range(self.depth):
            self.MLP.add_module("block_{}".format(i), Block(self.hidden_dim))
        self.MLP.add_module("final_layer", nn.Linear(self.hidden_dim, output_dim))
        self.MLP.to(self.device)

    def get_attention(self, previous_generation):
        if self.input_size == 1:

            pos_embed_1 = self.positional_embeder_1(previous_generation[:, 0])
            previous_generation = pos_embed_1
        elif self.input_size == 2:
            pos_embed_1 = self.positional_embeder_1(previous_generation[:, 0])
            pos_embed_2 = self.positional_embeder_2(previous_generation[:, 1])

            previous_generation = torch.cat([pos_embed_1, pos_embed_2], dim=-1)
        else:
            pos_embed_1 = self.positional_embeder_1(previous_generation[:, 0])
            pos_embed_2 = self.positional_embeder_2(previous_generation[:, 1])
            pos_embed_3 = self.positional_embeder_3(previous_generation[:, 2])
            previous_generation = torch.cat(
                [pos_embed_1, pos_embed_2, pos_embed_3], dim=-1
            )

        previous_generation = previous_generation.to(self.device)
        return previous_generation

    def forward(self, X, Timestamps, prev_batch, attention_module):
        assert (
            X.shape == prev_batch.shape
        ), "The previously generated batch should be of the same shape as the current one"
        time_embedding = self.time_embedding(Timestamps)
        if self.input_size == 1:

            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            X = torch.cat([time_embedding, pos_embed_1], dim=-1)
        elif self.input_size == 2:
            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, 1])
            X = torch.cat([time_embedding, pos_embed_1, pos_embed_2], dim=-1)
        else:
            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, 1])
            pos_embed_3 = self.positional_embeder_3(X[:, 2])
            X = torch.cat(
                [time_embedding, pos_embed_1, pos_embed_2, pos_embed_3], dim=-1
            )

        previous_generation = self.get_attention(prev_batch).to(self.device)
        context_vec = attention_module(previous_generation).to(self.device)
        final_input = torch.cat([context_vec, X], dim=-1).to(self.device)
        final_input = final_input.to(self.device)
        return self.MLP(final_input)


class MLP_Rolling_attention2(nn.Module):
    def __init__(
        self,
        depth: int,
        size: int,
        hidden_dim: int,
        output_dim: int,
        attention_module: nn.Module,
        pos_emb: str = "sinusoidal",
        input_size: int = 2,
        device="cpu",
    ):
        super(MLP_Rolling_attention2, self).__init__()
        self.device = torch.device(device)
        self.input_size = input_size
        self.depth = depth
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention_shape = attention_module.context_dim()
        self.time_embedding = Positional_embedding(
            size=size, type_=pos_emb, scale=1.0, device=self.device
        )
        self.positional_embeder_1 = Positional_embedding(
            size=size, type_=pos_emb, scale=25.0, device=self.device
        )

        if self.input_size == 2:
            self.positional_embeder_2 = Positional_embedding(
                size=size, type_=pos_emb, scale=25.0, device=self.device
            )
        if self.input_size == 3:
            self.positional_embeder_3 = Positional_embedding(
                size=size, type_=pos_emb, scale=25.0, device=self.device
            )

        if self.input_size == 1:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + self.attention_shape
            )
        elif self.input_size == 2:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + len(self.positional_embeder_2)
                + self.attention_shape
            )
        else:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + len(self.positional_embeder_2)
                + len(self.positional_embeder_3)
                + self.attention_shape
            )

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )

        for i in range(self.depth):
            self.MLP.add_module("block_{}".format(i), Block(self.hidden_dim))
        self.MLP.add_module("final_layer", nn.Linear(self.hidden_dim, output_dim))
        self.MLP.to(self.device)

    def forward(self, X, Timestamps, attention_module):
        X = X.to(self.device)
        time_embedding = self.time_embedding(Timestamps)
        if self.input_size == 1:

            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            X = torch.cat([time_embedding, pos_embed_1], dim=-1)
            attenton_input = torch.cat([pos_embed_1], dim=-1)
        elif self.input_size == 2:
            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, 1])
            X = torch.cat([time_embedding, pos_embed_1, pos_embed_2], dim=-1)
            attenton_input = torch.cat([pos_embed_1, pos_embed_2], dim=-1)
        else:
            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, 1])
            pos_embed_3 = self.positional_embeder_3(X[:, 2])
            X = torch.cat(
                [time_embedding, pos_embed_1, pos_embed_2, pos_embed_3], dim=-1
            )
            attenton_input = torch.cat([pos_embed_1, pos_embed_2, pos_embed_3], dim=-1)

        context_vec = attention_module(attenton_input).to(self.device)
        final_input = torch.cat([context_vec, X], dim=-1).to(self.device)
        final_input = final_input.to(self.device)
        return self.MLP(final_input)
