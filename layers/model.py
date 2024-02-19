import torch
from torch import nn
import torch.nn.functional as F
from .positional_embedding import Positional_embedding


class Block(nn.Module):
    def __init__(self, size):
        super(Block, self).__init__()
        self.linear_layer = nn.Linear(size, size)
        self.activate = nn.GELU()

    def forward(self, x):
        return x + self.activate(self.linear_layer(x))


class MLP(nn.Module):
    def __init__(
        self,
        depth: int,
        size: int,
        hidden_dim: int,
        output_dim: int,
        pos_emb: str = "sinusoidal",
        input_size: int = 2,
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.depth = depth
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.time_embedding = Positional_embedding(size=size, type_=pos_emb, scale=1.0)
        self.positional_embeder_1 = Positional_embedding(
            size=size, type_=pos_emb, scale=25.0
        )

        if self.input_size == 2:
            self.positional_embeder_2 = Positional_embedding(
                size=size, type_=pos_emb, scale=25.0
            )

        if self.input_size == 1:
            self.input_dim = len(self.time_embedding) + len(self.positional_embeder_1)
        else:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + len(self.positional_embeder_2)
            )

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )

        for i in range(self.depth):
            self.MLP.add_module("block_{}".format(i), Block(self.hidden_dim))
        self.MLP.add_module("final_layer", nn.Linear(self.hidden_dim, output_dim))

    def forward(self, X, Timestamps):
        if self.input_size == 1:

            time_embedding = self.time_embedding(Timestamps)
            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            X = torch.cat([time_embedding, pos_embed_1], dim=-1)
        else:
            time_embedding = self.time_embedding(Timestamps)
            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, 1])

            X = torch.cat([time_embedding, pos_embed_1, pos_embed_2], dim=-1)
        return self.MLP(X)


class Noise_Scheduer(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32
                )
                ** 2
            )
        else:
            raise ValueError("Invalid schedule")
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod, (1, 0), value=1.0)[
            :-1
        ]  # The cumulative alphas at the previous timestep

        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps
