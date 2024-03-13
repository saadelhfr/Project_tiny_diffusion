import torch
from torch import nn
import torch.nn.functional as F
from .positional_embedding import Positional_embedding
from src.utils.custom_cumprod import custom_cumprod

import math


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
        device="cpu",
    ):
        super(MLP, self).__init__()
        self.device = torch.device(device)
        self.input_size = input_size
        self.depth = depth
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
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
            self.input_dim = len(self.time_embedding) + len(self.positional_embeder_1)
        elif self.input_size == 2:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + len(self.positional_embeder_2)
            )
        else:
            self.input_dim = (
                len(self.time_embedding)
                + len(self.positional_embeder_1)
                + len(self.positional_embeder_2)
                + len(self.positional_embeder_3)
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
        elif self.input_size == 2:
            time_embedding = self.time_embedding(Timestamps)
            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, 1])

            X = torch.cat([time_embedding, pos_embed_1, pos_embed_2], dim=-1)
        else:
            time_embedding = self.time_embedding(Timestamps)
            pos_embed_1 = self.positional_embeder_1(X[:, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, 1])
            pos_embed_3 = self.positional_embeder_3(X[:, 2])
            X = torch.cat(
                [time_embedding, pos_embed_1, pos_embed_2, pos_embed_3], dim=-1
            )
        return self.MLP(X)


class Noise_Scheduler(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 250,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device="cpu",
    ):
        super(
            Noise_Scheduler, self
        ).__init__()  # Don't forget to call the superclass initializer
        self.device = torch.device(device)  # Ensure device is a torch.device object
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start,
                beta_end,
                num_timesteps,
                dtype=torch.float32,
                device=self.device,
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_timesteps,
                    dtype=torch.float32,
                    device=self.device,
                )
                ** 2
            )
        elif beta_schedule == "cosine":
            # Implementing the cosine schedule
            steps = torch.arange(
                num_timesteps, dtype=torch.float32, device=self.device
            ) / (num_timesteps - 1)
            self.betas = beta_start + (beta_end - beta_start) * 0.5 * (
                1 + torch.cos(math.pi * steps)
            )
        else:
            raise ValueError("Invalid schedule")

        self.alphas = 1 - self.betas
        # Ensure custom_cumprod and subsequent operations are on the correct device
        self.alphas_cumprod = custom_cumprod(self.alphas.to(self.device), dim=0).to(
            self.device
        )
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod, (1, 0), value=1.0)[
            :-1
        ].to(self.device)

        # Pre-calculate other necessary tensors
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(
            self.device
        )
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod).to(
            self.device
        )
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1
        ).to(self.device)

        # Calculate coefficients used in posterior mean calculation
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        ).to(self.device)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        ).to(self.device)

    # Make sure to adjust other methods to use self.device where appropriate
    def reconstruct_x0(self, x_t, t, noise):
        x_t = x_t.to(self.device)
        noise = noise.to(self.device)
        s1 = self.sqrt_inv_alphas_cumprod[t].to(self.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(self.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        x_0 = x_0.to(self.device)
        x_t = x_t.to(self.device)
        s1 = self.posterior_mean_coef1[t].to(self.device)
        s2 = self.posterior_mean_coef2[t].to(self.device)
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
        variance = variance.clip(1e-20).to(self.device)
        return variance

    def step(self, model_output, timestep, sample):
        model_output = model_output.to(self.device)
        sample = sample.to(self.device)
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output).to(
            self.device
        )
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t).to(
            self.device
        )

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance
        pred_prev_sample = pred_prev_sample.to(self.device)
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        x_start = x_start.to(self.device)
        x_noise = x_noise.to(self.device)
        s1 = self.sqrt_alphas_cumprod[timesteps].to(self.device)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].to(self.device)

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


class Noise_Scheduler3_D(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device="cpu",
    ):
        super(
            Noise_Scheduler3_D, self
        ).__init__()  # Don't forget to call the superclass initializer
        self.device = torch.device(device)  # Ensure device is a torch.device object
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start,
                beta_end,
                num_timesteps,
                dtype=torch.float32,
                device=self.device,
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_timesteps,
                    dtype=torch.float32,
                    device=self.device,
                )
                ** 2
            )
        elif beta_schedule == "sinusoidal":
            timesteps = torch.linspace(
                0,
                num_timesteps - 1,
                steps=num_timesteps,
                dtype=torch.float32,
                device=self.device,
            )
            self.betas = beta_start + (beta_end - beta_start) * 0.5 * (
                1 + torch.cos(torch.pi * timesteps / (num_timesteps - 1))
            )

        else:
            raise ValueError("Invalid schedule")

        self.alphas = 1 - self.betas
        # Ensure custom_cumprod and subsequent operations are on the correct device
        self.alphas_cumprod = custom_cumprod(self.alphas.to(self.device), dim=0).to(
            self.device
        )
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod, (1, 0), value=1.0)[
            :-1
        ].to(self.device)

        # Pre-calculate other necessary tensors
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(
            self.device
        )
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod).to(
            self.device
        )
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1
        ).to(self.device)

        # Calculate coefficients used in posterior mean calculation
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        ).to(self.device)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        ).to(self.device)

    # Make sure to adjust other methods to use self.device where appropriate
    def reconstruct_x0(self, x_t, t, noise):
        x_t = x_t.to(self.device)
        noise = noise.to(self.device)
        s1 = self.sqrt_inv_alphas_cumprod[t].to(self.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(self.device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        x_0 = x_0.to(self.device)
        x_t = x_t.to(self.device)
        s1 = self.posterior_mean_coef1[t].to(self.device)
        s2 = self.posterior_mean_coef2[t].to(self.device)
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
        variance = variance.clip(1e-20).to(self.device)
        return variance

    def step(self, model_output, timestep, sample):
        model_output = model_output.to(self.device)
        sample = sample.to(self.device)
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output).to(
            self.device
        )
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t).to(
            self.device
        )

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance
        pred_prev_sample = pred_prev_sample.to(self.device)
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        x_start = x_start.to(self.device)
        x_noise = x_noise.to(self.device)
        original_dim = x_start.shape
        if len(original_dim) == 2:
            x_start.unsqueeze(-1)
            x_noise.unsqueeze(-1)
        s1 = self.sqrt_alphas_cumprod[timesteps].unsqueeze(-1).unsqueeze(-1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(-1).unsqueeze(-1)

        out = s1 * x_start + s2 * x_noise
        if len(original_dim) == 2:
            out.squeeze()
        return out

    def __len__(self):
        return self.num_timesteps
