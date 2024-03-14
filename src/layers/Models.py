from typing import Dict, Any
from itertools import chain

from src.Trainers.Attention_trainer import AttentionTrainer
from src.Trainers.Old_attention_trainer import AttentionTrainerOld
from src.Trainers.Residual_only_trainer import TrainerResildualOnly
from src.layers.rolling_attention import (
    MLP_Rolling_attention2,
    RollingAttention2,
    MLP_Rolling_attention,
    RollingAttention,
)
from src.layers.seq2seq import Seq2Seq
from src.layers.model import MLP, Noise_Scheduler
import torch


class TinyDiffusion:
    def __init__(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        Diffusion_params: Dict[str, Any],
        device: str,
    ):
        self.model_name = model_name
        self.model_params = model_params
        self.train_params = train_params
        self.Diffusion_params = Diffusion_params
        self.diffusion_model = Noise_Scheduler(**Diffusion_params)
        self.initialise_model(self.model_name)
        self.optimizer_params = optimizer_params
        self.optimizer = torch.optim.AdamW(self.weights_models, **self.optimizer_params)

        self.criterion = torch.nn.MSELoss()
        self.device = device
        self.initialise_trainer(self.model_name)

    def initialise_trainer(self, model_name):
        if model_name == "Residual_only":
            self.trainer = TrainerResildualOnly(
                model_name,
                self.model,
                self.diffusion_model,
                self.optimizer,
                self.criterion,
                self.device,
                self.train_params,
            )
        elif model_name == "Residual_with_attention":
            models_dict = {"model": self.model, "attention": self.attention}
            self.trainer = AttentionTrainer(
                model_name,
                models_dict,
                self.diffusion_model,
                self.optimizer,
                self.criterion,
                self.device,
                self.train_params,
            )

        elif model_name == "Residual_with_old_attention":
            models_dict = {"model": self.model, "attention": self.attention}
            self.trainer = AttentionTrainerOld(
                model_name,
                models_dict,
                self.diffusion_model,
                self.optimizer,
                self.criterion,
                self.device,
                self.train_params,
            )

        elif model_name == "seq2seq":
            raise ValueError("The trainer is not yer implemented")
        else:
            raise ValueError(f"The model {self.model_name} is not defined")

    def initialise_model(self, model_name):
        if model_name == "Residual_only":
            self.model = MLP(**self.model_params)
            self.weights_models = self.model.parameters()
        elif model_name == "Residual_with_attention":
            params_model = self.model_params["Model"]
            params_attention = self.model_params["attention"]
            self.attention = RollingAttention2(**params_attention)
            self.model = MLP_Rolling_attention2(
                attention_module=self.attention, **params_model
            )
            self.weight_model = self.model.parameters()
            self.weight_attention = self.attention.parameters()
            self.weights_models = chain(self.weight_model, self.weight_attention)
        elif model_name == "Residual_with_old_attention":
            params_model = self.model_params["Model"]
            params_attention = self.model_params["attention"]
            self.attention = RollingAttention(**params_attention)
            self.model = MLP_Rolling_attention(
                attention_module=self.attention, **params_model
            )
            self.weight_model = self.model.parameters()
            self.weight_attention = self.attention.parameters()
            self.weights_models = chain(self.weight_model, self.weight_attention)
        elif model_name == "seq2seq":
            self.model = Seq2Seq(**self.model_params)
        else:
            raise ValueError(f"The model {self.model_name} is not defined")

    def train(self, train_loader):
        self.trainer.train(train_loader)
