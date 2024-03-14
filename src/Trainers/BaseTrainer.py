import os
from os.path import join
from sys import path
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import uuid
from tabulate import tabulate

from config import PROJECT_ROOT


class BaseTrainer:
    def __init__(
        self,
        model_name,
        model,
        noise_schedulr,
        optimizer,
        criterion,
        device,
        train_params,
    ):
        self.model_name = model_name
        self.best_loss = 1000
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.noise_scheduler = noise_schedulr
        self.device = device
        self.train_params = train_params  # Store additional training parameters
        self.batch_size = train_params["batch_size"]
        self.data_name = train_params["data_name"]
        self.model_save_path = self.create_save_dir()

    def train(self, train_loader):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def sample(self, sample_size):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save_training(self):
        raise NotImplementedError("this method should be implemented by subclasses.")

    def print_dict_as_table(self, data_dict, headers=["Key", "Value"]):
        table = [[key, value] for key, value in data_dict.items()]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def save_dict_as_table(self, data_dict, file_path, headers=["Key", "Value"]):
        file_path = os.path.join(file_path, "report.txt")
        table = [[key, value] for key, value in data_dict.items()]
        table_str = tabulate(table, headers=headers, tablefmt="grid")

        with open(file_path, "w") as file:
            file.write(table_str)

    def create_save_dir(self):
        dat_str = datetime.now().strftime("%Y-%m-%d_%H")
        unique_id = uuid.uuid4().hex[:6]
        dir_name = f"{self.model_name}_date_{dat_str}_{unique_id}"
        full_path = os.path.join(PROJECT_ROOT, "Outputs", dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        model_dir_path = os.path.join(full_path, "Checkpoints")

        self.checkpoints_dir_path = model_dir_path
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)

        return full_path
