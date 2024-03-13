import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from .BaseTrainer import BaseTrainer


class AttentionTrainerOld(BaseTrainer):
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
        super(AttentionTrainerOld, self).__init__(
            model_name,
            model,
            noise_schedulr,
            optimizer,
            criterion,
            device,
            train_params,
        )
        self.check_args(model)
        self.model = model["model"]
        self.attention_model = model["attention"]
        self.num_epochs = self.train_params["num_epoch"]
        self.batch_size = self.train_params["batch_size"]
        self.gradient_clip_threshhold = self.train_params["gradient_clip_threshhold"]
        self.eval_frequency = self.train_params["eval_frequency"]
        self.sample_size = self.train_params["sample_size"]

    def check_args(self, model):
        error_message = """
        Provided argument is not a dictionary.
        For training the attention models please provide an argument as described bellow : 
        {
            "model": <instance of a class derived from torch.nn.Module representing the model>,
            "attention": <instance of a class derived from torch.nn.Module representing the attention mechanism>
        }
        Each value should be derived of torch.nn.Module
        """
        assert isinstance(model, dict), error_message
        return True

    def train(self, train_loader):
        self.model.to(self.device)
        self.attention_model.to(self.device)
        losses = []
        frames = []
        for epoch in range(self.num_epochs):
            self.model.train()
            self.attention_model.train()
            progress_bar = tqdm(
                total=len(train_loader), desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            loss_batch = []
            for _, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_data = batch[0][0].to(self.device)
                prev_batch = batch[1][0].to(self.device)
                noise = torch.randn_like(batch_data)
                single_timestep = torch.randint(
                    0, self.noise_scheduler.num_timesteps, (1,)
                ).item()
                timesteps = torch.full(
                    (self.batch_size,), single_timestep, dtype=torch.long
                )
                noisy_data = self.noise_scheduler.add_noise(
                    batch_data, noise, timesteps
                )
                prediction = self.model(
                    noisy_data, timesteps, prev_batch, self.attention_model
                )  # forward pass
                loss = self.criterion(prediction, noise)  # compute the loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_threshhold
                )
                nn.utils.clip_grad_norm_(
                    self.attention_model.parameters(), self.gradient_clip_threshhold
                )
                self.optimizer.step()
                progress_bar.update(1)
                loss_batch.append(loss.item())
            epoch_loss = sum(loss_batch) / self.batch_size
            losses.append(epoch_loss)
            print(f"The loss for epoch {epoch+1} is {epoch_loss}")
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoints_dir_path, "model_best_loss.pth"),
                )
                torch.save(
                    self.attention_model.state_dict(),
                    os.path.join(self.checkpoints_dir_path, "attention_best_loss.pth"),
                )
            progress_bar.close()

            if epoch % self.eval_frequency or epoch == self.num_epochs - 1:
                sample = self.sample(self.sample_size)
                sample_obtained = np.concatenate(sample, axis=0)
                frames.append(sample_obtained)

        self.losses = losses
        self.frames = frames
        print("Training Report : ")
        report = self.train_params
        report["best loss achieved "] = self.best_loss
        self.save_training()
        self.save_dict_as_table(report, self.model_save_path)
        self.print_dict_as_table(report)

    def sample(self, sample_size):
        num_batches = sample_size // self.batch_size
        self.model.eval()
        self.attention_model.eval()
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        batch_samples = []
        batch_prev = torch.zeros((self.batch_size, self.model.input_size))
        for _ in tqdm(range(num_batches)):
            sample = torch.randn(self.batch_size, self.model.output_dim)
            for _, t in enumerate(timesteps):
                t_tensor = torch.full((self.batch_size,), t, dtype=torch.long)
                with torch.no_grad():
                    residual = self.model(
                        sample, t_tensor, batch_prev, self.attention_model
                    )
                sample = self.noise_scheduler.step(residual, t_tensor[0], sample)
            batch_prev = sample
            batch_samples.append(sample.cpu().detach().numpy())
        return batch_samples

    def save_training(self):
        np.save(os.path.join(self.model_save_path, "framse.npy"), self.frames)
        np.save(os.path.join(self.model_save_path, "losses.npy"), self.losses)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoints_dir_path, "model_last_epoch.pth"),
        )
        torch.save(
            self.attention_model.state_dict(),
            os.path.join(self.checkpoints_dir_path, "attention_last_epoch.pth"),
        )
