import os
import torch
from tqdm import tqdm
import numpy as np
from .BaseTrainer import BaseTrainer


class TrainerResildualOnly(BaseTrainer):
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
        super(TrainerResildualOnly, self).__init__(
            model_name,
            model,
            noise_schedulr,
            optimizer,
            criterion,
            device,
            train_params,
        )

        self.num_epochs = self.train_params["num_epoch"]
        self.batch_size = self.train_params["batch_size"]
        self.gradient_clip_threshhold = self.train_params["gradient_clip_threshhold"]
        self.eval_frequency = self.train_params["eval_frequency"]
        self.sample_size = self.train_params["sample_size"]

    def train(self, train_loader):
        losses = []
        frames = []
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(
                total=len(train_loader), desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            loss_batch = []
            for _, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_data = batch[0].to(self.device)
                noise = torch.randn_like(batch_data)
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, (batch_data.shape[0],)
                ).long()
                noisy_data = self.noise_scheduler.add_noise(
                    batch_data, noise, timesteps
                )
                prediction = self.model(noisy_data, timesteps)
                loss = self.criterion(prediction, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), self.gradient_clip_threshhold
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
                    os.path.join(
                        self.checkpoints_dir_path, "Residual_only_best_loss.pth"
                    ),
                )
            progress_bar.close()

            if epoch % self.eval_frequency == 0 or epoch == self.num_epochs - 1:
                sample = self.sample(self.sample_size)
                frames.append(sample.cpu().detach().numpy())

        self.losses = losses
        self.frames = frames
        print("Training Report : ")
        report = self.train_params
        report["best loss achieved "] = self.best_loss
        self.save_training()
        self.save_dict_as_table(report, self.model_save_path)
        self.print_dict_as_table(report)

    def sample(self, sample_size):
        self.model.eval()
        sample = torch.randn(sample_size, self.model.output_dim)
        timesteps = list(range(len(self.noise_scheduler)))
        timesteps.reverse()
        for _, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, sample_size)).long()
            with torch.no_grad():
                residual = self.model(sample, t)
            sample = self.noise_scheduler.step(residual, t[0], sample)
        return sample

    def save_training(self):
        np.save(os.path.join(self.model_save_path, "framse.npy"), self.frames)
        np.save(os.path.join(self.model_save_path, "losses.npy"), self.losses)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoints_dir_path, "Residual_only_last_epoch.pth"),
        )
