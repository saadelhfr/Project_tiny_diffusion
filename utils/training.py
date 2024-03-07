import numpy as np
import torch
from tqdm.auto import tqdm
import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        noise_scheduler,
        optimizer,
        criterion,
        device,
        save_path="best_model.pth",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.save_path = save_path  # Path to save the best model
        self.best_loss = float("inf")  # Initialize the best loss to infinity

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        gradient_clipthres: float,
        train_loader: torch.utils.data.DataLoader,
    ):
        self.model.to(self.device)
        global_step = 0
        losses = []
        frames = []
        for epoch in range(num_epochs):
            self.model.train()
            progress_bar = tqdm(
                total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"
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

                prediction = self.model(noisy_data, timesteps)  # forward pass
                loss = self.criterion(prediction, noise)  # compute the loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clipthres
                )
                self.optimizer.step()
                global_step += 1
                loss_batch.append(loss.item())

            epoch_loss = sum(loss_batch) / len(loss_batch)
            losses.append(epoch_loss)
            print(f"the loss for epoch {epoch} is {epoch_loss} ")

            # Check if current epoch loss is lower than the best loss
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                torch.save(self.model.state_dict(), self.save_path)

            progress_bar.close()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.model.eval()
                sample = torch.randn(10000, self.model.output_dim)
                timesteps = list(range(len(self.noise_scheduler)))[::-1]
                for _, t in enumerate(tqdm(timesteps)):
                    t = torch.from_numpy(np.repeat(t, 10000)).long()
                    with torch.no_grad():
                        residual = self.model(sample, t)
                    sample = self.noise_scheduler.step(residual, t[0], sample)

                frames.append(sample.cpu().detach().numpy())

        return losses, frames

    def train_seq2seq(
        self,
        num_epochs: int,
        batch_size: int,
        gradient_clipthres: float,
        train_loader: torch.utils.data.DataLoader,
    ):
        self.model.to(self.device)
        global_step = 0

        losses = []
        frames = []
        for epoch in range(num_epochs):
            processed_batches = 0
            self.model.train()
            progress_bar = tqdm(total=20, desc=f"Epoch {epoch + 1}/{num_epochs}")
            loss_batch = []

            for batch, _ in train_loader:
                self.optimizer.zero_grad()
                batch_data = batch.to(self.device)
                noise = torch.randn_like(batch_data)
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, (batch_data.shape[0],)
                ).long()

                noisy_data = self.noise_scheduler.add_noise(
                    batch_data, noise, timesteps
                )

                prediction = self.model(noisy_data, timesteps)  # forward pass
                loss = self.criterion(prediction, noise)  # compute the loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clipthres
                )
                self.optimizer.step()
                global_step += 1
                processed_batches += 1
                if processed_batches == 20:
                    break
                progress_bar.update(1)
                loss_batch.append(loss.item())

            epoch_loss = sum(loss_batch) / len(loss_batch)
            losses.append(epoch_loss)
            print(f"the loss for epoch {epoch} is {epoch_loss} ")

            # Check if current epoch loss is lower than the best loss
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                torch.save(self.model.state_dict(), self.save_path)

            progress_bar.close()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.model.eval()
                sample = torch.randn(784, self.model.output_dim)
                timesteps = list(range(len(self.noise_scheduler)))[::-1]
                for _, t in enumerate(tqdm(timesteps)):
                    t = torch.tensor(t).long()
                    with torch.no_grad():
                        residual = self.model(sample, t)
                    sample = self.noise_scheduler.step(residual, t.item(), sample)

                frames.append(sample.cpu().detach().numpy())
        return losses, frames
