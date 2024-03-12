import numpy as np
import random
from sympy import preview
import torch
from tqdm.auto import tqdm
import torch
import numpy as np
from tqdm import tqdm
import ot


class Trainer:
    def __init__(
        self,
        model,
        noise_scheduler,
        optimizer,
        criterion,
        device,
        attention_model=None,
        save_path="best_model.pth",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.save_path = save_path  # Path to save the best model
        self.best_loss = float("inf")  # Initialize the best loss to infinity
        self.attention_model = attention_model

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
                print(batch_data.shape, "form train")
                noise = torch.randn_like(batch_data)
                print(noise.shape, "sda3")
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
            progress_bar = tqdm(
                total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"
            )
            loss_batch = []

            for batch in train_loader:
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
                processed_batches += 1
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

            if (epoch % 100 == 0 or epoch == num_epochs - 1) and epoch != 0:
                self.model.eval()
                sample = torch.randn(1500, self.model.output_dim)
                timesteps = list(range(len(self.noise_scheduler)))[::-1]
                for _, t in enumerate(tqdm(timesteps)):
                    t = torch.tensor(t).long()
                    with torch.no_grad():
                        residual = self.model(sample, t)
                    sample = self.noise_scheduler.step(residual, t.item(), sample)

                frames.append(sample.cpu().detach().numpy())
        return losses, frames

    def train_attention(
        self,
        num_epochs: int,
        batch_size: int,
        gradient_clipthres: float,
        train_loader: torch.utils.data.DataLoader,
    ):
        self.model.to(self.device)
        assert (
            self.attention_model is not None
        ), "The attention model shouldn't be None for the train_attention function"
        self.attention_model.to(self.device)
        global_step = 0
        losses = []
        frames = []
        for epoch in range(num_epochs):
            self.attention_model.train()
            self.model.train()
            progress_bar = tqdm(
                total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"
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
                timesteps = torch.full((batch_size,), single_timestep, dtype=torch.long)

                noisy_data = self.noise_scheduler.add_noise(
                    batch_data, noise, timesteps
                )

                prediction = self.model(
                    noisy_data, timesteps, prev_batch, self.attention_model
                )  # forward pass
                loss = self.criterion(prediction, noise)  # compute the loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clipthres
                )
                self.optimizer.step()
                global_step += 1
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

            if (epoch % 100 == 0 or epoch == num_epochs - 1) and epoch != 0:

                num_batches = 50000 // batch_size
                self.model.eval()
                self.attention_model.eval()

                timesteps = list(range(len(self.noise_scheduler)))[::-1]

                batch_samples = []
                batch_prev = torch.zeros((batch_size, self.model.input_size))
                for _ in tqdm(range(num_batches)):

                    sample = torch.randn(batch_size, self.model.output_dim)
                    for _, t in enumerate(timesteps):
                        t_tensor = torch.full((batch_size,), t, dtype=torch.long)

                        with torch.no_grad():
                            residual = self.model(
                                sample, t_tensor, batch_prev, self.attention_model
                            )
                        sample = self.noise_scheduler.step(
                            residual, t_tensor[0], sample
                        )
                    batch_prev = sample
                    batch_samples.append(sample.cpu().detach().numpy())

                frames.append(np.concatenate(batch_samples, axis=0))

        return losses, frames

    def train_attention2(
        self,
        num_epochs: int,
        batch_size: int,
        len_data: int,
        gradient_clipthres: float,
        reset_probs: float,
        train_loader: torch.utils.data.DataLoader,
    ):
        self.model.to(self.device)
        assert (
            self.attention_model is not None
        ), "The attention model shouldn't be None for the train_attention function"
        self.attention_model.to(self.device)
        global_step = 0
        losses = []
        frames = []
        for epoch in range(num_epochs):
            self.attention_model.train()
            self.model.train()
            progress_bar = tqdm(
                total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}"
            )
            loss_batch = []
            prev_batch = None
            for j, batch in enumerate(train_loader):
                if prev_batch is None or j % len_data == 0:
                    self.attention_model.reset_history()
                    unused_prev_batch = batch[1][0]
                    prev_batch = (
                        torch.randn_like(unused_prev_batch) * 0.1
                    )  # if random.uniform(0, 1) < reset_probs:
                # self.attention_model.reset_history()
                # unused_prev_batch = batch[1][0].to(self.device)
                # prev_batch = torch.randn_like(unused_prev_batch) * 0.1
                # else:
                # prev_batch = batch[1][0].to(self.device)

                self.optimizer.zero_grad()
                batch_data = batch[0][0].to(self.device)
                noise = torch.randn_like(batch_data)
                self.attention_model.update_history(prev_batch)

                single_timestep = torch.randint(
                    0, self.noise_scheduler.num_timesteps, (1,)
                ).item()
                timesteps = torch.full((batch_size,), single_timestep, dtype=torch.long)
                timesteps_gen = list(range(single_timestep))[::-1]

                noisy_data = self.noise_scheduler.add_noise(
                    batch_data, noise, timesteps
                )

                prediction = self.model(
                    noisy_data, timesteps, self.attention_model
                )  # forward pass
                ## Sample From Data
                prev_batch = noisy_data.detach()
                for _, t in enumerate(timesteps_gen):
                    t_tensor = torch.full((batch_size,), t, dtype=torch.long)

                    prev_batch = self.noise_scheduler.step(
                        prediction, t_tensor[0], prev_batch
                    )

                loss = self.criterion(
                    prediction, noise
                ) + 0.2 * ot.sliced_wasserstein_distance(
                    prev_batch, batch_data, n_projections=100, seed=10
                )
                # compute the loss
                loss.backward()

                ## Sample From Data
                prev_batch = noisy_data.detach()
                for _, t in enumerate(timesteps_gen):
                    with torch.no_grad():
                        t_tensor = torch.full((batch_size,), t, dtype=torch.long)

                        prev_batch = self.noise_scheduler.step(
                            prediction, t_tensor[0], prev_batch
                        )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), gradient_clipthres
                )
                self.optimizer.step()
                global_step += 1
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

            if (epoch % 150 == 0 or epoch == num_epochs - 1) and epoch != 0:
                for _ in range(15):

                    batch_samples = self.sample_frames(batch_size)
                    frames.append(np.concatenate(batch_samples, axis=0))

        return losses, frames

    def sample_frames(self, batch_size):
        num_batches = 5000 // batch_size
        self.model.eval()
        self.attention_model.eval()
        self.attention_model.reset_history()

        timesteps = list(range(len(self.noise_scheduler)))[::-1]

        batch_samples = []
        batch_prev = torch.randn((batch_size, self.model.input_size)) * 0.1
        for _ in tqdm(range(num_batches)):

            sample = torch.randn(batch_size, self.model.output_dim)
            self.attention_model.update_history(batch_prev)
            for _, t in enumerate(timesteps):
                t_tensor = torch.full((batch_size,), t, dtype=torch.long)

                with torch.no_grad():
                    residual = self.model(sample, t_tensor, self.attention_model)
                sample = self.noise_scheduler.step(residual, t_tensor[0], sample)
            batch_prev = sample
            batch_samples.append(sample.cpu().detach().numpy())
        return batch_samples
