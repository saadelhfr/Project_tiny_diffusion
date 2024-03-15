from os import times
from numpy import random
import torch
import torch.nn as nn
from .positional_embedding import Positional_embedding


class Seq2Seq(nn.Module):
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
        super(Seq2Seq, self).__init__()

        self.depth = depth
        self.size = size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pos_emb = pos_emb
        self.input_size = input_size
        self.device = device
        self.time_embedding = Positional_embedding(
            size=size, type_=pos_emb, scale=1.0, device=self.device
        )

        self.positional_embeder_1 = Positional_embedding(
            size=size, type_=pos_emb, scale=25.0, device=self.device
        )

        self.positional_embeder_2 = Positional_embedding(
            size=size, type_=pos_emb, scale=25.0, device=self.device
        )

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
        self.encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.depth,
            device=self.device,
        )
        self.decoder = Decoder(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.depth,
            device=self.device,
        )

    def forward(self, X, Timestamps, target_tensor=None):
        # X of shape (batch_size , seqlength , 1 )
        # Timestamps of shape (batch_size , 1 )
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        print("from file " , __file__)
        print(X.shape)
        batch_size, seq_length, _ = X.shape
        time_embedding = self.time_embedding(Timestamps)  # shape (batch_size , size)

        time_embedding = time_embedding.unsqueeze(1)
        time_embedding = time_embedding.repeat(1, seq_length, 1)

        if self.input_size == 1:

            pos_embed_1 = self.positional_embeder_1(X[:, :, 0])
            X = torch.cat([time_embedding, pos_embed_1], dim=-1)
        elif self.input_size == 2:
            pos_embed_1 = self.positional_embeder_1(X[:, :, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, :, 1])

            X = torch.cat([time_embedding, pos_embed_1, pos_embed_2], dim=-1)
        else:
            pos_embed_1 = self.positional_embeder_1(X[:, :, 0])
            pos_embed_2 = self.positional_embeder_2(X[:, :, 1])
            pos_embed_3 = self.positional_embeder_3(X[:, :, 2])
            X = torch.cat(
                [time_embedding, pos_embed_1, pos_embed_2, pos_embed_3], dim=-1
            )

        if target_tensor is not None:
            target_embed1 = self.positional_embeder_1(target_tensor[:, :, 0])
            target_embed2 = self.positional_embeder_2(target_tensor[:, :, 1])
            embed_target = torch.cat(
                [time_embedding, target_embed1, target_embed2], dim=-1
            )
        else:
            embed_target = None

        hidden, cell = self.encoder(X)
        output = self.decoder(hidden, cell, seq_length, batch_size, embed_target)

        return output


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).to(
            device
        )

    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        device,
        use_learned_vector=True,
    ):
        self.input_dim = input_dim
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, num_layers, batch_first=True).to(
            device
        )

        self.fc = nn.Linear(input_dim, output_dim).to(device)

        self.use_learned_vector = use_learned_vector
        self.device = device

        # Initialize the learned vector as a parameter, if required
        if self.use_learned_vector:
            self.initial_vector = nn.Parameter(torch.randn(1, 1, input_dim))
            print(self.initial_vector.shape)

    def forward(self, hidden, cell, seq_length, batch_size, target_tensor=None):
        if self.use_learned_vector:
            # Use the learned initial vector as the first input
            decoder_input = self.initial_vector.repeat(batch_size, 1, 1)
        else:
            # Use zeros as the initial input for the first timestep
            decoder_input = torch.zeros(
                batch_size, 1, self.input_dim, device=self.device
            )

        decoder_hidden = (hidden, cell)
        decoder_outputs = []

        outputs = []
        for i in range(seq_length):
            decoder_output, decoder_hidden = self.lstm(decoder_input, decoder_hidden)
            decoder_outputs.append(self.fc(decoder_output))

            if target_tensor is not None and random.random() > 0.5:
                # Teacher forcing: use the next target tensor as the next input
                decoder_input = target_tensor[:, i : i + 1, :].to(self.device)
            else:
                # Use the model's own prediction as the next input
                decoder_input = decoder_output

            outputs.append(self.fc(decoder_output))
        # Concatenate outputs (now outside the loop)
        final_outputs = torch.cat(outputs, dim=1)
        return final_outputs
