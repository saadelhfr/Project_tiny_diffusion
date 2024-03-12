from numpy import random
import torch
import torch.nn as nn
from .positional_embedding import Positional_embedding


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        size,
        hidden_dim,
        output_dim,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        pos_emb="sinusoidal",
        input_size=2,
        device="cpu",
    ):
        super(TransformerSeq2Seq, self).__init__()

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

        # Assuming your Positional_embedding class can handle 'size' to define embedding size

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )

        # Input linear layer to match Transformer d_model size if necessary
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)

        # Output linear layer to project from d_model to output_dim
        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, src, timestamps, tgt=None):
        if len(src.shape) == 2:
            src = src.unsqueeze(0)
        batch_size, seq_length, _ = src.shape
        time_embedding = self.time_embedding(Timestamps)  # shape (batch_size , size)

        time_embedding = time_embedding.unsqueeze(1)
        time_embedding = time_embedding.repeat(1, seq_length, 1)

        if self.input_size == 1:

            pos_embed_1 = self.positional_embeder_1(src[:, :, 0])
            src = torch.cat([time_embedding, pos_embed_1], dim=-1)
            print(src)
        elif self.input_size == 2:
            pos_embed_1 = self.positional_embeder_1(src[:, :, 0])
            pos_embed_2 = self.positional_embeder_2(src[:, :, 1])

            src = torch.cat([time_embedding, pos_embed_1, pos_embed_2], dim=-1)
        else:
            pos_embed_1 = self.positional_embeder_1(src[:, :, 0])
            pos_embed_2 = self.positional_embeder_2(src[:, :, 1])
            pos_embed_3 = self.positional_embeder_3(src[:, :, 2])
            src = torch.cat(
                [time_embedding, pos_embed_1, pos_embed_2, pos_embed_3], dim=-1
            )

        src = self.input_linear(src)

        # The Transformer expects inputs in the shape (seq_length, batch_size, feature_size)

        if tgt is not None:
            tgt = tgt.permute(1, 0, 2)  # Same permutation for target if provided

        # Assuming a mask and target mask (tgt_mask) is handled if necessary
        output = self.transformer(src, tgt)

        # Permute back to (batch_size, seq_length, feature_size) and apply output linear layer
        output = self.output_linear(output)

        return output
