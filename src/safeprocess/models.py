"""Projected LSTM autoencoder used by the fresh-training recipe.

The parameter layout follows the retained LSTMProjMLPNet implementation.
There is no dependency on the thesis pipeline or its orchestration.
"""

from __future__ import annotations

import math

from torch import Tensor, nn
import torch


class _ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class LSTMProjMLPNet(nn.Module):
    """Timewise bottleneck with projected encoder and residual MLP decoder."""

    def __init__(
        self, input_dim: int, input_proj_dim: int = 64,
        hidden_size: int = 128, latent_size: int = 16,
        n_layers: int = 1, dropout: float = 0.1,
        post_mlp_ratio: int = 4, post_mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, input_proj_dim, hidden_size, latent_size, n_layers) < 1:
            raise ValueError("All network dimensions must be positive")
        recurrent_dropout = dropout if n_layers > 1 else 0.0
        self.input_proj = nn.Linear(input_dim, input_proj_dim)
        self.input_norm = nn.LayerNorm(input_proj_dim)
        self.encoder = nn.LSTM(input_proj_dim, hidden_size, num_layers=n_layers,
                               batch_first=True, dropout=recurrent_dropout)
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.expand = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers,
                               batch_first=True, dropout=recurrent_dropout)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.post_mlp = _ResidualMLP(hidden_size,
                                    max(hidden_size * post_mlp_ratio, hidden_size),
                                    post_mlp_dropout)
        self.output = nn.Linear(hidden_size, input_dim)

    def encode(self, x: Tensor) -> Tensor:
        enc_out, _ = self.encoder(self.input_norm(self.input_proj(x)))
        return torch.tanh(self.bottleneck(enc_out))

    def decode(self, latent: Tensor) -> Tensor:
        dec_out, _ = self.decoder(torch.tanh(self.expand(latent)))
        return self.output(dec_out + self.post_mlp(self.post_norm(dec_out)))

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def architecture(input_dim: int, config: dict) -> dict:
    """Resolve and persist every architectural choice, including the bottleneck."""
    return {
        "input_dim": int(input_dim),
        "input_proj_dim": int(config.get("input_proj_dim", 64)),
        "hidden_size": int(config.get("hidden_size", 128)),
        "latent_size": max(1, min(int(config.get("latent_cap", 16)),
                                  math.floor(float(config.get("latent_fraction", .5))
                                             * input_dim))),
        "n_layers": int(config.get("n_layers", 1)),
        "dropout": float(config.get("dropout", .1)),
        "post_mlp_ratio": int(config.get("post_mlp_ratio", 4)),
        "post_mlp_dropout": float(config.get("post_mlp_dropout", 0.0)),
    }


def build_model(input_dim: int, config: dict) -> LSTMProjMLPNet:
    return LSTMProjMLPNet(**architecture(input_dim, config))
