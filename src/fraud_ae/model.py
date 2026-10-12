"""Symmetric MLP autoencoder (ARCHITECTURE.md §3.2)."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class Autoencoder(nn.Module):
    """``input_dim → hidden_dims… → latent_dim → reversed(hidden_dims) → input_dim``.

    Linear output layer (inputs are standardized, so no squashing).
    BatchNorm is off by default: inference-time batch statistics distort anomaly scores (ADR 6).
    """

    def __init__(self, input_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.cfg = cfg
        self.encoder = self._stack([input_dim, *cfg.hidden_dims, cfg.latent_dim], final_act=True)
        self.decoder = self._stack(
            [cfg.latent_dim, *reversed(cfg.hidden_dims), input_dim], final_act=False
        )

    def _stack(self, dims: list[int], *, final_act: bool) -> nn.Sequential:
        act = _ACTIVATIONS[self.cfg.activation]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(a, b))
            last = i == len(dims) - 2
            if last and not final_act:
                break
            if self.cfg.batch_norm:
                layers.append(nn.BatchNorm1d(b))
            layers.append(act())
            if self.cfg.dropout > 0:
                layers.append(nn.Dropout(self.cfg.dropout))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_model(input_dim: int, cfg: ModelConfig) -> Autoencoder:
    return Autoencoder(input_dim, cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
