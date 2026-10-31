"""Symmetric MLP autoencoder (ARCHITECTURE.md §3.2) and, for Roadmap v2 Phase 8, a variational
autoencoder with a Gaussian decoder whose reconstruction probability is an anomaly score
(An & Cho 2015). Both expose ``encode``, ``forward`` (deterministic reconstruction) and
``loss(x, generator)`` so the training loop is shared."""

from __future__ import annotations

import math

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

    def loss(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Mean squared reconstruction error (the training objective)."""
        return ((self(x) - x) ** 2).mean()


LOGVAR_MIN, LOGVAR_MAX = -6.0, 3.0
_LOG_2PI = math.log(2 * math.pi)


class VariationalAutoencoder(Autoencoder):
    """Encoder trunk ``input_dim → hidden_dims`` with two heads (μ_z, log σ_z²) of ``latent_dim``;
    decoder trunk ``latent_dim → reversed(hidden_dims)`` with two heads (μ_x, log σ_x²) over the
    inputs. ``forward`` returns the deterministic reconstruction μ_x(μ_z(x)) so that residual
    scores, early stopping and the per-feature analysis work unchanged; ``loss`` is the negative
    ELBO per row; ``reconstruction_nll`` is −log p(x | z) averaged over ``n_samples`` draws of
    z ~ q(z | x) (the reconstruction probability score, higher = more anomalous)."""

    def __init__(self, input_dim: int, cfg: ModelConfig) -> None:
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.cfg = cfg
        h = list(cfg.hidden_dims)
        self.encoder = self._stack([input_dim, *h], final_act=True)
        self.z_mean = nn.Linear(h[-1], cfg.latent_dim)
        self.z_logvar = nn.Linear(h[-1], cfg.latent_dim)
        self.decoder = self._stack([cfg.latent_dim, *reversed(h)], final_act=True)
        self.x_mean = nn.Linear(h[0], input_dim)
        self.x_logvar = nn.Linear(h[0], input_dim)

    def encode_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.encoder(x)
        return self.z_mean(t), self.z_logvar(t).clamp(LOGVAR_MIN, LOGVAR_MAX)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_dist(x)[0]

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.decoder(z)
        return self.x_mean(t), self.x_logvar(t).clamp(LOGVAR_MIN, LOGVAR_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))[0]

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None):
        eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def _nll(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """−log N(x; mean, exp(logvar)) summed over the input dimensions; shape (n,)."""
        return 0.5 * (_LOG_2PI + logvar + (x - mean) ** 2 / torch.exp(logvar)).sum(dim=1)

    def loss(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        mu, logvar = self.encode_dist(x)
        mean_x, logvar_x = self.decode(self._sample(mu, logvar, generator))
        kl = -0.5 * (1 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)
        return (self._nll(x, mean_x, logvar_x) + kl).mean()

    @torch.no_grad()
    def reconstruction_nll(
        self, x: torch.Tensor, n_samples: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        mu, logvar = self.encode_dist(x)
        total = torch.zeros(x.shape[0], dtype=x.dtype)
        for _ in range(n_samples):
            mean_x, logvar_x = self.decode(self._sample(mu, logvar, generator))
            total += self._nll(x, mean_x, logvar_x)
        return total / n_samples


def build_model(input_dim: int, cfg: ModelConfig) -> Autoencoder:
    if cfg.kind == "vae":
        return VariationalAutoencoder(input_dim, cfg)
    return Autoencoder(input_dim, cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
