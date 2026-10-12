"""Phase 2 – training loop with early stopping on validation-legit MSE (ARCHITECTURE.md §3.3).

Run-directory contract (``runs/<name>/``)::

    config.toml   verbatim snapshot of the config used
    model.pt      best checkpoint (state_dict + input_dim + model config)
    history.csv   epoch, train_mse, val_legit_mse, lr
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .model import Autoencoder


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` and enable deterministic algorithms."""
    raise NotImplementedError("Phase 2")


def train(cfg: Config, X_train: np.ndarray, X_val_legit: np.ndarray, run_dir: Path) -> Autoencoder:
    """Train on legitimate rows only; keep the checkpoint with the lowest val-legit MSE."""
    raise NotImplementedError("Phase 2")


def load_model(run_dir: Path) -> Autoencoder:
    """Rebuild the model from ``model.pt`` (no config file needed)."""
    raise NotImplementedError("Phase 2")
