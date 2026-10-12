"""Phase 3a – anomaly scores from reconstruction residuals (ARCHITECTURE.md §3.4).

``mse``          per-sample mean squared residual (the abstract's method; primary)
``standardized`` residuals divided by per-feature residual std from val-legit, squared, summed
``mahalanobis``  residual vector vs. covariance of val-legit residuals
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ScoreConfig
from .model import Autoencoder


@dataclass(frozen=True)
class ScoreCalibration:
    """Statistics of val-legit residuals needed by the non-plain methods."""

    residual_mean: np.ndarray
    residual_std: np.ndarray
    residual_cov_inv: np.ndarray | None


def residuals(model: Autoencoder, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """``X - model(X)`` computed in eval mode without gradients; shape (n, d)."""
    raise NotImplementedError("Phase 3")


def calibrate(model: Autoencoder, X_val_legit: np.ndarray) -> ScoreCalibration:
    raise NotImplementedError("Phase 3")


def score(
    model: Autoencoder, X: np.ndarray, cfg: ScoreConfig, calib: ScoreCalibration | None = None
) -> np.ndarray:
    """Higher = more anomalous. Shape (n,)."""
    raise NotImplementedError("Phase 3")
