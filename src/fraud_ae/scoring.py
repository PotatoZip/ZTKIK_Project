"""Phase 3a – anomaly scores from reconstruction residuals (ARCHITECTURE.md §3).

``mse``          per-sample mean squared residual (the abstract's method; primary)
``standardized`` residuals centred and divided by the per-feature residual std of val-legit,
                 squared and summed
``mahalanobis``  squared Mahalanobis distance of the residual vector w.r.t. the mean and
                 covariance of val-legit residuals

Higher = more anomalous for every method. The calibration statistics come from *legitimate
validation rows only*, so no fraud label is used.

Artifacts (``runs/<name>/``): ``scores_val.npz`` and ``scores_test.npz`` with ``scores`` and
``y`` arrays, plus ``score_calibration.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ScoreConfig
from .model import Autoencoder

CALIBRATION_FILE = "score_calibration.npz"
EPS = 1e-12


@dataclass(frozen=True)
class ScoreCalibration:
    """Statistics of val-legit residuals needed by the non-plain methods."""

    residual_mean: np.ndarray
    residual_std: np.ndarray
    residual_cov_inv: np.ndarray | None


@torch.no_grad()
def residuals(model: Autoencoder, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """``X - model(X)`` computed in eval mode without gradients; shape (n, d), float64."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    was_training = model.training
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32)
    out = [(xb - model(xb)).double().numpy() for xb in xt.split(batch_size)]
    model.train(was_training)
    return np.concatenate(out) if out else np.empty((0, X.shape[1]))


def calibrate(model: Autoencoder, X_val_legit: np.ndarray, ridge: float = 1e-6) -> ScoreCalibration:
    """Residual mean / std / inverse covariance on legitimate validation rows."""
    R = residuals(model, X_val_legit)
    if len(R) < 2:
        raise ValueError("need at least two legitimate validation rows to calibrate")
    mean = R.mean(axis=0)
    std = R.std(axis=0)
    std = np.where(std > EPS, std, EPS)
    cov = np.cov(R, rowvar=False, bias=True) + ridge * np.eye(R.shape[1])  # ddof 0, like std
    cov_inv = np.linalg.pinv(cov)
    return ScoreCalibration(residual_mean=mean, residual_std=std, residual_cov_inv=cov_inv)


def score_from_residuals(
    R: np.ndarray, cfg: ScoreConfig, calib: ScoreCalibration | None = None
) -> np.ndarray:
    """Turn a residual matrix into scores; see module docstring for the three methods."""
    R = np.asarray(R, dtype=np.float64)
    if cfg.method == "mse":
        return (R**2).mean(axis=1)
    if calib is None:
        raise ValueError(f"score method {cfg.method!r} needs a ScoreCalibration")
    Z = R - calib.residual_mean
    if cfg.method == "standardized":
        return ((Z / calib.residual_std) ** 2).sum(axis=1)
    if cfg.method == "mahalanobis":
        if calib.residual_cov_inv is None:
            raise ValueError("calibration has no covariance")
        return np.einsum("ij,jk,ik->i", Z, calib.residual_cov_inv, Z)
    raise ValueError(f"unknown score method {cfg.method!r}")


def score(
    model: Autoencoder, X: np.ndarray, cfg: ScoreConfig, calib: ScoreCalibration | None = None
) -> np.ndarray:
    """Higher = more anomalous. Shape (n,)."""
    return score_from_residuals(residuals(model, X), cfg, calib)


# --------------------------------------------------------------------------- artifact I/O


def save_scores(run_dir: Path, name: str, scores: np.ndarray, y: np.ndarray) -> Path:
    """Write ``scores_<name>.npz`` with ``scores`` and ``y``."""
    out = Path(run_dir) / f"scores_{name}.npz"
    np.savez_compressed(out, scores=np.asarray(scores, dtype=np.float64), y=np.asarray(y))
    return out


def load_scores(run_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(run_dir) / f"scores_{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – run `fraud-ae threshold` first")
    with np.load(path) as z:
        return z["scores"], z["y"]


def save_calibration(calib: ScoreCalibration, run_dir: Path) -> Path:
    out = Path(run_dir) / CALIBRATION_FILE
    np.savez_compressed(
        out,
        residual_mean=calib.residual_mean,
        residual_std=calib.residual_std,
        residual_cov_inv=calib.residual_cov_inv,
    )
    return out


def load_calibration(run_dir: Path) -> ScoreCalibration:
    with np.load(Path(run_dir) / CALIBRATION_FILE) as z:
        return ScoreCalibration(z["residual_mean"], z["residual_std"], z["residual_cov_inv"])
