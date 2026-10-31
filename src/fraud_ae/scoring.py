"""Phase 3a – anomaly scores from reconstruction residuals (ARCHITECTURE.md §3).

``mse``          per-sample mean squared residual (the abstract's method; primary)
``standardized`` residuals centred and divided by the per-feature residual std of val-legit,
                 squared and summed
``mahalanobis``  squared Mahalanobis distance of the residual vector w.r.t. the mean and
                 covariance of val-legit residuals
``recon_prob``   (VAE only) −log p(x | z) averaged over ``n_samples`` draws z ~ q(z | x): the
                 reconstruction probability of An & Cho (2015); Phase 8
``latent_nll``   negative log-likelihood of the encoder code under a Gaussian mixture fitted on
                 val-legit codes (latent density); Phase 8
``latent_gmm``   z(log MSE) + z(latent NLL), both standardised on val-legit: a residual-plus-code
                 score aimed at frauds that reconstruct well but sit in an unusual code region;
                 Phase 8 (DAGMM-lite)

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
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from .config import ScoreConfig
from .model import Autoencoder, VariationalAutoencoder

CALIBRATION_FILE = "score_calibration.npz"
EPS = 1e-12


@dataclass(frozen=True)
class ScoreCalibration:
    """Statistics of val-legit residuals needed by the non-plain methods."""

    residual_mean: np.ndarray
    residual_std: np.ndarray
    residual_cov_inv: np.ndarray | None
    # Phase 8 latent-density calibration (None unless the score method needs it)
    gmm_weights: np.ndarray | None = None
    gmm_means: np.ndarray | None = None
    gmm_covs: np.ndarray | None = None
    log_mse_stats: np.ndarray | None = None  # [mean, std] of log(mse) on val-legit
    latent_nll_stats: np.ndarray | None = None  # [mean, std] of the latent NLL on val-legit


@torch.no_grad()
def codes(model: Autoencoder, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Encoder output per row (the posterior mean μ_z for a VAE); shape (n, latent), float64."""
    X = np.asarray(X)
    was_training = model.training
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32)
    out = [model.encode(xb).double().numpy() for xb in xt.split(batch_size)]
    model.train(was_training)
    return np.concatenate(out) if out else np.empty((0, model.cfg.latent_dim))


def fit_latent_gmm(
    Z: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-covariance Gaussian mixture on legit codes → ``(weights, means, covariances)``."""
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=seed,
        n_init=1,
    ).fit(np.asarray(Z, dtype=np.float64))
    return gmm.weights_, gmm.means_, gmm.covariances_


def gmm_nll(Z: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """−log Σ_k w_k N(z; μ_k, Σ_k) per row (equals ``−GaussianMixture.score_samples``)."""
    Z = np.asarray(Z, dtype=np.float64)
    d = Z.shape[1]
    comp = np.empty((Z.shape[0], len(weights)))
    for k, (w, mu, cov) in enumerate(zip(weights, means, covs, strict=True)):
        L = np.linalg.cholesky(cov)
        sol = solve_triangular(L, (Z - mu).T, lower=True)
        maha = (sol**2).sum(axis=0)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        comp[:, k] = np.log(w) - 0.5 * (d * np.log(2 * np.pi) + logdet + maha)
    return -logsumexp(comp, axis=1)


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


def calibrate(
    model: Autoencoder,
    X_val_legit: np.ndarray,
    ridge: float = 1e-6,
    score_cfg: ScoreConfig | None = None,
    seed: int = 0,
) -> ScoreCalibration:
    """Residual mean / std / inverse covariance on legitimate validation rows; with a
    ``latent_gmm`` / ``latent_nll`` score config also the Gaussian mixture on the val-legit codes
    and the standardisation statistics of log-MSE and latent NLL."""
    R = residuals(model, X_val_legit)
    if len(R) < 2:
        raise ValueError("need at least two legitimate validation rows to calibrate")
    mean = R.mean(axis=0)
    std = R.std(axis=0)
    std = np.where(std > EPS, std, EPS)
    cov = np.cov(R, rowvar=False, bias=True) + ridge * np.eye(R.shape[1])  # ddof 0, like std
    cov_inv = np.linalg.pinv(cov)
    extra: dict[str, np.ndarray] = {}
    if score_cfg is not None and score_cfg.method in ("latent_gmm", "latent_nll"):
        Z = codes(model, X_val_legit)
        w, m, c = fit_latent_gmm(Z, score_cfg.gmm_components, seed)
        nll = gmm_nll(Z, w, m, c)
        log_mse = np.log((R**2).mean(axis=1) + EPS)
        extra = {
            "gmm_weights": w,
            "gmm_means": m,
            "gmm_covs": c,
            "log_mse_stats": np.array([log_mse.mean(), max(log_mse.std(), EPS)]),
            "latent_nll_stats": np.array([nll.mean(), max(nll.std(), EPS)]),
        }
    return ScoreCalibration(residual_mean=mean, residual_std=std, residual_cov_inv=cov_inv, **extra)


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
    model: Autoencoder,
    X: np.ndarray,
    cfg: ScoreConfig,
    calib: ScoreCalibration | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Higher = more anomalous. Shape (n,). ``seed`` drives the Monte-Carlo draws of
    ``recon_prob`` so that scoring is reproducible."""
    if cfg.method == "recon_prob":
        if not isinstance(model, VariationalAutoencoder):
            raise ValueError("score method 'recon_prob' needs a variational autoencoder")
        return reconstruction_probability(model, X, cfg.n_samples, seed)
    if cfg.method in ("latent_gmm", "latent_nll"):
        if calib is None or calib.gmm_weights is None:
            raise ValueError(f"score method {cfg.method!r} needs a latent-GMM calibration")
        nll = gmm_nll(codes(model, X), calib.gmm_weights, calib.gmm_means, calib.gmm_covs)
        if cfg.method == "latent_nll":
            return nll
        R = residuals(model, X)
        z_mse = (np.log((R**2).mean(axis=1) + EPS) - calib.log_mse_stats[0]) / calib.log_mse_stats[
            1
        ]
        z_nll = (nll - calib.latent_nll_stats[0]) / calib.latent_nll_stats[1]
        return z_mse + z_nll
    return score_from_residuals(residuals(model, X), cfg, calib)


@torch.no_grad()
def reconstruction_probability(
    model: VariationalAutoencoder, X: np.ndarray, n_samples: int, seed: int, batch_size: int = 4096
) -> np.ndarray:
    """−log p(x | z) averaged over ``n_samples`` seeded draws of z ~ q(z | x); float64 (n,)."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(int(seed))
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    out = [
        model.reconstruction_nll(xb, n_samples, gen).double().numpy() for xb in xt.split(batch_size)
    ]
    model.train(was_training)
    return np.concatenate(out) if out else np.empty(0)


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
    arrays = {k: v for k, v in calib.__dict__.items() if v is not None}
    np.savez_compressed(out, **arrays)
    return out


def load_calibration(run_dir: Path) -> ScoreCalibration:
    with np.load(Path(run_dir) / CALIBRATION_FILE) as z:
        return ScoreCalibration(**{k: z[k] for k in z.files})
