"""Phase 5 – classical baselines on the same splits, scored by the same evaluator.

* PCA reconstruction error with ``k`` components (a linear autoencoder).
* Isolation Forest fitted on train-legit.
Each baseline writes a run directory that follows the same artifact contract as the AE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .data import Splits


def pca_scores(X_train: np.ndarray, X: np.ndarray, n_components: int) -> np.ndarray:
    """Per-sample squared reconstruction error after projecting onto ``n_components`` PCs."""
    raise NotImplementedError("Phase 5")


def iforest_scores(X_train: np.ndarray, X: np.ndarray, seed: int) -> np.ndarray:
    """Negated ``score_samples`` so that higher = more anomalous."""
    raise NotImplementedError("Phase 5")


def run_baselines(cfg: Config, splits: Splits, runs_root: Path) -> dict[str, dict[str, Any]]:
    """Return ``{run_name: metrics}`` and write ``results/comparison.md``."""
    raise NotImplementedError("Phase 5")
