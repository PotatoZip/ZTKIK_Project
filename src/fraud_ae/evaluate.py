"""Phase 4 – metrics on the held-out test set (ARCHITECTURE.md §3.6).

Threshold-free: ROC-AUC, PR-AUC (average precision), recall at each FPR in ``eval.fpr_grid``.
At the chosen threshold: confusion matrix, precision, recall, FPR, F1, Wilson CIs, bootstrap CI.
``meets_target`` = recall ≥ eval.target_recall and FPR < eval.target_fpr at the threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``."""
    raise NotImplementedError("Phase 4")


def recall_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
    """Recall obtained when the threshold is set so that exactly ``fpr`` of legit rows alarm."""
    raise NotImplementedError("Phase 4")


def confusion_at_threshold(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, int]:
    """``{"tp", "fp", "tn", "fn"}`` with alarms defined as ``score > threshold``."""
    raise NotImplementedError("Phase 4")


def evaluate(
    scores: np.ndarray, y: np.ndarray, threshold: float, cfg: EvalConfig, seed: int = 0
) -> dict[str, Any]:
    """Full metric dictionary; serialisable to ``metrics.json``."""
    raise NotImplementedError("Phase 4")


def save_metrics(metrics: dict[str, Any], run_dir: Path) -> Path:
    raise NotImplementedError("Phase 4")
