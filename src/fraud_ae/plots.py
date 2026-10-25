"""All figures (matplotlib, headless ``Agg``). Each function saves a PNG and returns its path.

Figures referenced by the report (ARCHITECTURE.md §3.6): training curves, score histogram with
threshold, ROC, PR, recall-vs-FPR with the (1 %, 85 %) target marker, threshold sweep,
per-feature residuals (explainability), seed variance, baseline comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def training_curves(history_csv: Path, out: Path) -> Path:
    raise NotImplementedError("Phase 2")


def score_histogram(scores: np.ndarray, y: np.ndarray, threshold: float, out: Path) -> Path:
    raise NotImplementedError("Phase 3")


def roc_pr_curves(scores: np.ndarray, y: np.ndarray, out: Path) -> Path:
    raise NotImplementedError("Phase 4")


def recall_vs_fpr(
    scores: np.ndarray, y: np.ndarray, target: tuple[float, float], out: Path
) -> Path:
    raise NotImplementedError("Phase 4")


def threshold_sweep(scores: np.ndarray, y: np.ndarray, threshold: float, out: Path) -> Path:
    raise NotImplementedError("Phase 4")


def per_feature_residuals(
    residuals: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...], out: Path
) -> Path:
    raise NotImplementedError("Phase 4")


def baseline_comparison(metrics_by_run: dict[str, dict], out: Path) -> Path:
    raise NotImplementedError("Phase 5")
