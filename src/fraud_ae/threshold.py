"""Phase 3b – threshold selection (ARCHITECTURE.md §3.5).

Primary strategy ``quantile`` uses *legitimate validation scores only*: the threshold is the
``1 - target_fpr`` quantile, so the FPR criterion holds by construction and no fraud label is
touched. ``youden`` / ``max_f1`` are semi-supervised alternatives that use val fraud labels and
are reported for comparison only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import ThresholdConfig


@dataclass(frozen=True)
class ThresholdResult:
    strategy: str
    target_fpr: float
    value: float
    val_fpr: float
    val_recall: float | None  # None when no frauds are present in val

    def to_dict(self) -> dict:
        return asdict(self)


def quantile_threshold(scores_legit: np.ndarray, target_fpr: float) -> float:
    """``np.quantile(scores_legit, 1 - target_fpr)``."""
    raise NotImplementedError("Phase 3")


def choose_threshold(
    scores_val: np.ndarray, y_val: np.ndarray, cfg: ThresholdConfig
) -> ThresholdResult:
    raise NotImplementedError("Phase 3")


def save_threshold(result: ThresholdResult, run_dir: Path) -> Path:
    raise NotImplementedError("Phase 3")


def load_threshold(run_dir: Path) -> ThresholdResult:
    raise NotImplementedError("Phase 3")
