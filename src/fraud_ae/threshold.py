"""Phase 3b – threshold selection (ARCHITECTURE.md §3).

Primary strategy ``quantile`` uses *legitimate validation scores only*: the threshold is the
``1 - target_fpr`` quantile, so the FPR criterion holds by construction and no fraud label is
touched. ``youden`` / ``max_f1`` are semi-supervised alternatives that use val fraud labels and
are reported for comparison only.

Alarm rule everywhere in the project: ``score > threshold``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import ThresholdConfig

THRESHOLD_FILE = "threshold.json"


@dataclass(frozen=True)
class ThresholdResult:
    strategy: str
    target_fpr: float
    value: float
    val_fpr: float
    val_recall: float | None  # None when no frauds are present in val
    score_method: str = "mse"
    n_val_legit: int = 0
    n_val_fraud: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def quantile_threshold(scores_legit: np.ndarray, target_fpr: float) -> float:
    """``np.quantile(scores_legit, 1 - target_fpr)`` – uses legitimate rows only."""
    scores_legit = np.asarray(scores_legit, dtype=np.float64)
    if scores_legit.size == 0:
        raise ValueError("no legitimate scores to take the quantile of")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.quantile(scores_legit, 1.0 - target_fpr))


def youden_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising Youden's J = TPR − FPR (needs both classes)."""
    _check_both_classes(y)
    fpr, tpr, thr = roc_curve(y, scores)
    j = tpr - fpr
    best = int(np.argmax(j))
    # roc_curve thresholds are "score >= thr"; our alarm rule is "score > threshold", so step
    # down to the largest score strictly below the chosen one.
    return _strictly_below(scores, thr[best])


def max_f1_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximising F1 on the given (val) labels."""
    _check_both_classes(y)
    precision, recall, thr = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return _strictly_below(scores, thr[best])


def _check_both_classes(y: np.ndarray) -> None:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("semi-supervised strategies need both classes in the validation set")


def _strictly_below(scores: np.ndarray, t: float) -> float:
    """Largest score < t, so that ``score > result`` is equivalent to ``score >= t``."""
    below = scores[scores < t]
    return float(below.max()) if below.size else float(t - 1e-12)


def fpr_at(scores_legit: np.ndarray, threshold: float) -> float:
    return float((np.asarray(scores_legit) > threshold).mean()) if len(scores_legit) else 0.0


def recall_at(scores_fraud: np.ndarray, threshold: float) -> float | None:
    return float((np.asarray(scores_fraud) > threshold).mean()) if len(scores_fraud) else None


def choose_threshold(
    scores_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ThresholdConfig,
    score_method: str = "mse",
) -> ThresholdResult:
    scores_val = np.asarray(scores_val, dtype=np.float64)
    y_val = np.asarray(y_val)
    if scores_val.shape != y_val.shape:
        raise ValueError("scores_val and y_val must have the same shape")
    legit, fraud = scores_val[y_val == 0], scores_val[y_val == 1]

    if cfg.strategy == "quantile":
        value = quantile_threshold(legit, cfg.target_fpr)
    elif cfg.strategy == "youden":
        value = youden_threshold(scores_val, y_val)
    elif cfg.strategy == "max_f1":
        value = max_f1_threshold(scores_val, y_val)
    else:
        raise ValueError(f"unknown threshold strategy {cfg.strategy!r}")

    return ThresholdResult(
        strategy=cfg.strategy,
        target_fpr=cfg.target_fpr,
        value=value,
        val_fpr=fpr_at(legit, value),
        val_recall=recall_at(fraud, value),
        score_method=score_method,
        n_val_legit=int(len(legit)),
        n_val_fraud=int(len(fraud)),
    )


def save_threshold(result: ThresholdResult, run_dir: Path) -> Path:
    out = Path(run_dir) / THRESHOLD_FILE
    out.write_text(json.dumps(result.to_dict(), indent=2))
    return out


def load_threshold(run_dir: Path) -> ThresholdResult:
    path = Path(run_dir) / THRESHOLD_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – run `fraud-ae threshold` first")
    return ThresholdResult(**json.loads(path.read_text()))
