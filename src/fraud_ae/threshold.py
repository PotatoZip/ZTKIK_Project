"""Phase 3b – threshold selection (ARCHITECTURE.md §3).

Primary strategy ``quantile`` uses *legitimate validation scores only*: the threshold is the
``1 - target_fpr`` quantile, so the FPR criterion holds by construction and no fraud label is
touched. ``youden`` / ``max_f1`` are semi-supervised alternatives that use val fraud labels and
are reported for comparison only. ``quantile_by_hour`` (Phase 8) takes the legit quantile per
hour segment (night / day) and divides every score by its segment's quantile, so the stored
scores are segment-normalised and the single threshold is 1.0; the global val FPR equals the
target because each segment is at the target.

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
    segments: dict | None = None  # quantile_by_hour: per-segment thresholds in raw score units

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
    elif cfg.strategy == "quantile_by_hour":
        raise ValueError(
            "quantile_by_hour needs the hour of each row – use hour_segmented_threshold"
        )
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


def is_night(hours: np.ndarray, night_hours: tuple[float, float]) -> np.ndarray:
    h = np.asarray(hours, dtype=np.float64)
    return (h >= night_hours[0]) & (h < night_hours[1])


def normalize_by_segment(scores: np.ndarray, night: np.ndarray, q_night: float, q_day: float):
    """Divide each score by its segment's legit quantile; the common threshold becomes 1.0."""
    scores = np.asarray(scores, dtype=np.float64)
    return np.where(np.asarray(night, bool), scores / q_night, scores / q_day)


def hour_segmented_threshold(
    scores_val: np.ndarray,
    y_val: np.ndarray,
    hours_val: np.ndarray,
    scores_test: np.ndarray,
    hours_test: np.ndarray,
    cfg: ThresholdConfig,
    score_method: str = "mse",
) -> tuple[ThresholdResult, np.ndarray, np.ndarray]:
    """Phase 8 ``quantile_by_hour``: the ``1 − target_fpr`` legit quantile is taken separately for
    night (``cfg.night_hours``) and day validation rows; scores of both splits are divided by
    their segment's quantile. Returns ``(result, scores_val_normalised, scores_test_normalised)``
    with ``result.value == 1.0`` and the raw per-segment thresholds in ``result.segments``."""
    scores_val, y_val = np.asarray(scores_val, dtype=np.float64), np.asarray(y_val)
    night_val, night_test = (
        is_night(hours_val, cfg.night_hours),
        is_night(hours_test, cfg.night_hours),
    )
    legit = y_val == 0
    q_night = quantile_threshold(scores_val[legit & night_val], cfg.target_fpr)
    q_day = quantile_threshold(scores_val[legit & ~night_val], cfg.target_fpr)
    sv = normalize_by_segment(scores_val, night_val, q_night, q_day)
    st = normalize_by_segment(scores_test, night_test, q_night, q_day)
    result = ThresholdResult(
        strategy="quantile_by_hour",
        target_fpr=cfg.target_fpr,
        value=1.0,
        val_fpr=fpr_at(sv[legit], 1.0),
        val_recall=recall_at(sv[y_val == 1], 1.0),
        score_method=score_method,
        n_val_legit=int(legit.sum()),
        n_val_fraud=int((y_val == 1).sum()),
        segments={
            "night_hours": [float(cfg.night_hours[0]), float(cfg.night_hours[1])],
            "night": q_night,
            "day": q_day,
            "n_val_legit_night": int((legit & night_val).sum()),
            "n_val_legit_day": int((legit & ~night_val).sum()),
            "val_fpr_night": fpr_at(sv[legit & night_val], 1.0),
            "val_fpr_day": fpr_at(sv[legit & ~night_val], 1.0),
            "test_rows_night": int(night_test.sum()),
        },
    )
    return result, sv, st


def save_threshold(result: ThresholdResult, run_dir: Path) -> Path:
    out = Path(run_dir) / THRESHOLD_FILE
    out.write_text(json.dumps(result.to_dict(), indent=2))
    return out


def load_threshold(run_dir: Path) -> ThresholdResult:
    path = Path(run_dir) / THRESHOLD_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – run `fraud-ae threshold` first")
    return ThresholdResult(**json.loads(path.read_text()))
