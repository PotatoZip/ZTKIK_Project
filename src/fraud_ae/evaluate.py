"""Phase 4 – metrics on the held-out test set (ARCHITECTURE.md §3).

Threshold-free: ROC-AUC, PR-AUC (average precision), recall at each FPR in ``eval.fpr_grid``
(threshold set on the *test* legit scores themselves, i.e. the ranking quality of the score).

At the threshold chosen in Phase 3 on validation data: confusion matrix, precision, recall, FPR,
F1, Wilson score intervals for every proportion, and a stratified bootstrap CI for recall and
FPR at the threshold and for recall@``eval.target_fpr``.

``meets_target`` = recall ≥ ``eval.target_recall`` **and** FPR < ``eval.target_fpr`` at the
threshold (point estimates, as the criterion is stated in the abstract). The CIs say how much to
trust that verdict. Alarm rule everywhere: ``score > threshold``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import EvalConfig

METRICS_FILE = "metrics.json"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n`` (``(nan, nan)`` if ``n == 0``)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def threshold_for_fpr(scores_legit: np.ndarray, fpr: float) -> float:
    """Threshold at which exactly ``floor(fpr * n_legit)`` legit rows satisfy ``score > t``
    (barring ties). ``fpr = 0`` returns the maximum legit score."""
    s = np.sort(np.asarray(scores_legit, dtype=np.float64))
    n = s.size
    if n == 0:
        raise ValueError("no legitimate scores")
    if not 0.0 <= fpr < 1.0:
        raise ValueError("fpr must be in [0, 1)")
    k = int(math.floor(fpr * n))  # number of legit alarms allowed
    return float(s[n - k - 1])


def recall_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
    """Recall obtained when the threshold is set so that exactly ``fpr`` of legit rows alarm."""
    scores, y = _check(scores, y)
    t = threshold_for_fpr(scores[y == 0], fpr)
    fraud = scores[y == 1]
    return float((fraud > t).mean()) if fraud.size else float("nan")


def confusion_at_threshold(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, int]:
    """``{"tp", "fp", "tn", "fn"}`` with alarms defined as ``score > threshold``."""
    scores, y = _check(scores, y)
    alarm = scores > threshold
    return {
        "tp": int(np.sum(alarm & (y == 1))),
        "fp": int(np.sum(alarm & (y == 0))),
        "tn": int(np.sum(~alarm & (y == 0))),
        "fn": int(np.sum(~alarm & (y == 1))),
    }


def rates_from_confusion(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 and not math.isnan(precision)
        else 0.0
    )
    return {"recall": recall, "fpr": fpr, "precision": precision, "f1": f1}


def bootstrap_ci(
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    target_fpr: float,
    n_samples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Stratified bootstrap (legit and fraud resampled separately, class sizes fixed).

    Returns percentile CIs for recall and FPR at ``threshold`` and for recall at ``target_fpr``
    (threshold re-derived on each resampled legit set).
    """
    scores, y = _check(scores, y)
    legit, fraud = scores[y == 0], scores[y == 1]
    rng = np.random.default_rng(seed)
    rec_t = np.empty(n_samples)
    fpr_t = np.empty(n_samples)
    rec_f = np.empty(n_samples)
    n_l, n_f = legit.size, fraud.size
    for i in range(n_samples):
        bl = legit[rng.integers(0, n_l, n_l)]
        bf = fraud[rng.integers(0, n_f, n_f)] if n_f else fraud
        rec_t[i] = (bf > threshold).mean() if n_f else np.nan
        fpr_t[i] = (bl > threshold).mean()
        rec_f[i] = (bf > threshold_for_fpr(bl, target_fpr)).mean() if n_f else np.nan

    def summary(a: np.ndarray) -> dict[str, float]:
        if np.isnan(a).all():
            return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        lo, hi = np.percentile(a, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"mean": float(a.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}

    return {
        "samples": int(n_samples),
        "seed": int(seed),
        "alpha": alpha,
        "recall_at_threshold": summary(rec_t),
        "fpr_at_threshold": summary(fpr_t),
        "recall_at_target_fpr": summary(rec_f),
    }


def evaluate(
    scores: np.ndarray, y: np.ndarray, threshold: float, cfg: EvalConfig, seed: int = 0
) -> dict[str, Any]:
    """Full metric dictionary; serialisable to ``metrics.json``."""
    scores, y = _check(scores, y)
    legit, fraud = scores[y == 0], scores[y == 1]
    n_l, n_f = int(legit.size), int(fraud.size)
    if n_l == 0:
        raise ValueError("evaluation needs legitimate rows")
    both = n_f > 0

    # --- threshold-free -------------------------------------------------------------------
    grid = []
    for f in cfg.fpr_grid:
        t = threshold_for_fpr(legit, f)
        fp = int((legit > t).sum())
        tp = int((fraud > t).sum())
        lo, hi = wilson_interval(tp, n_f)
        grid.append(
            {
                "fpr_target": float(f),
                "threshold": t,
                "fpr": fp / n_l,
                "fp": fp,
                "recall": tp / n_f if both else float("nan"),
                "tp": tp,
                "recall_ci_wilson": [lo, hi],
            }
        )

    # --- at the Phase 3 threshold ---------------------------------------------------------
    cm = confusion_at_threshold(scores, y, threshold)
    rates = rates_from_confusion(cm)
    rec_ci = wilson_interval(cm["tp"], cm["tp"] + cm["fn"])
    fpr_ci = wilson_interval(cm["fp"], cm["fp"] + cm["tn"])
    prec_ci = wilson_interval(cm["tp"], cm["tp"] + cm["fp"])

    boot = bootstrap_ci(scores, y, threshold, cfg.target_fpr, cfg.bootstrap_samples, seed)

    # --- criterion --------------------------------------------------------------------------
    fp_allowed = math.ceil(cfg.target_fpr * n_l) - 1  # largest fp with fp / n_l < target_fpr
    tp_required = math.ceil(cfg.target_recall * n_f) if both else 0
    meets_fpr = rates["fpr"] < cfg.target_fpr
    meets_recall = both and rates["recall"] >= cfg.target_recall
    meets_target = bool(meets_fpr and meets_recall)

    return {
        "n_legit": n_l,
        "n_fraud": n_f,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, scores)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, scores)) if both else float("nan"),
        "recall_at_fpr": grid,
        "at_threshold": {
            **cm,
            **rates,
            "recall_ci_wilson": list(rec_ci),
            "fpr_ci_wilson": list(fpr_ci),
            "precision_ci_wilson": list(prec_ci),
        },
        "bootstrap": boot,
        "criterion": {
            "target_recall": cfg.target_recall,
            "target_fpr": cfg.target_fpr,
            "fp_allowed": int(fp_allowed),
            "tp_required": int(tp_required),
            "meets_fpr": bool(meets_fpr),
            "meets_recall": bool(meets_recall),
        },
        "meets_target": meets_target,
    }


def per_feature_summary(
    residuals: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...]
) -> dict[str, Any]:
    """Mean squared residual per input feature for legit and fraud rows (explainability)."""
    R = np.asarray(residuals, dtype=np.float64)
    y = np.asarray(y)
    if R.ndim != 2 or R.shape[0] != y.shape[0] or R.shape[1] != len(feature_names):
        raise ValueError("residuals must be (n, d) with d == len(feature_names) and n == len(y)")
    legit = (R[y == 0] ** 2).mean(axis=0) if (y == 0).any() else np.full(R.shape[1], np.nan)
    fraud = (R[y == 1] ** 2).mean(axis=0) if (y == 1).any() else np.full(R.shape[1], np.nan)
    ratio = fraud / np.maximum(legit, 1e-12)
    order = np.argsort(-ratio)
    return {
        "features": list(feature_names),
        "legit_msr": legit.tolist(),
        "fraud_msr": fraud.tolist(),
        "ratio": ratio.tolist(),
        "top_features": [feature_names[i] for i in order[:5]],
    }


def save_metrics(metrics: dict[str, Any], run_dir: Path) -> Path:
    out = Path(run_dir) / METRICS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, allow_nan=True))
    return out


def load_metrics(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / METRICS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – run `fraud-ae evaluate` first")
    return json.loads(path.read_text())


def _check(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if scores.ndim != 1 or scores.shape != y.shape:
        raise ValueError("scores and y must be 1-D arrays of the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    return scores, y
