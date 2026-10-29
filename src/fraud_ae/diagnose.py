"""Phase 7 – diagnosis of the frauds a run misses (ROADMAP.md, Roadmap v2).

Given a run directory (AE or baseline) with ``scores_test.npz`` and ``threshold.json``, map the
missed test frauds back to their raw rows and describe them next to the detected frauds and the
legitimate test rows: anomaly score and its percentile among legit scores, amount, hour of day
and the raw features that separate detected frauds from legit traffic. With ``--against``
runs on the *same* test split it also reports how many of the missed frauds the other
detectors catch, the Jaccard overlap of the missed sets, and ensemble upper bounds (union at the
runs' own thresholds, union at per-run oracle thresholds, rank-average fusion at an oracle FPR).

Raw rows are recovered through the processed directory's ``data_report.json`` (``raw_csv`` and
``drop_duplicates`` of the config that produced the split), so the same code works for the real
CSV, the synthetic CSV and splits without the hour feature. Writes ``diagnostics.json`` and
``figures/missed_frauds.png`` into the run directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .data import V_COLUMNS, drop_duplicates, hour_of_day, load_raw, load_report, load_splits
from .evaluate import threshold_for_fpr, wilson_interval
from .scoring import load_scores
from .threshold import load_threshold
from .train import CONFIG_FILE

DIAGNOSTICS_FILE = "diagnostics.json"
FIGURE_FILE = "figures/missed_frauds.png"
NIGHT_HOURS = (0.0, 6.0)  # [start, end) in hours of day
TOP_K = 5


@dataclass(frozen=True)
class DiagnosisArrays:
    """Row-level arrays the figure needs (the JSON summary holds only aggregates)."""

    scores: np.ndarray
    y: np.ndarray
    missed: np.ndarray
    detected: np.ndarray
    hours: np.ndarray
    amount: np.ndarray
    threshold: float
    top_features: dict[str, Any]


# --------------------------------------------------------------------------- pure functions


def missed_masks(
    scores: np.ndarray, y: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(missed, detected)`` boolean masks over all rows; both are subsets of ``y == 1``."""
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y).astype(int)
    if scores.shape != y.shape:
        raise ValueError("scores and y must have the same shape")
    alarm = scores > threshold
    return (y == 1) & ~alarm, (y == 1) & alarm


def legit_percentile(scores_legit: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Share of legit scores strictly below each value (0 = lowest legit score, 1 = above all)."""
    s = np.sort(np.asarray(scores_legit, dtype=np.float64))
    if s.size == 0:
        raise ValueError("no legitimate scores")
    return np.searchsorted(s, np.asarray(values, dtype=np.float64), side="left") / s.size


def _median(a: np.ndarray) -> float:
    return float(np.median(a)) if len(a) else float("nan")


def group_profile(
    scores: np.ndarray, pct: np.ndarray, hours: np.ndarray, amount: np.ndarray
) -> dict[str, Any]:
    """Aggregates for one group of rows (legit / detected frauds / missed frauds)."""
    hours = np.asarray(hours, dtype=float)
    night = (hours >= NIGHT_HOURS[0]) & (hours < NIGHT_HOURS[1])
    return {
        "n": int(len(scores)),
        "score_median": _median(scores),
        "legit_percentile_median": _median(pct),
        "amount_median": _median(amount),
        "amount_mean": float(np.mean(amount)) if len(amount) else float("nan"),
        "night_share": float(night.mean()) if len(hours) else float("nan"),
        "hour_counts": np.bincount(np.floor(hours).astype(int) % 24, minlength=24).tolist(),
    }


def top_features(
    raw: pd.DataFrame,
    legit: np.ndarray,
    detected: np.ndarray,
    missed: np.ndarray,
    k: int = TOP_K,
) -> dict[str, Any]:
    """The ``k`` raw features (V1..V28, Amount) that separate detected frauds from legit rows
    most, measured as |mean(detected) − mean(legit)| / std(legit); means of the three groups."""
    cols = [*V_COLUMNS, "Amount"]
    X = raw[cols].to_numpy(dtype=np.float64)
    mu_l, sd_l = X[legit].mean(axis=0), X[legit].std(axis=0)
    mean_or_nan = lambda m: X[m].mean(axis=0) if m.any() else np.full(X.shape[1], np.nan)  # noqa: E731
    mu_d, mu_m = mean_or_nan(detected), mean_or_nan(missed)
    effect = np.abs(np.nan_to_num(mu_d - mu_l)) / np.maximum(sd_l, 1e-12)
    order = np.argsort(-effect)[:k]
    return {
        "features": [cols[i] for i in order],
        "legit_mean": mu_l[order].tolist(),
        "detected_mean": mu_d[order].tolist(),
        "missed_mean": mu_m[order].tolist(),
        "effect_size_detected": effect[order].tolist(),
    }


def jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def overlap_matrix(missed_sets: dict[str, set[int]]) -> dict[str, Any]:
    names = list(missed_sets)
    return {
        "runs": names,
        "n_missed": [len(missed_sets[n]) for n in names],
        "jaccard": [[jaccard(missed_sets[a], missed_sets[b]) for b in names] for a in names],
    }


def fusion_bounds(
    scores_by_run: dict[str, np.ndarray],
    thresholds: dict[str, float],
    y: np.ndarray,
    fpr: float,
) -> dict[str, Any]:
    """Ensemble upper bounds over the given runs (same rows, same ``y``):

    * ``union_own`` – alarm if any run alarms at its own threshold (FPR adds up);
    * ``union_oracle`` – each run at its oracle threshold for ``fpr``, then the union;
    * ``rank_fusion`` – mean of per-run rank-normalised scores, oracle threshold at ``fpr``.
    """
    y = np.asarray(y).astype(int)
    legit, n_fraud = y == 0, int((y == 1).sum())
    names = list(scores_by_run)
    S = np.column_stack([np.asarray(scores_by_run[n], dtype=np.float64) for n in names])
    if S.shape[0] != y.shape[0]:
        raise ValueError("scores and y must have the same length")
    own = np.column_stack([S[:, i] > thresholds[n] for i, n in enumerate(names)])
    oracle = np.column_stack(
        [S[:, i] > threshold_for_fpr(S[legit, i], fpr) for i in range(len(names))]
    )
    ranks = np.argsort(np.argsort(S, axis=0), axis=0) / max(S.shape[0] - 1, 1)
    fused = ranks.mean(axis=1)
    fused_alarm = fused > threshold_for_fpr(fused[legit], fpr)

    def summarise(alarm: np.ndarray) -> dict[str, Any]:
        tp, fp = int((alarm & (y == 1)).sum()), int((alarm & legit).sum())
        lo, hi = wilson_interval(tp, n_fraud)
        return {
            "tp": tp,
            "fp": fp,
            "recall": tp / n_fraud if n_fraud else float("nan"),
            "recall_ci_wilson": [lo, hi],
            "fpr": fp / int(legit.sum()),
        }

    return {
        "runs": names,
        "fpr_target": float(fpr),
        "union_own": summarise(own.any(axis=1)),
        "union_oracle": summarise(oracle.any(axis=1)),
        "rank_fusion": summarise(fused_alarm),
    }


# --------------------------------------------------------------------------- orchestration


def raw_test_rows(processed_dir: Path, root: Path, idx_test: np.ndarray) -> pd.DataFrame:
    """The raw (un-scaled) rows of the test split, recovered through ``data_report.json``."""
    dcfg = load_report(processed_dir)["config"]
    raw = load_raw(Path(root) / dcfg["raw_csv"])
    if dcfg["drop_duplicates"]:
        raw, _ = drop_duplicates(raw)
    else:
        raw = raw.reset_index(drop=True)
    if idx_test.max() >= len(raw):
        raise ValueError(f"{processed_dir} does not match {dcfg['raw_csv']} (index out of range)")
    return raw.iloc[idx_test].reset_index(drop=True)


def diagnose(
    run: Path,
    against: list[Path],
    processed_dir: Path,
    root: Path,
    fpr: float | None = None,
) -> tuple[dict[str, Any], DiagnosisArrays]:
    """Full diagnosis of ``run``; returns ``(summary, arrays)``. ``against`` runs must have been
    evaluated on the same test split (checked through their ``y``)."""
    run, against = Path(run), [Path(a) for a in against]
    cfg = load_config(run / CONFIG_FILE)
    fpr = cfg.eval.target_fpr if fpr is None else float(fpr)
    th = load_threshold(run)
    scores, y = load_scores(run, "test")
    y = y.astype(int)
    splits = load_splits(processed_dir)
    if not np.array_equal(splits.y_test.astype(int), y):
        raise ValueError(f"{run} was not scored on the test split in {processed_dir}")
    missed, detected = missed_masks(scores, y, th.value)
    legit = y == 0
    if legit.sum() == 0 or (y == 1).sum() == 0:
        raise ValueError("diagnosis needs legit and fraud rows in the test split")

    raw = raw_test_rows(processed_dir, root, splits.idx_test)
    hours = hour_of_day(raw["Time"])
    amount = raw["Amount"].to_numpy(dtype=np.float64)
    pct = legit_percentile(scores[legit], scores)
    groups = {
        name: group_profile(scores[m], pct[m], hours[m], amount[m])
        for name, m in (("legit", legit), ("detected", detected), ("missed", missed))
    }
    feats = top_features(raw, legit, detected, missed)

    scores_by_run, thresholds = {run.name: scores}, {run.name: th.value}
    missed_sets = {run.name: set(np.flatnonzero(missed).tolist())}
    others = []
    for other in against:
        s_o, y_o = load_scores(other, "test")
        if not np.array_equal(y_o.astype(int), y):
            raise ValueError(f"{other} was scored on a different test split than {run}")
        t_o = load_threshold(other).value
        m_o, _ = missed_masks(s_o, y, t_o)
        own = (s_o > t_o) & missed
        oracle = (s_o > threshold_for_fpr(s_o[legit], fpr)) & missed
        others.append(
            {
                "run": other.name,
                "threshold": float(t_o),
                "recall_own": float(((s_o > t_o) & (y == 1)).sum() / (y == 1).sum()),
                "caught_of_missed_own": int(own.sum()),
                "caught_of_missed_oracle": int(oracle.sum()),
            }
        )
        scores_by_run[other.name], thresholds[other.name] = s_o, t_o
        missed_sets[other.name] = set(np.flatnonzero(m_o).tolist())

    n_fraud = int((y == 1).sum())
    tp = int(detected.sum())
    summary = {
        "run": run.name,
        "score_method": th.score_method,
        "threshold": float(th.value),
        "fpr_target": fpr,
        "n_legit": int(legit.sum()),
        "n_fraud": n_fraud,
        "n_detected": tp,
        "n_missed": int(missed.sum()),
        "recall": tp / n_fraud,
        "recall_ci_wilson": list(wilson_interval(tp, n_fraud)),
        "fpr": float((scores[legit] > th.value).mean()),
        "groups": groups,
        "top_features": feats,
        "missed_rows": sorted(int(i) for i in splits.idx_test[missed]),
        "against": others,
        "overlap": overlap_matrix(missed_sets),
        "fusion": fusion_bounds(scores_by_run, thresholds, y, fpr) if others else None,
    }
    arrays = DiagnosisArrays(scores, y, missed, detected, hours, amount, float(th.value), feats)
    return summary, arrays


def save_diagnostics(summary: dict[str, Any], run_dir: Path) -> Path:
    out = Path(run_dir) / DIAGNOSTICS_FILE
    out.write_text(json.dumps(summary, indent=2, allow_nan=True))
    return out


def load_diagnostics(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / DIAGNOSTICS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found – run `fraud-ae diagnose` first")
    return json.loads(path.read_text())
