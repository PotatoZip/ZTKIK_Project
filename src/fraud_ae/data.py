"""Phase 1 – data preprocessing for the Credit Card Fraud dataset.

Pipeline (see ARCHITECTURE.md §3.1)::

    dataset.zip → raw CSV (sha256 checked) → drop duplicates → feature engineering
    → seeded split (train: legit only) → StandardScaler fit on train → processed artifacts

Artifacts written to ``DataConfig.processed_dir``:
    splits.npz          X_train, X_val, y_val, X_test, y_test, idx_train, idx_val, idx_test
    scaler.joblib       fitted sklearn StandardScaler (train statistics only)
    feature_names.json  ordered list of model input columns
    data_report.json    row counts, duplicates dropped, class ratio, sha256, config snapshot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

RAW_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


@dataclass(frozen=True)
class Splits:
    """Scaled feature matrices and labels. ``y_train`` is implicitly all zeros."""

    X_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    feature_names: tuple[str, ...]


def extract_dataset(zip_path: Path, raw_csv: Path) -> Path:
    """Unzip ``creditcard.csv`` into ``raw_csv`` if it is not already there."""
    raise NotImplementedError("Phase 1")


def sha256_of(path: Path) -> str:
    """Hex digest of a file, used to pin the dataset version in data_report.json."""
    raise NotImplementedError("Phase 1")


def load_raw(raw_csv: Path) -> pd.DataFrame:
    """Read the CSV and assert the expected 31 columns."""
    raise NotImplementedError("Phase 1")


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows *before* splitting; returns (df, n_dropped)."""
    raise NotImplementedError("Phase 1")


def engineer_features(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Return the model input frame (no ``Class``).

    * ``Amount`` → ``log1p(Amount)`` when ``cfg.log_amount``
    * ``Time``   → ``hour_sin``, ``hour_cos`` of hour-of-day when ``cfg.time_feature == "sincos"``,
      dropped otherwise
    * ``V1..V28`` unchanged
    """
    raise NotImplementedError("Phase 1")


def split_indices(
    y: np.ndarray, cfg: DataConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded split. Legit rows follow ``cfg.legit_fractions``; frauds go to val/test only
    according to ``cfg.fraud_val_fraction``. Returns (idx_train, idx_val, idx_test)."""
    raise NotImplementedError("Phase 1")


def fit_scaler(X_train: np.ndarray):
    """Fit a ``StandardScaler`` on the training (legit-only) matrix."""
    raise NotImplementedError("Phase 1")


def prepare(cfg: DataConfig, project_root: Path) -> Splits:
    """Run the full Phase 1 pipeline and write the processed artifacts."""
    raise NotImplementedError("Phase 1")


def load_splits(processed_dir: Path) -> Splits:
    """Load ``splits.npz`` + ``feature_names.json`` written by :func:`prepare`."""
    raise NotImplementedError("Phase 1")
