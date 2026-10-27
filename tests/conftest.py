"""Shared fixtures. Tests never touch the real 150 MB CSV: a tiny synthetic frame with the same
31 columns (``fraud_ae.synthetic``) stands in for it so the suite runs in seconds anywhere."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud_ae.config import DataConfig
from fraud_ae.data import Splits, prepare
from fraud_ae.synthetic import make_synthetic_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def base_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "base.toml"


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """2 000 legit + 40 fraud rows shaped like creditcard.csv, plus 10 exact duplicates."""
    return make_synthetic_frame(n_legit=2000, n_fraud=40, n_duplicates=10, seed=0)


@pytest.fixture(scope="session")
def synthetic_splits(tmp_path_factory) -> Splits:
    """Phase 1 output (scaled splits) for the synthetic frame, built once per session."""
    root = tmp_path_factory.mktemp("proj")
    raw = root / "data" / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True)
    make_synthetic_frame(n_legit=2000, n_fraud=40, n_duplicates=10, seed=0).to_csv(raw, index=False)
    splits, _ = prepare(DataConfig(zip_path=""), root)
    return splits
