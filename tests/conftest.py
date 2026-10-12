"""Shared fixtures. Tests never touch the real 150 MB CSV: a tiny synthetic frame with the same
31 columns stands in for it so the suite runs in seconds anywhere."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def base_config_path(project_root: Path) -> Path:
    return project_root / "configs" / "base.toml"


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """2 000 legit + 40 fraud rows shaped like creditcard.csv, with 10 exact duplicates."""
    rng = np.random.default_rng(0)
    n_legit, n_fraud = 2000, 40
    n = n_legit + n_fraud
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", rng.uniform(0, 2 * 86400, size=n).round())
    df["Amount"] = rng.lognormal(mean=3.0, sigma=1.2, size=n).round(2)
    df["Class"] = np.r_[np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]
    # frauds live further from the origin so a sane detector can find them
    df.loc[df["Class"] == 1, [f"V{i}" for i in range(1, 29)]] += 3.0
    dup = df.iloc[:10]
    return pd.concat([df, dup], ignore_index=True)
