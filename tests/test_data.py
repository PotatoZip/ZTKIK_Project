"""Phase 1 contract tests. They are expected to fail until data.py is implemented."""

import numpy as np
import pytest

from fraud_ae import data
from fraud_ae.config import DataConfig

phase1 = pytest.mark.xfail(raises=NotImplementedError, reason="Phase 1 pending", strict=True)


@phase1
def test_drop_duplicates_removes_exact_copies(synthetic_df):
    df, n = data.drop_duplicates(synthetic_df)
    assert n == 10 and len(df) == len(synthetic_df) - 10


@phase1
def test_engineer_features_columns(synthetic_df):
    X = data.engineer_features(synthetic_df, DataConfig())
    assert "Class" not in X.columns and "Time" not in X.columns
    assert {"hour_sin", "hour_cos"} <= set(X.columns)
    assert X.shape[1] == DataConfig().n_features


@phase1
def test_split_train_has_no_fraud_and_sets_are_disjoint(synthetic_df):
    y = synthetic_df["Class"].to_numpy()
    tr, va, te = data.split_indices(y, DataConfig())
    assert y[tr].sum() == 0
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(y)


@phase1
def test_scaler_uses_train_statistics_only(synthetic_df):
    X = np.random.default_rng(0).normal(5, 2, size=(100, 4))
    scaler = data.fit_scaler(X)
    assert np.allclose(scaler.mean_, X.mean(axis=0))
