"""Phase 3a contract tests for the three residual-based scores, on an untrained and on a trained
autoencoder over the synthetic split."""

import numpy as np
import pytest
import torch

from fraud_ae import scoring as sc
from fraud_ae.config import ModelConfig, ScoreConfig
from fraud_ae.model import build_model
from fraud_ae.train import mse_per_sample


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model(31, ModelConfig())


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(size=(300, 31)).astype(np.float32)


def test_residuals_shape_and_definition(model, X):
    R = sc.residuals(model, X, batch_size=64)
    assert R.shape == (300, 31) and R.dtype == np.float64
    with torch.no_grad():
        expected = (torch.as_tensor(X) - model(torch.as_tensor(X))).numpy()
    np.testing.assert_allclose(R, expected, atol=1e-6)


def test_residuals_restore_training_flag(model, X):
    model.train()
    sc.residuals(model, X)
    assert model.training


def test_mse_score_matches_train_helper(model, X):
    s = sc.score(model, X, ScoreConfig(method="mse"))
    expected = mse_per_sample(model, torch.as_tensor(X)).numpy()
    np.testing.assert_allclose(s, expected, rtol=1e-5)
    assert (s >= 0).all()


def test_calibrated_scores_have_expected_mean_on_calibration_data(model, X):
    calib = sc.calibrate(model, X)
    d = X.shape[1]
    z = sc.score(model, X, ScoreConfig(method="standardized"), calib)
    m = sc.score(model, X, ScoreConfig(method="mahalanobis"), calib)
    # sum of squared z-scores and squared Mahalanobis distance both average to d on the data
    # they were calibrated on (population statistics, tiny ridge)
    assert z.mean() == pytest.approx(d, rel=1e-6)
    assert m.mean() == pytest.approx(d, rel=1e-3)
    assert (z >= 0).all() and (m >= 0).all()


def test_mahalanobis_equals_standardized_for_diagonal_covariance():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(500, 4)) * np.array([1.0, 2.0, 0.5, 3.0])
    calib = sc.ScoreCalibration(
        residual_mean=R.mean(axis=0),
        residual_std=R.std(axis=0),
        residual_cov_inv=np.diag(1.0 / R.var(axis=0)),
    )
    z = sc.score_from_residuals(R, ScoreConfig(method="standardized"), calib)
    m = sc.score_from_residuals(R, ScoreConfig(method="mahalanobis"), calib)
    np.testing.assert_allclose(z, m, rtol=1e-10)


def test_non_plain_methods_require_calibration(model, X):
    for method in ("standardized", "mahalanobis"):
        with pytest.raises(ValueError):
            sc.score(model, X, ScoreConfig(method=method))


def test_calibration_round_trip(tmp_path, model, X):
    calib = sc.calibrate(model, X)
    sc.save_calibration(calib, tmp_path)
    loaded = sc.load_calibration(tmp_path)
    np.testing.assert_array_equal(loaded.residual_mean, calib.residual_mean)
    np.testing.assert_array_equal(loaded.residual_cov_inv, calib.residual_cov_inv)


def test_scores_round_trip(tmp_path):
    s, y = np.array([0.1, 2.0]), np.array([0, 1])
    sc.save_scores(tmp_path, "val", s, y)
    s2, y2 = sc.load_scores(tmp_path, "val")
    np.testing.assert_array_equal(s, s2)
    np.testing.assert_array_equal(y, y2)
    with pytest.raises(FileNotFoundError):
        sc.load_scores(tmp_path, "test")


@pytest.mark.parametrize("method", ["mse", "standardized", "mahalanobis"])
def test_trained_model_separates_synthetic_frauds(method, tmp_path, synthetic_splits):
    """Every score method must rank synthetic frauds above most legit rows after training."""
    from fraud_ae.config import Config, DataConfig, ThresholdConfig, TrainConfig
    from fraud_ae.threshold import choose_threshold
    from fraud_ae.train import train

    cfg = Config(
        data=DataConfig(zip_path=""),
        train=TrainConfig(epochs=30, batch_size=128, patience=10, seed=3, num_threads=2),
    )
    X_val_legit = synthetic_splits.X_val[synthetic_splits.y_val == 0]
    model = train(cfg, synthetic_splits.X_train, X_val_legit, tmp_path / "run")
    calib = sc.calibrate(model, X_val_legit)
    s = sc.score(model, synthetic_splits.X_val, ScoreConfig(method=method), calib)
    y = synthetic_splits.y_val
    assert np.median(s[y == 1]) > np.median(s[y == 0])
    res = choose_threshold(s, y, ThresholdConfig(strategy="quantile", target_fpr=0.05), method)
    assert res.val_fpr <= 0.05 + 1e-9
    assert res.val_recall is not None and res.val_recall >= 0.5
