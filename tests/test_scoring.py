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


# --------------------------------------------------------------------------- Phase 8 scores


def test_gmm_nll_matches_sklearn():
    from sklearn.mixture import GaussianMixture

    Z = np.random.default_rng(0).normal(size=(500, 3))
    Z[:250] += 4
    gmm = GaussianMixture(2, covariance_type="full", random_state=0).fit(Z)
    ours = sc.gmm_nll(Z, gmm.weights_, gmm.means_, gmm.covariances_)
    np.testing.assert_allclose(ours, -gmm.score_samples(Z), rtol=1e-8, atol=1e-8)
    w, m, c = sc.fit_latent_gmm(Z, 2, seed=0)
    assert w.shape == (2,) and m.shape == (2, 3) and c.shape == (2, 3, 3)


def test_latent_calibration_scores_and_round_trip(tmp_path, model, X):
    cfg = ScoreConfig(method="latent_gmm", gmm_components=2)
    plain = sc.calibrate(model, X)
    assert plain.gmm_weights is None
    with pytest.raises(ValueError):
        sc.score(model, X, cfg, plain)
    calib = sc.calibrate(model, X, score_cfg=cfg, seed=0)
    assert calib.gmm_weights is not None and calib.gmm_means.shape == (2, 5)
    s = sc.score(model, X, cfg, calib)
    assert s.shape == (300,) and abs(s.mean()) < 1e-6  # sum of two standardised terms
    nll = sc.score(model, X, ScoreConfig(method="latent_nll", gmm_components=2), calib)
    np.testing.assert_allclose(
        nll, sc.gmm_nll(sc.codes(model, X), calib.gmm_weights, calib.gmm_means, calib.gmm_covs)
    )
    assert abs(((nll - calib.latent_nll_stats[0]) / calib.latent_nll_stats[1]).mean()) < 1e-6
    sc.save_calibration(calib, tmp_path)
    back = sc.load_calibration(tmp_path)
    for k, v in calib.__dict__.items():
        np.testing.assert_array_equal(getattr(back, k), v)
    np.testing.assert_array_equal(sc.score(model, X, cfg, back), s)


def test_recon_prob_requires_vae_and_is_seeded(model, X):
    with pytest.raises(ValueError):
        sc.score(model, X, ScoreConfig(method="recon_prob"))
    torch.manual_seed(0)
    vae = build_model(31, ModelConfig(kind="vae"))
    cfg = ScoreConfig(method="recon_prob", n_samples=3)
    a, b, c = (sc.score(vae, X, cfg, seed=s) for s in (1, 1, 2))
    assert a.shape == (300,) and np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.isfinite(a).all()
    # residual-based methods still work on the VAE (deterministic reconstruction)
    assert sc.score(vae, X, ScoreConfig(method="mse")).shape == (300,)
    assert sc.codes(vae, X).shape == (300, 5)


def test_trained_vae_separates_synthetic_frauds(tmp_path, synthetic_splits):
    from sklearn.metrics import roc_auc_score

    from fraud_ae.config import Config, TrainConfig
    from fraud_ae.train import train

    sp = synthetic_splits
    cfg = Config(model=ModelConfig(kind="vae"), train=TrainConfig(epochs=25, patience=10))
    vae = train(cfg, sp.X_train, sp.X_val[sp.y_val == 0], tmp_path / "vae")
    s = sc.score(vae, sp.X_test, ScoreConfig(method="recon_prob", n_samples=5), seed=0)
    assert roc_auc_score(sp.y_test, s) > 0.9
    import json

    summary = json.loads((tmp_path / "vae" / "train_summary.json").read_text())
    assert summary["model_kind"] == "vae" and summary["objective"] == "neg_elbo"
    from fraud_ae.model import VariationalAutoencoder
    from fraud_ae.train import load_model

    assert isinstance(load_model(tmp_path / "vae"), VariationalAutoencoder)
