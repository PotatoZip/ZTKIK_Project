"""Phase 2 contract tests: early-stopping logic, run-directory artifacts, checkpoint round-trip,
convergence on the synthetic split and bit-for-bit determinism across two seeded runs."""

import dataclasses

import numpy as np
import pytest
import torch

from fraud_ae import train as tr
from fraud_ae.config import Config, DataConfig, ModelConfig, TrainConfig
from fraud_ae.data import prepare


def _cfg(**train_kwargs) -> Config:
    base = TrainConfig(epochs=5, batch_size=128, patience=3, seed=7, num_threads=2)
    return Config(
        data=DataConfig(zip_path=""),
        model=ModelConfig(),
        train=dataclasses.replace(base, **train_kwargs),
    )


@pytest.fixture(scope="module")
def synthetic_splits(tmp_path_factory, synthetic_module_df):
    root = tmp_path_factory.mktemp("proj")
    raw = root / "data" / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True)
    synthetic_module_df.to_csv(raw, index=False)
    splits, _ = prepare(DataConfig(zip_path=""), root)
    return splits


@pytest.fixture(scope="module")
def synthetic_module_df():
    from fraud_ae.synthetic import make_synthetic_frame

    return make_synthetic_frame(n_legit=2000, n_fraud=40, n_duplicates=10, seed=0)


# ----------------------------------------------------------------------------- early stopping


def test_early_stopping_counts_non_improving_epochs():
    es = tr.EarlyStopping(patience=2)
    assert es.step(1.0, 1) and not es.should_stop
    assert es.step(0.5, 2) and es.best_epoch == 2
    assert not es.step(0.6, 3) and not es.should_stop
    assert not es.step(0.55, 4) and es.should_stop
    assert es.best == 0.5 and es.best_epoch == 2


def test_early_stopping_min_delta():
    es = tr.EarlyStopping(patience=1, min_delta=0.1)
    es.step(1.0, 1)
    assert not es.step(0.95, 2)  # improvement below min_delta does not count
    assert es.should_stop


# ----------------------------------------------------------------------------- training


def test_train_writes_run_directory(tmp_path, synthetic_splits):
    run = tmp_path / "runs" / "smoke"
    cfg = _cfg(epochs=4)
    X_val_legit = synthetic_splits.X_val[synthetic_splits.y_val == 0]
    model = tr.train(cfg, synthetic_splits.X_train, X_val_legit, run)

    for name in (tr.CONFIG_FILE, tr.MODEL_FILE, tr.HISTORY_FILE, tr.SUMMARY_FILE):
        assert (run / name).exists()
    assert (run / "figures" / "training_curves.png").exists()

    hist = tr.load_history(run)
    assert [int(h["epoch"]) for h in hist] == [1, 2, 3, 4]
    assert all(np.isfinite(h["val_legit_mse"]) and h["val_legit_mse"] > 0 for h in hist)
    summary = tr.load_summary(run)
    assert summary["epochs_run"] == 4 and not summary["stopped_early"]
    assert summary["n_train"] == 1200 and summary["input_dim"] == 31
    assert 1000 < summary["n_parameters"] < 5000
    assert summary["best_val_legit_mse"] == pytest.approx(min(h["val_legit_mse"] for h in hist))
    assert "[train]" in (run / tr.CONFIG_FILE).read_text()
    assert not model.training  # returned in eval mode


def test_checkpoint_round_trip_and_best_epoch(tmp_path, synthetic_splits):
    run = tmp_path / "runs" / "ckpt"
    X_val_legit = synthetic_splits.X_val[synthetic_splits.y_val == 0]
    trained = tr.train(_cfg(epochs=3), synthetic_splits.X_train, X_val_legit, run)
    loaded = tr.load_model(run)
    x = torch.as_tensor(synthetic_splits.X_test[:16])
    assert torch.equal(trained(x), loaded(x))
    # the checkpoint reproduces the best val-legit MSE recorded in history
    best = tr.load_summary(run)["best_val_legit_mse"]
    val_mse = float(tr.mse_per_sample(loaded, torch.as_tensor(X_val_legit)).mean())
    assert val_mse == pytest.approx(best, rel=1e-5)


def test_val_mse_converges_on_synthetic_data(tmp_path, synthetic_splits):
    run = tmp_path / "runs" / "conv"
    X_val_legit = synthetic_splits.X_val[synthetic_splits.y_val == 0]
    tr.train(_cfg(epochs=30, patience=10), synthetic_splits.X_train, X_val_legit, run)
    hist = tr.load_history(run)
    first, best = hist[0]["val_legit_mse"], min(h["val_legit_mse"] for h in hist)
    assert best < 0.9 * first  # loss moves, not just noise
    assert all(h["train_mse"] > 0 for h in hist)


def test_frauds_reconstruct_worse_than_legit(tmp_path, synthetic_splits):
    """The whole premise of the project on the small synthetic set."""
    run = tmp_path / "runs" / "sep"
    X_val_legit = synthetic_splits.X_val[synthetic_splits.y_val == 0]
    model = tr.train(_cfg(epochs=30, patience=10), synthetic_splits.X_train, X_val_legit, run)
    mse = tr.mse_per_sample(model, torch.as_tensor(synthetic_splits.X_test)).numpy()
    y = synthetic_splits.y_test
    assert np.median(mse[y == 1]) > 2 * np.median(mse[y == 0])


def test_same_seed_gives_identical_history(tmp_path, synthetic_splits):
    X_val_legit = synthetic_splits.X_val[synthetic_splits.y_val == 0]
    runs = []
    for name in ("a", "b"):
        run = tmp_path / "runs" / name
        tr.train(_cfg(epochs=4, seed=11), synthetic_splits.X_train, X_val_legit, run)
        runs.append(run)
    ha, hb = tr.load_history(runs[0]), tr.load_history(runs[1])
    for ra, rb in zip(ha, hb, strict=True):
        assert ra["train_mse"] == rb["train_mse"] and ra["val_legit_mse"] == rb["val_legit_mse"]
    ma, mb = tr.load_model(runs[0]), tr.load_model(runs[1])
    for pa, pb in zip(ma.parameters(), mb.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_different_seed_gives_different_history(tmp_path, synthetic_splits):
    X_val_legit = synthetic_splits.X_val[synthetic_splits.y_val == 0]
    tr.train(_cfg(epochs=2, seed=1), synthetic_splits.X_train, X_val_legit, tmp_path / "s1")
    tr.train(_cfg(epochs=2, seed=2), synthetic_splits.X_train, X_val_legit, tmp_path / "s2")
    h1, h2 = tr.load_history(tmp_path / "s1"), tr.load_history(tmp_path / "s2")
    assert h1[0]["train_mse"] != h2[0]["train_mse"]


def test_early_stopping_triggers(tmp_path, synthetic_splits):
    """With lr = 1e-12 float32 weights cannot change, so val MSE is bit-identical every epoch
    and the loop must stop after exactly ``patience`` non-improving epochs."""
    X_val_legit = synthetic_splits.X_val[synthetic_splits.y_val == 0]
    run = tmp_path / "runs" / "es"
    tr.train(_cfg(epochs=60, patience=3, lr=1e-12), synthetic_splits.X_train, X_val_legit, run)
    s = tr.load_summary(run)
    assert s["stopped_early"] and s["epochs_run"] == 4 and s["best_epoch"] == 1
    hist = tr.load_history(run)
    assert len({h["val_legit_mse"] for h in hist}) == 1


def test_train_rejects_bad_shapes(tmp_path, synthetic_splits):
    with pytest.raises(ValueError):
        tr.train(_cfg(), synthetic_splits.X_train, synthetic_splits.X_val[:, :5], tmp_path / "x")
