"""Phase 1 contract tests: de-duplication, feature engineering, split invariants (no fraud in
train, disjoint and covering, seeded), scaler fitted on train only, artifact round-trip."""

import dataclasses
import json

import numpy as np
import pytest

from fraud_ae import data
from fraud_ae.config import DataConfig
from fraud_ae.synthetic import make_synthetic_frame

# ----------------------------------------------------------------------------- synthetic data


def test_synthetic_frame_has_raw_schema(synthetic_df):
    assert list(synthetic_df.columns) == data.RAW_COLUMNS
    assert len(synthetic_df) == 2050
    assert synthetic_df["Class"].sum() == 40
    assert synthetic_df.duplicated().sum() == 10


def test_synthetic_frame_is_seeded():
    a = make_synthetic_frame(seed=3)
    b = make_synthetic_frame(seed=3)
    c = make_synthetic_frame(seed=4)
    assert a.equals(b) and not a.equals(c)


# ----------------------------------------------------------------------------- transforms


def test_drop_duplicates_removes_exact_copies(synthetic_df):
    df, n = data.drop_duplicates(synthetic_df)
    assert n == 10 and len(df) == len(synthetic_df) - 10
    assert not df.duplicated().any()
    assert list(df.index) == list(range(len(df)))  # positional index after dedupe


def test_engineer_features_columns(synthetic_df):
    X = data.engineer_features(synthetic_df, DataConfig())
    assert "Class" not in X.columns and "Time" not in X.columns
    assert {"hour_sin", "hour_cos"} <= set(X.columns)
    assert X.shape[1] == DataConfig().n_features == 31
    assert list(X.columns[:28]) == [f"V{i}" for i in range(1, 29)]


def test_engineer_features_drop_time(synthetic_df):
    X = data.engineer_features(synthetic_df, DataConfig(time_feature="drop"))
    assert X.shape[1] == 29 and "hour_sin" not in X.columns


def test_engineer_features_values(synthetic_df):
    X = data.engineer_features(synthetic_df, DataConfig())
    np.testing.assert_allclose(X["Amount"], np.log1p(synthetic_df["Amount"]))
    np.testing.assert_allclose(X["hour_sin"] ** 2 + X["hour_cos"] ** 2, 1.0)
    # Time = 6 h (quarter day) → sin = 1, cos = 0; Time = 1 day → same as Time = 0
    df = synthetic_df.iloc[:2].copy()
    df["Time"] = [6 * 3600.0, 86400.0]
    Xt = data.engineer_features(df, DataConfig())
    np.testing.assert_allclose(
        Xt[["hour_sin", "hour_cos"]].to_numpy(), [[1, 0], [0, 1]], atol=1e-12
    )
    X_raw = data.engineer_features(synthetic_df, DataConfig(log_amount=False))
    np.testing.assert_allclose(X_raw["Amount"], synthetic_df["Amount"])


# ----------------------------------------------------------------------------- split


def test_split_train_has_no_fraud_and_sets_are_disjoint(synthetic_df):
    y = synthetic_df["Class"].to_numpy()
    tr, va, te = data.split_indices(y, DataConfig())
    assert y[tr].sum() == 0
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(y)


def test_split_sizes_follow_config(synthetic_df):
    y = synthetic_df["Class"].to_numpy()
    n_legit, n_fraud = int((y == 0).sum()), int(y.sum())
    tr, va, te = data.split_indices(y, DataConfig(legit_fractions=(0.6, 0.2, 0.2)))
    assert len(tr) == round(0.6 * n_legit)
    assert (y[va] == 0).sum() == round(0.2 * n_legit)
    assert y[va].sum() == y[te].sum() == n_fraud / 2
    # all frauds to test
    _, va0, te0 = data.split_indices(y, DataConfig(fraud_val_fraction=0.0))
    assert y[va0].sum() == 0 and y[te0].sum() == n_fraud


def test_split_is_seeded(synthetic_df):
    y = synthetic_df["Class"].to_numpy()
    a = data.split_indices(y, DataConfig(seed=1))
    b = data.split_indices(y, DataConfig(seed=1))
    c = data.split_indices(y, DataConfig(seed=2))
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))
    assert not np.array_equal(a[0], c[0])
    assert all(np.all(np.diff(x) > 0) for x in a)  # sorted, unique


# ----------------------------------------------------------------------------- scaler


def test_scaler_uses_train_statistics_only():
    X = np.random.default_rng(0).normal(5, 2, size=(100, 4))
    scaler = data.fit_scaler(X)
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    assert np.allclose(scaler.scale_, X.std(axis=0))


# ----------------------------------------------------------------------------- pipeline


@pytest.fixture
def synthetic_project(tmp_path, synthetic_df):
    """A throw-away project root with the synthetic frame written as the raw CSV."""
    raw = tmp_path / "data" / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True)
    synthetic_df.to_csv(raw, index=False)
    return tmp_path


def test_prepare_writes_artifacts_and_invariants(synthetic_project, synthetic_df):
    cfg = DataConfig(zip_path="")
    splits, report = data.prepare(cfg, synthetic_project)
    out = synthetic_project / cfg.processed_dir

    for name in (data.SPLITS_FILE, data.SCALER_FILE, data.FEATURES_FILE, data.REPORT_FILE):
        assert (out / name).exists()
    for fig in report["figures"].values():
        assert (out / "figures" / fig.split("/")[-1]).exists()

    # counts
    assert report["rows"] == {
        "raw": 2050,
        "duplicates_dropped": 10,
        "after_dedupe": 2040,
        "frauds_dropped_train_window": 0,
    }
    s = report["splits"]
    assert s["train"]["fraud"] == 0 and s["train"]["legit"] == 1200
    assert s["val"]["fraud"] == 20 and s["test"]["fraud"] == 20
    assert s["train"]["total"] + s["val"]["total"] + s["test"]["total"] == 2040
    assert report["class_balance"]["fraud_rate"] == pytest.approx(40 / 2040)
    assert len(report["by_hour"]["fraud_rate"]) == 24
    assert report["source"]["sha256"] == data.sha256_of(
        synthetic_project / "data" / "raw" / "creditcard.csv"
    )

    # scaled train has zero mean / unit variance; val/test were transformed, not fitted
    assert splits.X_train.dtype == np.float32
    np.testing.assert_allclose(splits.X_train.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(splits.X_train.std(axis=0), 1.0, atol=1e-4)
    assert abs(splits.X_test.mean()) > 0 or abs(splits.X_val.mean()) > 0
    assert splits.y_val.sum() == 20 and splits.y_test.sum() == 20

    # no exact duplicate feature rows across splits (dedupe happened before the split)
    all_rows = np.vstack([splits.X_train, splits.X_val, splits.X_test])
    assert len(np.unique(all_rows, axis=0)) == len(all_rows)

    # scaler statistics equal train statistics in feature space
    import joblib

    scaler = joblib.load(out / data.SCALER_FILE)
    deduped, _ = data.drop_duplicates(synthetic_df)
    X = data.engineer_features(deduped, cfg).to_numpy()
    np.testing.assert_allclose(scaler.mean_, X[splits.idx_train].mean(axis=0))
    assert json.loads((out / data.FEATURES_FILE).read_text()) == list(splits.feature_names)


def test_prepare_is_deterministic(synthetic_project):
    cfg = DataConfig(zip_path="")
    a, _ = data.prepare(cfg, synthetic_project)
    b, _ = data.prepare(cfg, synthetic_project)
    for f in dataclasses.fields(data.Splits):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        assert np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb


def test_load_splits_round_trip(synthetic_project):
    cfg = DataConfig(zip_path="")
    written, _ = data.prepare(cfg, synthetic_project)
    loaded = data.load_splits(synthetic_project / cfg.processed_dir)
    for f in dataclasses.fields(data.Splits):
        va, vb = getattr(written, f.name), getattr(loaded, f.name)
        assert np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb
    assert loaded.feature_names == written.feature_names


def test_prepare_checks_sha256(synthetic_project):
    with pytest.raises(data.DataError, match="sha256 mismatch"):
        data.prepare(DataConfig(zip_path="", expected_sha256="00" * 32), synthetic_project)


def test_extract_dataset_from_zip(tmp_path, synthetic_df):
    import zipfile

    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("creditcard.csv", synthetic_df.to_csv(index=False))
    raw = tmp_path / "data" / "raw" / "creditcard.csv"
    assert data.extract_dataset(zip_path, raw) == raw
    assert len(data.load_raw(raw)) == len(synthetic_df)
    with pytest.raises(data.DataError):
        data.extract_dataset(tmp_path / "missing.zip", tmp_path / "nope.csv")


def test_load_raw_rejects_wrong_schema(tmp_path, synthetic_df):
    bad = tmp_path / "bad.csv"
    synthetic_df.drop(columns=["V7"]).to_csv(bad, index=False)
    with pytest.raises(data.DataError):
        data.load_raw(bad)


def test_split_table_markdown(synthetic_project):
    _, report = data.prepare(DataConfig(zip_path=""), synthetic_project)
    md = data.split_table_markdown(report)
    assert md.splitlines()[0].startswith("| Split |")
    assert "| train | 1,200 | 0 |" in md


# --------------------------------------------------------------------------- temporal split


def test_temporal_split_is_chronological_and_legit_only_in_train(synthetic_df):
    df, _ = data.drop_duplicates(synthetic_df)
    y = df["Class"].to_numpy()
    t = df["Time"].to_numpy()
    cfg = DataConfig(split="temporal")
    tr, va, te = data.split_indices(y, cfg, time=t)
    assert y[tr].sum() == 0
    assert t[tr].max() <= t[va].min() and t[va].max() <= t[te].min()
    assert len(set(tr) | set(va) | set(te)) == len(tr) + len(va) + len(te)
    n_legit = int((y == 0).sum())
    assert len(tr) == round(0.6 * n_legit)
    assert y[va].sum() + y[te].sum() <= y.sum()  # train-window frauds are dropped
    assert y[va].sum() + y[te].sum() > 0
    with pytest.raises(ValueError):
        data.split_indices(y, cfg)  # needs Time


def test_temporal_split_ignores_seed(synthetic_df):
    df, _ = data.drop_duplicates(synthetic_df)
    y, t = df["Class"].to_numpy(), df["Time"].to_numpy()
    a = data.split_indices(y, DataConfig(split="temporal", seed=1), time=t)
    b = data.split_indices(y, DataConfig(split="temporal", seed=2), time=t)
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))


def test_prepare_temporal_reports_dropped_frauds(synthetic_project):
    root = synthetic_project
    cfg = DataConfig(zip_path="", split="temporal", processed_dir="data/processed_temporal")
    splits, report = data.prepare(cfg, root)
    assert report["split_strategy"] == "temporal"
    rows = report["rows"]
    used = sum(report["splits"][s]["fraud"] for s in ("train", "val", "test"))
    assert rows["frauds_dropped_train_window"] == report["class_balance"]["fraud"] - used
    w = report["time_windows"]
    assert w["train"][1] <= w["val"][0] and w["val"][1] <= w["test"][0]
    assert splits.y_val.sum() == report["splits"]["val"]["fraud"]


# ----------------------------------------------------------------------------- Phase 8


def test_hours_from_features_recovers_hour_of_day(synthetic_df):
    cfg = DataConfig()
    X = data.engineer_features(synthetic_df, cfg)
    scaler = data.fit_scaler(X.to_numpy())
    Xs = scaler.transform(X.to_numpy())
    hours = data.hours_from_features(Xs, tuple(X.columns), scaler)
    expected = data.hour_of_day(synthetic_df["Time"])
    diff = np.abs(hours - expected)
    np.testing.assert_allclose(np.minimum(diff, 24 - diff), 0.0, atol=1e-6)
    with pytest.raises(data.DataError):
        X29 = data.engineer_features(synthetic_df, DataConfig(time_feature="drop"))
        data.hours_from_features(
            X29.to_numpy(), tuple(X29.columns), data.fit_scaler(X29.to_numpy())
        )
