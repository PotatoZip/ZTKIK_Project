"""Phase 10 – rolling threshold, variance groups, evasion probe, temporal frame, CLI."""

import json
import shutil

import numpy as np
import pytest

from fraud_ae import robustness as rb
from fraud_ae.cli import main
from fraud_ae.config import Config, DataConfig, TrainConfig
from fraud_ae.data import load_report, prepare
from fraud_ae.synthetic import make_synthetic_frame


def test_window_ids_equal_legit_counts_and_frauds_follow_time():
    y = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    ids = rb.window_ids(y, 3)
    assert ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert np.bincount(ids[y == 0]).tolist() == [3, 3, 3]
    assert np.all(np.diff(ids) >= 0)
    with pytest.raises(ValueError):
        rb.window_ids(np.zeros(2, int), 3)
    with pytest.raises(ValueError):
        rb.window_ids(np.zeros(10, int), 1)
    assert rb.window_ids(np.array([1, 1, 0, 0, 0, 0]), 2).tolist() == [0, 0, 0, 0, 1, 1]


def _windows(rng, n_windows, scale_from, factor, n_legit=20_000, n_fraud=20):
    """Time-ordered scores: every window holds a permutation of the same legit score set (so the
    0.9 % quantile is identical), scaled by ``factor`` from window ``scale_from`` on; frauds
    score far above everything."""
    base = rng.gamma(2.0, 1.0, n_legit)
    scores, y = [], []
    for k in range(n_windows):
        s = rng.permutation(base) * (factor if k >= scale_from else 1.0)
        scores += [s, np.full(n_fraud, 50.0)]
        y += [np.zeros(n_legit, int), np.ones(n_fraud, int)]
    return np.concatenate(scores), np.concatenate(y)


def test_rolling_threshold_holds_without_drift_and_recovers_after_a_step():
    rng = np.random.default_rng(0)
    flat = rb.rolling_evaluation(*_windows(rng, 4, scale_from=99, factor=1.0), 4, 0.009)
    for name in ("fixed", "rolling_legit"):
        assert flat["summary"][name]["all_in_band"], name
        assert flat["summary"][name]["fpr_min"] == pytest.approx(0.009, abs=1e-4)
        assert flat["summary"][name]["pooled_recall"] == 1.0
    # label-free calibration counts the frauds among the top rows → FPR below the legit target
    assert flat["summary"]["rolling_all"]["fpr_mean"] < 0.009
    # a 1.6× step at the start of window 2: the fixed threshold stays wrong, the rolling one
    # is wrong for one window and exact again on window 3
    step = rb.rolling_evaluation(*_windows(rng, 4, scale_from=2, factor=1.6), 4, 0.009)
    w = {r["window"]: r for r in step["windows"]}
    assert w[2]["fixed"]["fpr"] > 0.02 and w[3]["fixed"]["fpr"] > 0.02
    assert w[2]["rolling_legit"]["fpr"] > 0.02
    assert w[3]["rolling_legit"]["fpr"] == pytest.approx(0.009, abs=1e-4)
    s = step["summary"]
    assert s["fixed"]["windows_in_band"] == 1 and s["rolling_legit"]["windows_in_band"] == 2
    assert s["fixed"]["all_in_band"] is False and len(step["windows"]) == 3
    assert s["fixed"]["tp"] == sum(r["fixed"]["tp"] for r in step["windows"]) == 60
    assert s["fixed"]["n_fraud"] == 60 and len(s["fixed"]["pooled_recall_ci_wilson"]) == 2


def _fake_run(d, recall, fpr):
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(
        json.dumps({"at_threshold": {"recall": recall, "fpr": fpr, "tp": int(recall * 473)}})
    )


def test_variance_groups_and_summary(tmp_path):
    runs = tmp_path / "runs"
    _fake_run(runs / "sup_hgb", 0.89, 0.009)
    _fake_run(runs / "sup_hgb_seed0", 0.88, 0.0091)
    _fake_run(runs / "sup_hgb_seed1", 0.90, 0.0089)
    _fake_run(runs / "sup_hgb_dseed0", 0.87, 0.0095)
    _fake_run(runs / "sup_hgb_under", 0.5, 0.009)  # another family, must not be picked up
    _fake_run(runs / "sup_hgb_seed7_synthetic", 0.1, 0.01)  # synthetic twin, different suffix
    g = rb.variance_groups(runs, "sup_hgb")
    assert [r["seed"] for r in g["model_seed"]] == ["base", 0, 1]
    assert [r["seed"] for r in g["data_seed"]] == ["base", 0]
    s = rb.variance_summary(g["model_seed"])
    assert s["n"] == 3 and s["recall_mean"] == pytest.approx(0.89) and s["all_meet_recall"]
    assert not rb.variance_summary(g["model_seed"], target_recall=0.895)["all_meet_recall"]
    assert s["recall_std"] == pytest.approx(0.01) and s["fpr_max"] == 0.0091
    assert rb.variance_summary([])["n"] == 0
    g_syn = rb.variance_groups(runs, "sup_hgb", suffix="_synthetic")
    assert [r["seed"] for r in g_syn["model_seed"]] == [7] and g_syn["data_seed"] == []


@pytest.fixture(scope="module")
def pooled_ae(tmp_path_factory):
    from fraud_ae.train import train

    root = tmp_path_factory.mktemp("rob")
    raw = root / "data" / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True)
    make_synthetic_frame(n_legit=2000, n_fraud=40, n_duplicates=10, seed=0).to_csv(raw, index=False)
    splits, _ = prepare(DataConfig(zip_path="", fraud_val_fraction=0.0), root)
    ae = train(Config(train=TrainConfig(epochs=20)), splits.X_train, splits.X_val, root / "ae")
    return splits, ae


def test_evasion_probe_is_monotone_in_the_budget(pooled_ae):
    import torch

    from fraud_ae.train import mse_per_sample

    splits, ae = pooled_ae
    s = mse_per_sample(ae, torch.as_tensor(splits.X_test)).numpy()
    thr = float(np.quantile(s[splits.y_test == 0], 0.99))
    detected = (splits.y_test == 1) & (s > thr)
    assert detected.sum() > 10
    ev = rb.evasion_probe(
        ae,
        splits.X_test[detected],
        thr,
        splits.feature_names,
        budgets=(0.1, 0.5, 2.0, 8.0),
        steps=60,
    )
    frac = [b["evaded_fraction"] for b in ev["per_budget"]]
    assert frac == sorted(frac) and frac[-1] > frac[0] and ev["n_detected"] == int(detected.sum())
    assert ev["per_budget"][0]["budget"] == 0.1 and ev["fraud_l2_norm_median"] > 0
    if ev["median_minimal_budget"] is not None:
        assert ev["median_minimal_budget"] in (0.1, 0.5, 2.0, 8.0)
    if ev["movement"]:
        assert (
            len(ev["movement"]["features"]) <= 8 and sum(ev["movement"]["share_of_l2"]) <= 1 + 1e-9
        )
    for key in ("q25_minimal_budget", "median_minimal_budget", "q75_minimal_budget"):
        assert ev[key] is None or np.isfinite(ev[key])
    # the model is left untouched and trainable again after the probe
    assert np.allclose(mse_per_sample(ae, torch.as_tensor(splits.X_test)).numpy(), s)
    assert all(p.requires_grad for p in ae.parameters())
    # a probe where nothing evades reports n/a instead of None
    none = rb.evasion_probe(ae, splits.X_test[detected][:5], 1e-9, splits.feature_names, (0.1,), 2)
    assert none["median_minimal_budget"] is None and none["never_evaded"] == 5
    md = rb.robustness_markdown(
        {
            "rolling": {"n_windows": 2, "band": [0.008, 0.01], "target_fpr": 0.009, "models": {}},
            "variance": {},
            "evasion": none,
        }
    )
    assert "n/a (fewer than half" in md
    with pytest.raises(ValueError):
        rb.evasion_probe(ae, splits.X_test[:0], thr, splits.feature_names)


def test_temporal_frame_recovers_train_window_frauds(tmp_path):
    raw = tmp_path / "data" / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True)
    make_synthetic_frame(n_legit=2000, n_fraud=40, n_duplicates=10, seed=0).to_csv(raw, index=False)
    cfg = DataConfig(zip_path="", split="temporal", processed_dir="data/processed_temporal")
    splits, report = prepare(cfg, tmp_path)
    fr = rb.temporal_frame(tmp_path / cfg.processed_dir, tmp_path)
    assert len(fr.X_train_fraud) == report["rows"]["frauds_dropped_train_window"] > 0
    assert fr.X_train_legit.shape == splits.X_train.shape
    assert len(fr.y_eval) == len(splits.y_val) + len(splits.y_test)
    assert fr.y_eval.sum() == splits.y_val.sum() + splits.y_test.sum()
    assert np.all(np.diff(fr.time_eval) >= 0)
    assert fr.X_eval.shape[1] == len(fr.feature_names) == 31
    np.testing.assert_allclose(
        np.sort(fr.X_eval[:, 0]), np.sort(np.r_[splits.X_val[:, 0], splits.X_test[:, 0]]), atol=1e-6
    )
    with pytest.raises(ValueError):
        prepare(DataConfig(zip_path="", processed_dir="data/processed_random"), tmp_path)
        rb.temporal_frame(tmp_path / "data" / "processed_random", tmp_path)
    assert load_report(tmp_path / cfg.processed_dir)["split_strategy"] == "temporal"


def test_robustness_cli_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    for name in ("v2_pooled.toml", "v2_supervised.toml", "ablation_temporal.toml"):
        shutil.copy(project_root / "configs" / name, tmp_path / "configs" / name)
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "all",
                "--config",
                "configs/ablation_temporal.toml",
                "--run",
                "runs/ae_temporal",
                "--synthetic",
            ]
        )
        == 0
    )
    assert (
        main(
            ["all", "--config", "configs/v2_pooled.toml", "--run", "runs/ae_pooled", "--synthetic"]
        )
        == 0
    )
    sup = [
        "supervised",
        "--config",
        "configs/v2_supervised.toml",
        "--synthetic",
        "--methods",
        "hgb",
        "--budgets",
    ]
    assert main(sup) == 0
    assert main([*sup, "--seed", "1", "--suffix", "_seed1"]) == 0
    snap = json.loads(
        (tmp_path / "runs" / "sup_hgb_seed1_synthetic" / "train_summary.json").read_text()
    )
    assert snap["seed"] == 1
    capsys.readouterr()
    rc = main(
        [
            "robustness",
            "--config",
            "configs/v2_supervised.toml",
            "--synthetic",
            "--variance-runs",
            "sup_hgb",
            "--windows",
            "3",
            "--evasion-steps",
            "5",
            "--out",
            "results/rob.md",
            "--figures-dir",
            "figs",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert (
        "rolling ae:" in out and "rolling hgb:" in out and "variance sup_hgb model_seed: n=2" in out
    )
    md = (tmp_path / "results" / "rob.md").read_text()
    assert "## 1. Rolling threshold" in md and "| `sup_hgb` | model seed | 2 |" in md
    res = json.loads((tmp_path / "results" / "rob.json").read_text())
    assert (
        set(res["rolling"]["models"]) == {"ae", "hgb", "neg_ae"}
        and len(res["rolling"]["models"]["ae"]["windows"]) == 2
    )
    assert res["evasion"]["n_detected"] > 0
    for f in ("v2_rolling_fpr.png", "v2_evasion.png"):
        assert (tmp_path / "figs" / f).exists()
