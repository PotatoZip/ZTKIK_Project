"""Phase 9 – out-of-fold protocol over the frauds, supervised / hybrid / fine-tuned AE methods."""

import json
import shutil

import numpy as np
import pytest

from fraud_ae import supervised as sp
from fraud_ae.cli import main
from fraud_ae.config import Config, ConfigError, DataConfig, SupervisedConfig, TrainConfig
from fraud_ae.data import prepare
from fraud_ae.synthetic import make_synthetic_frame

CFG = Config(
    supervised=SupervisedConfig(n_folds=4, hgb_max_iter=30, mlp_epochs=15, finetune_epochs=2),
    train=TrainConfig(epochs=3),
)


@pytest.fixture(scope="module")
def pooled(tmp_path_factory):
    """Pooled synthetic split (all 40 frauds in test) and a tiny legit-only AE trained on it."""
    from fraud_ae.train import train

    root = tmp_path_factory.mktemp("pooled")
    raw = root / "data" / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True)
    make_synthetic_frame(n_legit=2000, n_fraud=40, n_duplicates=10, seed=0).to_csv(raw, index=False)
    splits, _ = prepare(DataConfig(zip_path="", fraud_val_fraction=0.0), root)
    ae = train(CFG, splits.X_train, splits.X_val[splits.y_val == 0], root / "ae")
    return splits, ae


def test_folds_cover_frauds_once_and_are_balanced(pooled):
    splits, _ = pooled
    f = sp.make_folds(splits, 5, seed=1)
    assert len(f.fraud_rows) == 40 and np.array_equal(f.fraud_rows, np.sort(f.fraud_rows))
    assert np.bincount(f.fraud_fold, minlength=5).tolist() == [8] * 5
    assert len(f.val_fold) == len(splits.y_val) and len(f.test_fold) == len(splits.y_test)
    fraud_pos = np.flatnonzero(splits.y_test == 1)
    assert np.array_equal(f.test_fold[fraud_pos], f.fraud_fold)
    assert sp.make_folds(splits, 5, seed=1).to_dict() == f.to_dict()
    assert not np.array_equal(sp.make_folds(splits, 5, seed=2).fraud_fold, f.fraud_fold)
    with pytest.raises(ValueError):
        sp.make_folds(splits, 50, seed=1)


def test_no_fraud_is_trained_on_and_evaluated_in_the_same_fold(pooled):
    splits, _ = pooled
    f = sp.make_folds(splits, 5, seed=42)
    fraud_pos = np.flatnonzero(splits.y_test == 1)
    seen = set()
    for k in range(5):
        train = set(fraud_pos[f.fraud_fold != k].tolist())
        held = set(fraud_pos[f.fraud_fold == k].tolist())
        assert not (train & held) and len(train) + len(held) == 40
        assert not (held & seen)
        seen |= held
    assert seen == set(fraud_pos.tolist())


def test_fraud_folds_are_identical_across_legit_splits(tmp_path):
    """The fraud assignment depends only on the fraud rows, not on the data seed."""
    raw = tmp_path / "data" / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True)
    make_synthetic_frame(n_legit=2000, n_fraud=40, n_duplicates=10, seed=0).to_csv(raw, index=False)
    folds = []
    for s in (1, 2):
        splits, _ = prepare(DataConfig(zip_path="", fraud_val_fraction=0.0, seed=s), tmp_path)
        folds.append(sp.make_folds(splits, 5, seed=42))
    assert np.array_equal(folds[0].fraud_rows, folds[1].fraud_rows)
    assert np.array_equal(folds[0].fraud_fold, folds[1].fraud_fold)


def test_hybrid_features_and_classifier_methods(pooled):
    splits, ae = pooled
    H = sp.hybrid_features(ae, splits.X_test[:10])
    assert H.shape == (10, 2 * splits.X_test.shape[1] + 1)
    np.testing.assert_allclose(H[:, -1], (H[:, 31:62] ** 2).mean(axis=1))
    Xl, Xf = splits.X_train[:400], splits.X_test[splits.y_test == 1][:20]
    for method in ("logreg", "hgb", "hgb_under", "mlp", "hybrid_logreg", "hybrid_hgb"):
        (s,), info = sp.fit_score(method, CFG.supervised, Xl, Xf, [splits.X_test], seed=0, ae=ae)
        assert s.shape == (len(splits.X_test),) and (s > 0).all() and (s <= 1).all(), method
        assert np.median(s[splits.y_test == 1]) > np.median(s[splits.y_test == 0]), method
    assert info["n_features"] == 63
    with pytest.raises(ValueError):
        sp.fit_score("hybrid_hgb", CFG.supervised, Xl, Xf, [splits.X_test], 0, ae=None)
    with pytest.raises(ValueError):
        sp.fit_score("svm", CFG.supervised, Xl, Xf, [splits.X_test], 0)


def test_finetuned_autoencoders(pooled):
    from fraud_ae import semi

    splits, ae = pooled
    Xl, Xf = splits.X_train[:600], splits.X_test[splits.y_test == 1][:16]
    for mode in ("margin", "sad"):
        model, info = semi.finetune(ae, CFG.supervised, Xl, Xf, seed=0, mode=mode)
        assert info["mode"] == mode and info["margin"] > 0 and info["centre"].shape == (5,)
        s = semi.semi_scores(model, splits.X_test, info)
        assert s.shape == (len(splits.X_test),) and (s >= 0).all()
        assert np.median(s[splits.y_test == 1]) > np.median(s[splits.y_test == 0]), mode
        assert info["diverged"] is False
    # the original AE is untouched by fine-tuning
    (s_ae,), _ = sp.fit_score("neg_ae", CFG.supervised, Xl, Xf, [Xl[:5]], 0, ae=ae)
    assert not np.array_equal(s_ae, semi.semi_scores(ae, Xl[:5], {"mode": "margin"}))
    with pytest.raises(ValueError):
        semi.finetune(ae, CFG.supervised, Xl, Xf[:0], 0)
    with pytest.raises(ValueError):
        semi.finetune(ae, CFG.supervised, Xl, Xf, 0, mode="other")


def test_run_oof_scores_every_row_once_with_per_fold_thresholds(pooled):
    splits, ae = pooled
    folds = sp.make_folds(splits, 4, seed=0)
    res = sp.run_oof("logreg", CFG, splits, folds, budget=None, ae=None)
    assert (
        res.scores_val.shape == splits.y_val.shape and res.scores_test.shape == splits.y_test.shape
    )
    assert np.isfinite(res.scores_val).all() and np.isfinite(res.scores_test).all()
    assert len(res.thresholds) == 4 and all(t > 0 for t in res.thresholds)
    assert res.labels_per_fold == [30] * 4
    assert all(f <= 0.0125 for f in res.val_fpr_per_fold)  # 0.9 % quantile on 400 legit rows
    legit_val = res.scores_val[splits.y_val == 0]
    assert abs((legit_val > 1.0).mean() - 0.01) < 0.01
    small = sp.run_oof("logreg", CFG, splits, folds, budget=5)
    assert small.labels_per_fold == [5] * 4
    same = sp.run_oof("logreg", CFG, splits, folds, budget=5)
    np.testing.assert_array_equal(small.scores_test, same.scores_test)  # seeded sub-sampling


def test_run_supervised_writes_contract_and_few_label(pooled, tmp_path):
    splits, ae = pooled
    out = sp.run_supervised(
        CFG, splits, tmp_path / "runs", ("hgb_under", "neg_ae"), (5, 30, 99), ae
    )
    assert set(out) == {"sup_hgb_under", "sup_neg_ae"}
    run = tmp_path / "runs" / "sup_neg_ae"
    for f in (
        "config.toml",
        "folds.json",
        "scores_val.npz",
        "scores_test.npz",
        "threshold.json",
        "metrics.json",
        "train_summary.json",
        "few_label.json",
    ):
        assert (run / f).exists(), f
    assert not (run / "model.pt").exists()
    fl = json.loads((run / "few_label.json").read_text())
    assert [b["budget"] for b in fl["budgets"]] == [5, "all"]  # 30 and 99 ≥ available → skipped
    assert fl["budgets"][0]["labels_per_fold"] == [5] * 4
    m = out["sup_neg_ae"]
    assert m["threshold_strategy"] == "quantile_oof" and m["protocol"]["ae_run"] == "runs/ae_pooled"
    assert m["few_label"][-1]["recall"] == m["at_threshold"]["recall"]
    th = json.loads((run / "threshold.json").read_text())
    assert th["value"] == 1.0 and len(th["segments"]["per_fold"]) == 4
    md, rows = sp.few_label_markdown([run, tmp_path / "runs" / "sup_hgb_under"])
    assert "| 5 labels | all labels |" in md and len(rows) == 2
    from fraud_ae.plots import few_label_curve

    ref = {"run": "ae", "recall": 0.8, "recall_ci_wilson": [0.7, 0.9], "fpr": 0.009}
    assert few_label_curve(rows, tmp_path / "fl.png", ref).exists()
    with pytest.raises(FileNotFoundError):
        sp.few_label_markdown([tmp_path / "nothing"])


def test_supervised_config_validation(tmp_path):
    from fraud_ae.config import load_config

    for bad in (
        "[supervised]\nmethods = ['svm']\n",
        "[supervised]\nbudgets = [0]\n",
        "[supervised]\nn_folds = 1\n",
        "[supervised]\nmargin_quantile = 1.5\n",
    ):
        (tmp_path / "b.toml").write_text(bad)
        with pytest.raises(ConfigError):
            load_config(tmp_path / "b.toml")


def test_supervised_cli_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    for name in ("v2_pooled.toml", "v2_supervised.toml"):
        shutil.copy(project_root / "configs" / name, tmp_path / "configs" / name)
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/v2_pooled.toml", "--synthetic"]) == 0
    # hybrid / fine-tuned methods need the legit-only AE on the same split
    assert main(["semi", "--config", "configs/v2_supervised.toml", "--synthetic", "--budgets"]) == 1
    assert "config.toml" in capsys.readouterr().err
    assert (
        main(
            ["all", "--config", "configs/v2_pooled.toml", "--run", "runs/ae_pooled", "--synthetic"]
        )
        == 0
    )
    # … and that AE must come from the very same split (a data-seed split is rejected)
    assert (
        main(["prepare", "--config", "configs/v2_pooled.toml", "--synthetic", "--data-seed", "1"])
        == 0
    )
    rc = main(["semi", "--config", "configs/v2_supervised.toml", "--synthetic", "--data-seed", "1"])
    assert rc == 1 and "was trained on data/processed_pooled, not on" in capsys.readouterr().err
    # `semi` only accepts the fine-tuned AE methods
    with pytest.raises(SystemExit):
        main(["semi", "--config", "configs/v2_supervised.toml", "--methods", "logreg"])
    rc = main(
        [
            "supervised",
            "--config",
            "configs/v2_supervised.toml",
            "--synthetic",
            "--methods",
            "logreg",
            "hybrid_logreg",
            "--budgets",
            "4",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "sup_logreg_synthetic: out-of-fold recall" in out and "4→" in out and "all→" in out
    runs = tmp_path / "runs"
    for name in ("sup_logreg_synthetic", "sup_hybrid_logreg_synthetic"):
        assert (runs / name / "few_label.json").exists() and (runs / name / "folds.json").exists()
    rc = main(
        [
            "few-label",
            "--runs",
            "runs/sup_logreg_synthetic",
            "runs/sup_hybrid_logreg_synthetic",
            "--out",
            "results/fl.md",
            "--figure",
            "figs/fl.png",
            "--reference",
            "runs/ae_pooled",
        ]
    )
    assert rc == 0
    md = (tmp_path / "results" / "fl.md").read_text()
    assert "| 4 labels | all labels |" in md and "Legit-only reference `ae_pooled`" in md
    assert (tmp_path / "figs" / "fl.png").exists()
    capsys.readouterr()
    rc = main(
        [
            "compare",
            "--runs",
            "runs/sup_logreg_synthetic",
            "runs/ae_pooled",
            "--out",
            "results/c.md",
        ]
    )
    assert rc == 0
    assert "LogReg (balanced)" in capsys.readouterr().out
    assert "logreg (oof)" in (tmp_path / "results" / "c.md").read_text()
