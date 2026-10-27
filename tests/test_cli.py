import shutil

import pytest

from fraud_ae.cli import build_parser, main


def test_help_lists_all_phases(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("prepare", "train", "threshold", "evaluate", "baselines", "all"):
        assert cmd in out


def test_unimplemented_phase_exits_with_code_2(capsys, monkeypatch, project_root):
    import fraud_ae.cli as cli

    def stub(cfg):
        raise NotImplementedError("Phase 5")

    monkeypatch.setattr(cli, "cmd_baselines", stub)
    monkeypatch.chdir(project_root)
    assert main(["baselines", "--config", "configs/base.toml"]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_prepare_synthetic_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/base.toml", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "| train |" in out and "sha256(" in out
    processed = tmp_path / "data" / "processed_synthetic"
    assert (processed / "splits.npz").exists()
    assert (processed / "data_report.json").exists()
    assert (processed / "figures" / "eda_class_ratio.png").exists()
    assert not (tmp_path / "data" / "processed").exists()  # real artifacts untouched


def test_prepare_missing_data_exits_with_code_1(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/base.toml"]) == 1
    assert "dataset.zip" in capsys.readouterr().err


def test_train_synthetic_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/base.toml", "--synthetic"]) == 0
    rc = main(
        [
            "train",
            "--config",
            "configs/base.toml",
            "--run",
            "runs/t",
            "--synthetic",
            "--epochs",
            "2",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "epochs: 2/2" in out
    run = tmp_path / "runs" / "t"
    assert (run / "model.pt").exists() and (run / "history.csv").exists()
    assert (run / "figures" / "training_curves.png").exists()
    # --epochs changed the config, so the snapshot is the effective config, not a verbatim copy
    from fraud_ae.config import load_config

    snap = load_config(run / "config.toml")
    assert snap.train.epochs == 2 and "overrides applied" in (run / "config.toml").read_text()
    assert snap.model == load_config(tmp_path / "configs" / "base.toml").model


def test_train_without_prepare_exits_with_code_1(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["train", "--config", "configs/base.toml", "--run", "runs/t"]) == 1
    assert "prepare" in capsys.readouterr().err


def test_threshold_synthetic_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/base.toml", "--synthetic"]) == 0
    assert (
        main(
            [
                "train",
                "--config",
                "configs/base.toml",
                "--run",
                "runs/t",
                "--synthetic",
                "--epochs",
                "3",
            ]
        )
        == 0
    )
    assert main(["threshold", "--run", "runs/t", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "strategy: quantile" in out and "val FPR" in out
    run = tmp_path / "runs" / "t"
    for name in ("scores_val.npz", "scores_test.npz", "threshold.json", "score_calibration.npz"):
        assert (run / name).exists()
    assert (run / "figures" / "score_histogram_val.png").exists()
    # overrides are honoured and recorded
    assert (
        main(
            [
                "threshold",
                "--run",
                "runs/t",
                "--synthetic",
                "--method",
                "mahalanobis",
                "--strategy",
                "max_f1",
            ]
        )
        == 0
    )
    import json

    t = json.loads((run / "threshold.json").read_text())
    assert t["score_method"] == "mahalanobis" and t["strategy"] == "max_f1"


def test_threshold_before_train_exits_with_code_1(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["threshold", "--run", "runs/none"]) == 1
    assert "config.toml" in capsys.readouterr().err


def test_evaluate_synthetic_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/base.toml", "--synthetic"]) == 0
    assert (
        main(
            [
                "train",
                "--config",
                "configs/base.toml",
                "--run",
                "runs/t",
                "--synthetic",
                "--epochs",
                "20",
            ]
        )
        == 0
    )
    assert main(["threshold", "--run", "runs/t", "--synthetic"]) == 0
    assert main(["evaluate", "--run", "runs/t", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "ROC-AUC" in out and "criterion" in out and "test set: 400 legit / 20 fraud" in out
    run = tmp_path / "runs" / "t"
    import json

    m = json.loads((run / "metrics.json").read_text())
    assert "meets_target" in m and m["score_method"] == "mse" and "per_feature" in m
    assert m["bootstrap"]["samples"] == 1000
    for name in (
        "roc_pr_test.png",
        "recall_vs_fpr_test.png",
        "threshold_sweep_test.png",
        "score_histogram_test.png",
        "per_feature_residuals_test.png",
    ):
        assert (run / "figures" / name).exists()


def test_evaluate_before_threshold_exits_with_code_1(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/base.toml", "--synthetic"]) == 0
    rc = main(
        [
            "train",
            "--config",
            "configs/base.toml",
            "--run",
            "runs/t",
            "--synthetic",
            "--epochs",
            "1",
        ]
    )
    assert rc == 0
    assert main(["evaluate", "--run", "runs/t", "--synthetic"]) == 1
    assert "threshold" in capsys.readouterr().err


def test_all_synthetic_runs_four_phases(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["all", "--config", "configs/base.toml", "--run", "runs/t", "--synthetic"]) == 0
    assert (tmp_path / "runs" / "t" / "metrics.json").exists()


def test_train_rejects_splits_with_wrong_feature_count(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    cfg = (
        (tmp_path / "configs" / "base.toml")
        .read_text()
        .replace('time_feature = "sincos"', 'time_feature = "drop"')
    )
    (tmp_path / "configs" / "no_time.toml").write_text(cfg)
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/base.toml", "--synthetic"]) == 0  # 31 features
    rc = main(["train", "--config", "configs/no_time.toml", "--run", "runs/t", "--synthetic"])
    assert rc == 1
    assert "29" in capsys.readouterr().err
