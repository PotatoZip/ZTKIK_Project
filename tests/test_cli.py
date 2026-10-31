import shutil

import pytest

from fraud_ae.cli import build_parser, main


def test_help_lists_all_phases(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in (
        "prepare",
        "train",
        "threshold",
        "evaluate",
        "baselines",
        "compare",
        "report",
        "all",
    ):
        assert cmd in out


def test_unimplemented_phase_exits_with_code_2(capsys, monkeypatch, project_root):
    import fraud_ae.cli as cli

    def stub(*args):  # every phase is implemented; keep the exit-code path covered
        raise NotImplementedError("Phase 7")

    monkeypatch.setattr(cli, "cmd_compare", stub)
    monkeypatch.chdir(project_root)
    assert main(["compare"]) == 2
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


def test_baselines_and_compare_synthetic_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    (tmp_path / "configs").mkdir()
    shutil.copy(project_root / "configs" / "base.toml", tmp_path / "configs" / "base.toml")
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/base.toml", "--synthetic"]) == 0
    assert main(["baselines", "--config", "configs/base.toml", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "baseline_pca_k5_synthetic: PCA k=5" in out and "baseline_iforest_synthetic" in out
    runs = tmp_path / "runs"
    for name in ("baseline_pca_k5", "baseline_pca_k10", "baseline_pca_k15", "baseline_iforest"):
        assert (runs / f"{name}_synthetic" / "metrics.json").exists()
    # default compare excludes synthetic runs → no metrics → exit 1
    assert main(["compare", "--out", "results/c.md", "--figures-dir", "figs"]) == 1
    assert "metrics.json" in capsys.readouterr().err
    rc = main(
        [
            "compare",
            "--runs",
            *[str(p) for p in sorted(runs.glob("baseline_*"))],
            "--out",
            "results/c.md",
            "--figures-dir",
            "figs",
        ]
    )
    assert rc == 0
    assert "4 runs" in capsys.readouterr().out
    md = (tmp_path / "results" / "c.md").read_text()
    assert md.count("| `baseline_") == 4
    assert (tmp_path / "figs" / "comparison_recall.png").exists()


# --------------------------------------------------------------------------- Phase 7


def _copy_configs(tmp_path, project_root, *names):
    (tmp_path / "configs").mkdir()
    for name in names:
        shutil.copy(project_root / "configs" / name, tmp_path / "configs" / name)


def test_help_lists_diagnose_and_data_seed(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    assert "diagnose" in capsys.readouterr().out
    for cmd in ("prepare", "train", "baselines", "all"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([cmd, "--help"])
        assert "--data-seed" in capsys.readouterr().out


def test_data_seed_is_recorded_and_followed_by_threshold_and_evaluate(
    tmp_path, monkeypatch, project_root
):
    _copy_configs(tmp_path, project_root, "base.toml")
    monkeypatch.chdir(tmp_path)
    assert (
        main(["prepare", "--config", "configs/base.toml", "--synthetic", "--data-seed", "1"]) == 0
    )
    processed = tmp_path / "data" / "processed_dseed1_synthetic"
    assert (processed / "splits.npz").exists() and (processed / "figures").exists()
    assert not (tmp_path / "data" / "processed_synthetic").exists()
    assert not (tmp_path / "report").exists()  # EDA figures of a re-drawn split stay local
    import json

    assert json.loads((processed / "data_report.json").read_text())["config"]["seed"] == 1
    # train without the flag must fail (base processed dir does not exist) …
    assert main(["train", "--config", "configs/base.toml", "--run", "runs/t", "--synthetic"]) == 1
    # … and with it the snapshot records the effective data config
    rc = main(
        [
            "train",
            "--config",
            "configs/base.toml",
            "--run",
            "runs/t",
            "--synthetic",
            "--data-seed",
            "1",
            "--epochs",
            "2",
        ]
    )
    assert rc == 0
    from fraud_ae.config import load_config

    snap = load_config(tmp_path / "runs" / "t" / "config.toml")
    assert snap.data.seed == 1 and snap.data.processed_dir == "data/processed_dseed1"
    assert snap.train.seed == 42
    # threshold / evaluate need no --data-seed: they read the snapshot
    assert main(["threshold", "--run", "runs/t", "--synthetic"]) == 0
    assert main(["evaluate", "--run", "runs/t", "--synthetic"]) == 0
    assert (tmp_path / "runs" / "t" / "metrics.json").exists()


def test_two_data_seeds_give_different_legit_splits_same_frauds(
    tmp_path, monkeypatch, project_root
):
    _copy_configs(tmp_path, project_root, "v2_pooled.toml")
    monkeypatch.chdir(tmp_path)
    import numpy as np

    from fraud_ae.data import load_splits

    splits = {}
    for s in (0, 1):
        assert (
            main(
                [
                    "prepare",
                    "--config",
                    "configs/v2_pooled.toml",
                    "--synthetic",
                    "--data-seed",
                    str(s),
                ]
            )
            == 0
        )
        splits[s] = load_splits(tmp_path / "data" / f"processed_pooled_dseed{s}_synthetic")
    assert not np.array_equal(splits[0].idx_train, splits[1].idx_train)
    for s in (0, 1):
        assert splits[s].y_val.sum() == 0 and splits[s].y_test.sum() == 40
    # the same 40 frauds are test frauds under both seeds (positions in the de-duplicated frame)
    frauds = {s: set(splits[s].idx_test[splits[s].y_test == 1].tolist()) for s in (0, 1)}
    assert frauds[0] == frauds[1]


def test_pooled_protocol_end_to_end_synthetic(tmp_path, monkeypatch, project_root, capsys):
    _copy_configs(tmp_path, project_root, "v2_pooled.toml")
    monkeypatch.chdir(tmp_path)
    assert (
        main(["all", "--config", "configs/v2_pooled.toml", "--run", "runs/p", "--synthetic"]) == 0
    )
    out = capsys.readouterr().out
    assert "| val | 400 | 0 |" in out and "val recall: n/a (0 frauds)" in out
    assert "test set: 400 legit / 40 fraud" in out
    import json

    th = json.loads((tmp_path / "runs" / "p" / "threshold.json").read_text())
    assert th["val_recall"] is None and th["n_val_fraud"] == 0
    assert json.loads((tmp_path / "runs" / "p" / "metrics.json").read_text())["n_fraud"] == 40


def test_baselines_suffix_and_data_seed_name_runs(tmp_path, monkeypatch, project_root, capsys):
    _copy_configs(tmp_path, project_root, "v2_pooled.toml")
    monkeypatch.chdir(tmp_path)
    cfg = ["--config", "configs/v2_pooled.toml", "--synthetic"]
    assert main(["prepare", *cfg, "--data-seed", "2"]) == 0
    assert main(["baselines", *cfg, "--suffix", "_pooled", "--data-seed", "2"]) == 0
    runs = tmp_path / "runs"
    assert (runs / "baseline_pca_k5_pooled_dseed2_synthetic" / "metrics.json").exists()
    assert (runs / "baseline_iforest_pooled_dseed2_synthetic" / "threshold.json").exists()
    assert not (runs / "baseline_pca_k5_synthetic").exists()
    capsys.readouterr()
    rc = main(
        [
            "compare",
            "--runs",
            *[str(p) for p in sorted(runs.glob("baseline_*"))],
            "--out",
            "results/v2.md",
            "--figures-dir",
            "figs",
            "--figure-prefix",
            "v2_",
        ]
    )
    assert rc == 0
    md = (tmp_path / "results" / "v2.md").read_text()
    assert "| Data seed | Test frauds |" in md and "| 42 | 2 | 40 |" in md and "| n/a |" in md
    assert (tmp_path / "figs" / "v2_comparison_recall.png").exists()
    assert not (tmp_path / "figs" / "comparison_recall.png").exists()


# --------------------------------------------------------------------------- Phase 8


def test_hour_segmented_threshold_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    _copy_configs(tmp_path, project_root, "v2_unsup_by_hour.toml")
    monkeypatch.chdir(tmp_path)
    args = ["--config", "configs/v2_unsup_by_hour.toml", "--run", "runs/h", "--synthetic"]
    assert main(["all", *args]) == 0
    out = capsys.readouterr().out
    assert "hour segments: night 0–6 h" in out and "threshold: 1 " in out
    import json

    import numpy as np

    t = json.loads((tmp_path / "runs" / "h" / "threshold.json").read_text())
    assert t["strategy"] == "quantile_by_hour" and t["value"] == 1.0
    assert t["segments"]["n_val_legit_night"] + t["segments"]["n_val_legit_day"] == 400
    with np.load(tmp_path / "runs" / "h" / "scores_val.npz") as z:
        legit = z["scores"][z["y"] == 0]
    assert abs((legit > 1.0).mean() - t["val_fpr"]) < 1e-12  # stored scores are normalised
    m = json.loads((tmp_path / "runs" / "h" / "metrics.json").read_text())
    assert m["threshold_strategy"] == "quantile_by_hour" and m["n_fraud"] == 40
    # the strategy is also available as a threshold override on an existing run
    assert main(["threshold", "--run", "runs/h", "--synthetic", "--strategy", "quantile"]) == 0
    assert json.loads((tmp_path / "runs" / "h" / "threshold.json").read_text())["segments"] is None
    assert (
        main(["threshold", "--run", "runs/h", "--synthetic", "--strategy", "quantile_by_hour"]) == 0
    )
    assert json.loads((tmp_path / "runs" / "h" / "threshold.json").read_text())["value"] == 1.0


def test_vae_and_latent_scores_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    _copy_configs(tmp_path, project_root, "v2_unsup_vae.toml", "v2_unsup_latent_gmm4.toml")
    monkeypatch.chdir(tmp_path)
    import json

    assert main(["prepare", "--config", "configs/v2_unsup_vae.toml", "--synthetic"]) == 0
    for cfg, run in (("v2_unsup_vae", "v"), ("v2_unsup_latent_gmm4", "g")):
        c = ["--config", f"configs/{cfg}.toml", "--run", f"runs/{run}", "--synthetic"]
        assert main(["train", *c, "--epochs", "15"]) == 0
        assert main(["threshold", "--run", f"runs/{run}", "--synthetic"]) == 0
        assert main(["evaluate", "--run", f"runs/{run}", "--synthetic"]) == 0
    sv = json.loads((tmp_path / "runs" / "v" / "train_summary.json").read_text())
    assert sv["model_kind"] == "vae" and sv["objective"] == "neg_elbo"
    mv = json.loads((tmp_path / "runs" / "v" / "metrics.json").read_text())
    assert mv["score_method"] == "recon_prob" and mv["roc_auc"] > 0.9
    mg = json.loads((tmp_path / "runs" / "g" / "metrics.json").read_text())
    assert mg["score_method"] == "latent_gmm"
    import numpy as np

    with np.load(tmp_path / "runs" / "g" / "score_calibration.npz") as z:
        assert z["gmm_weights"].shape == (4,)
    # latent_nll re-uses the same calibration path through a CLI override
    assert main(["threshold", "--run", "runs/g", "--synthetic", "--method", "latent_nll"]) == 0
    assert (
        json.loads((tmp_path / "runs" / "g" / "threshold.json").read_text())["score_method"]
        == "latent_nll"
    )
    capsys.readouterr()
    rc = main(["compare", "--runs", "runs/v", "runs/g", "--out", "results/c.md"])
    assert rc == 0 and "VAE 31-20-10-5" in capsys.readouterr().out
