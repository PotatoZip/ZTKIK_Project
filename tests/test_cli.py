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
    monkeypatch.chdir(project_root)
    assert main(["threshold", "--run", "runs/none"]) == 2
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
