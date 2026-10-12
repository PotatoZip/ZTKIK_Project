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
    assert main(["prepare", "--config", "configs/base.toml"]) == 2
    assert "not implemented" in capsys.readouterr().err
