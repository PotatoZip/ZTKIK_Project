"""Phase 6 – report rendering from artifacts (traceability of every number)."""

import json
from pathlib import Path

import pytest

from fraud_ae import report as rp
from fraud_ae.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store_root(tmp_path):
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "at_threshold": {"recall": 0.7721518, "fp": 541, "recall_ci_wilson": [0.71, 0.82]},
                "meets_target": False,
                "score_method": "mse",
                "recall_at_fpr": [{"recall": 0.47}, {"recall": 0.77}],
            }
        )
    )
    (run / "threshold.json").write_text(json.dumps({"val_recall": 0.8432}))
    for i, r in enumerate((0.76, 0.77, 0.78)):
        d = tmp_path / "runs" / f"s{i}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"at_threshold": {"recall": r}}))
    return tmp_path


def test_refs_and_formats(store_root):
    r = lambda t: rp.render(t, store_root)  # noqa: E731
    assert r("{{ ${runs/r:at_threshold.recall} | pct1 }}") == "77.2 %"
    assert r("{{ ${runs/r:at_threshold.recall} | pct3 }}") == "77.215 %"
    assert r("{{ ${runs/r:at_threshold.fp} | int }}") == "541"
    assert r("{{ ${runs/r:at_threshold.fp} * 100 | int }}") == "54 100"
    assert r("{{ ${runs/r:at_threshold.recall_ci_wilson.1} | pct0 }}") == "82 %"
    assert r("{{ ${runs/r:recall_at_fpr.1.recall} | f2 }}") == "0.77"
    assert r("{{ ${runs/r:meets_target} | verdict }}") == "not met"
    assert r("{{ ${runs/r:score_method} }}") == "mse"
    assert r("{{ ${runs/r/threshold.json:val_recall} | pct1 }}") == "84.3 %"
    assert r("{{ ${runs/r:at_threshold.recall} }}") == "0.772152"


def test_expressions(store_root):
    r = lambda t: rp.render(t, store_root)  # noqa: E731
    assert (
        r("{{ ${runs/r/threshold.json:val_recall} - ${runs/r:at_threshold.recall} | pp1 }}")
        == "7.1 pp"
    )
    assert (
        r(
            "{{ mean(${runs/s0:at_threshold.recall}, ${runs/s1:at_threshold.recall}, "
            "${runs/s2:at_threshold.recall}) | pct1 }}"
        )
        == "77.0 %"
    )
    assert (
        r(
            "{{ std(${runs/s0:at_threshold.recall}, ${runs/s1:at_threshold.recall}, "
            "${runs/s2:at_threshold.recall}) | pct1 }}"
        )
        == "1.0 %"
    )
    assert (
        r("{{ max(${runs/s0:at_threshold.recall}, ${runs/s2:at_threshold.recall}) | pct0 }}")
        == "78 %"
    )
    assert r("{{ (1 - ${runs/r:at_threshold.recall}) * 237 | f0 }}") == "54"
    assert r("plain text {{ ${runs/r:at_threshold.fp} }} stays") == "plain text 541 stays"


def test_errors_are_collected(store_root):
    with pytest.raises(rp.ReportError) as exc:
        rp.render(
            "{{ ${runs/missing:x} }} {{ ${runs/r:nope.key} }} {{ 1 + 1 }} "
            "{{ ${runs/r:at_threshold.fp} | bogus }} {{ ${runs/r:score_method} | pct1 }}",
            store_root,
        )
    msg = str(exc.value)
    for frag in (
        "not found",
        "key 'nope.key'",
        "unknown format",
        "no artifact reference",
        "needs a number",
    ):
        assert frag in msg
    with pytest.raises(rp.ReportError):
        rp.render("{{ __import__('os') + ${runs/r:at_threshold.fp} }}", store_root)


def test_render_report_writes_file(store_root):
    (store_root / "report").mkdir()
    (store_root / "report" / "t.md").write_text("recall {{ ${runs/r:at_threshold.recall} | pct1 }}")
    out = rp.render_report(store_root, "report/t.md", "report/out.md")
    assert out.read_text() == "recall 77.2 %"
    with pytest.raises(FileNotFoundError):
        rp.render_report(store_root, "report/none.md")


def test_cli_report_fails_cleanly_without_runs(tmp_path, monkeypatch, capsys):
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "report_template.md").write_text("{{ ${runs/ae_base:roc_auc} | f3 }}")
    monkeypatch.chdir(tmp_path)
    assert main(["report"]) == 1
    assert "runs/ae_base" in capsys.readouterr().err
    assert not (tmp_path / "report" / "report.md").exists()


# --------------------------------------------------------------------------- the real template


def test_template_references_only_tracked_artifact_locations():
    tpl = (PROJECT_ROOT / rp.DEFAULT_TEMPLATE).read_text()
    refs = rp.references(tpl)
    assert len(refs) > 50
    for path, keys in refs:
        assert path.startswith(("runs/", "data/processed")), path
        assert keys and " " not in keys, keys
    # every placeholder parses (format is a known one)
    for body in rp.PLACEHOLDER.findall(tpl):
        fmt = body.rsplit("|", 1)[1].strip() if "|" in body else None
        assert fmt is None or fmt in ("str", "verdict", "int") or fmt[:-1] in ("pct", "pp", "f"), (
            body
        )


@pytest.mark.skipif(
    not (PROJECT_ROOT / "runs" / "ae_base" / "metrics.json").exists(),
    reason="real run artifacts not present",
)
def test_tracked_report_matches_artifacts():
    """The committed report.md must be exactly what the artifacts render to."""
    rendered = rp.render((PROJECT_ROOT / rp.DEFAULT_TEMPLATE).read_text(), PROJECT_ROOT)
    assert "{{" not in rendered
    assert rendered == (PROJECT_ROOT / rp.DEFAULT_OUTPUT).read_text()
