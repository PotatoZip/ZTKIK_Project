"""Phase 7 – diagnosis of missed frauds: masks, percentiles, overlap, fusion bounds, CLI."""

import json
import shutil

import numpy as np
import pytest

from fraud_ae import diagnose as dg
from fraud_ae.cli import main
from fraud_ae.evaluate import recall_at_fpr


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000, int), np.ones(50, int)]
    legit = rng.normal(0, 1, 1000)
    fraud = np.r_[rng.normal(5, 1, 30), rng.normal(0, 1, 20)]  # 30 obvious, 20 on the manifold
    return np.r_[legit, fraud], y


def test_missed_masks_partition_the_frauds(toy):
    s, y = toy
    missed, detected = dg.missed_masks(s, y, threshold=2.5)
    assert not (missed & detected).any()
    assert np.array_equal(missed | detected, y == 1)
    assert 15 <= missed.sum() <= 20 and detected.sum() >= 30
    with pytest.raises(ValueError):
        dg.missed_masks(s[:-1], y, 1.0)


def test_legit_percentile_bounds_and_monotone(toy):
    s, y = toy
    p = dg.legit_percentile(s[y == 0], s)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert p[np.argmax(s)] == 1.0
    assert np.median(p[y == 0]) == pytest.approx(0.5, abs=0.01)
    assert np.all(np.diff(p[np.argsort(s)]) >= 0)


def test_group_profile_and_top_features(toy):
    s, y = toy
    hours = np.linspace(0, 24, len(y), endpoint=False)
    amount = np.full(len(y), 10.0)
    prof = dg.group_profile(s, dg.legit_percentile(s[y == 0], s), hours, amount)
    assert prof["n"] == len(y) and sum(prof["hour_counts"]) == len(y)
    assert prof["night_share"] == pytest.approx(0.25, abs=0.01) and prof["amount_median"] == 10.0
    empty = dg.group_profile(s[:0], s[:0], hours[:0], amount[:0])
    assert empty["n"] == 0 and np.isnan(empty["score_median"])

    import pandas as pd

    from fraud_ae.data import V_COLUMNS

    raw = pd.DataFrame(np.random.default_rng(1).normal(size=(len(y), 28)), columns=V_COLUMNS)
    raw["Amount"] = amount
    raw.loc[y == 1, "V14"] -= 8  # the separating feature
    missed, detected = dg.missed_masks(s, y, 2.5)
    tf = dg.top_features(raw, y == 0, detected, missed, k=3)
    assert tf["features"][0] == "V14" and len(tf["features"]) == 3
    assert tf["detected_mean"][0] < -6 and abs(tf["legit_mean"][0]) < 0.5
    assert tf["effect_size_detected"] == sorted(tf["effect_size_detected"], reverse=True)


def test_overlap_matrix():
    a, b = {1, 2, 3}, {3, 4}
    m = dg.overlap_matrix({"a": a, "b": b})
    assert m["runs"] == ["a", "b"] and m["n_missed"] == [3, 2]
    assert m["jaccard"][0][0] == 1.0 and m["jaccard"][1][1] == 1.0
    assert m["jaccard"][0][1] == m["jaccard"][1][0] == pytest.approx(0.25)
    assert dg.jaccard(set(), set()) == 1.0


def test_fusion_bounds_are_upper_bounds(toy):
    s, y = toy
    rng = np.random.default_rng(3)
    other = s + rng.normal(0, 0.5, len(s))
    runs = {"a": s, "b": other}
    ths = {"a": 2.5, "b": 2.0}
    fu = dg.fusion_bounds(runs, ths, y, fpr=0.01)
    single = {n: float(((runs[n] > ths[n]) & (y == 1)).mean() * len(y) / 50) for n in runs}
    single_fpr = {n: float((runs[n][y == 0] > ths[n]).mean()) for n in runs}
    assert fu["union_own"]["recall"] >= max(single.values())
    assert fu["union_own"]["fpr"] >= max(single_fpr.values())
    assert fu["union_oracle"]["recall"] >= max(recall_at_fpr(runs[n], y, 0.01) for n in runs)
    assert fu["union_oracle"]["fpr"] <= 0.02 + 1e-9
    assert fu["rank_fusion"]["fp"] == 10  # exactly floor(0.01 * 1000) legit alarms
    assert 0 <= fu["rank_fusion"]["recall"] <= 1
    # fusing a score with itself reproduces its own oracle recall
    same = dg.fusion_bounds({"a": s, "b": s.copy()}, {"a": 2.5, "b": 2.5}, y, 0.01)
    assert same["rank_fusion"]["recall"] == pytest.approx(recall_at_fpr(s, y, 0.01))
    assert same["union_own"]["recall"] == pytest.approx(single["a"])
    with pytest.raises(ValueError):
        dg.fusion_bounds({"a": s[:-1]}, {"a": 1.0}, y, 0.01)


def _project(tmp_path, project_root):
    (tmp_path / "configs").mkdir()
    for name in ("base.toml", "v2_pooled.toml"):
        shutil.copy(project_root / "configs" / name, tmp_path / "configs" / name)


def test_diagnose_cli_end_to_end(tmp_path, monkeypatch, project_root, capsys):
    _project(tmp_path, project_root)
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "--config", "configs/v2_pooled.toml", "--synthetic"]) == 0
    args = ["--config", "configs/v2_pooled.toml", "--synthetic"]
    assert main(["train", *args, "--run", "runs/ae", "--epochs", "5"]) == 0
    assert main(["threshold", "--run", "runs/ae", "--synthetic"]) == 0
    assert main(["evaluate", "--run", "runs/ae", "--synthetic"]) == 0
    assert main(["baselines", *args, "--suffix", "_pooled"]) == 0
    capsys.readouterr()
    rc = main(
        [
            "diagnose",
            "--run",
            "runs/ae",
            "--against",
            "runs/baseline_pca_k5_pooled_synthetic",
            "runs/baseline_iforest_pooled_synthetic",
            "--synthetic",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "test frauds missed" in out and "bounds over 3 runs" in out
    d = json.loads((tmp_path / "runs" / "ae" / "diagnostics.json").read_text())
    assert d["n_fraud"] == 40 and d["n_missed"] + d["n_detected"] == 40
    assert d["groups"]["legit"]["n"] == 400 and len(d["groups"]["missed"]["hour_counts"]) == 24
    assert len(d["top_features"]["features"]) == dg.TOP_K
    assert [o["run"] for o in d["against"]] == [
        "baseline_pca_k5_pooled_synthetic",
        "baseline_iforest_pooled_synthetic",
    ]
    assert d["overlap"]["runs"][0] == "ae" and d["fusion"]["runs"] == d["overlap"]["runs"]
    assert len(d["missed_rows"]) == d["n_missed"]
    assert (tmp_path / "runs" / "ae" / "figures" / "missed_frauds.png").exists()
    # without --against: no fusion block, still a figure
    assert main(["diagnose", "--run", "runs/ae", "--synthetic"]) == 0
    assert dg.load_diagnostics(tmp_path / "runs" / "ae")["fusion"] is None


def test_diagnose_rejects_runs_from_another_split(tmp_path, monkeypatch, project_root, capsys):
    _project(tmp_path, project_root)
    monkeypatch.chdir(tmp_path)
    for cfg in ("base", "v2_pooled"):
        assert main(["prepare", "--config", f"configs/{cfg}.toml", "--synthetic"]) == 0
    assert main(["baselines", "--config", "configs/base.toml", "--synthetic"]) == 0
    pooled = ["--config", "configs/v2_pooled.toml", "--synthetic", "--suffix", "_p"]
    assert main(["baselines", *pooled]) == 0
    rc = main(
        [
            "diagnose",
            "--run",
            "runs/baseline_pca_k5_p_synthetic",
            "--against",
            "runs/baseline_pca_k5_synthetic",  # scored on the v1 split (20 test frauds)
            "--synthetic",
        ]
    )
    assert rc == 1
    assert "different test split" in capsys.readouterr().err
