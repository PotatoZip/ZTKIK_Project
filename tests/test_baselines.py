"""Phase 5 – PCA / Isolation Forest baselines, run-directory contract and the comparison."""

import json

import numpy as np
import pytest

from fraud_ae import baselines as bl
from fraud_ae.config import Config, EvalConfig
from fraud_ae.evaluate import load_metrics
from fraud_ae.threshold import load_threshold

CFG = Config(eval=EvalConfig(bootstrap_samples=50))


def test_pca_error_is_zero_with_all_components_and_grows_with_fewer(synthetic_splits):
    X = synthetic_splits.X_train
    d = X.shape[1]
    full = bl.pca_scores(X, synthetic_splits.X_val, d)
    assert np.allclose(full, 0.0, atol=1e-8)
    e5 = bl.pca_scores(X, synthetic_splits.X_val, 5).mean()
    e15 = bl.pca_scores(X, synthetic_splits.X_val, 15).mean()
    assert e5 > e15 > 0
    with pytest.raises(ValueError):
        bl.pca_scores(X, synthetic_splits.X_val, d + 1)


def test_pca_error_on_train_matches_unexplained_variance(synthetic_splits):
    X = synthetic_splits.X_train
    pca = bl.fit_pca(X, 5)
    err = bl.pca_reconstruction_error(pca, X).mean() * X.shape[1]
    unexplained = pca.explained_variance_.sum() * (1 / pca.explained_variance_ratio_.sum() - 1)
    assert err == pytest.approx(unexplained * (len(X) - 1) / len(X), rel=1e-6)


@pytest.mark.parametrize(
    "fn",
    [
        lambda sp: bl.pca_scores(sp.X_train, sp.X_test, 5),
        lambda sp: bl.iforest_scores(sp.X_train, sp.X_test, seed=0),
    ],
)
def test_baseline_scores_separate_frauds(synthetic_splits, fn):
    from sklearn.metrics import roc_auc_score

    s = fn(synthetic_splits)
    y = synthetic_splits.y_test
    assert s.shape == y.shape
    assert np.median(s[y == 1]) > np.median(s[y == 0])
    assert roc_auc_score(y, s) > 0.95


def test_iforest_is_seeded(synthetic_splits):
    sp = synthetic_splits
    a = bl.iforest_scores(sp.X_train, sp.X_val, seed=1)
    b = bl.iforest_scores(sp.X_train, sp.X_val, seed=1)
    c = bl.iforest_scores(sp.X_train, sp.X_val, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_run_baselines_writes_contract_and_meets_target_on_toy_set(synthetic_splits, tmp_path):
    results = bl.run_baselines(CFG, synthetic_splits, tmp_path / "runs", pca_components=(5, 10))
    assert set(results) == {"baseline_pca_k5", "baseline_pca_k10", "baseline_iforest"}
    for name, m in results.items():
        run = tmp_path / "runs" / name
        for f in (
            "config.toml",
            "train_summary.json",
            "scores_val.npz",
            "scores_test.npz",
            "threshold.json",
            "metrics.json",
        ):
            assert (run / f).exists(), f"{name}/{f}"
        assert not (run / "model.pt").exists()
        assert (run / "figures" / "recall_vs_fpr_test.png").exists()
        th = load_threshold(run)
        assert th.strategy == "quantile" and th.n_val_fraud == 20
        assert load_metrics(run)["score_method"] == th.score_method
        assert m["baseline"]["kind"] in ("pca", "iforest")
        assert m["roc_auc"] > 0.95
        assert m["at_threshold"]["recall"] >= 0.85
    assert results["baseline_pca_k5"]["baseline"]["explained_variance_ratio"] < 1.0
    assert (
        json.loads((tmp_path / "runs" / "baseline_iforest" / "train_summary.json").read_text())[
            "seed"
        ]
        == CFG.train.seed
    )


def test_run_baselines_skips_k_above_feature_count(synthetic_splits, tmp_path):
    results = bl.run_baselines(CFG, synthetic_splits, tmp_path / "runs", pca_components=(5, 99))
    assert set(results) == {"baseline_pca_k5", "baseline_iforest"}


def test_write_comparison_table_and_seed_families(synthetic_splits, tmp_path):
    runs = tmp_path / "runs"
    bl.run_baselines(CFG, synthetic_splits, runs, pca_components=(5,))
    # fake a seed family by copying the iforest run under two seed names
    import shutil

    for s in (0, 1):
        shutil.copytree(runs / "baseline_iforest", runs / f"baseline_iforest_seed{s}")
    rows = bl.write_comparison(
        sorted(runs.iterdir()), tmp_path / "results" / "comparison.md", tmp_path / "figs"
    )
    assert [r["run"] for r in rows][0] in {
        "baseline_pca_k5",
        "baseline_iforest",
        "baseline_iforest_seed0",
        "baseline_iforest_seed1",
    }
    assert rows == sorted(rows, key=lambda r: (-r["recall"], r["run"]))
    md = (tmp_path / "results" / "comparison.md").read_text()
    assert "| `baseline_pca_k5` | PCA k=5 |" in md and "## Seed variance" in md
    assert "`baseline_iforest` | 3 |" in md  # family of three (base + 2 seeds)
    assert (tmp_path / "figs" / "comparison_recall.png").exists()
    assert (tmp_path / "figs" / "seed_variance.png").exists()
    fams = bl.seed_families(rows, min_size=3)
    assert list(fams) == ["baseline_iforest"] and len(fams["baseline_iforest"]["test"]) == 3


def test_write_comparison_without_runs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bl.write_comparison([tmp_path / "nothing"], tmp_path / "c.md")
