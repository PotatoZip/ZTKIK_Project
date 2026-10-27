"""Phase 4 – evaluation metrics, CIs, criterion check and figures."""

import json
import math

import numpy as np
import pytest

from fraud_ae import evaluate as ev
from fraud_ae.config import EvalConfig
from fraud_ae.plots import per_feature_residuals, recall_vs_fpr, roc_pr_curves, threshold_sweep

CFG = EvalConfig(
    fpr_grid=(0.001, 0.01, 0.05), target_recall=0.85, target_fpr=0.01, bootstrap_samples=200
)


def _scores(n_legit=5000, n_fraud=100, shift=4.0, seed=0):
    """Log-normal legit scores and a shifted fraud population – a well-separated toy problem."""
    rng = np.random.default_rng(seed)
    legit = rng.lognormal(0.0, 0.5, n_legit)
    fraud = rng.lognormal(0.0, 0.5, n_fraud) * shift
    y = np.r_[np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]
    return np.r_[legit, fraud], y


# --------------------------------------------------------------------------- primitives


def test_perfect_scores_give_full_recall_at_zero_fpr():
    y = np.r_[np.zeros(1000), np.ones(50)]
    scores = y.astype(float)
    assert ev.recall_at_fpr(scores, y, fpr=0.01) == 1.0
    assert ev.recall_at_fpr(scores, y, fpr=0.0) == 1.0


def test_wilson_interval_contains_point_estimate():
    lo, hi = ev.wilson_interval(42, 50)
    assert lo < 42 / 50 < hi
    assert ev.wilson_interval(0, 10)[0] == 0.0 and ev.wilson_interval(10, 10)[1] == 1.0
    assert all(math.isnan(v) for v in ev.wilson_interval(0, 0))
    with pytest.raises(ValueError):
        ev.wilson_interval(11, 10)


def test_wilson_interval_matches_known_value():
    # k = 199, n = 236 (Phase 3 val recall): Wilson 95 % ≈ [0.791, 0.884]
    lo, hi = ev.wilson_interval(199, 236)
    assert abs(lo - 0.791) < 0.002 and abs(hi - 0.884) < 0.002


def test_threshold_for_fpr_gives_exact_alarm_count():
    rng = np.random.default_rng(1)
    legit = rng.normal(size=10_000)
    for fpr in (0.0, 0.001, 0.01, 0.1):
        t = ev.threshold_for_fpr(legit, fpr)
        assert int((legit > t).sum()) == int(math.floor(fpr * len(legit)))


def test_recall_at_fpr_is_monotone_in_fpr():
    scores, y = _scores(shift=2.0)
    r = [ev.recall_at_fpr(scores, y, f) for f in (0.001, 0.01, 0.05, 0.2)]
    assert r == sorted(r)


def test_confusion_at_threshold_counts_and_rule():
    scores = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    y = np.array([0, 0, 1, 0, 1])
    cm = ev.confusion_at_threshold(scores, y, threshold=1.0)  # alarm is strictly greater
    assert cm == {"tp": 1, "fp": 1, "tn": 2, "fn": 1}
    rates = ev.rates_from_confusion(cm)
    assert rates["recall"] == 0.5 and rates["fpr"] == pytest.approx(1 / 3)
    assert rates["precision"] == 0.5 and rates["f1"] == pytest.approx(0.5)


def test_input_validation():
    with pytest.raises(ValueError):
        ev.confusion_at_threshold(np.zeros(3), np.zeros(2), 0.0)
    with pytest.raises(ValueError):
        ev.confusion_at_threshold(np.zeros(3), np.array([0, 2, 1]), 0.0)
    with pytest.raises(ValueError):
        ev.threshold_for_fpr(np.array([]), 0.01)


# --------------------------------------------------------------------------- evaluate()


def test_evaluate_meets_target_on_separable_scores():
    scores, y = _scores(shift=6.0)
    t = ev.threshold_for_fpr(scores[y == 0], 0.009)
    m = ev.evaluate(scores, y, t, CFG, seed=0)
    assert m["meets_target"] is True
    assert m["criterion"]["meets_fpr"] and m["criterion"]["meets_recall"]
    assert m["at_threshold"]["fpr"] < 0.01 and m["at_threshold"]["recall"] >= 0.85
    assert m["roc_auc"] > 0.99 and m["pr_auc"] > 0.85
    assert [g["fpr_target"] for g in m["recall_at_fpr"]] == list(CFG.fpr_grid)
    for g in m["recall_at_fpr"]:
        assert g["fp"] == math.floor(g["fpr_target"] * m["n_legit"])
    assert m["criterion"]["fp_allowed"] == math.ceil(0.01 * m["n_legit"]) - 1
    assert m["criterion"]["tp_required"] == math.ceil(0.85 * m["n_fraud"])


def test_evaluate_fails_target_when_threshold_too_high_or_too_low():
    scores, y = _scores(shift=6.0)
    legit = scores[y == 0]
    m_low = ev.evaluate(scores, y, ev.threshold_for_fpr(legit, 0.05), CFG)
    assert not m_low["meets_target"] and not m_low["criterion"]["meets_fpr"]
    m_high = ev.evaluate(scores, y, float(scores.max()), CFG)
    assert not m_high["meets_target"] and m_high["at_threshold"]["recall"] == 0.0
    assert m_high["criterion"]["meets_fpr"]


def test_evaluate_confidence_intervals_bracket_point_estimates():
    scores, y = _scores(shift=2.5)
    t = ev.threshold_for_fpr(scores[y == 0], 0.009)
    m = ev.evaluate(scores, y, t, CFG, seed=3)
    at, b = m["at_threshold"], m["bootstrap"]
    assert at["recall_ci_wilson"][0] <= at["recall"] <= at["recall_ci_wilson"][1]
    assert at["fpr_ci_wilson"][0] <= at["fpr"] <= at["fpr_ci_wilson"][1]
    assert b["recall_at_threshold"]["ci_lo"] <= at["recall"] <= b["recall_at_threshold"]["ci_hi"]
    assert b["fpr_at_threshold"]["ci_lo"] <= at["fpr"] <= b["fpr_at_threshold"]["ci_hi"]
    r1 = next(g for g in m["recall_at_fpr"] if g["fpr_target"] == 0.01)["recall"]
    assert b["recall_at_target_fpr"]["ci_lo"] <= r1 <= b["recall_at_target_fpr"]["ci_hi"]
    assert b["samples"] == CFG.bootstrap_samples


def test_evaluate_is_deterministic_for_a_seed():
    scores, y = _scores(shift=2.5)
    a = ev.evaluate(scores, y, 2.0, CFG, seed=7)["bootstrap"]
    b = ev.evaluate(scores, y, 2.0, CFG, seed=7)["bootstrap"]
    c = ev.evaluate(scores, y, 2.0, CFG, seed=8)["bootstrap"]
    assert a == b and a != c


def test_evaluate_without_frauds_reports_nan_and_fails_target():
    scores = np.random.default_rng(0).lognormal(size=500)
    y = np.zeros(500, dtype=int)
    m = ev.evaluate(scores, y, 1.0, CFG)
    assert math.isnan(m["roc_auc"]) and math.isnan(m["at_threshold"]["recall"])
    assert m["meets_target"] is False


def test_metrics_json_round_trip(tmp_path):
    scores, y = _scores()
    m = ev.evaluate(scores, y, 2.0, CFG)
    path = ev.save_metrics(m, tmp_path)
    assert path.name == ev.METRICS_FILE
    loaded = ev.load_metrics(tmp_path)
    assert loaded["meets_target"] == m["meets_target"]
    assert loaded["at_threshold"] == m["at_threshold"]
    json.loads(path.read_text())  # valid JSON
    with pytest.raises(FileNotFoundError):
        ev.load_metrics(tmp_path / "missing")


def test_per_feature_summary_ranks_shifted_feature_first():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(300, 4))
    y = np.r_[np.zeros(250, dtype=int), np.ones(50, dtype=int)]
    R[y == 1, 2] += 5.0
    s = ev.per_feature_summary(R, y, ("a", "b", "c", "d"))
    assert s["top_features"][0] == "c"
    assert s["fraud_msr"][2] > 10 * s["legit_msr"][2]
    with pytest.raises(ValueError):
        ev.per_feature_summary(R, y, ("a", "b"))


# --------------------------------------------------------------------------- trained model


def test_trained_model_meets_target_on_synthetic_split(synthetic_splits, tmp_path):
    """End-to-end on the synthetic frame: train → score → val threshold → test verdict."""
    from fraud_ae.config import Config, ScoreConfig, ThresholdConfig, TrainConfig
    from fraud_ae.scoring import calibrate, residuals, score
    from fraud_ae.threshold import choose_threshold
    from fraud_ae.train import train

    sp = synthetic_splits
    cfg = Config(train=TrainConfig(epochs=30, patience=5, seed=0))
    model = train(cfg, sp.X_train, sp.X_val[sp.y_val == 0], tmp_path / "run")
    calib = calibrate(model, sp.X_val[sp.y_val == 0])
    s_val = score(model, sp.X_val, ScoreConfig(), calib)
    s_test = score(model, sp.X_test, ScoreConfig(), calib)
    th = choose_threshold(s_val, sp.y_val, ThresholdConfig(target_fpr=0.009))
    m = ev.evaluate(s_test, sp.y_test, th.value, CFG, seed=0)
    assert m["n_fraud"] == 20 and m["n_legit"] == 400
    assert m["roc_auc"] > 0.95
    assert m["at_threshold"]["recall"] >= 0.85
    assert m["at_threshold"]["fpr"] <= 0.0125  # 5 of 400 (granularity 0.25 %)
    summ = ev.per_feature_summary(residuals(model, sp.X_test), sp.y_test, sp.feature_names)
    assert set(summ["top_features"]) <= set(sp.feature_names)
    # the synthetic frauds are shifted on V1..V14, so those should dominate the ranking
    assert sum(f in {f"V{i}" for i in range(1, 15)} for f in summ["top_features"]) >= 3


# --------------------------------------------------------------------------- figures


def test_phase4_figures_are_written(tmp_path):
    scores, y = _scores(n_legit=2000, n_fraud=40)
    R = np.random.default_rng(0).normal(size=(len(y), 3))
    paths = [
        roc_pr_curves(scores, y, tmp_path / "roc_pr.png"),
        recall_vs_fpr(scores, y, (0.01, 0.85), tmp_path / "rvf.png", threshold=2.0),
        threshold_sweep(scores, y, 2.0, tmp_path / "sweep.png"),
        per_feature_residuals(R, y, ("a", "b", "c"), tmp_path / "pf.png"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
