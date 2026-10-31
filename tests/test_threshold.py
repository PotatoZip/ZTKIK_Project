"""Phase 3b contract tests: quantile property, semi-supervised strategies, result bookkeeping,
JSON round-trip."""

import numpy as np
import pytest

from fraud_ae import threshold as th
from fraud_ae.config import ThresholdConfig


def test_quantile_threshold_yields_target_fpr():
    scores = np.random.default_rng(0).exponential(size=100_000)
    t = th.quantile_threshold(scores, target_fpr=0.01)
    assert abs((scores > t).mean() - 0.01) < 1e-3


@pytest.mark.parametrize("target", [0.001, 0.009, 0.05])
def test_quantile_threshold_property_small_samples(target):
    for seed in range(5):
        scores = np.random.default_rng(seed).lognormal(size=5_000)
        t = th.quantile_threshold(scores, target)
        assert (scores > t).mean() <= target + 1e-9  # never above the target on the data used
        assert (scores > t).mean() >= target - 1 / len(scores)


def test_quantile_threshold_rejects_bad_input():
    with pytest.raises(ValueError):
        th.quantile_threshold(np.array([]), 0.01)
    with pytest.raises(ValueError):
        th.quantile_threshold(np.ones(10), 1.5)


def _separable():
    rng = np.random.default_rng(1)
    legit = rng.normal(0, 1, 1000)
    fraud = rng.normal(8, 1, 50)
    return np.r_[legit, fraud], np.r_[np.zeros(1000, int), np.ones(50, int)]


def test_youden_and_max_f1_on_separable_scores():
    scores, y = _separable()
    for fn in (th.youden_threshold, th.max_f1_threshold):
        t = fn(scores, y)
        assert (scores[y == 1] > t).all() and not (scores[y == 0] > t).any()


def test_semi_supervised_strategies_need_both_classes():
    scores = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        th.youden_threshold(scores, np.zeros(10, int))
    with pytest.raises(ValueError):
        th.choose_threshold(scores, np.zeros(10, int), ThresholdConfig(strategy="max_f1"))


def test_choose_threshold_quantile_reports_val_rates():
    scores, y = _separable()
    res = th.choose_threshold(scores, y, ThresholdConfig(strategy="quantile", target_fpr=0.01))
    assert res.strategy == "quantile" and res.score_method == "mse"
    assert res.n_val_legit == 1000 and res.n_val_fraud == 50
    assert res.val_fpr == pytest.approx(0.01, abs=1e-3)
    assert res.val_recall == 1.0
    assert (scores[y == 0] > res.value).sum() == 10


def test_choose_threshold_without_frauds_has_none_recall():
    rng = np.random.default_rng(0)
    scores = rng.exponential(size=500)
    res = th.choose_threshold(scores, np.zeros(500, int), ThresholdConfig(target_fpr=0.01))
    assert res.val_recall is None and res.n_val_fraud == 0
    assert (scores > res.value).sum() == 5


def test_threshold_json_round_trip(tmp_path):
    scores, y = _separable()
    res = th.choose_threshold(scores, y, ThresholdConfig(), score_method="mahalanobis")
    th.save_threshold(res, tmp_path)
    assert th.load_threshold(tmp_path) == res
    with pytest.raises(FileNotFoundError):
        th.load_threshold(tmp_path / "nope")


# --------------------------------------------------------------------------- Phase 8


def test_hour_segmented_threshold_normalises_each_segment_to_the_target():
    from fraud_ae import threshold as th
    from fraud_ae.config import ThresholdConfig

    rng = np.random.default_rng(0)
    n = 20_000
    hours = rng.uniform(0, 24, n)
    night = (hours >= 0) & (hours < 6)
    s_val = np.where(night, rng.gamma(2.0, 2.0, n), rng.gamma(2.0, 1.0, n))  # night scores higher
    y_val = np.zeros(n, int)
    s_test, h_test = rng.gamma(2.0, 1.5, 1000), rng.uniform(0, 24, 1000)
    cfg = ThresholdConfig(strategy="quantile_by_hour", target_fpr=0.01, night_hours=(0.0, 6.0))
    res, sv, st = th.hour_segmented_threshold(s_val, y_val, hours, s_test, h_test, cfg)
    assert res.strategy == "quantile_by_hour" and res.value == 1.0 and res.val_recall is None
    seg = res.segments
    assert seg["night"] > seg["day"] > 0 and seg["night_hours"] == [0.0, 6.0]
    assert seg["val_fpr_night"] == pytest.approx(0.01, abs=0.002)
    assert seg["val_fpr_day"] == pytest.approx(0.01, abs=0.002)
    assert res.val_fpr == pytest.approx(0.01, abs=0.002)
    assert seg["n_val_legit_night"] + seg["n_val_legit_day"] == n
    night_test = th.is_night(h_test, (0.0, 6.0))
    np.testing.assert_allclose(st[night_test], s_test[night_test] / seg["night"])
    np.testing.assert_allclose(st[~night_test], s_test[~night_test] / seg["day"])
    assert seg["test_rows_night"] == int(night_test.sum())
    with pytest.raises(ValueError):
        th.choose_threshold(s_val, y_val, cfg)


def test_threshold_json_round_trip_keeps_segments(tmp_path):
    from fraud_ae import threshold as th

    r = th.ThresholdResult("quantile_by_hour", 0.009, 1.0, 0.009, None, segments={"night": 2.0})
    th.save_threshold(r, tmp_path)
    assert th.load_threshold(tmp_path) == r
    plain = th.ThresholdResult("quantile", 0.009, 1.5, 0.009, 0.8)
    th.save_threshold(plain, tmp_path)
    assert th.load_threshold(tmp_path).segments is None
