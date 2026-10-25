import numpy as np
import pytest

from fraud_ae import evaluate

phase4 = pytest.mark.xfail(raises=NotImplementedError, reason="Phase 4 pending", strict=True)


@phase4
def test_perfect_scores_give_full_recall_at_zero_fpr():
    y = np.r_[np.zeros(1000), np.ones(50)]
    scores = y.astype(float)
    assert evaluate.recall_at_fpr(scores, y, fpr=0.01) == 1.0


@phase4
def test_wilson_interval_contains_point_estimate():
    lo, hi = evaluate.wilson_interval(42, 50)
    assert lo < 42 / 50 < hi
