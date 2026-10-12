import numpy as np
import pytest

from fraud_ae import threshold

phase3 = pytest.mark.xfail(raises=NotImplementedError, reason="Phase 3 pending", strict=True)


@phase3
def test_quantile_threshold_yields_target_fpr():
    scores = np.random.default_rng(0).exponential(size=100_000)
    t = threshold.quantile_threshold(scores, target_fpr=0.01)
    assert abs((scores > t).mean() - 0.01) < 1e-3
