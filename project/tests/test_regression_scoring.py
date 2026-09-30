import numpy as np

from inference.regression import fit_linear
from inference.scoring import score_candidate


def test_regression_recovers_scale_offset_and_metrics():
    x = np.arange(-20, 21, dtype=float)
    y = 0.25 * x - 7.5
    metrics = fit_linear(x, y)
    assert metrics is not None
    assert metrics.scale == pytest.approx(0.25)
    assert metrics.offset == pytest.approx(-7.5)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(0.0, abs=1e-12)
    assert metrics.mae == pytest.approx(0.0, abs=1e-12)


def test_constant_raw_or_target_is_rejected():
    assert fit_linear([1, 1, 1], [1, 2, 3]) is None
    assert fit_linear([1, 2, 3], [7, 7, 7]) is None


def test_low_pair_coverage_is_penalized():
    metrics = fit_linear([1, 2, 3, 4], [3, 5, 7, 9])
    assert metrics is not None
    full_score = score_candidate(metrics, 4, 4)[0]
    partial_score = score_candidate(metrics, 4, 40)[0]
    assert full_score > partial_score


import pytest

