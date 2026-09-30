"""Least-squares fitting for the linear signal model y = a*x + b."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    scale: float
    offset: float
    r2: float
    rmse: float
    mae: float
    raw_variance: float
    target_variance: float


def fit_linear(x: ArrayLike, y: ArrayLike) -> RegressionMetrics | None:
    """Fit ``theta = argmin ||X theta - y||²`` for ``X = [x, 1]``.

    For this one-predictor problem the centered least-squares formula is both
    stable and much faster than invoking a general matrix decomposition for
    every candidate. No matrix inverse is formed. Constant inputs and targets
    are rejected because they cannot identify a meaningful signal mapping.
    """

    raw = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if raw.ndim != 1 or target.ndim != 1 or raw.size != target.size:
        raise ValueError("x and y must be equal-length one-dimensional arrays")
    if raw.size < 2 or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(target)):
        return None
    raw_variance = float(np.var(raw))
    target_variance = float(np.var(target))
    if raw_variance <= 1e-15 or target_variance <= 1e-15:
        return None
    raw_mean = float(np.mean(raw))
    target_mean = float(np.mean(target))
    raw_centered = raw - raw_mean
    target_centered = target - target_mean
    denominator = float(np.dot(raw_centered, raw_centered))
    if denominator <= 1e-15:
        return None
    scale = float(np.dot(raw_centered, target_centered) / denominator)
    offset = target_mean - scale * raw_mean
    prediction = scale * raw + offset
    errors = target - prediction
    squared_error = float(np.dot(errors, errors))
    centered = target - float(np.mean(target))
    total_error = float(np.dot(centered, centered))
    r2 = 1.0 - squared_error / total_error
    return RegressionMetrics(
        scale=scale,
        offset=offset,
        r2=r2,
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        raw_variance=raw_variance,
        target_variance=target_variance,
    )
