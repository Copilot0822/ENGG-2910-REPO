"""Transparent candidate confidence/ranking calculation."""

from __future__ import annotations

import math

from inference.regression import RegressionMetrics


def score_candidate(
    metrics: RegressionMetrics,
    observation_count: int,
    target_observation_count: int,
) -> tuple[float, float, float]:
    target_std = math.sqrt(metrics.target_variance)
    normalized_rmse = metrics.rmse / max(target_std, 1e-15)
    pairing_fraction = min(1.0, observation_count / target_observation_count)
    # Exact/high-coverage mappings approach 1.05. A coincidental fit based on a
    # small overlap is heavily discounted even if its local R² is high.
    confidence = 0.35 + 0.65 * pairing_fraction
    ranking_score = metrics.r2 * confidence - 0.10 * normalized_rmse + 0.05 * pairing_fraction
    return ranking_score, normalized_rmse, pairing_fraction
