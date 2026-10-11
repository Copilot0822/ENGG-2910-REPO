"""Robust measured-only effort suggestions and explicit editable-input state."""

import numpy as np
import pandas as pd

from .model import recency_weight


def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    if not len(values):
        return np.nan
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - weights / 2) / weights.sum()
    return float(np.interp(q, positions, values))


def measured_history(rides):
    if rides.empty:
        return rides.assign(recommendation_hr=pd.Series(dtype=float))
    value = pd.to_numeric(
        rides.get(
            "hr_recommendation_value", rides.get("measured_avg_hr", pd.Series(np.nan, index=rides.index))
        ),
        errors="coerce",
    )
    reliable = rides.get(
        "hr_reliable",
        rides.get("hr_source", pd.Series("missing", index=rides.index)).isin(["samples", "summary"]),
    )
    reliable = reliable.fillna(False) & ~rides.get(
        "hr_imputed", pd.Series(False, index=rides.index)
    ).fillna(False)
    coverage = rides.get("hr_moving_coverage", pd.Series(1.0, index=rides.index)).fillna(0)
    summary = rides.get("hr_recommendation_source", pd.Series("", index=rides.index)).str.startswith(
        "recorded summary"
    )
    reliable &= (coverage >= 0.7) | summary
    result = rides[reliable & value.between(35, 230) & rides.distance_m.gt(0)].copy()
    result["recommendation_hr"] = value.loc[result.index]
    return result


def recommend_hr(
    rides,
    distance_km,
    reference_date,
    half_life=180,
    features=None,
    hr_range=None,
    minimum=4,
    bandwidth=0.2,
):
    history = measured_history(rides)
    dates = pd.to_datetime(history.date, utc=True)
    reference = pd.Timestamp(reference_date)
    reference = reference.tz_localize("UTC") if reference.tzinfo is None else reference.tz_convert("UTC")
    history = history[dates < reference].copy()
    empty = {
        "suggested_hr": None,
        "comparable_rides": 0,
        "total_measured_rides": len(history),
        "evidence": "unavailable",
        "method": "manual entry required",
        "warnings": [],
        "ride_ids": [],
    }
    if not len(history) or distance_km <= 0:
        empty["warnings"] = ["No reliable earlier measured HR is available; enter a target manually."]
        return empty
    distance = history.distance_m.to_numpy() / 1000
    difference = abs(np.log(distance / distance_km))
    similar = (distance >= distance_km * 0.8) & (distance <= distance_km * 1.2)
    width = np.log1p(bandwidth)
    age = (reference - pd.to_datetime(history.date, utc=True)).dt.total_seconds().to_numpy() / 86400
    recent = recency_weight(age, half_life)
    for _ in range(8):
        weights = np.exp(-0.5 * (difference / width) ** 2) * recent
        effective = weights.sum() ** 2 / max(1e-30, float((weights**2).sum()))
        if effective >= min(minimum, len(history)) or width >= 0.7:
            break
        width *= 1.3
    fallback = difference.min() > np.log(1.8) or len(history) < 3
    if fallback:
        weights = recent.copy()
    # Terrain is a weak modifier only with adequate distance support.
    features = features or {}
    if int(similar.sum()) >= 8 and not fallback:
        if features.get("gain_per_km") is not None and "elevation_gain_m" in history:
            gain = history.elevation_gain_m.fillna(0).to_numpy() / np.maximum(0.1, distance)
            weights *= np.clip(np.exp(-abs(gain - features["gain_per_km"]) / 25), 0.6, 1)
        if features.get("bike") and "bike" in history:
            weights *= np.where(history.bike.eq(features["bike"]), 1, 0.8)
        if features.get("paved_fraction") is not None and "paved_fraction" in history:
            weights *= np.clip(
                1 - abs(history.paved_fraction.fillna(0.5) - features["paved_fraction"]), 0.7, 1
            )
    selected = weights >= weights.max() * 0.05
    values = history.recommendation_hr.to_numpy()
    estimate = weighted_quantile(values, weights, 0.5)
    bounds = [max(35, float(values.min())), min(230, float(values.max()))]
    if hr_range and all(v is not None for v in hr_range):
        bounds = [max(bounds[0], hr_range[0]), min(bounds[1], hr_range[1])]
    if bounds[0] > bounds[1]:
        empty["warnings"] = [
            "Reliable ride HR and the fitted effort model have no common support; enter HR manually."
        ]
        return empty
    estimate = int(round(np.clip(estimate, *bounds)))
    warnings = []
    extrapolation = distance_km > distance.max() * 1.1
    if extrapolation:
        warnings.append("Route exceeds the longest reliable measured-HR ride; pacing is extrapolated.")
    if int(similar.sum()) < minimum:
        warnings.append("Sparse similar-distance HR evidence; distance matching was broadened.")
    effective = weights.sum() ** 2 / max(1e-30, float((weights**2).sum()))
    return {
        "suggested_hr": estimate,
        "comparable_rides": int(selected.sum()),
        "within_20_percent": int(similar.sum()),
        "total_measured_rides": len(history),
        "median_distance_km": weighted_quantile(distance, weights, 0.5),
        "distance_range_km": [float(distance[selected].min()), float(distance[selected].max())],
        "effective_rides": float(effective),
        "half_life_days": half_life,
        "evidence": "moderate"
        if not fallback and not extrapolation and similar.sum() >= minimum and effective >= minimum
        else "low",
        "method": "general measured-HR baseline"
        if fallback
        else "smooth distance and recency weighted median",
        "distance_specific": not fallback,
        "warnings": warnings,
        "ride_ids": history.loc[selected, "id"].astype(str).tolist(),
    }


def update_hr_state(state, route_token, recommendation):
    """A new route updates the suggestion; a manual target persists until reset."""
    state["hr_recommendation"] = recommendation
    if not state.get("hr_manual", False):
        state["target_hr"] = recommendation.get("suggested_hr")
    if state.get("hr_route_token") != route_token:
        state["hr_route_token"] = route_token
        if not state.get("hr_manual", False):
            state["target_hr"] = recommendation.get("suggested_hr")
    elif "target_hr" not in state:
        state["target_hr"] = recommendation.get("suggested_hr")


def mark_manual_hr(state):
    state["hr_manual"] = True


def use_suggested_hr(state):
    state["target_hr"] = state.get("hr_recommendation", {}).get("suggested_hr")
    state["hr_manual"] = False


def validate_hr(rides, half_life=180, before=None):
    history = measured_history(rides).sort_values("date")
    rows = []
    for test in history.itertuples():
        if before is not None and pd.Timestamp(test.date) >= pd.Timestamp(before):
            continue
        earlier = rides[pd.to_datetime(rides.date, utc=True) < pd.Timestamp(test.date)]
        if len(measured_history(earlier)) < 4:
            continue
        suggestion = recommend_hr(earlier, test.distance_m / 1000, test.date, half_life)
        if suggestion["suggested_hr"] is None:
            continue
        baseline = float(measured_history(earlier).recommendation_hr.median())
        rows.append(
            {
                "id": str(test.id),
                "date": test.date,
                "distance_km": test.distance_m / 1000,
                "actual_hr": test.recommendation_hr,
                "suggested_hr": suggestion["suggested_hr"],
                "error_bpm": suggestion["suggested_hr"] - test.recommendation_hr,
                "baseline_error_bpm": baseline - test.recommendation_hr,
                "evidence": suggestion["evidence"],
                "training_ids": suggestion["ride_ids"],
            }
        )
    return rows
