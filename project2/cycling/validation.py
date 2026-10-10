"""Strict chronological backtesting, with a separate late calibration set."""

import numpy as np
import pandas as pd

from .features import build_segments
from .geometry import clean_points
from .model import fit_model
from .prediction import predict_route, speed_from_time
from .events import route_events

OSM_COLUMNS = [
    "surface",
    "surface_source",
    "surface_confidence",
    "surface_raw",
    "smoothness",
    "tracktype",
    "highway",
    "bicycle",
    "junction",
    "way_id",
    "map_confidence",
    "map_distance_m",
    "forward",
    "osm_tags",
    "control_kind",
    "control_id",
    "control_uncertain",
]


def held_out_route(points, ride, settings):
    """Only geometry/elevation and reusable OSM enter the test route, never observed ride dynamics."""
    raw = points[["lat", "lon", "elevation"]].copy()
    raw["track_group"] = points.group
    clean, _ = clean_points(raw, settings, route=True)
    for col in OSM_COLUMNS:
        clean[col] = points[col].to_numpy()
    events = route_events(clean)
    neutral = {"id": "route", "date": ride["date"], "bike": ride.get("bike", "unknown")}
    return build_segments(clean, neutral, events, historical=False), events


def chronological_splits(rides, minimum_train=4, max_tests=8):
    ordered = rides.sort_values("date").reset_index(drop=True)
    splits = []
    for _, test in ordered.iterrows():
        # Strict date cutoff also excludes rides at the same timestamp.
        train = ordered[pd.to_datetime(ordered.date, utc=True) < pd.Timestamp(test.date)]
        if len(train) >= minimum_train:
            splits.append((train, test))
    return splits[-max_tests:]


def _fitness_multiplier(previous):
    if not previous:
        return 1.0
    ratios = [row["actual_rolling_kmh"] / row["raw_predicted_kmh"] for row in previous[-4:]]
    # Shrink a sparse recent-fitness correction toward 1; its use is chosen on earlier folds only.
    return float(np.clip(1 + (np.median(ratios) - 1) * len(ratios) / (len(ratios) + 4), 0.9, 1.1))


def _score(model, test, route, events, fitness=False):
    target = test.get("measured_avg_hr")
    if target is not None and pd.isna(target):
        target = None
    result = predict_route(
        model,
        route,
        events,
        target_hr=target,
        bike=test.bike,
        scenario="observed",
        departure=test.date,
        use_fitness=fitness,
    )
    actual_speed = speed_from_time(test.distance_m, test.moving_s)
    baseline_time = float((route.distance_m / (model.baseline_predict(route) / 3.6)).sum())
    gradient_time = float((route.distance_m / (model.baseline_predict(route, True) / 3.6)).sum())
    # Gradient baseline is a cruising model, so it receives the same exclusive interruption ledger.
    gradient_time += sum(e["moving_loss_s"] for e in result["events"])
    known = float(
        route.loc[route.surface_source.eq("known"), "distance_m"].sum() / route.distance_m.sum()
    )
    primary = route.groupby("surface").distance_m.sum().idxmax()
    matched = float(
        route.loc[route.map_confidence.ge(0.25), "distance_m"].sum() / route.distance_m.sum()
    )
    reliable = (
        bool(test.get("timestamp_reliable", False))
        and result["unknown_wait_events"] == 0
        and matched >= 0.7
    )
    return {
        "ride_id": str(test.id),
        "date": str(test.date),
        "distance_km": float(test.distance_m / 1000),
        "elevation_gain_m": test.get("elevation_gain_m"),
        "dominant_surface": primary,
        "known_surface_fraction": known,
        "map_matched_fraction": matched,
        "actual_rolling_kmh": actual_speed,
        "predicted_rolling_kmh": result["rolling_kmh"],
        "speed_absolute_error": abs(result["rolling_kmh"] - actual_speed),
        "actual_moving_s": float(test.moving_s),
        "predicted_moving_s": result["moving_s"],
        "moving_error_s": result["moving_s"] - float(test.moving_s),
        "moving_absolute_percentage_error": abs(result["moving_s"] / test.moving_s - 1) * 100,
        "actual_elapsed_s": float(test.elapsed_s),
        "predicted_elapsed_s": result["elapsed_s"],
        "elapsed_error_s": result["elapsed_s"] - float(test.elapsed_s) if reliable else None,
        "elapsed_reliable": reliable,
        "elapsed_lower_bound_error_s": result["elapsed_s"] - float(test.elapsed_s),
        "median_baseline_error_kmh": abs(
            speed_from_time(route.distance_m.sum(), baseline_time) - actual_speed
        ),
        "gradient_baseline_error_kmh": abs(
            speed_from_time(route.distance_m.sum(), gradient_time) - actual_speed
        ),
        "target_hr_measured": target is not None,
        "train_count": len(model.train_ids),
        "train_latest": model.support["last_ride"],
        "fitness_multiplier": model.fitness_adjustment,
        "warnings": result["warnings"],
    }


def backtest(store, rides, segments, observations, settings, progress=None):
    splits = chronological_splits(rides, max_tests=settings.validation_rides)
    if len(splits) < 3:
        return {
            "status": "insufficient",
            "reason": "At least seven timestamped rides are needed for separate chronological selection and calibration.",
            "selected_half_life": settings.half_life_days,
            "rows": [],
            "calibration_speed_residuals": [],
        }
    tuning_count = max(1, len(splits) // 2)
    tuning, calibration = splits[:tuning_count], splits[tuning_count:]
    candidates = [60, 90, 120, 180, 365] if settings.optimize_half_life else [settings.half_life_days]
    routes = {str(test.id): held_out_route(store.points(test.id), test, settings) for _, test in splits}
    scores, candidate_rows = [], {}
    total_jobs = len(candidates) * len(tuning) + len(calibration)
    completed = 0
    for half_life in candidates:
        rows = []
        for train, test in tuning:
            if progress:
                progress(
                    completed / total_jobs,
                    f"Earlier-only validation: half-life {half_life:g}, ride {completed + 1}/{total_jobs}",
                )
            model = fit_model(train, segments, observations, half_life, reference_date=test.date)
            route, events = routes[str(test.id)]
            row = _score(model, test, route, events)
            row["raw_predicted_kmh"] = row["predicted_rolling_kmh"]
            model.fitness_adjustment = _fitness_multiplier(rows)
            trend = _score(model, test, route, events, fitness=True)
            row["fitness_speed_error"] = trend["speed_absolute_error"]
            row["fitness_prediction_kmh"] = trend["predicted_rolling_kmh"]
            row["fitness_result"] = trend
            rows.append(row)
            completed += 1
        candidate_rows[half_life] = rows
        scores.append(
            {
                "half_life": half_life,
                "speed_mae_kmh": float(np.mean([r["speed_absolute_error"] for r in rows])),
                "fitness_speed_mae_kmh": float(np.mean([r["fitness_speed_error"] for r in rows])),
            }
        )
    selected = min(scores, key=lambda s: s["speed_mae_kmh"])["half_life"]
    selected_scores = next(s for s in scores if s["half_life"] == selected)
    use_fitness = (
        len(tuning) >= 3
        and selected_scores["fitness_speed_mae_kmh"] < selected_scores["speed_mae_kmh"] * 0.97
    )
    rows = candidate_rows[selected]
    for row in rows:
        trend = row.pop("fitness_result")
        if use_fitness:
            row.update(trend)
        row["split"] = "selection"
    residuals = []
    coverage = []
    for train, test in calibration:
        if progress:
            progress(completed / total_jobs, f"Chronological calibration: ride {test.id}")
        model = fit_model(train, segments, observations, selected, reference_date=test.date)
        route, events = routes[str(test.id)]
        raw = _score(model, test, route, events)
        model.fitness_adjustment = _fitness_multiplier(rows) if use_fitness else 1.0
        row = _score(model, test, route, events, fitness=use_fitness)
        row["raw_predicted_kmh"] = raw["predicted_rolling_kmh"]
        row["split"] = "calibration"
        # Coverage uses earlier residuals only; current error never sets its own interval.
        if len(residuals) >= 3:
            rank = min(len(residuals) - 1, int(np.ceil((len(residuals) + 1) * 0.9)) - 1)
            margin = float(np.sort(residuals)[rank])
            covered = row["speed_absolute_error"] <= margin
            row["interval_covered"] = bool(covered)
            coverage.append(covered)
        residuals.append(row["speed_absolute_error"])
        rows.append(row)
        completed += 1
    heldout = [r for r in rows if r["split"] == "calibration"]
    elapsed = [abs(r["elapsed_error_s"]) for r in heldout if r["elapsed_error_s"] is not None]
    frame = pd.DataFrame(heldout)
    frame["distance_band"] = pd.cut(
        frame.distance_km,
        [0, 30, 75, 150, np.inf],
        labels=["<30 km", "30–75 km", "75–150 km", ">150 km"],
    )
    frame["elevation_band"] = pd.cut(
        pd.to_numeric(frame.elevation_gain_m),
        [-1, 300, 1000, np.inf],
        labels=["<300 m", "300–1000 m", ">1000 m"],
    )
    groups = {}
    for dimension in ("distance_band", "dominant_surface", "elevation_band"):
        grouped = frame.groupby(dimension, observed=True).speed_absolute_error.agg(["count", "mean"])
        groups[dimension] = [
            {"group": str(key), "rides": int(value["count"]), "speed_mae_kmh": float(value["mean"])}
            for key, value in grouped.iterrows()
        ]
    metrics = {
        "speed_mae_kmh": float(np.mean([r["speed_absolute_error"] for r in heldout])),
        "moving_mae_minutes": float(np.mean([abs(r["moving_error_s"]) for r in heldout]) / 60),
        "elapsed_mae_minutes": float(np.mean(elapsed) / 60) if elapsed else None,
        "median_absolute_percentage_error": float(
            np.median([r["moving_absolute_percentage_error"] for r in heldout])
        ),
        "interval_coverage": float(np.mean(coverage)) if coverage else None,
        "coverage_rides": len(coverage),
        "calibration_rides": len(heldout),
        "median_baseline_mae_kmh": float(np.mean([r["median_baseline_error_kmh"] for r in heldout])),
        "gradient_baseline_mae_kmh": float(np.mean([r["gradient_baseline_error_kmh"] for r in heldout])),
    }
    if progress:
        progress(1, "Chronological validation complete")
    return {
        "status": "complete",
        "selected_half_life": selected,
        "candidates": scores,
        "fitness_selected": bool(use_fitness),
        "fitness_multiplier": _fitness_multiplier(rows) if use_fitness else 1.0,
        "rows": rows,
        "metrics": metrics,
        "groups": groups,
        "calibration_speed_residuals": residuals,
        "interval_provisional": len(residuals) < 20,
        "elapsed_note": "Elapsed accuracy excludes rides with uncertain recording gaps, insufficient road/control coverage or unlearned waits. Personal breaks in test recordings are not treated as predictable traffic waits.",
    }
