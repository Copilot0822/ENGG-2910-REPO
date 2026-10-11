"""Paired prequential elapsed backtests; effort and stop models see only the past."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .accuracy import (
    DURATION_LABELS,
    PRIMARY_VARIANTS,
    duration_group,
    paired_percentage_rows,
    percentage_metrics,
    percentage_rows,
)
from .comparison import comparison_folds
from .daylight import daylight_plan
from .elapsed import ElapsedCalibration
from .hr import recommend_hr, validate_hr, weighted_quantile
from .model import fit_model, recency_weight
from .prediction import predict_route, speed_from_time
from .stopping import StoppingModel, route_features
from .storage import Store
from .validation import held_out_route
from .weather import HistoricalWeather

METHODS = ("context", "distance", "residual")
DISTANCE_LABELS = ["under 50 km", "50–100 km", "100–150 km", "150–200 km", "over 200 km"]


def distance_group(distance):
    return DISTANCE_LABELS[
        4 if distance > 200 else int(np.searchsorted([50, 100, 150], distance, side="right"))
    ]


def _initialize(root, settings, segments, observations, halves, hr_half, method, residual_rows):
    global _context
    _context = (root, settings, segments, observations, halves, hr_half, method, residual_rows)


def _fit_prefix(store, train, test, segments, observations, half, variant):
    # Source hash invalidates this private cache when predictive logic changes.
    code = b"".join(
        (Path(__file__).parent / name).read_bytes()
        for name in ["model.py", "prediction.py", "hr.py", "stopping.py"]
    )
    token = hashlib.sha256(
        code + json.dumps([train.id.astype(str).tolist(), str(test.date), half, variant]).encode()
    ).hexdigest()
    path = store.root / "journey/folds" / token / f"model-{variant}.joblib"
    if path.exists():
        return joblib.load(path), path
    model = fit_model(
        train, segments, observations, half, reference_date=test.date, weather=variant == "CW"
    )
    model.variant = variant
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return model, path


def _row(result, test, key, suggestion, osm_coverage, destination):
    actual = float(test.elapsed_truth_s)
    daylight = daylight_plan(test.date, actual, destination)
    cutoff = pd.Timestamp(daylight["cutoff"]) if daylight["cutoff"] else None
    date = pd.Timestamp(test.date)
    finish = date + pd.Timedelta(seconds=result["elapsed_s"])
    actual_finish = date + pd.Timedelta(seconds=actual)
    hr = test.get("hr_recommendation_value") if test.get("hr_reliable", False) else None
    known = hr is not None and pd.notna(hr)
    row = {
        "id": str(test.id),
        "date": str(test.date),
        "distance_km": float(test.distance_m / 1000),
        "variant": key,
        "elapsed_reliable": bool(test.elapsed_reliable),
        "actual_elapsed_s": actual,
        "predicted_elapsed_s": result["elapsed_s"],
        "elapsed_error_s": result["elapsed_s"] - actual,
        "absolute_elapsed_error_s": abs(result["elapsed_s"] - actual),
        "actual_overall_kmh": speed_from_time(test.distance_m, actual),
        "predicted_overall_kmh": result["overall_kmh"],
        "actual_moving_s": float(test.truth_moving_s),
        "predicted_moving_s": result["moving_s"],
        "actual_rolling_kmh": speed_from_time(test.distance_m, test.truth_moving_s),
        "predicted_rolling_kmh": result["rolling_kmh"],
        "actual_stationary_s": float(test.total_stationary_s),
        "identified_traffic_s": float(test.identified_traffic_s),
        "actual_unattributed_s": float(test.unattributed_stationary_s),
        "predicted_traffic_s": result["stationary_s"],
        "predicted_personal_s": result["personal_break_s"],
        "predicted_stopped_s": result["stationary_s"] + result["personal_break_s"],
        "moving_error_s": result["moving_s"] - test.truth_moving_s,
        "traffic_attribution_error_s": result["stationary_s"] - test.identified_traffic_s,
        "unattributed_error_s": result["personal_break_s"] - test.unattributed_stationary_s,
        "actual_measured_hr": float(hr) if known else None,
        "suggested_hr": suggestion.get("suggested_hr"),
        "hr_error_bpm": suggestion.get("suggested_hr") - float(hr)
        if known and suggestion.get("suggested_hr") is not None
        else None,
        "hr_evidence": suggestion.get("evidence"),
        "hr_training_ids": suggestion.get("ride_ids", []),
        "weather_coverage": result["weather_distance_coverage"],
        "osm_coverage": osm_coverage,
        "actual_before_cutoff": bool(actual_finish <= cutoff) if cutoff is not None else None,
        "predicted_before_cutoff": bool(finish <= cutoff) if cutoff is not None else None,
        "cutoff": cutoff.isoformat() if cutoff is not None else None,
        "unknown_wait_events": result["unknown_wait_events"],
        "warnings": result["warnings"],
    }
    if result.get("baseline_only"):
        for key in [
            "predicted_moving_s",
            "predicted_rolling_kmh",
            "predicted_traffic_s",
            "predicted_personal_s",
            "predicted_stopped_s",
            "moving_error_s",
            "traffic_attribution_error_s",
            "unattributed_error_s",
            "weather_coverage",
            "unknown_wait_events",
        ]:
            row[key] = None
        row["hr_evidence"] = "not used by direct elapsed baseline"
    return row


def _fold(task):
    split, train, test = task
    root, settings, segments, observations, halves, hr_half, method, residual_rows = _context
    store = Store(root)
    points = store.points(test.id)
    route, events = held_out_route(points, test, settings)
    destination = (float(points.lat.iloc[-1]), float(points.lon.iloc[-1]))
    osm = float(points.loc[points.map_confidence.ge(0.5), "ds_m"].sum() / max(1, points.ds_m.sum()))
    features = route_features(route, test.bike)
    provider = HistoricalWeather(store.root, replace(settings, weather_offline=True), details=False)
    catalog = store.root / "weather/stations.json"
    provider.stations = json.loads(catalog.read_text(encoding="utf-8")) if catalog.exists() else []
    rows, paths, base_results = [], {}, {}
    suggestion = recommend_hr(
        train,
        features["distance_km"],
        test.date,
        hr_half,
        features,
        minimum=settings.hr_min_comparable,
        bandwidth=settings.hr_distance_bandwidth,
    )
    for variant in ("C", "CW"):
        model, path = _fit_prefix(store, train, test, segments, observations, halves[variant], variant)
        paths[variant] = str(path)
        result = predict_route(
            model,
            route,
            events,
            suggestion["suggested_hr"],
            test.bike,
            "observed",
            departure=test.date,
            weather_provider=provider if variant == "CW" else None,
            break_mode="minimal",
        )
        base_results[variant] = result
        rows.append(_row(result, test, variant + "-auto", suggestion, osm, destination))
        if variant == "CW":
            methods = ("context", "distance") if split == "selection" else (method,)
            for candidate in methods:
                stops = StoppingModel.fit(
                    train, candidate, settings.stopping_half_life_days, residual_rows
                )
                improved = predict_route(
                    model,
                    route,
                    events,
                    suggestion["suggested_hr"],
                    test.bike,
                    "observed",
                    departure=test.date,
                    weather_provider=provider,
                    stopping_model=stops,
                    break_mode="historical",
                )
                key = f"candidate-{candidate}" if split == "selection" else "CW+stops-auto"
                rows.append(_row(improved, test, key, suggestion, osm, destination))
                if split != "selection" and test.get("hr_reliable", False):
                    known = predict_route(
                        model,
                        route,
                        events,
                        float(test.hr_recommendation_value),
                        test.bike,
                        "observed",
                        departure=test.date,
                        weather_provider=provider,
                        stopping_model=stops,
                        break_mode="historical",
                    )
                    rows.append(_row(known, test, "CW+stops-known", suggestion, osm, destination))
            if test.get("hr_reliable", False):
                known = predict_route(
                    model,
                    route,
                    events,
                    float(test.hr_recommendation_value),
                    test.bike,
                    "observed",
                    departure=test.date,
                    weather_provider=provider,
                    break_mode="minimal",
                )
                rows.append(_row(known, test, "CW-known", suggestion, osm, destination))
    reliable = train[train.elapsed_reliable].copy()
    age = (pd.Timestamp(test.date) - pd.to_datetime(reliable.date, utc=True)).dt.total_seconds() / 86400
    seconds_per_km = reliable.elapsed_truth_s / (reliable.distance_m / 1000)
    baseline_s = features["distance_km"] * weighted_quantile(
        seconds_per_km, recency_weight(age, 180), 0.5
    )
    baseline = {
        **base_results["C"],
        "elapsed_s": baseline_s,
        "overall_kmh": speed_from_time(route.distance_m.sum(), baseline_s),
        "baseline_only": True,
    }
    rows.append(_row(baseline, test, "simple-elapsed", suggestion, osm, destination))
    for row in rows:
        row["split"] = split
    raw = next(r for r in rows if r["variant"] == "CW-auto")
    residual = {
        "id": str(test.id),
        "date": str(test.date),
        "predicted_moving_s": raw["predicted_moving_s"],
        "residual_stop_s": max(
            0, raw["actual_elapsed_s"] - raw["predicted_moving_s"] - raw["predicted_traffic_s"]
        ),
    }
    assert provider.client.requests == 0
    return {"rows": rows, "paths": paths, "residual": residual, "test_id": str(test.id), "split": split}


def _run(
    tasks, store, settings, segments, observations, halves, hr_half, method, residual_rows, progress=None
):
    workers = min(3, max(1, settings.comparison_workers))
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize,
        initargs=(
            str(store.root),
            settings,
            segments,
            observations,
            halves,
            hr_half,
            method,
            residual_rows,
        ),
    ) as executor:
        futures = [executor.submit(_fold, task) for task in tasks]
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if progress:
                progress(
                    (i + 1) / len(tasks),
                    f"Elapsed/HR paired {tasks[0][0]} backtest {i + 1}/{len(tasks)}",
                )
    return results


def elapsed_metrics(rows):
    percentages = percentage_metrics(rows)
    frame = percentage_rows(rows)
    frame = frame[frame.percentage_eligible]
    if frame.empty:
        return {**percentages, "elapsed_mae_minutes": None}

    def numeric(name):
        return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce")

    error = numeric("predicted_elapsed_s") - numeric("actual_elapsed_s")
    actual_daylight = frame.get("actual_before_cutoff", pd.Series(None, index=frame.index))
    predicted_daylight = frame.get("predicted_before_cutoff", pd.Series(None, index=frame.index))
    usable = actual_daylight.notna() & predicted_daylight.notna()
    actual_daylight, predicted_daylight = (
        actual_daylight[usable].astype(bool),
        predicted_daylight[usable].astype(bool),
    )
    moving_errors = (numeric("predicted_moving_s") - numeric("actual_moving_s")).abs().dropna()
    rolling_errors = (numeric("predicted_rolling_kmh") - numeric("actual_rolling_kmh")).abs().dropna()
    result = {
        **percentages,
        "elapsed_mae_minutes": float(error.abs().mean() / 60),
        "median_absolute_elapsed_minutes": float(error.abs().median() / 60),
        "elapsed_bias_minutes": float(error.mean() / 60),
        "overall_speed_mae_kmh": float(
            (numeric("predicted_overall_kmh") - numeric("actual_overall_kmh")).abs().mean()
        )
        if numeric("predicted_overall_kmh").notna().any()
        else None,
        "moving_mae_minutes": float(moving_errors.mean() / 60) if len(moving_errors) else None,
        "rolling_mae_kmh": float(rolling_errors.mean()) if len(rolling_errors) else None,
        "daylight_rides": len(actual_daylight),
        "daylight_accuracy": float((actual_daylight == predicted_daylight).mean())
        if len(actual_daylight)
        else None,
        "optimistic_daylight_errors": int((predicted_daylight & ~actual_daylight).sum())
        if len(actual_daylight)
        else 0,
    }
    for coverage in (80, 90):
        column = f"interval_{coverage}_covered"
        values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        result[f"coverage_{coverage}"] = float(values.mean()) if len(values) else None
        result[f"coverage_{coverage}_rides"] = len(values)
    return result


def select_stopping_method(selection_rows):
    """Only earlier selection folds choose a method, now by per-ride MAPE."""
    candidates = [
        {
            "method": method,
            **elapsed_metrics(
                [
                    row
                    for row in selection_rows
                    if row.get("split") == "selection" and row["variant"] == f"candidate-{method}"
                ]
            ),
        }
        for method in METHODS
    ]
    eligible = [candidate for candidate in candidates if candidate["elapsed_mape"] is not None]
    if not eligible:
        raise ValueError("No valid earlier selection durations for stopping-model selection.")
    return min(eligible, key=lambda candidate: candidate["elapsed_mape"])["method"], candidates


def percentage_report(report):
    """Re-score saved predictions, preserving selection, fits and calibration."""
    result = deepcopy(report)
    if not result.get("rows"):
        return result
    frame, cohort = paired_percentage_rows(result["rows"])
    rows = frame.to_dict("records")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None
    result["rows"] = rows

    def scored(key):
        return [
            row
            for row in rows
            if row["variant"] == key
            and (key not in PRIMARY_VARIANTS or row["paired_comparison_eligible"])
        ]

    keys = list(PRIMARY_VARIANTS) + sorted(set(frame.variant) - set(PRIMARY_VARIANTS))
    result["comparisons"] = [{"variant": key, **elapsed_metrics(scored(key))} for key in keys]
    result["groups"] = {
        key: {
            label: elapsed_metrics(
                [row for row in scored(key) if distance_group(row["distance_km"]) == label]
            )
            for label in DISTANCE_LABELS
        }
        for key in keys
    }
    result["duration_groups"] = {
        key: {
            label: elapsed_metrics(
                [
                    row
                    for row in scored(key)
                    if row["percentage_eligible"] and duration_group(row["actual_elapsed_s"]) == label
                ]
            )
            for label in DURATION_LABELS
        }
        for key in keys
    }
    measured_ids = {
        row["id"] for row in rows if row["variant"] == "CW+stops-known" and row["percentage_eligible"]
    }
    result["measured_hr_comparisons"] = [
        {"variant": key, **elapsed_metrics([row for row in scored(key) if row["id"] in measured_ids])}
        for key in ("CW-auto", "CW-known", "CW+stops-auto", "CW+stops-known")
    ]
    eligible = [r for r in result["comparisons"] if r["variant"] in PRIMARY_VARIANTS and r["rides"]]
    mape_winner = min(eligible, key=lambda r: r["elapsed_mape"])["variant"] if eligible else None
    minute_winner = (
        min(eligible, key=lambda r: r["elapsed_mae_minutes"])["variant"] if eligible else None
    )
    result["percentage_comparison_cohort"] = cohort
    result["primary_accuracy_metric"] = "elapsed_mape"
    result["percentage_ranking"] = {
        "lowest_mape_variant": mape_winner,
        "lowest_minute_mae_variant": minute_winner,
        "winners_differ": mape_winner != minute_winner if eligible else None,
        "note": "Final-set ranking is descriptive, not hyperparameter selection. Existing deployed models and C+Weather default are retained.",
    }
    result.setdefault("stopping_selection_metric", "elapsed_mae_minutes")
    result["percentage_method"] = (
        "One absolute/signed percentage per valid ride, divided by its actual elapsed duration, then averaged. "
        "P90 uses linear empirical quantile interpolation; it is an error statistic, not a prediction interval. "
        "Underprediction magnitudes are positive; their conditional P90 excludes overpredicted/exact rides, "
        "whereas late-arrival P90 over all rides assigns zero to those rides."
    )
    return result


def refresh_percentage_reporting(store):
    report = percentage_report(store.get_state("journey_validation", {}))
    if not report.get("rows"):
        raise ValueError("No saved elapsed validation predictions. Validate planning first.")
    store.set_state("journey_validation", report)
    (store.root / "reports/journey-validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(report["rows"]).to_csv(store.root / "reports/journey-validation.csv", index=False)
    return report


def run_validation(store, settings, progress=None):
    rides, segments = store.rides(), store.segments()
    if "activity_start_utc" in rides:
        rides["date"] = rides.activity_start_utc.fillna(rides.date)
    if "elapsed_reliable" not in rides:
        raise ValueError("Audit activity timing before elapsed validation.")
    observations = store.get_state("observations", [])
    folds = comparison_folds(
        rides,
        final_count=max(12, settings.validation_rides * 2 // 3),
        selection_count=max(6, settings.validation_rides // 4),
    )
    selection, final = (
        [f for f in folds if f[0] == "selection"],
        [f for f in folds if f[0] != "selection"],
    )
    boundary = min(pd.Timestamp(t.date) for _, _, t in final)
    hr_candidates = []
    for half in (60, 120, 180, 365):
        values = validate_hr(rides, half, boundary)
        hr_candidates.append(
            {
                "half_life": half,
                "rides": len(values),
                "mae_bpm": float(np.mean([abs(r["error_bpm"]) for r in values])) if values else None,
            }
        )
    available = [c for c in hr_candidates if c["mae_bpm"] is not None]
    hr_half = (
        min(available, key=lambda r: r["mae_bpm"])["half_life"]
        if available
        else settings.hr_half_life_days
    )
    old = store.get_state("comparison", {})
    comparisons = old.get("comparisons", [])
    halves = {
        v: next(
            (
                c.get("half_life", c.get("selected_half_life", settings.half_life_days))
                for c in comparisons
                if c.get("variant") == v
            ),
            settings.half_life_days,
        )
        for v in ("C", "CW")
    }
    # Actual stored comparison exposes selected half-life under this key.
    for v in halves:
        saved = store.root / "models" / f"model-{v}.joblib"
        if saved.exists():
            halves[v] = joblib.load(saved).half_life
    first = _run(
        selection, store, settings, segments, observations, halves, hr_half, "context", [], progress
    )
    first.sort(
        key=lambda r: next(pd.Timestamp(t.date) for _, _, t in selection if str(t.id) == r["test_id"])
    )
    residual_pool, selection_rows = [], [row for item in first for row in item["rows"]]
    # Residual candidate is evaluated in order: its target uses only already
    # completed out-of-sample predictions, never its own held-out elapsed time.
    for item in first:
        _, train, test = next(f for f in selection if str(f[2].id) == item["test_id"])
        model = joblib.load(item["paths"]["CW"])
        points = store.points(test.id)
        route, events = held_out_route(points, test, settings)
        provider = HistoricalWeather(store.root, replace(settings, weather_offline=True), details=False)
        provider.stations = json.loads(
            (store.root / "weather/stations.json").read_text(encoding="utf-8")
        )
        suggestion = recommend_hr(
            train, route.distance_m.sum() / 1000, test.date, hr_half, route_features(route, test.bike)
        )
        stop = StoppingModel.fit(train, "residual", settings.stopping_half_life_days, residual_pool)
        result = predict_route(
            model,
            route,
            events,
            suggestion["suggested_hr"],
            test.bike,
            "observed",
            departure=test.date,
            weather_provider=provider,
            stopping_model=stop,
            break_mode="historical",
        )
        row = _row(
            result,
            test,
            "candidate-residual",
            suggestion,
            next(r["osm_coverage"] for r in item["rows"]),
            (points.lat.iloc[-1], points.lon.iloc[-1]),
        )
        row["split"] = "selection"
        selection_rows.append(row)
        if test.elapsed_reliable:
            residual_pool.append(item["residual"])
    selected, stopping_candidates = select_stopping_method(selection_rows)
    results = _run(
        final,
        store,
        settings,
        segments,
        observations,
        halves,
        hr_half,
        selected,
        residual_pool,
        progress,
    )
    rows = [row for item in results for row in item["rows"]]
    calibration = {}
    selected_first = []
    for row in selection_rows:
        copy = row.copy()
        if copy["variant"] == f"candidate-{selected}":
            copy["variant"] = "CW+stops-auto"
        if not copy["variant"].startswith("candidate"):
            selected_first.append(copy)
    # Interval evaluation is prequential, with no current/future errors used.
    for row in sorted(selected_first + rows, key=lambda r: (r["date"], r["variant"])):
        key = row["variant"]
        prior = calibration.setdefault(key, [])
        estimate = ElapsedCalibration(prior).estimate(
            row["predicted_elapsed_s"], row["distance_km"], row["date"]
        )
        for coverage in (80, 90):
            limits = estimate["intervals_s"].get(str(coverage))
            row[f"interval_{coverage}_covered"] = (
                bool(limits[0] <= row["actual_elapsed_s"] <= limits[1]) if limits else None
            )
            row[f"interval_{coverage}_lower_s"] = limits[0] if limits else None
            row[f"interval_{coverage}_upper_s"] = limits[1] if limits else None
        if row["elapsed_reliable"]:
            prior.append(
                {k: row[k] for k in ["date", "distance_km", "actual_elapsed_s", "predicted_elapsed_s"]}
            )
    for row in rows:
        if row["variant"] == "CW+stops-auto":
            match = next(
                (r for r in rows if r["id"] == row["id"] and r["variant"] == "CW+stops-known"), None
            )
            row["hr_selection_time_effect_s"] = (
                row["predicted_elapsed_s"] - match["predicted_elapsed_s"] if match else None
            )
    keys = sorted({r["variant"] for r in rows})
    comparisons = [
        {"variant": key, **elapsed_metrics([r for r in rows if r["variant"] == key])} for key in keys
    ]
    groups = {
        key: {
            label: elapsed_metrics(
                [r for r in rows if r["variant"] == key and distance_group(r["distance_km"]) == label]
            )
            for label in DISTANCE_LABELS
        }
        for key in keys
    }
    measured_ids = {r["id"] for r in rows if r["variant"] == "CW+stops-known" and r["elapsed_reliable"]}
    measured_comparisons = [
        {
            "variant": key,
            **elapsed_metrics([r for r in rows if r["variant"] == key and r["id"] in measured_ids]),
        }
        for key in ["CW-auto", "CW-known", "CW+stops-auto", "CW+stops-known"]
    ]
    hr_rows = validate_hr(rides, hr_half)
    # Report post-selection HR errors separately from earlier model selection.
    hr_final = [r for r in hr_rows if pd.Timestamp(r["date"]) >= boundary]

    def hr_metrics(values):
        return {
            "rides": len(values),
            "mae_bpm": float(np.mean([abs(r["error_bpm"]) for r in values])) if values else None,
            "median_absolute_bpm": float(np.median([abs(r["error_bpm"]) for r in values]))
            if values
            else None,
            "global_median_mae_bpm": float(np.mean([abs(r["baseline_error_bpm"]) for r in values]))
            if values
            else None,
        }

    report = {
        "status": "complete",
        "selected_hr_half_life": hr_half,
        "hr_candidates": hr_candidates,
        "hr_metrics": hr_metrics(hr_final),
        "hr_all_prequential_metrics": hr_metrics(hr_rows),
        "hr_groups": {
            label: hr_metrics([r for r in hr_final if distance_group(r["distance_km"]) == label])
            for label in DISTANCE_LABELS
        },
        "hr_rows": hr_rows,
        "selected_stopping_method": selected,
        "stopping_selection_metric": "elapsed_mape",
        "stopping_candidates": stopping_candidates,
        "selection_last": max(str(t.date) for _, _, t in selection),
        "final_first": boundary.isoformat(),
        "selection_rides": len(selection),
        "final_rides": len(final),
        "comparisons": comparisons,
        "measured_hr_comparisons": measured_comparisons,
        "groups": groups,
        "rows": rows,
        "calibration_rows": calibration,
        "residual_rows": residual_pool
        + [
            i["residual"]
            for i in results
            if next(t.elapsed_reliable for _, _, t in final if str(t.id) == i["test_id"])
        ],
        "weather_evaluation": "Retrospective ECCC observations queried at predicted arrivals using only the route and departure; held-out GPS passage timestamps are not prediction inputs. Issued forecast skill remains unvalidated.",
        "probability_note": "Fewer than 60 comparable held-out elapsed errors; before-cutoff probabilities are withheld. Correlated rides, sparse long rides and forecast error limit calibration.",
        "speed_halves": halves,
        "timing_quality": store.get_state("timing_quality"),
    }
    report = percentage_report(report)
    store.set_state("journey_validation", report)
    destination = store.root / "reports/journey-validation.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(store.root / "reports/journey-validation.csv", index=False)
    return report
