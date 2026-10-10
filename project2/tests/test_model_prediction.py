import numpy as np
import pandas as pd
import pytest

from cycling.events import route_events
from cycling.features import build_segments
from cycling.model import fit_model, recency_weight, training_weights
from cycling.pipeline import prepare_route, process_history, train
from cycling.prediction import predict_route, speed_from_time
from cycling.validation import backtest, chronological_splits, held_out_route
from cycling.importer import import_archive
from conftest import gpx_bytes, strava_zip


def test_recency_half_life_and_ride_normalization():
    assert recency_weight([0, 120, 240]).tolist() == [1, 0.5, 0.25]
    frame = pd.DataFrame(
        {
            "ride_id": ["short"] * 2 + ["long"] * 20,
            "date": ["2025-01-01T00:00:00Z"] * 22,
            "surface": ["paved"] * 22,
        }
    )
    weights = training_weights(frame, "2025-01-01T00:00:00Z", 120)
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())
    with pytest.raises(ValueError):
        recency_weight([10], 0)


def test_adaptive_fatigue_features(synthetic_dataset):
    _, segments, _ = synthetic_dataset
    assert segments.completed_km.max() > 5
    assert segments.gain_m.max() > 0
    assert segments.moving_hours.max() > 0.2
    assert segments.groupby("ride_id").planned_km.nunique().eq(1).all()
    assert segments.distance_m.nunique() > 1


def test_strict_chronological_separation(synthetic_dataset):
    rides, _, _ = synthetic_dataset
    rides.loc[5, "date"] = rides.loc[4, "date"]
    for train_rides, test_ride in chronological_splits(rides):
        assert pd.to_datetime(train_rides.date, utc=True).max() < pd.Timestamp(test_ride.date)
        assert str(test_ride.id) not in set(train_rides.id)


def test_holding_out_route_strips_sensors_and_fatigue(synthetic_dataset, settings):
    rides, _, traces = synthetic_dataset
    route, _ = held_out_route(traces["9"], rides.iloc[-1], settings)
    assert route.moving_hours.eq(0).all()
    assert route.measured_hr.isna().all()
    assert route.power.isna().all()
    assert route.speed_kmh.isna().all()


def test_training_cutoff_excludes_future_hr(synthetic_dataset):
    rides, segments, _ = synthetic_dataset
    initial = rides.iloc[:4].copy()
    initial.loc[initial.index[-1], "measured_avg_hr"] = None
    model = fit_model(initial, segments, [], reference_date=rides.date.iloc[4])
    assert model.train_ids == ["0", "1", "2", "3"]
    assert model.median_hr == pytest.approx(np.median(initial.measured_avg_hr.dropna()))
    assert model.support["imputed_hr_rides"] == 1


def test_no_hr_baseline_without_fabrication(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    rides["measured_avg_hr"] = np.nan
    segments["measured_hr"] = np.nan
    segments["hr_measured"] = False
    model = fit_model(rides, segments, [])
    route, events = held_out_route(traces["9"], rides.iloc[-1], settings)
    result = predict_route(model, route, events, target_hr=145)
    assert model.median_hr is None and not model.hr_available
    assert result["achieved_hr"] is None
    assert result["segments"].predicted_hr.isna().all()
    assert result["moving_s"] > 0
    assert any("Insufficient measured HR" in w for w in result["warnings"])


def test_average_hr_constraint_and_variable_profile(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    assert model.hr_available
    route, events = held_out_route(traces["9"], rides.iloc[-1], settings)
    result = predict_route(model, route, events, target_hr=140)
    assert result["achieved_hr"] == pytest.approx(140, abs=0.5)
    assert result["segments"].predicted_hr.std() > 0.1
    assert result["segments"].predicted_hr.between(*model.hr_range).all()
    limited = predict_route(model, route, events, target_hr=225)
    assert limited["modeled_target_hr"] == pytest.approx(model.hr_range[1])
    assert any("outside measured support" in w for w in limited["warnings"])


def test_correct_time_and_speed_aggregation(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    route, events = held_out_route(traces["9"], rides.iloc[-1], settings)
    result = predict_route(model, route, events, personal_break_minutes=20)
    assert result["elapsed_s"] == pytest.approx(result["moving_s"] + result["stationary_s"] + 1200)
    assert result["rolling_kmh"] == pytest.approx(
        speed_from_time(result["distance_m"], result["moving_s"])
    )
    assert result["overall_kmh"] < result["rolling_kmh"]
    assert result["moving_s"] == pytest.approx(result["segments"].moving_time_s.sum())


def test_interruption_ledger_and_full_stop(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    points = traces["9"].copy()
    points.loc[80, "control_kind"] = "stop"
    points.loc[80, "control_id"] = "test:f"
    points.loc[80, "turn_angle_deg"] = 90.0
    events = route_events(points)
    assert len(events) == 1  # Corner and stop are one interruption.
    route = build_segments(points, rides.iloc[-1], events)
    full = predict_route(model, route, events, scenario="full_stop")
    observed = predict_route(model, route, events, scenario="observed")
    assert full["moving_s"] > observed["moving_s"]
    assert full["moving_s"] == pytest.approx(
        full["cruising_time_s"] + sum(full["losses_by_kind"].values())
    )
    assert full["stationary_s"] == 0  # Unknown dwell is not invented.
    assert full["unknown_wait_events"] == 1


def test_minimum_data_fallback(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides.iloc[:1], segments, [])
    assert model.base is None and "baseline" in model.kind
    route, events = held_out_route(traces["0"], rides.iloc[0], settings)
    result = predict_route(model, route, events, target_hr=145)
    assert result["rolling_kmh"] > 0
    with pytest.raises(ValueError, match="Import and process"):
        fit_model(rides.iloc[:0], segments, [])


def test_prediction_does_not_read_historical_outcomes(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    route, events = held_out_route(traces["9"], rides.iloc[-1], settings)
    original = predict_route(model, route, events, target_hr=140)
    manipulated = route.copy()
    for col in ("speed_kmh", "measured_hr", "power", "moving_s", "moving_hours", "stationary_s"):
        manipulated[col] = 99999
    result = predict_route(model, manipulated, events, target_hr=140)
    assert original["moving_s"] == pytest.approx(result["moving_s"])


def test_backtest_and_calibration_use_earlier_rides(store, synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    for _, ride in rides.iterrows():
        store.add_ride(ride.to_dict(), ride.id, traces[ride.id])
    settings.validation_rides = 6
    result = backtest(store, rides, segments, [], settings)
    assert result["status"] == "complete"
    for row in result["rows"]:
        assert pd.Timestamp(row["train_latest"]) < pd.Timestamp(row["date"])
    assert len(result["calibration_speed_residuals"]) == 3
    assert result["interval_provisional"]


def test_real_end_to_end_synthetic_workflow(store, settings):
    report = import_archive(strava_zip(), store, settings)
    assert report["imported"] == 1
    processed = process_history(store, settings)
    assert processed["processed"] == 1
    model = train(store, settings, validate=False)
    route = prepare_route(gpx_bytes(timed=False), "planned.gpx", store, settings)
    result = predict_route(model, route["segments"], route["events"], target_hr=145)
    assert result["distance_m"] > 4000
    assert result["rolling_kmh"] > 10
    assert store.model_path().exists()


def test_optional_elevation_correction(store, settings):
    original = gpx_bytes(timed=False, elevation=False)
    correction = b"distance_m,elevation_m\n0,100\n10000,200\n"
    route = prepare_route(original, "missing-elevation.gpx", store, settings, correction)
    assert not route["points"].elevation_missing.any()
    assert route["diagnostics"]["elevation_gain_m"] > 40
    assert route["segments"].gradient_pct.mean() == pytest.approx(1, abs=0.1)


def test_selected_fitness_variant_keeps_speed_and_time_consistent(
    monkeypatch, synthetic_dataset, settings
):
    from types import SimpleNamespace
    import cycling.validation as validation_module

    rides, segments, _ = synthetic_dataset
    settings.validation_rides = 6
    monkeypatch.setattr(validation_module, "held_out_route", lambda *args: (None, None))
    monkeypatch.setattr(
        validation_module, "fit_model", lambda *args, **kwargs: SimpleNamespace(fitness_adjustment=1.0)
    )

    def score(model, test, route, events, fitness=False):
        moving = 300.0 if fitness else 600.0
        speed = 36.0 if fitness else 18.0
        return {
            "ride_id": test.id,
            "date": test.date,
            "distance_km": 3.0,
            "elevation_gain_m": 100.0,
            "dominant_surface": "paved",
            "known_surface_fraction": 1.0,
            "actual_rolling_kmh": 36.0,
            "predicted_rolling_kmh": speed,
            "speed_absolute_error": abs(speed - 36),
            "predicted_moving_s": moving,
            "moving_error_s": moving - 300,
            "moving_absolute_percentage_error": abs(moving / 300 - 1) * 100,
            "elapsed_error_s": moving - 300,
            "median_baseline_error_kmh": 5.0,
            "gradient_baseline_error_kmh": 4.0,
        }

    monkeypatch.setattr(validation_module, "_score", score)
    fake_store = SimpleNamespace(points=lambda ride_id: None)
    result = backtest(fake_store, rides, segments, [], settings)
    assert result["fitness_selected"]
    for row in result["rows"]:
        assert row["predicted_rolling_kmh"] == 36.0
        assert row["predicted_moving_s"] == 300.0
        assert row["moving_error_s"] == 0.0
