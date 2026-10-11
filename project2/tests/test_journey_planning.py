from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from cycling.importer import import_archive
from cycling.model import fit_model
from cycling.pipeline import prepare_route, process_history, train
from cycling.prediction import predict_route
from cycling.stopping import StoppingModel, allocate_breaks
from cycling.validation import held_out_route
from cycling.weather import neutral_weather
from conftest import gpx_bytes, strava_zip


def stopping_history(rides):
    result = rides.copy()
    result["elapsed_reliable"] = True
    result["truth_moving_s"] = result.moving_s
    result["unattributed_stationary_s"] = np.linspace(30, 300, len(result))
    result["break_progress_weights"] = [[1.0] * 10] * len(result)
    return result


def test_personalized_custom_minimal_and_complete_ledger(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    route, events = held_out_route(traces["9"], rides.iloc[-1], settings)
    stops = StoppingModel.fit(stopping_history(rides))

    class Traffic:
        def predict(self, *args):
            return {"moving_loss_s": 12.0, "stationary_s": 30.0, "stationary_known": True}

    model.behavior = Traffic()
    event = {"kind": "signal", "distance_m": float(route.end_m.iloc[2])}
    results = {
        mode: predict_route(
            model,
            route,
            [event],
            departure="2025-06-01T12:00Z",
            stopping_model=stops,
            break_mode=mode,
            personal_break_minutes=3,
        )
        for mode in ["historical", "custom", "minimal"]
    }
    assert results["historical"]["personal_break_s"] > 0
    assert results["custom"]["personal_break_s"] == pytest.approx(180)
    assert results["minimal"]["personal_break_s"] == 0
    for result in results.values():
        assert result["stationary_s"] == 30
        assert result["moving_slowdown_s"] == 12
        assert result["elapsed_s"] == pytest.approx(
            result["cruising_time_s"]
            + result["moving_slowdown_s"]
            + result["stationary_s"]
            + result["personal_break_s"]
        )
        assert result["segments"].elapsed_time_s.sum() == pytest.approx(result["elapsed_s"])
        assert result["segments"].accumulated_elapsed_s.iloc[-1] == pytest.approx(result["elapsed_s"])
        assert (
            pd.Timestamp(result["finish_time_utc"]) - pd.Timestamp(result["departure_utc"])
        ).total_seconds() == pytest.approx(result["elapsed_s"], abs=1e-6)
    assert results["minimal"]["moving_s"] == pytest.approx(results["custom"]["moving_s"])


def test_breaks_shift_downstream_weather_without_network(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    route, events = held_out_route(traces["9"], rides.iloc[-1], settings)

    class Provider:
        note = "Synthetic forecast"
        client = type("Client", (), {"errors": []})()

        def __init__(self):
            self.calls = []

        def at(self, frame, times):
            self.calls.append(times.copy())
            result = neutral_weather(frame)
            result["weather_available"] = False
            return result

    minimal, custom = Provider(), Provider()
    a = predict_route(
        model,
        route,
        events,
        departure="2025-06-01T12:00Z",
        weather_provider=minimal,
        break_mode="minimal",
    )
    b = predict_route(
        model,
        route,
        events,
        departure="2025-06-01T12:00Z",
        weather_provider=custom,
        break_mode="custom",
        personal_break_minutes=20,
    )
    assert b["elapsed_s"] - a["elapsed_s"] == pytest.approx(1200)
    assert (custom.calls[-1].iloc[-1] - minimal.calls[-1].iloc[-1]).total_seconds() > 1100
    assert custom.calls[-1].is_monotonic_increasing
    assert b["weather_distance_coverage"] == 0


def test_stopping_fit_excludes_unreliable_and_future(synthetic_dataset):
    rides, segments, _ = synthetic_dataset
    history = stopping_history(rides)
    history.loc[9, "elapsed_reliable"] = False
    model = StoppingModel.fit(history)
    assert "9" not in set(model.rides.id)
    first = model.predict(segments[segments.ride_id.eq("0")], 1000, "2025-01-01T00:00Z")
    assert first["samples"] == 0 and first["evidence"] == "unavailable"
    long = segments[segments.ride_id.eq("9")].copy()
    scale = 200000 / long.distance_m.sum()
    for col in ["distance_m", "start_m", "end_m"]:
        long[col] *= scale
    estimate = model.predict(long, 30000, "2026-01-01T00:00Z")
    assert estimate["evidence"] == "low"
    assert any("extrapolated" in w for w in estimate["warnings"])
    assert allocate_breaks(
        long, estimate["expected_s"], estimate["progress_weights"]
    ).sum() == pytest.approx(estimate["expected_s"])


def test_streamlit_manual_hr_survives_rerun_new_route_and_reset(monkeypatch, store, settings):
    monkeypatch.setenv("CYCLING_DATA_DIR", settings.data_dir)
    import_archive(strava_zip(), store, settings)
    process_history(store, settings)
    train(store, settings, validate=False)
    route = prepare_route(gpx_bytes(timed=False), "first.gpx", store, settings)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60)
    app.session_state["route"] = route
    app.session_state["loaded_route_token"] = "a"
    app.run()
    app.sidebar.radio[0].set_value("Route Prediction").run()
    assert not app.exception
    target = next(x for x in app.number_input if x.label == "Target average heart rate (BPM)")
    assert target.value is not None
    target.set_value(135).run()
    assert app.session_state["hr_manual"]
    app.run()
    assert next(x for x in app.number_input if x.label == "Target average heart rate (BPM)").value == 135
    replacement = prepare_route(gpx_bytes(count=230, timed=False), "second.gpx", store, settings)
    app.session_state["route"] = replacement
    app.session_state["loaded_route_token"] = "b"
    app.run()
    assert next(x for x in app.number_input if x.label == "Target average heart rate (BPM)").value == 135
    next(b for b in app.button if b.label == "Use suggested HR").click().run()
    assert not app.session_state["hr_manual"]
    assert (
        next(x for x in app.number_input if x.label == "Target average heart rate (BPM)").value
        == app.session_state["hr_recommendation"]["suggested_hr"]
    )
    next(x for x in app.number_input if x.label == "Target average heart rate (BPM)").set_value(
        138
    ).run()
    next(b for b in app.button if b.label == "Predict total ride time").click().run()
    assert not app.exception
    assert app.session_state["result"]["requested_hr"] == 138
    assert app.session_state["result"]["hr_selection"] == "manual override"
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception
    assert any(m.label == "Expected finish time" for m in app.metric)
