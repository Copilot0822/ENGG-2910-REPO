import numpy as np
import pytest

from cycling.events import BehaviorModel, observe_events, route_events
from cycling.osm import annotate_network
from test_geometry_osm import trace


def recording(profile, settings):
    p = annotate_network(trace(profile, settings), {"elements": []})
    p.loc[len(p) // 2, "control_kind"] = "stop"
    p.loc[len(p) // 2, "control_id"] = "one:forward"
    p["map_confidence"] = 1.0
    return p


def test_gap_and_slow_rolling_are_not_confirmed_stop(settings):
    p = recording([6] * 10 + [0] * 5 + [6] * 10, settings)
    e = route_events(p)
    full = observe_events(p, e, "one")[0]
    assert full["behavior"] == "full_stop" and full["stationary_s"] == 25
    p.loc[12, "unobserved_s"] = 30
    gap = observe_events(p, e, "two")[0]
    assert not gap["reliable"] and gap["behavior"] == "unknown"
    assert np.isnan(gap["stationary_s"])
    rolling = recording([6] * 10 + [0.5] * 5 + [6] * 10, settings)
    assert observe_events(rolling, route_events(rolling), "three")[0]["behavior"] == "rolling"


def test_encounter_metrics_and_probability_shrinkage(settings):
    p = recording([6] * 10 + [0] * 5 + [6] * 10, settings)
    event = route_events(p)[0]
    full = observe_events(p, [event], "1")[0]
    for key in (
        "speed_at_sign_mps",
        "braking_distance_m",
        "braking_s",
        "acceleration_s",
        "recovery_distance_m",
        "turn_direction",
        "approach_heading",
    ):
        assert key in full
    minimal = {
        **full,
        "identity": "other:forward",
        "behavior": "minimal",
        "stationary_s": 0,
        "moving_loss_s": 0,
        "minimum_mps": 6,
    }
    model = BehaviorModel([full] + [minimal] * 30)
    estimate = model.predict(event, 6, "observed", full["hour"], full["weekday"])
    assert 0 < estimate["stop_probability"] < 0.5
    assert sum(estimate["behavior_probabilities"].values()) == pytest.approx(1)
    assert estimate["local_encounters"] == 1
    assert "generalized" in estimate["source"]
    forced = model.predict(event, 6, "full_stop")
    assert forced["expected_minimum_mps"] == 0 and forced["stop_probability"] == 1


def test_default_c_uses_traffic_and_b_remains_available(store, synthetic_dataset, settings):
    from cycling.model import fit_model, load_model
    from cycling.prediction import predict_route
    from cycling.validation import held_out_route

    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    model.variant = "B"
    model.save(store.model_path())
    model.save(store.root / "models/model-B.joblib")
    model.variant = "C"
    model.save(store.root / "models/model-C.joblib")
    store.set_state("training", {"variant": "B"})
    assert load_model(store).variant == "C"
    p = traces["9"].copy()
    p.loc[90, "control_kind"] = "stop"
    p.loc[90, "control_id"] = "test:forward"
    route, events = held_out_route(p, rides.iloc[-1], settings)
    full = predict_route(model, route, events)
    model.variant = "B"
    baseline = predict_route(model, route, events)
    assert full["moving_s"] > baseline["moving_s"]
    assert any(e["kind"] == "stop" for e in full["events"])
    assert not any(e["kind"] == "stop" for e in baseline["events"])
