import numpy as np
import pandas as pd
import pytest

from cycling.events import BehaviorModel, observe_events, physics_loss, route_events
from cycling.geometry import clean_points
from cycling.osm import OSMProvider, annotate_network, classify_surface, control_applies


def trace(speeds, settings, gap_index=None):
    speeds = np.asarray(speeds, dtype=float)
    seconds = np.arange(len(speeds)) * 5
    if gap_index is not None:
        seconds[gap_index:] += 120
    distance = np.cumsum(speeds * 5)
    raw = pd.DataFrame(
        {
            "lat": 43.0,
            "lon": -80 + distance / (111195 * np.cos(np.radians(43))),
            "time": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(seconds, unit="s"),
            "speed": speeds,
            "elevation": 100.0,
            "distance": distance,
        }
    )
    return clean_points(raw, settings)[0]


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("asphalt", "paved"),
        ("fine_gravel", "hardpack_gravel"),
        ("gravel", "loose_gravel"),
        ("sand", "rough_natural"),
        ("", "unknown"),
    ],
)
def test_surfaces(tag, expected):
    assert classify_surface({"surface": tag})[0] == expected


def test_surface_inference_kept_separate():
    assert classify_surface({"highway": "residential"}) == ("unknown", "unknown", 0)
    assert classify_surface({"tracktype": "grade2"})[:2] == ("hardpack_gravel", "inferred")


def test_directional_and_side_road_controls():
    assert control_applies({"highway": "stop", "direction": "forward"}, True)[0] is True
    assert control_applies({"highway": "stop", "stop:direction": "backward"}, True)[0] is False
    assert control_applies({"highway": "stop"}, True, member=False)[0] is False
    assert control_applies({"highway": "stop"}, True, junction_shared=True)[0] is None


def network_fixture(points):
    middle = len(points) // 2
    nodes = [
        {"type": "node", "id": i + 1, "lat": float(points.lat.iloc[i]), "lon": float(points.lon.iloc[i])}
        for i in (0, middle, len(points) - 1)
    ]
    nodes[1]["tags"] = {"highway": "stop", "direction": "forward"}
    return {
        "elements": nodes
        + [
            {
                "type": "way",
                "id": 99,
                "nodes": [n["id"] for n in nodes],
                "tags": {"highway": "residential", "surface": "asphalt"},
            }
        ]
    }


def test_network_match_preserves_direction_and_geometry(settings):
    points = trace([6] * 30, settings)
    payload = network_fixture(points)
    forward = annotate_network(points, payload)
    assert forward.surface.eq("paved").all()
    assert forward.control_kind.eq("stop").sum() == 1
    assert np.array_equal(points.lon, forward.lon)
    reverse = points.copy()
    reverse["heading"] = 270.0
    backward = annotate_network(reverse, payload)
    assert backward.control_kind.eq("stop").sum() == 0


@pytest.mark.parametrize(
    "profile,expected",
    [
        ([6] * 8 + [0] * 3 + [6] * 9, "full_stop"),
        ([6] * 8 + [2] * 3 + [6] * 9, "rolling"),
        ([6] * 20, "minimal"),
    ],
)
def test_stop_behavior_detection(profile, expected, settings):
    points = annotate_network(trace(profile, settings), {"elements": []})
    midpoint = len(points) // 2
    points.loc[midpoint, "control_kind"] = "stop"
    points.loc[midpoint, "control_id"] = "123:f"
    points["map_confidence"] = 1.0  # Synthetic known control on the ridden road.
    observations = observe_events(points, route_events(points), "ride")
    assert observations[0]["behavior"] == expected


def test_signal_wait_stationary_not_moving(settings):
    points = annotate_network(trace([6] * 10 + [0] * 5 + [6] * 10, settings), {"elements": []})
    points.loc[12, "control_kind"] = "signal"
    points.loc[12, "control_id"] = "light:f"
    points["map_confidence"] = 1.0
    observations = observe_events(points, route_events(points), "r1")
    assert observations[0]["stationary_s"] == 25
    behavior = BehaviorModel(observations * 10)
    result = behavior.predict(route_events(points)[0], 6, "observed")
    assert result["stationary_s"] > 15
    assert result["moving_loss_s"] >= 0


def test_smart_recording_gap_keeps_unknown_wait(settings):
    points = trace([6] * 20, settings, gap_index=10)
    assert points.unobserved_s.sum() >= 115
    assert points.stationary_s.sum() == 0
    assert points.moving_s.sum() < points.dt_s.sum()


def test_recorded_timer_pause_is_observed(settings):
    raw = pd.DataFrame(
        {
            "lat": [43, 43, 43],
            "lon": [-80, -79.9996, -79.9992],
            "time": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:00:05Z", "2025-01-01T00:02:10Z"]
            ),
            "speed": [6, 6, 6],
            "timer_pause_s": [0, 0, 120],
        }
    )
    points, _ = clean_points(raw, settings)
    assert points.stationary_s.sum() == 120
    assert points.unobserved_s.sum() == pytest.approx(0, abs=1)


def test_disconnected_route_does_not_bridge(settings):
    raw = pd.DataFrame(
        {"lat": [43, 43, 44, 44], "lon": [-80, -79.999, -80, -79.999], "track_group": [0, 0, 1, 1]}
    )
    points, diag = clean_points(raw, settings, route=True)
    assert points.ds_m.iloc[2] == 0
    assert diag["distance_m"] < 200


def test_offline_osm_never_requests_network(store, settings):
    class NoNetwork:
        headers = {}

        def post(self, *args, **kwargs):
            raise AssertionError("Offline attempted a network request")

    provider = OSMProvider(store, settings, NoNetwork())
    points = provider.annotate(trace([6] * 20, settings))
    assert points.surface.eq("unknown").all()
    assert provider.requested == 0


def test_overpass_cache_and_bounded_requests(store, settings):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": []}

    class Session:
        headers = {}
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return Response()

    settings.osm_provider, settings.osm_max_tiles, settings.osm_min_interval_s = "overpass", 1, 0
    session = Session()
    provider = OSMProvider(store, settings, session)
    provider._tile(1, 1)
    provider._tile(1, 1)
    provider._tile(2, 2)
    assert session.calls == 1
    OSMProvider(store, settings, session)._tile(1, 1)
    assert session.calls == 1


def test_physics_loss_nonnegative_and_full_stop_slower():
    assert physics_loss(7, 7) == 0
    assert physics_loss(7, 0) > physics_loss(7, 3) > 0


def test_incidental_pause_at_corner_does_not_become_a_personal_break(settings):
    points = annotate_network(trace([6] * 10 + [0] * 5 + [6] * 10, settings), {"elements": []})
    points.loc[12, "control_kind"] = "corner"
    points.loc[12, "control_id"] = "geometry-only"
    events = route_events(points)
    observations = observe_events(points, events, "ride")
    assert observations[0]["stationary_s"] == 25
    result = BehaviorModel(observations * 10).predict(events[0], 6)
    assert result["stationary_s"] == 0
    assert result["stop_probability"] is None
