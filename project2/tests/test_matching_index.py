"""Offline topology, ambiguity, control applicability and cache regression fixtures."""

import json

import numpy as np
import pandas as pd
import pytest

from cycling.comparison import compare_models, comparison_folds, variant_data, variant_points
from cycling.coverage import trace_coverage
from cycling.events import route_events
from cycling.geometry import clean_points
from cycling.matching import annotate_hmm
from cycling.network import NetworkIndex
from cycling.osm import OSMProvider
from cycling.pipeline import prepare_route, process_history

SX = 111195 * np.cos(np.radians(43))


def node(identifier, x, y, tags=None):
    return {
        "type": "node",
        "id": identifier,
        "lat": 43 + y / 111195,
        "lon": -80 + x / SX,
        "tags": tags or {},
    }


def way(identifier, nodes, tags=None):
    return {
        "type": "way",
        "id": identifier,
        "nodes": [n["id"] for n in nodes],
        "geometry": [{"lat": n["lat"], "lon": n["lon"]} for n in nodes],
        "tags": tags or {"highway": "cycleway", "surface": "compacted"},
    }


def track(x, y, settings, timed=False):
    x, y = np.asarray(x), np.broadcast_to(y, len(x))
    raw = pd.DataFrame({"lat": 43 + y / 111195, "lon": -80 + x / SX, "elevation": 100.0})
    if timed:
        raw["time"] = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
            np.arange(len(x)) * 5, unit="s"
        )
    return clean_points(raw, settings, route=not timed)[0]


def test_parallel_cycleway_with_noise_stays_on_connected_path(settings):
    path = [node(1, 0, 0), node(2, 150, 0), node(3, 300, 0)]
    road = [node(4, 0, 24), node(5, 300, 24)]
    payload = {
        "elements": path
        + road
        + [way(10, path), way(20, road, {"highway": "primary", "surface": "asphalt"})]
    }
    rng = np.random.default_rng(42)
    points = track(np.arange(0, 301, 10), rng.normal(0, 2, 31), settings)
    matched = annotate_hmm(points, payload)
    assert matched.way_id.eq("10").all()
    assert matched.surface.eq("hardpack_gravel").all()
    pd.testing.assert_series_equal(matched.lon, points.lon)
    assert matched.matched_geometry.ne("[]").all()


def test_indistinguishable_parallel_roads_are_explicitly_uncertain(settings):
    a, b = [node(1, 0, 0), node(2, 300, 0)], [node(3, 0, 8), node(4, 300, 8)]
    matched = annotate_hmm(
        track(np.arange(0, 301, 10), 4, settings), {"elements": a + b + [way(10, a), way(20, b)]}
    )
    assert matched.map_confidence.lt(0.5).all()
    assert matched.surface.eq("unknown").all()


def test_continuity_beats_nearer_unconnected_parallel_road(settings):
    path = [node(1, 0, 0), node(2, 150, 0), node(3, 300, 0)]
    adjacent = [node(4, 90, 12), node(5, 210, 12)]
    x = np.arange(0, 301, 10)
    # A GPS excursion is nearer the automobile road, but its disconnected endpoints
    # cannot explain the preceding/following observed path. Nearest-point matching fails.
    y = np.where((x >= 100) & (x <= 200), 10, 0)
    points = track(x, y, settings)
    payload = {
        "elements": path
        + adjacent
        + [way(10, path), way(20, adjacent, {"highway": "primary", "surface": "asphalt"})]
    }
    matched = annotate_hmm(points, payload)
    assert matched.way_id.eq("10").all()
    assert matched.surface.eq("hardpack_gravel").all()


@pytest.mark.parametrize("tags", [{"bridge": "yes", "layer": "1"}, {"tunnel": "yes", "layer": "-1"}, {}])
def test_unconnected_crossing_does_not_create_a_turn(settings, tags):
    a, b = [node(1, 0, 0), node(2, 300, 0)], [node(3, 150, -100), node(4, 150, 100)]
    payload = {"elements": a + b + [way(10, a), way(20, b, {"highway": "path", **tags})]}
    matched = annotate_hmm(track(np.arange(0, 301, 15), 0, settings), payload)
    assert matched.way_id.eq("10").all()
    assert not matched.control_kind.ne("").any()


def test_junction_and_sparse_turn_follow_node_connectivity(settings):
    a, junction, b = node(1, 0, 0), node(2, 150, 0), node(3, 150, 150)
    nearby = [node(7, 130, 40), node(8, 130, 160)]
    payload = {
        "elements": [a, junction, b]
        + nearby
        + [way(10, [a, junction]), way(20, [junction, b]), way(30, nearby)]
    }
    points = track([0, 60, 120, 150, 150, 150], [0, 0, 0, 0, 60, 120], settings)
    matched = annotate_hmm(points, payload)
    assert set(matched.way_id) <= {"10", "20"}
    assert matched.way_id.iloc[-1] == "20"


def test_sparse_directional_control_and_side_road(settings):
    a, stop, b = (
        node(1, 0, 0),
        node(2, 100, 0, {"highway": "stop", "direction": "forward"}),
        node(3, 300, 0),
    )
    side = [node(4, 100, 8, {"highway": "stop"}), node(5, 100, 100)]
    payload = {"elements": [a, stop, b] + side + [way(10, [a, stop, b]), way(20, side)]}
    matched = annotate_hmm(track([0, 60, 140, 210, 290], 0, settings), payload)
    assert matched.control_kind.eq("stop").sum() == 1
    reverse = annotate_hmm(track([290, 210, 140, 60, 0], 0, settings), payload)
    assert reverse.control_kind.eq("stop").sum() == 0


def test_shared_signals_and_pedestrian_only_signal(settings):
    a, signal, signal2, b = (
        node(1, 0, 0),
        node(2, 120, 0, {"highway": "traffic_signals"}),
        node(3, 130, 0, {"highway": "traffic_signals"}),
        node(4, 300, 0),
    )
    side = node(5, 120, 100)
    pedestrian = node(6, 220, 0, {"highway": "traffic_signals", "traffic_signals": "pedestrian"})
    payload = {
        "elements": [a, signal, signal2, b, side, pedestrian]
        + [way(10, [a, signal, signal2, pedestrian, b]), way(20, [signal, side])]
    }
    matched = annotate_hmm(track(np.arange(0, 301, 10), 0, settings), payload)
    assert matched.control_kind.eq("signal").sum() == 1


def test_untagged_and_bicycle_restricted_roads(settings):
    a, b = node(1, 0, 0), node(2, 300, 0)
    points = track(np.arange(0, 301, 15), 0, settings)
    matched = annotate_hmm(points, {"elements": [a, b, way(10, [a, b], {"highway": "residential"})]})
    assert matched.surface.eq("unknown").all()
    assert matched.highway.eq("residential").all()
    restricted = annotate_hmm(
        points, {"elements": [a, b, way(10, [a, b], {"highway": "primary", "bicycle": "no"})]}
    )
    assert restricted.map_confidence.eq(0).all()


def test_persistent_index_updates_only_relevant_signatures(tmp_path, settings):
    a, b = node(1, 0, 0), node(2, 300, 0)
    source = tmp_path / "map.json"
    source.write_text(json.dumps({"elements": [a, b, way(10, [a, b])]}))
    points = track([0, 100, 200, 300], 0, settings)
    outside = points.copy()
    outside["lat"] += 10
    first = NetworkIndex(tmp_path, [source])
    initial, outside_signature = first.signature(points), first.signature(outside)
    indexed_at = first.report()[0]["indexed_at"]
    assert NetworkIndex(tmp_path, [source]).report()[0]["indexed_at"] == indexed_at
    source.write_text(
        json.dumps({"elements": [a, b, way(10, [a, b], {"highway": "path", "surface": "gravel"})]})
    )
    second = NetworkIndex(tmp_path, [source])
    assert second.signature(points) != initial
    assert second.signature(outside) == outside_signature
    assert second.data_for(outside)["elements"] == []


def test_missing_extract_and_remote_match_do_not_upload_or_crash(store, settings):
    settings.osm_provider, settings.local_osm_path = "local", str(store.root / "absent.pbf")
    settings.valhalla_url = "https://remote.example"

    class NoNetwork:
        headers = {}

        def post(self, *a, **kw):
            raise AssertionError("Recorded GPS must remain local")

    result = OSMProvider(store, settings, NoNetwork()).annotate(track([0, 100, 200], 0, settings))
    assert result.surface.eq("unknown").all()


def test_coverage_is_distance_weighted(settings):
    points = annotate_hmm(track([0, 10, 20, 120], 0, settings), {"elements": []})
    points.loc[3, "surface"] = "paved"
    report = trace_coverage(points)
    assert report["surfaces"]["paved"] / report["total_km"] == pytest.approx(100 / 120)


def test_distinct_close_controls_are_not_merged(settings):
    points = annotate_hmm(track(np.arange(0, 301, 10), 0, settings), {"elements": []})
    for index, kind in ((10, "stop"), (14, "signal")):
        points.loc[index, "control_kind"] = kind
        points.loc[index, "control_id"] = str(index)
    events = route_events(points)
    assert len(events) == 2
    assert events[0]["end_m"] <= events[1]["start_m"]


def test_comparison_selection_precedes_final_and_preserves_long_rides():
    rides = pd.DataFrame(
        {
            "id": [str(i) for i in range(100)],
            "date": pd.date_range("2025-01-01", periods=100, tz="UTC").astype(str),
            "distance_m": [120000 if i % 10 == 0 else 30000 for i in range(100)],
        }
    )
    folds = comparison_folds(rides, 20, 8)
    earlier = [f[2] for f in folds if f[0] == "selection"]
    final = [f[2] for f in folds if f[0] == "calibration"]
    assert max(t.date for t in earlier) < min(t.date for t in final)
    assert len(final) == 20
    assert sum(t.distance_m >= 100000 for t in final) >= 3
    for _, train, test in folds:
        assert str(test.id) not in set(train.id)
        assert pd.to_datetime(train.date).max() < pd.Timestamp(test.date)


def test_history_incremental_and_dataset_invalidation(store, settings):
    points = track(np.arange(0, 601, 10), 0, settings, timed=True)
    meta = {
        "id": "one",
        "date": "2025-01-01T00:00:00+00:00",
        "bike": "unknown",
        "distance_m": 600,
        "moving_s": 300,
        "elapsed_s": 300,
        "measured_avg_hr": None,
    }
    store.add_ride(meta, "fingerprint", points)
    source = store.root / "map.json"
    a, b = node(1, 0, 0), node(2, 600, 0)
    source.write_text(json.dumps({"elements": [a, b, way(10, [a, b])]}))
    settings.osm_provider, settings.local_osm_path = "local", str(source)
    assert process_history(store, settings)["processed"] == 1
    assert process_history(store, settings)["processed"] == 0
    source.write_text(
        json.dumps({"elements": [a, b, way(10, [a, b], {"highway": "path", "surface": "gravel"})]})
    )
    assert process_history(store, settings)["processed"] == 1
    assert store.points("one").surface.eq("loose_gravel").all()


def test_planned_route_cache_and_unknown_out_of_region(store, settings, monkeypatch):
    gpx = b'<gpx><trk><trkseg><trkpt lat="43" lon="-80"><ele>100</ele></trkpt><trkpt lat="43" lon="-79.999"><ele>100</ele></trkpt></trkseg></trk></gpx>'
    first = prepare_route(gpx, "route.gpx", store, settings)
    monkeypatch.setattr(
        OSMProvider, "annotate", lambda *_: (_ for _ in ()).throw(AssertionError("cache missed"))
    )
    second = prepare_route(gpx, "route.gpx", store, settings)
    pd.testing.assert_frame_equal(first["points"], second["points"])
    assert second["coverage"]["matched_km"] == 0
    assert variant_points(first["points"], "A").surface.eq("unknown").all()


def test_variant_label_cache_is_specific_to_activity(store, settings):
    # Two rides in the same extract share a network signature but never training labels.
    for rid, length in (("a", 300), ("b", 600)):
        points = annotate_hmm(
            track(np.arange(0, length + 1, 10), 0, settings, timed=True), {"elements": []}
        )
        meta = {
            "id": rid,
            "date": "2025-01-01T00:00:00+00:00",
            "bike": "unknown",
            "distance_m": length,
            "moving_s": length / 2,
            "elapsed_s": length / 2,
            "measured_avg_hr": None,
            "enrichment_signature": "same-extract",
        }
        store.add_ride(meta, rid, points)
        from cycling.features import build_segments

        store.save_segments(rid, build_segments(points, meta, []))
    variants = variant_data(store, store.rides())
    assert set(variants["B"][0].ride_id) == {"a", "b"}
    assert variants["B"][0].groupby("ride_id").distance_m.sum()["b"] > 500


def test_complete_paired_comparison_on_offline_history(store, settings, synthetic_dataset):
    from cycling.model import load_model

    rides, _, traces = synthetic_dataset
    from cycling.features import build_segments

    for round_number in (0, 1):
        for _, source in rides.iterrows():
            meta = source.to_dict()
            rid = f"{round_number}-{source.id}"
            meta.update(
                id=rid,
                date=(pd.Timestamp(source.date) + pd.Timedelta(days=150 * round_number)).isoformat(),
                enrichment_signature="synthetic-network",
            )
            points = traces[str(source.id)].copy()
            store.add_ride(meta, rid, points)
            store.save_segments(rid, build_segments(points, meta, route_events(points)))
    settings.optimize_half_life = False
    report = compare_models(store, settings)
    paired = pd.DataFrame(report["paired_rows"])
    ids = paired.groupby("variant").ride_id.apply(set)
    assert ids["A"] == ids["B"] == ids["C"]
    assert pd.Timestamp(report["selection_last"]) < pd.Timestamp(report["final_first"])
    for row in report["paired_rows"]:
        assert pd.Timestamp(row["train_latest"]) < pd.Timestamp(row["date"])
    assert load_model(store).variant == report["selected_variant"]
    assert all((store.root / "models" / f"model-{v}.joblib").exists() for v in "ABC")


def test_passing_signals_do_not_invent_conditional_wait(settings):
    from cycling.events import BehaviorModel, observe_events

    points = annotate_hmm(track(np.arange(0, 601, 10), 0, settings, timed=True), {"elements": []})
    points.loc[30, "control_kind"] = "signal"
    points.loc[30, "control_id"] = "signal:test"
    points["map_confidence"] = 1.0  # Synthetic known control; road matching is tested separately.
    events = route_events(points)
    observed = observe_events(points, events, "past")
    assert observed[0]["stationary_s"] == 0
    estimated = BehaviorModel(observed * 10).predict(events[0], 6, "observed")
    assert estimated["stationary_s"] == 0
    assert not estimated["stationary_known"]


def test_short_full_stop_window_reaches_zero_with_limited_peak_speed():
    from cycling.events import physics_loss

    factor = 1 / 0.9 + 1 / 0.45
    travel = np.sqrt(2 * 10 * factor)
    assert physics_loss(10, 0, 10) == pytest.approx(travel - 10 / 10)
    assert physics_loss(10, 0, 10) > physics_loss(10, 7, 10)
