from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cycling.geometry import clean_points
from cycling.recordings import parse_fit
from cycling.timing import timing_quality


class Message:
    def __init__(self, name, **fields):
        self.name, self.fields = name, fields

    def __iter__(self):
        return iter([SimpleNamespace(name=k, value=v) for k, v in self.fields.items()])


def test_fit_timer_pause_and_total_elapsed_are_preserved(monkeypatch, settings):
    start = datetime(2025, 6, 1, 12)
    messages = [Message("event", event="timer", event_type="start", timestamp=start)]
    for second, distance, speed in [(0, 0, 5), (10, 50, 5), (70, 50, 0), (80, 100, 5)]:
        messages.append(
            Message(
                "record",
                timestamp=start + timedelta(seconds=second),
                position_lat=int(43 * 2**31 / 180),
                position_long=int((-80 + distance / 81000) * 2**31 / 180),
                distance=distance,
                speed=speed,
                heart_rate=145,
            )
        )
    messages += [
        Message("event", event="timer", event_type="stop_all", timestamp=start + timedelta(seconds=10)),
        Message("event", event="timer", event_type="start", timestamp=start + timedelta(seconds=70)),
        Message(
            "session",
            start_time=start,
            timestamp=start + timedelta(seconds=80),
            total_elapsed_time=80,
            total_timer_time=20,
            avg_heart_rate=145,
        ),
    ]
    monkeypatch.setattr(
        "cycling.recordings.FitFile", lambda *a, **k: SimpleNamespace(get_messages=lambda: messages)
    )
    raw, recorded = parse_fit(b"mocked")
    points, diagnostics = clean_points(raw, settings)
    assert recorded["timer_s"] == 20 and recorded["recorded_elapsed_s"] == 80
    assert recorded["timer_event_count"] == 3
    assert raw.timer_pause_s.sum() == 60
    assert points.stationary_s.sum() == 60 and points.unobserved_s.sum() == 0
    quality = timing_quality(
        {**diagnostics, "summary_elapsed_s": 80, "summary_moving_s": 20}, points, recorded
    )
    assert quality["elapsed_reliable"] and quality["elapsed_truth_s"] == 80
    assert quality["total_stationary_s"] == 60


def simple_points(settings):
    raw = pd.DataFrame(
        {
            "lat": 43.0,
            "lon": -80 + np.arange(61) * 0.00003,
            "time": pd.date_range("2025-01-01T12:00Z", periods=61, freq="1s"),
            "speed": np.r_[np.full(41, 2.0), np.zeros(20)],
            "hr": 145.0,
        }
    )
    return clean_points(raw, settings)


def test_duration_does_not_require_stop_reason(settings):
    points, diag = simple_points(settings)
    result = timing_quality({**diag, "summary_elapsed_s": 60}, points)
    assert result["elapsed_reliable"]
    assert result["unattributed_stationary_s"] > 0
    assert result["identified_traffic_s"] == 0


def test_traffic_wait_is_subtracted_once_and_capped(settings):
    points, diag = simple_points(settings)
    observed = points.stationary_s.sum()
    events = [
        {
            "reliable": True,
            "kind": "signal",
            "stationary_s": observed / 2,
            "distance_m": points.distance_m.iloc[-1],
        }
    ]
    result = timing_quality({**diag, "summary_elapsed_s": 60}, points, encounters=events)
    assert (
        result["identified_traffic_s"] + result["unattributed_stationary_s"]
        == result["total_stationary_s"]
    )
    assert result["identified_traffic_s"] == observed / 2
    events[0]["stationary_s"] = 1000
    capped = timing_quality({**diag, "summary_elapsed_s": 60}, points, encounters=events)
    assert capped["unattributed_stationary_s"] == 0


def test_conflicting_and_invalid_recording_durations(settings):
    points, diag = simple_points(settings)
    result = timing_quality({**diag, "summary_elapsed_s": 600, "summary_moving_s": 700}, points)
    assert not result["elapsed_reliable"]
    assert "invalid_summary_moving_duration" in result["elapsed_quality_reasons"]
    assert "recording_span_conflicts_with_activity" in result["elapsed_quality_reasons"]


def test_sparse_hr_cannot_be_rescued_by_imputed_or_sample_summary(settings):
    points, diag = simple_points(settings)
    points.loc[:35, "hr"] = np.nan
    result = timing_quality(
        {**diag, "summary_elapsed_s": 60, "summary_avg_hr": 150, "hr_source": "samples"}, points
    )
    assert not result["hr_reliable"] and result["hr_recommendation_value"] is None


def test_trustworthy_fit_summary_hr_without_samples(settings):
    points, diag = simple_points(settings)
    points["hr"] = np.nan
    result = timing_quality(
        {**diag, "summary_elapsed_s": 60, "summary_avg_hr": None, "hr_source": "summary"},
        points,
        {"recorded_avg_hr": 145},
    )
    assert result["hr_reliable"] and result["hr_recommendation_value"] == 145
    assert result["hr_recommendation_source"].startswith("recorded summary")


def test_unobserved_paused_intervals_can_have_reliable_elapsed(settings):
    points, diag = simple_points(settings)
    raw = points[["lat", "lon", "time", "hr"]].copy()
    raw["speed"] = points.speed_mps
    raw.loc[10:, "time"] += pd.Timedelta(seconds=70)
    points, diag = clean_points(raw, settings)
    result = timing_quality({**diag, "summary_elapsed_s": 130}, points)
    assert result["elapsed_reliable"]
    assert "unobserved_intervals_inside_trustworthy_activity_duration" in result["timing_notes"]
