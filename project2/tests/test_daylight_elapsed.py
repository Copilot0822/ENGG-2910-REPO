from datetime import date, time

import pandas as pd
import pytest

from cycling.daylight import daylight_plan, local_departure, latest_departure
from cycling.elapsed import ElapsedCalibration, cutoff_probability
from cycling.journey_validation import distance_group


def test_finish_sunset_twilight_and_margin():
    result = daylight_plan("2026-10-11T09:00:00-04:00", 5 * 3600, (43.65, -79.38))
    assert pd.Timestamp(result["finish_local"]).hour == 14
    sunset = pd.Timestamp(result["sunset"])
    twilight = pd.Timestamp(result["civil_twilight"])
    assert sunset.hour == 18 and 39 <= sunset.minute <= 41
    assert twilight.hour == 19 and 8 <= twilight.minute <= 11
    assert (sunset - pd.Timestamp(result["cutoff"])).total_seconds() == 1800
    assert result["remaining_daylight_minutes"] > 270
    civil = daylight_plan("2026-10-11T09:00:00-04:00", 5 * 3600, (43.65, -79.38), "civil_twilight")
    assert pd.Timestamp(civil["cutoff"]) > pd.Timestamp(result["cutoff"])
    assert any("not full daylight" in n for n in civil["notes"])


def test_destination_timezone_and_overnight():
    crossed = daylight_plan("2026-06-01T09:00:00-04:00", 6 * 3600, (49.28, -123.12))
    assert crossed["destination_timezone"] == "America/Vancouver"
    assert pd.Timestamp(crossed["finish_local"]).hour == 12
    overnight = daylight_plan("2026-10-11T23:00:00-04:00", 3 * 3600, (43.65, -79.38))
    assert pd.Timestamp(overnight["finish_local"]).day == 12
    assert overnight["remaining_daylight_minutes"] < 0
    assert not overnight["daylight_at_finish"]


def test_dst_nonexistent_and_repeated_hours():
    with pytest.raises(ValueError, match="daylight-saving"):
        local_departure(date(2026, 3, 8), time(2, 30), "America/Toronto")
    with pytest.raises(ValueError):
        local_departure(date(2026, 11, 1), time(1, 30), "America/Toronto")
    first = pd.Timestamp(local_departure(date(2026, 11, 1), time(1, 30), "America/Toronto", "first"))
    second = pd.Timestamp(local_departure(date(2026, 11, 1), time(1, 30), "America/Toronto", "second"))
    assert (second - first).total_seconds() == 3600
    spring = daylight_plan("2026-03-08T00:30:00-05:00", 3 * 3600, (43.65, -79.38))
    assert pd.Timestamp(spring["finish_local"]).hour == 4


def test_polar_daylight_condition_has_no_fake_sunset():
    result = daylight_plan("2026-12-21T12:00:00+00:00", 3600, (80, 15))
    assert result["sunset"] is None and result["cutoff"] is None
    assert not result["daylight_at_finish"]


def test_asymmetric_intervals_scale_with_duration_and_use_only_past():
    rows = [
        {
            "date": f"2025-01-{i + 1:02}T00:00Z",
            "distance_km": 40,
            "predicted_elapsed_s": 3600,
            "actual_elapsed_s": 3600 + x,
        }
        for i, x in enumerate([-200, -100, 0, 100, 200, 500, 1000, 1500, 2500, 3000])
    ]
    calibration = ElapsedCalibration(rows)
    small = calibration.estimate(3600, 40, "2026-01-01T00:00Z")
    large = calibration.estimate(36000, 200, "2026-01-01T00:00Z")
    assert small["conservative_s"] > 3600
    assert large["conservative_s"] - 36000 >= (small["conservative_s"] - 3600) * 9
    assert small["provisional"] and not small["probability_supported"]
    assert cutoff_probability(small, 100000) is None
    earlier = calibration.estimate(3600, 40, "2025-01-05T00:00Z")
    assert earlier["samples"] == 4 and earlier["conservative_s"] is None
    assert small["intervals_s"]["90"][1] - 3600 > 3600 - small["intervals_s"]["90"][0]


def test_latest_departure_recomputes_time_dependent_duration():
    calls = []

    def predictor(departure):
        moment = pd.Timestamp(departure)
        calls.append(moment)
        elapsed = 3600 if moment.hour < 14 else 7200
        return {
            "elapsed_s": elapsed,
            "elapsed_uncertainty": {"conservative_s": elapsed, "provisional": False},
        }

    result = latest_departure(predictor, "2026-10-11T18:00Z", "2026-10-11T12:00Z")
    assert (
        abs((pd.Timestamp(result["departure_utc"]) - pd.Timestamp("2026-10-11T16:00Z")).total_seconds())
        <= 60
    )
    assert len(calls) > 9 and not result["provisional"]


def test_latest_departure_can_use_expected_criterion():
    def predictor(departure):
        return {
            "elapsed_s": 3600,
            "elapsed_uncertainty": {"conservative_s": 7200, "median_s": 5400, "provisional": False},
        }

    result = latest_departure(predictor, "2026-10-11T18:00Z", "2026-10-11T12:00Z", criterion="expected")
    assert (
        abs((pd.Timestamp(result["departure_utc"]) - pd.Timestamp("2026-10-11T17:00Z")).total_seconds())
        <= 60
    )
    assert result["provisional"] and result["criterion"] == "expected"


def test_probability_requires_nominal_coverage_not_just_sample_count():
    rows = [
        {
            "date": (pd.Timestamp("2025-01-01T00:00Z") + pd.Timedelta(days=i)).isoformat(),
            "distance_km": 50,
            "actual_elapsed_s": 4000,
            "predicted_elapsed_s": 3600,
        }
        for i in range(70)
    ]
    unsupported = ElapsedCalibration(rows, 0.7, 0.85).estimate(3600, 50, "2026-01-01T00:00Z")
    assert unsupported["provisional"] and cutoff_probability(unsupported, 5000) is None
    supported = ElapsedCalibration(rows, 0.8, 0.9).estimate(3600, 50, "2026-01-01T00:00Z")
    assert not supported["provisional"]
    assert cutoff_probability(supported, 5000) == 1


@pytest.mark.parametrize(
    "distance,expected",
    [
        (49.9, "under 50 km"),
        (50, "50–100 km"),
        (100, "100–150 km"),
        (150, "150–200 km"),
        (201, "over 200 km"),
    ],
)
def test_long_distance_groups(distance, expected):
    assert distance_group(distance) == expected
