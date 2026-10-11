from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from cycling.config import Settings
from cycling.weather import (
    ECCCClient,
    HistoricalWeather,
    interpolate_observations,
    parse_archive,
    parse_observation,
    wind_components,
)


@pytest.mark.parametrize("bearing,head,cross", [(270, 20, 0), (90, -20, 0), (0, 0, -20)])
def test_westerly_components(bearing, head, cross):
    h, c = wind_components(20, 270, bearing)
    assert h == pytest.approx(head, abs=1e-10)
    assert c == pytest.approx(cross, abs=1e-10)


def obs(hour=12, **fields):
    return parse_observation(
        {
            "UTC_DATE": f"2025-06-01T{hour:02}:00:00",
            "STN_ID": 1,
            "WIND_SPEED": 20,
            "WIND_DIRECTION": 27,
            "TEMP": 15,
            **fields,
        }
    )


def test_parse_units_flags_calm_and_missing():
    assert obs()["wind_from_deg"] == 270
    assert obs(STATION_PRESSURE=100.1)["pressure_kpa"] == 100.1
    calm = obs(WIND_SPEED=0, WIND_DIRECTION=None)
    assert calm["calm"] and calm["wind_east_kmh"] == 0
    assert wind_components(0, np.nan, 90) == (0, 0)
    assert np.isnan(obs(WIND_SPEED=None)["wind_east_kmh"])
    assert np.isnan(obs(TEMP_FLAG="M")["air_temperature_c"])
    assert np.isnan(obs(WIND_DIRECTION=0)["wind_from_deg"])


def test_interpolation_circular_direction_no_long_gap():
    rows = pd.DataFrame([obs(12, WIND_DIRECTION=35), obs(13, WIND_DIRECTION=1)])
    values = interpolate_observations(
        rows, pd.Series(pd.to_datetime(["2025-06-01T12:30Z", "2025-06-01T16:00Z"]))
    )
    assert values.wind_north_kmh.iloc[0] > 19
    assert abs(values.wind_east_kmh.iloc[0]) < 1e-10
    assert values.interpolated.iloc[0]
    assert np.isnan(values.wind_north_kmh.iloc[1])


def test_archive_standard_time_ignores_dst():
    content = "Date/Time (LST),Temp (°C),Wind Spd (km/h),Wind Dir (10s deg)\n2025-07-01 12:00,20,15,27\n"
    rows = parse_archive(content, {"id": "1", "timezone": "EST"})
    assert rows.time.iloc[0] == pd.Timestamp("2025-07-01T17:00Z")
    local = datetime(2025, 7, 1, 12, tzinfo=ZoneInfo("America/Toronto"))
    assert pd.Timestamp(local).tz_convert("UTC") == pd.Timestamp("2025-07-01T16:00Z")


class Response:
    def __init__(self, value, status=200):
        self.value, self.status_code, self.headers, self.content = value, status, {}, b"{}"

    def json(self):
        return self.value

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError("failed")


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_station_discovery_pagination_cache_offline(tmp_path):
    feature = {
        "geometry": {"coordinates": [-79, 44]},
        "properties": {
            "STN_ID": 1,
            "STATION_NAME": "ONE",
            "CLIMATE_IDENTIFIER": "ID",
            "ELEVATION": 100,
            "TIMEZONE": "EST",
            "HLY_FIRST_DATE": "2000-01-01",
            "HLY_LAST_DATE": "2026-01-01",
        },
    }
    session = Session(
        [
            Response(
                {"features": [], "links": [{"rel": "next", "href": "https://api.weather.gc.ca/page2"}]}
            ),
            Response({"features": [feature]}),
        ]
    )
    settings = Settings(weather_min_interval_s=0)
    client = ECCCClient(tmp_path, settings, session)
    f = pd.DataFrame({"lat": [43.5], "lon": [-79.5]})
    assert client.stations(f)[0]["id"] == "1"
    assert len(session.calls) == 2
    assert client.stations(f)[0]["name"] == "ONE"
    assert len(session.calls) == 2
    settings.weather_offline = True
    assert client.stations(f)[0]["id"] == "1"
    assert client.request("https://api.weather.gc.ca/uncached") is None
    with pytest.raises(ValueError):
        client.request("https://example.com/gps")


def test_station_selection_skips_missing_and_respects_distance(tmp_path):
    settings = Settings(weather_candidates=3, weather_radius_km=50, weather_offline=True)
    provider = HistoricalWeather(tmp_path, settings)
    provider.stations = [
        {
            "id": str(i),
            "lat": 44 + dy,
            "lon": -79,
            "elevation": 100,
            "first": "2000-01-01",
            "last": "2026-01-01",
        }
        for i, dy in [(1, 0), (2, 0.1), (3, 2)]
    ]
    provider.months = {
        ("1", 2025, 6): pd.DataFrame([obs(WIND_SPEED=None)]),
        ("2", 2025, 6): pd.DataFrame([obs()]),
    }
    f = pd.DataFrame(
        {
            "lat": [44],
            "lon": [-79],
            "elevation_m": [100],
            "start_lat": [44],
            "end_lat": [44],
            "start_lon": [-79],
            "end_lon": [-79.001],
        }
    )
    result = provider.at(f, pd.Series([pd.Timestamp("2025-06-01T12:00Z")]))
    assert result.weather_station_ids.iloc[0] == "1|2"
    assert result.weather_wind_station_ids.iloc[0] == "2"
    assert result.headwind_kmh.iloc[0] == pytest.approx(20, abs=0.001)
    assert result.weather_station_distance_km.iloc[0] < 12


def test_scalar_weather_retains_provenance_without_wind(tmp_path):
    import json

    provider = HistoricalWeather(tmp_path, Settings(weather_offline=True))
    provider.stations = [
        {"id": "1", "lat": 44, "lon": -79, "elevation": 100, "first": "2000-01-01", "last": "2026-01-01"}
    ]
    provider.months = {("1", 2025, 6): pd.DataFrame([obs(WIND_SPEED=None)])}
    frame = pd.DataFrame({"lat": [44], "lon": [-79], "elevation_m": [100], "travel_bearing_deg": [90]})
    result = provider.at(frame, pd.Series([pd.Timestamp("2025-06-01T12:00Z")]))
    assert result.air_temperature_c.iloc[0] == 15
    assert not result.weather_available.iloc[0]
    assert np.isnan(result.wind_speed_kmh.iloc[0])
    assert np.isnan(result.weather_uncertainty_kmh.iloc[0])
    assert result.weather_confidence.iloc[0] == 0
    assert result.weather_station_ids.iloc[0] == "1"
    assert result.weather_wind_station_ids.iloc[0] == ""
    assert result.weather_station_distance_km.iloc[0] == 0
    assert json.loads(result.weather_station_distances_km.iloc[0]) == {"1": 0}
    assert "2025-06-01" in result.weather_observation_times.iloc[0]


def test_partial_failure_retry_does_not_poison_cache(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr("cycling.weather.time.sleep", lambda seconds: None)
    session = Session(
        [Response({}, 503), Response({}, 503), Response({}, 503), Response({"features": []})]
    )
    client = ECCCClient(tmp_path, Settings(weather_min_interval_s=0), session)
    assert client.request("https://api.weather.gc.ca/test") is None
    assert client.errors
    assert not list(client.root.glob("*.json"))
    assert client.request("https://api.weather.gc.ca/test") == {"features": []}
    assert client.request("https://api.weather.gc.ca/test") == {"features": []}
    assert len(session.calls) == 4
    assert issubclass(requests.HTTPError, requests.RequestException)


def test_api_subset_falls_back_to_official_archive(tmp_path, monkeypatch):
    client = ECCCClient(tmp_path, Settings(weather_min_interval_s=0), Session([]))
    monkeypatch.setattr(client, "features", lambda *a, **k: ([], True))
    seen = []

    def request(url, params, *args, **kwargs):
        seen.append((url, params))
        return (
            "Date/Time (LST),Temp (°C),Wind Spd (km/h),Wind Dir (10s deg)\n2025-07-01 12:00,20,15,27\n"
        )

    monkeypatch.setattr(client, "request", request)
    rows = client.month({"id": "1", "timezone": "EST"}, 2025, 7)
    assert len(rows) == 1 and rows.wind_from_deg.iloc[0] == 270
    assert "climate.weather.gc.ca" in seen[0][0]
    assert seen[0][1]["stationID"] == "1"


def test_retry_respects_request_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("cycling.weather.time.sleep", lambda seconds: None)
    session = Session([Response({}, 503)])
    client = ECCCClient(tmp_path, Settings(weather_min_interval_s=0, weather_max_requests=1), session)
    assert client.request("https://api.weather.gc.ca/test") is None
    assert client.requests == len(session.calls) == 1


def test_archive_utc_month_boundary_uses_previous_local_month(tmp_path):
    settings = Settings(weather_offline=True)
    provider = HistoricalWeather(tmp_path, settings)
    station = {
        "id": "1",
        "lat": 44,
        "lon": -79,
        "elevation": 100,
        "timezone": "EST",
        "first": "2000-01-01",
        "last": "2026-01-01",
    }
    provider.stations = [station]
    csv = "Date/Time (LST),Temp (°C),Wind Spd (km/h),Wind Dir (10s deg)\n2025-06-30 22:00,20,20,27\n"
    provider.months[("1", 2025, 6)] = parse_archive(csv, station)
    provider.months[("1", 2025, 7)] = pd.DataFrame()
    f = pd.DataFrame(
        {
            "lat": [44],
            "lon": [-79],
            "start_lat": [44],
            "end_lat": [44],
            "start_lon": [-79],
            "end_lon": [-79.001],
        }
    )
    result = provider.at(f, pd.Series([pd.Timestamp("2025-07-01T03:00Z")]))
    assert result.weather_available.iloc[0]
    assert result.headwind_kmh.iloc[0] == pytest.approx(20, abs=0.001)
    assert "archive" in result.weather_source.iloc[0]


def test_city_forecast_compass_and_pop_not_rainfall():
    from cycling.forecast import parse_city

    p = {
        "properties": {
            "identifier": "on-test",
            "hourlyForecastGroup": {
                "timestamp": {"en": "2025-06-01T12:00Z"},
                "hourlyForecasts": [
                    {
                        "timestamp": "2025-06-01T13:00Z",
                        "temperature": {"value": {"en": 20}, "units": {"en": "C"}},
                        "wind": {
                            "speed": {"value": {"en": 20}, "units": {"en": "km/h"}},
                            "direction": {"value": {"en": "W"}},
                        },
                        "lop": {"value": {"en": 50}},
                    }
                ],
            },
        }
    }
    rows = parse_city(p)
    assert rows.wind_from_deg.iloc[0] == 270
    assert np.isnan(rows.precipitation_mm.iloc[0])
    p["properties"]["hourlyForecastGroup"]["hourlyForecasts"][0]["wind"]["direction"]["value"]["en"] = (
        "N"
    )
    assert parse_city(p).wind_north_kmh.iloc[0] == pytest.approx(20)
    p["properties"]["hourlyForecastGroup"]["hourlyForecasts"][0]["wind"]["direction"]["value"]["en"] = (
        "VR"
    )
    assert np.isnan(parse_city(p).wind_east_kmh.iloc[0])


def test_weather_model_neutral_fallback_and_prefix(synthetic_dataset, settings):
    from cycling.model import fit_model
    from cycling.weather import WEATHER_NUMERIC, neutral_weather

    rides, segments, _ = synthetic_dataset
    for i, k in enumerate(WEATHER_NUMERIC):
        segments[k] = i + segments.gradient_pct
    segments["weather_available"] = True
    model = fit_model(rides.iloc[:6], segments, [], weather=True)
    assert model.train_ids == list(map(str, range(6)))
    f = neutral_weather(segments.iloc[:20])
    np.testing.assert_allclose(model.predict_cruise(f), model.neutral_model.predict_cruise(f))
    assert np.isfinite(model.predict_cruise(segments.iloc[:20])).all()


def test_iterative_forecast_uses_segment_arrivals(synthetic_dataset, settings, monkeypatch):
    from cycling.model import fit_model
    from cycling.prediction import predict_route
    from cycling.validation import held_out_route
    from cycling.weather import neutral_weather

    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    route, events = held_out_route(traces["9"], rides.iloc[-1], settings)
    original = model.predict_cruise

    def wind_sensitive(f, hr=None, use_fitness=False):
        wind = f.get("headwind_kmh", pd.Series(0.0, index=f.index)).fillna(0).to_numpy()
        return original(f, hr, use_fitness) * (1 - 0.01 * wind)

    monkeypatch.setattr(model, "predict_cruise", wind_sensitive)

    class Provider:
        note = "Mock forecast"
        client = type("Client", (), {"errors": []})()

        def __init__(self):
            self.calls = []

        def at(self, f, t):
            self.calls.append(t.copy())
            result = neutral_weather(f)
            result["headwind_kmh"] = 10 + t.dt.minute / 60
            result["weather_available"] = True
            return result

    provider = Provider()
    result = predict_route(
        model,
        route,
        events,
        departure="2025-06-01T12:00Z",
        weather_provider=provider,
        personal_break_minutes=5,
    )
    assert len(provider.calls) >= 2
    assert provider.calls[-1].is_monotonic_increasing
    assert provider.calls[-1].max() - provider.calls[-1].min() > pd.Timedelta(minutes=10)
    assert result["elapsed_s"] == pytest.approx(result["moving_s"] + result["stationary_s"] + 300)
