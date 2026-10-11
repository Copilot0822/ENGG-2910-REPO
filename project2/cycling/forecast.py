"""Initial modular ECCC forecast provider: nearby City Page hourly regional forecasts.

No gridded-resolution claim: each segment records its selected city's distance and issue
time. Outside the city's short hourly horizon, weather is unavailable, never invented.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from .weather import API, HistoricalWeather, distances_km, parse_observation

COMPASS = dict(zip("N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split(), np.arange(16) * 22.5))


def english(value, default=None):
    return value.get("en", default) if isinstance(value, dict) else value


def parse_city(feature):
    p = feature["properties"]
    group = p.get("hourlyForecastGroup") or {}
    issue = english(group.get("timestamp"))
    rows = []
    for raw in group.get("hourlyForecasts", []):
        wind = raw.get("wind", {})
        speed = english(wind.get("speed", {}).get("value"))
        unit = english(wind.get("speed", {}).get("units"))
        if unit not in ("km/h", None):
            speed = None
        direction = COMPASS.get(english(wind.get("direction", {}).get("value")), np.nan)
        # Historical code 0 means calm; a forecast's compass N means 360 degrees.
        if direction == 0:
            direction = 360.0
        temperature = english(raw.get("temperature", {}).get("value"))
        if english(raw.get("temperature", {}).get("units")) not in ("C", None):
            temperature = None
        row = parse_observation(
            {
                "UTC_DATE": raw.get("timestamp"),
                "STN_ID": p["identifier"],
                "WIND_SPEED": speed,
                "WIND_DIRECTION": direction / 10,
                "TEMP": temperature,
                "WEATHER_ENG_DESC": english(raw.get("condition"), ""),
            },
            "ECCC City Page hourly regional forecast",
        )
        row["forecast_issue_utc"] = issue
        # POP is not a measured precipitation amount; leave accumulation missing.
        rows.append(row)
    return pd.DataFrame(rows)


class CityForecast(HistoricalWeather):
    def __init__(self, root, settings, client=None):
        # One selected city per segment; interpolation is temporal, not a fabricated grid.
        super().__init__(
            root,
            replace(
                settings, weather_candidates=1, weather_radius_km=min(50, settings.weather_radius_km)
            ),
            client,
        )
        self.forecasts = {}
        self.note = "ECCC City Page hourly regional forecast, nearest city within 50 km; short horizon, not a road-scale or gridded forecast. Wind and temperature are provided; humidity, rainfall amounts and station pressure remain missing."

    def prepare(self, frame, refresh=False):
        bounds = [
            np.floor(frame.lon.min()) - 1,
            np.floor(frame.lat.min()) - 1,
            np.ceil(frame.lon.max()) + 1,
            np.ceil(frame.lat.max()) + 1,
        ]
        items, _ = self.client.features(
            "citypageweather-realtime",
            {"bbox": ",".join(map(str, bounds))},
            ttl=1800,
            refresh=refresh,
        )
        candidates = []
        for feature in items:
            coords = feature["geometry"]["coordinates"]
            distance = distances_km(frame.lat.to_numpy(), frame.lon.to_numpy(), coords[1], coords[0])
            candidates.append((float(distance.min()), feature))
        self.stations = []
        for distance, feature in sorted(candidates, key=lambda x: x[0])[:12]:
            if distance > 50:
                continue
            identifier = str(feature["properties"]["identifier"])
            data = self.client.request(
                f"{API}/citypageweather-realtime/items/{feature['id']}",
                {"f": "json"},
                ttl=1800,
                refresh=refresh,
            )
            if not data:
                continue
            rows = parse_city(data)
            if rows.empty:
                continue
            # Only use genuinely future values relative to their issuance.
            issued = pd.to_datetime(rows.forecast_issue_utc, utc=True, errors="coerce")
            rows = rows[rows.time >= issued].reset_index(drop=True)
            if rows.empty:
                continue
            self.forecasts[identifier] = rows
            coords = data["geometry"]["coordinates"]
            self.stations.append(
                {
                    "id": identifier,
                    "name": english(data["properties"]["name"]),
                    "lat": coords[1],
                    "lon": coords[0],
                    "elevation": np.nan,
                    "first": rows.time.min().isoformat(),
                    "last": rows.time.max().isoformat(),
                }
            )

    def observations(self, station, months, refresh=False):
        return self.forecasts.get(station["id"], pd.DataFrame())

    def at(self, frame, times, refresh=False):
        result = super().at(frame, times, refresh)
        result["weather_forecast_issue_utc"] = ""
        for i, ids in enumerate(result.weather_station_ids):
            if not ids:
                continue
            obs = self.forecasts[ids.split("|")[0]]
            moment = pd.Timestamp(times.iloc[i])
            if moment < obs.time.min() or moment > obs.time.max():
                result.at[i, "weather_available"] = False
                from .weather import WEATHER_NUMERIC

                result.loc[i, WEATHER_NUMERIC] = np.nan
                result.at[i, "weather_source"] = "Outside ECCC hourly forecast horizon"
            else:
                result.at[i, "weather_forecast_issue_utc"] = str(obs.forecast_issue_utc.iloc[0])
        return result
