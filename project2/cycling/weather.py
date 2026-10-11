"""Cached ECCC station observations. GPS stays local; remote queries use coarse catalog bounds.

Historical wind is from-direction in tens of degrees, speed km/h, pressure kPa.
Station winds (usually 10 m) are environmental predictors, not cyclist-height measurements.
"""

from datetime import timezone, timedelta
import hashlib
import io
import json
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

API = "https://api.weather.gc.ca/collections"
ARCHIVE = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
VERSION = "eccc-1"
WEATHER_NUMERIC = [
    "headwind_kmh",
    "crosswind_kmh",
    "crosswind_abs_kmh",
    "wind_speed_kmh",
    "air_temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "pressure_kpa",
]
VALUES = [
    "wind_north_kmh",
    "wind_east_kmh",
    "air_temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "pressure_kpa",
]
FIELDS = {
    "TEMP": ("air_temperature_c", -80, 60),
    "RELATIVE_HUMIDITY": ("humidity_pct", 0, 100),
    "PRECIP_AMOUNT": ("precipitation_mm", 0, 300),
    "STATION_PRESSURE": ("pressure_kpa", 50, 110),
    "WIND_SPEED": ("wind_speed_kmh", 0, 250),
    "WIND_DIRECTION": ("wind_from_deg", 0, 36),
}
OFFSETS = {"PST": -8, "MST": -7, "CST": -6, "EST": -5, "AST": -4, "NST": -3.5}


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def wind_components(speed, from_deg, bearing_deg):
    speed, direction, bearing = np.broadcast_arrays(speed, from_deg, bearing_deg)
    angle = np.radians(direction - bearing)
    head, cross = speed * np.cos(angle), speed * np.sin(angle)
    return np.where(speed == 0, 0, head), np.where(speed == 0, 0, cross)


def travel_bearing(frame):
    a, b = np.radians(frame.start_lat), np.radians(frame.end_lat)
    dl = np.radians(frame.end_lon - frame.start_lon)
    return (
        np.degrees(
            np.arctan2(
                np.sin(dl) * np.cos(b), np.cos(a) * np.sin(b) - np.sin(a) * np.cos(b) * np.cos(dl)
            )
        )
        % 360
    )


def distances_km(lat, lon, station_lat, station_lon):
    a, b = np.radians(lat), np.radians(station_lat)
    h = np.sin((a - b) / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(np.radians(lon - station_lon) / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def parse_observation(properties, source="ECCC climate-hourly"):
    row = {
        "time": pd.to_datetime(properties.get("UTC_DATE"), utc=True, errors="coerce"),
        "station_id": str(properties.get("STN_ID", "")),
        "source": source,
        "description": properties.get("WEATHER_ENG_DESC") or "",
        "quality": 1.0,
    }
    flags = {}
    for field, (name, low, high) in FIELDS.items():
        value = number(properties.get(field))
        flag = str(properties.get(field + "_FLAG") or "").strip()
        flags[field] = flag
        # Unknown quality codes are retained, but excluded rather than guessed valid.
        if flag not in ("", "E", "T") or not low <= value <= high:
            value = np.nan
        if flag == "E":
            row["quality"] = min(row["quality"], 0.7)
        row[name] = value
    row["wind_from_deg"] *= 10
    speed, direction = row["wind_speed_kmh"], row["wind_from_deg"]
    row["calm"] = bool(speed == 0)
    if direction == 0 and speed > 0:
        direction = row["wind_from_deg"] = np.nan
    row["wind_north_kmh"] = 0.0 if speed == 0 else speed * np.cos(np.radians(direction))
    row["wind_east_kmh"] = 0.0 if speed == 0 else speed * np.sin(np.radians(direction))
    row["quality_flags"] = json.dumps(flags)
    return row


def parse_archive(content, station):
    frame = pd.read_csv(io.StringIO(content), encoding="utf-8-sig")
    offset = OFFSETS.get(station.get("timezone"))
    if offset is None:
        raise ValueError("Archive local-standard timezone is unknown; refusing to guess UTC.")
    columns = {
        "Temp (°C)": "TEMP",
        "Rel Hum (%)": "RELATIVE_HUMIDITY",
        "Precip. Amount (mm)": "PRECIP_AMOUNT",
        "Stn Press (kPa)": "STATION_PRESSURE",
        "Wind Spd (km/h)": "WIND_SPEED",
        "Wind Dir (10s deg)": "WIND_DIRECTION",
    }
    rows = []
    for raw in frame.to_dict("records"):
        stamp = pd.Timestamp(raw["Date/Time (LST)"]).tz_localize(timezone(timedelta(hours=offset)))
        props = {
            "UTC_DATE": stamp.tz_convert("UTC"),
            "STN_ID": station["id"],
            "WEATHER_ENG_DESC": raw.get("Weather") if pd.notna(raw.get("Weather")) else "",
        }
        for column, field in columns.items():
            props[field] = raw.get(column)
            flag = raw.get(column.split(" (")[0] + " Flag")
            props[field + "_FLAG"] = flag if pd.notna(flag) else None
        rows.append(parse_observation(props, "ECCC downloadable historical archive"))
    return pd.DataFrame(rows)


class ECCCClient:
    def __init__(self, root, settings, session=None):
        self.root = root / "weather" / "requests"
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings, self.session = settings, session or requests.Session()
        self.session.headers.update({"User-Agent": "PersonalCyclingPredictor/0.3 (cached ECCC access)"})
        self.requests, self.cache_hits, self.errors, self.last_request = 0, 0, [], 0.0

    def request(self, url, params=None, ttl=86400 * 180, refresh=False, text=False):
        if urlparse(url).hostname not in ("api.weather.gc.ca", "climate.weather.gc.ca"):
            raise ValueError("Untrusted weather endpoint.")
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
        path = self.root / (key + ".json")
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if (
            cached
            and not refresh
            and (time.time() - cached["fetched"] < ttl or self.settings.weather_offline)
        ):
            self.cache_hits += 1
            return cached["value"]
        if self.settings.weather_offline:
            self.errors.append("Offline: required ECCC records are not cached.")
            return None
        if self.requests >= self.settings.weather_max_requests:
            self.errors.append("ECCC request budget reached; cached data remain available.")
            return None
        for attempt in range(3):
            if self.requests >= self.settings.weather_max_requests:
                self.errors.append("ECCC request budget reached during retry.")
                break
            time.sleep(
                max(0, self.settings.weather_min_interval_s - (time.monotonic() - self.last_request))
            )
            self.last_request = time.monotonic()
            self.requests += 1
            try:
                response = self.session.get(url, params=params, timeout=(10, 50))
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(
                        min(
                            20,
                            number(response.headers.get("Retry-After"))
                            if response.headers.get("Retry-After", "").isdigit()
                            else 2**attempt,
                        )
                    )
                    response.raise_for_status()
                response.raise_for_status()
                value = response.text if text else response.json()
                if len(response.content) > 25_000_000:
                    raise ValueError("ECCC response exceeds the bounded payload limit.")
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"fetched": time.time(), "url": url, "params": params, "value": value}),
                    encoding="utf-8",
                )
                temporary.replace(path)
                return value
            except (requests.RequestException, ValueError) as exc:
                if attempt == 2:
                    self.errors.append(f"ECCC retrieval failed: {type(exc).__name__}")
        # A transient failure is never saved as a successful empty response.
        return cached["value"] if cached else None

    def features(self, collection, params, ttl=86400 * 180, refresh=False):
        url, output, seen = f"{API}/{collection}/items", [], set()
        params = {"f": "json", "limit": 1000, **params}
        for _ in range(12):
            data = self.request(url, params, ttl, refresh)
            if data is None:
                return output, False
            output.extend(data.get("features", []))
            next_url = next((x["href"] for x in data.get("links", []) if x.get("rel") == "next"), None)
            if not next_url:
                return output, True
            if next_url in seen:
                break
            seen.add(next_url)
            url, params = next_url, None
        self.errors.append("ECCC pagination limit reached; coverage is partial.")
        return output, False

    def stations(self, frame, refresh=False):
        # Round outward to whole degrees. This is station-metadata discovery, not trace upload.
        bounds = [
            np.floor(frame.lon.min()) - 1,
            np.floor(frame.lat.min()) - 1,
            np.ceil(frame.lon.max()) + 1,
            np.ceil(frame.lat.max()) + 1,
        ]
        features, _ = self.features(
            "climate-stations",
            {"bbox": ",".join(map(str, bounds)), "HAS_HOURLY_DATA": "Y"},
            ttl=86400 * 30,
            refresh=refresh,
        )
        stations = []
        for feature in features:
            p, coords = feature["properties"], feature["geometry"]["coordinates"]
            if not p.get("HLY_FIRST_DATE"):
                continue
            stations.append(
                {
                    "id": str(p["STN_ID"]),
                    "climate_id": p["CLIMATE_IDENTIFIER"],
                    "name": p["STATION_NAME"],
                    "lon": coords[0],
                    "lat": coords[1],
                    "elevation": number(p.get("ELEVATION")),
                    "timezone": p.get("TIMEZONE"),
                    "first": p["HLY_FIRST_DATE"],
                    "last": p.get("HLY_LAST_DATE"),
                }
            )
        return stations

    def month(self, station, year, month, refresh=False):
        begin = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        end = begin + pd.offsets.MonthBegin(1)
        ttl = 21600 if end > pd.Timestamp.now(tz="UTC") else 86400 * 180
        items, complete = self.features(
            "climate-hourly",
            {
                "STN_ID": station["id"],
                "datetime": f"{begin.isoformat()}/{(end - pd.Timedelta(seconds=1)).isoformat()}",
            },
            ttl,
            refresh,
        )
        rows = pd.DataFrame([parse_observation(x["properties"]) for x in items])
        if complete and (
            rows.empty or not rows.get("wind_speed_kmh", pd.Series(dtype=float)).notna().any()
        ):
            # API exposes a station subset. Official monthly CSV is a real fallback, not an empty marker.
            content = self.request(
                ARCHIVE,
                {
                    "format": "csv",
                    "stationID": station["id"],
                    "Year": year,
                    "Month": month,
                    "Day": 1,
                    "timeframe": 1,
                    "submit": "Download Data",
                },
                ttl,
                refresh,
                text=True,
            )
            if content:
                try:
                    archive = parse_archive(content, station)
                    rows = pd.concat([rows, archive], ignore_index=True)
                except (ValueError, KeyError, pd.errors.ParserError) as exc:
                    self.errors.append(
                        f"Archive unavailable for station {station['id']}: {type(exc).__name__}"
                    )
        if rows.empty:
            return rows
        return rows.dropna(subset=["time"]).drop_duplicates("time", keep="last").sort_values("time")


def interpolate_observations(observations, times):
    """Vector interpolation; no long gap bridging, no circular-direction averaging."""
    times = pd.to_datetime(times, utc=True, errors="coerce")
    result = pd.DataFrame(index=np.arange(len(times)))
    for field in VALUES:
        result[field] = np.nan
    result["age_minutes"], result["quality"] = np.nan, 0.0
    result["interpolated"], result["description"], result["quality_flags"] = False, "", ""
    result["observation_times"] = ""
    if observations.empty:
        return result
    obs = observations.sort_values("time").reset_index(drop=True)
    x = pd.to_datetime(obs.time, utc=True).dt.as_unit("ns").astype("int64").to_numpy() / 1e9
    t = times.dt.as_unit("ns").astype("int64").to_numpy() / 1e9
    right = np.clip(np.searchsorted(x, t), 0, len(x) - 1)
    left = np.clip(right - 1, 0, len(x) - 1)
    left = np.where(x[right] <= t, right, left)
    near = np.where(abs(t - x[left]) <= abs(t - x[right]), left, right)
    age = abs(t - x[near]) / 60
    bracket = (x[left] <= t) & (x[right] >= t) & (x[right] - x[left] <= 5400) & (left != right)
    alpha = np.divide(t - x[left], x[right] - x[left], out=np.zeros(len(t)), where=left != right)
    for field in VALUES:
        y = pd.to_numeric(obs[field], errors="coerce").to_numpy()
        valid = bracket & np.isfinite(y[left]) & np.isfinite(y[right])
        # Precipitation is an hourly accumulation, not a smoothly changing instantaneous rate.
        if field == "precipitation_mm":
            valid[:] = False
        values = np.where(valid, y[left] * (1 - alpha) + y[right] * alpha, y[near])
        result[field] = np.where((age <= 60) & times.notna(), values, np.nan)
    result["age_minutes"] = age
    result["quality"] = np.where(age <= 60, obs.quality.to_numpy()[near], 0)
    result["interpolated"] = bracket
    for column in ("description", "quality_flags"):
        result[column] = obs[column].fillna("").to_numpy()[near]
    result["observation_times"] = [
        f"{obs.time.iloc[a].isoformat()}|{obs.time.iloc[b].isoformat()}"
        if flag
        else obs.time.iloc[n].isoformat()
        for a, b, n, flag in zip(left, right, near, bracket)
    ]
    return result


def neutral_weather(frame):
    result = frame.copy()
    for column in WEATHER_NUMERIC + ["wind_north_kmh", "wind_east_kmh", "wind_from_deg"]:
        result[column] = np.nan
    result["weather_available"] = False
    result["weather_confidence"] = 0.0
    result["weather_source"] = "unavailable / weather-neutral terrain baseline"
    return result


class HistoricalWeather:
    def __init__(self, root, settings, client=None, details=True):
        self.settings, self.client = settings, client or ECCCClient(root, settings)
        self.details = details
        self.stations, self.months = [], {}

    def prepare(self, frame, refresh=False):
        self.stations = self.client.stations(frame, refresh)

    def observations(self, station, months, refresh=False):
        parts = []
        for month in months:
            key = station["id"], month.year, month.month
            if key not in self.months:
                self.months[key] = self.client.month(station, month.year, month.month, refresh)
            parts.append(self.months[key])
        return pd.concat(parts, ignore_index=True)

    def at(self, frame, times, refresh=False):
        output = neutral_weather(frame).reset_index(drop=True)
        output["weather_station_ids"], output["weather_description"] = "", ""
        output["weather_wind_station_ids"], output["weather_station_distances_km"] = "", "{}"
        output["weather_station_distance_km"], output["weather_age_minutes"] = np.nan, np.nan
        output["weather_uncertainty_kmh"], output["weather_interpolated"] = np.nan, False
        output["weather_observation_times"], output["weather_quality_flags"] = "", ""
        times = pd.to_datetime(times, utc=True, errors="coerce").reset_index(drop=True)
        output["weather_time_utc"] = times
        if times.notna().sum() == 0 or not self.stations:
            return output
        stations = [
            s
            for s in self.stations
            if pd.Timestamp(s["first"], tz="UTC") <= times.max()
            and (
                not s["last"] or pd.Timestamp(s["last"], tz="UTC") + pd.Timedelta(days=2) >= times.min()
            )
        ]
        if not stations:
            return output
        distances = np.column_stack(
            [
                distances_km(frame.lat.to_numpy(), frame.lon.to_numpy(), s["lat"], s["lon"])
                for s in stations
            ]
        )
        elevation = frame.get("elevation_m", pd.Series(np.nan, index=frame.index)).to_numpy()
        delta = np.column_stack([np.abs(elevation - s["elevation"]) for s in stations])
        score = distances + np.nan_to_num(delta, nan=0) / 100
        order = np.argsort(score, axis=1)[:, : self.settings.weather_candidates]
        candidates = sorted(
            set(
                order[
                    distances[np.arange(len(frame))[:, None], order] <= self.settings.weather_radius_km
                ]
            )
        )
        values, weights, ages, interpolated, descriptions, flags, stamps, sources = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        selected_stations = []
        months = pd.period_range(
            times.min().tz_localize(None) - pd.Timedelta(hours=12),
            times.max().tz_localize(None) + pd.Timedelta(hours=12),
            freq="M",
        )
        for index in candidates:
            station = stations[index]
            obs = self.observations(station, months, refresh)
            interp = interpolate_observations(obs, times)
            candidate = (order == index).any(axis=1) & (
                distances[:, index] <= self.settings.weather_radius_km
            )
            # A nearby station without readings receives zero weight; other stations remain eligible.
            weight = (
                interp.quality.to_numpy()
                * np.exp(-np.nan_to_num(delta[:, index], nan=0) / 500)
                / np.maximum(5, distances[:, index]) ** 2
            )
            weight *= candidate
            values.append(interp[VALUES].to_numpy())
            weights.append(weight)
            ages.append(interp.age_minutes.to_numpy())
            interpolated.append(interp.interpolated.to_numpy())
            descriptions.append(interp.description.to_numpy())
            flags.append(interp.quality_flags.to_numpy())
            stamps.append(interp.observation_times.to_numpy())
            sources.append(" + ".join(obs.source.unique()) if len(obs) else "")
            selected_stations.append((station, index))
        if not values:
            return output
        value, weight = np.stack(values, axis=1), np.column_stack(weights)
        usable = np.isfinite(value[:, :, 0]) & np.isfinite(value[:, :, 1])
        # Only combine the best two valid stations; preserve field-specific missingness.
        ranked = np.argsort(-weight * usable, axis=1)
        keep = np.zeros_like(weight, dtype=bool)
        np.put_along_axis(keep, ranked[:, :2], True, axis=1)
        weight *= keep
        for k, field in enumerate(VALUES):
            w = weight * np.isfinite(value[:, :, k])
            den = w.sum(axis=1)
            output[field] = np.divide(
                (np.nan_to_num(value[:, :, k]) * w).sum(axis=1),
                den,
                out=np.full(len(frame), np.nan),
                where=den > 0,
            )
        wind_weight = weight * usable
        den = wind_weight.sum(axis=1)
        measurement_weight = weight * np.isfinite(value).any(axis=2)
        # Preserve wind-specific distance/age when wind is available, and retain
        # provenance for temperature-only or other scalar observations otherwise.
        provenance_weight = np.where(den[:, None] > 0, wind_weight, measurement_weight)
        provenance_den = provenance_weight.sum(axis=1)
        distance_matrix = np.column_stack([distances[:, i] for _, i in selected_stations])
        output["weather_station_distance_km"] = np.divide(
            (distance_matrix * provenance_weight).sum(axis=1),
            provenance_den,
            out=np.full(len(frame), np.nan),
            where=provenance_den > 0,
        )
        output["weather_age_minutes"] = np.divide(
            (np.nan_to_num(np.column_stack(ages)) * provenance_weight).sum(axis=1),
            provenance_den,
            out=np.full(len(frame), np.nan),
            where=provenance_den > 0,
        )
        output["weather_available"] = den > 0
        speed = np.hypot(output.wind_north_kmh, output.wind_east_kmh)
        direction = np.degrees(np.arctan2(output.wind_east_kmh, output.wind_north_kmh)) % 360
        output["wind_speed_kmh"], output["wind_from_deg"] = speed, direction
        bearing = frame.travel_bearing_deg if "travel_bearing_deg" in frame else travel_bearing(frame)
        head, cross = wind_components(speed, direction, bearing)
        output["headwind_kmh"], output["crosswind_kmh"], output["crosswind_abs_kmh"] = (
            head,
            cross,
            abs(cross),
        )
        spread = np.sqrt(
            np.nan_to_num(
                (
                    (value[:, :, 0] - output.wind_north_kmh.to_numpy()[:, None]) ** 2
                    + (value[:, :, 1] - output.wind_east_kmh.to_numpy()[:, None]) ** 2
                )
            )
            * wind_weight
        )
        output["weather_uncertainty_kmh"] = (
            np.divide((spread**2).sum(axis=1), den, out=np.zeros(len(frame)), where=den > 0) ** 0.5
            + 2
            + output.weather_station_distance_km * 0.12
        )
        output.loc[den == 0, "weather_uncertainty_kmh"] = np.nan
        output["weather_confidence"] = np.where(
            den > 0,
            np.exp(-output.weather_station_distance_km / 50) * np.exp(-output.weather_age_minutes / 180),
            0,
        )
        output["weather_interpolated"] = (np.column_stack(interpolated) & (measurement_weight > 0)).any(
            axis=1
        )
        if not self.details:
            return output
        for i in range(len(output)):
            used = np.flatnonzero(measurement_weight[i] > 0)
            output.at[i, "weather_station_ids"] = "|".join(selected_stations[j][0]["id"] for j in used)
            output.at[i, "weather_wind_station_ids"] = "|".join(
                selected_stations[j][0]["id"] for j in np.flatnonzero(wind_weight[i] > 0)
            )
            output.at[i, "weather_station_distances_km"] = json.dumps(
                {selected_stations[j][0]["id"]: float(distance_matrix[i, j]) for j in used}
            )
            output.at[i, "weather_source"] = (
                "; ".join(sorted({sources[j] for j in used})) or output.at[i, "weather_source"]
            )
            output.at[i, "weather_description"] = " | ".join(str(descriptions[j][i]) for j in used)
            output.at[i, "weather_quality_flags"] = " | ".join(str(flags[j][i]) for j in used)
            output.at[i, "weather_observation_times"] = "; ".join(str(stamps[j][i]) for j in used)
        return output


def segment_timestamps(points, segments):
    valid = points.time.notna() & points.distance_m.notna()
    p = points[valid]
    if len(p) < 2:
        return pd.Series(pd.NaT, index=segments.index, dtype="datetime64[ns, UTC]")
    # Segment labels already refer to the original cumulative geometry. No ride start-time shortcut.
    seconds = pd.to_datetime(p.time, utc=True).dt.as_unit("ns").astype("int64").to_numpy() / 1e9
    mids = (segments.start_m.to_numpy() + segments.end_m.to_numpy()) / 2
    result = pd.to_datetime(np.interp(mids, p.distance_m, seconds), unit="s", utc=True)
    # Do not bridge discontinuities or long recording gaps.
    gap = points[points.unobserved_s.gt(30)]
    bad = np.zeros(len(segments), dtype=bool)
    for row in gap.itertuples():
        bad |= (segments.start_m.to_numpy() <= row.distance_m) & (
            segments.end_m.to_numpy() >= row.distance_m - row.ds_m
        )
    result = pd.Series(result, index=segments.index)
    result.loc[bad] = pd.NaT
    return result


def enrich_history(store, settings, progress=None, refresh=False):
    rides, frames = store.rides(), []
    if rides.empty:
        raise ValueError("Import and process rides before weather enrichment.")
    for ride in rides.itertuples():
        segments = store.segments([ride.id])
        if len(segments):
            frames.append(segments)
    all_segments = pd.concat(frames, ignore_index=True)
    provider = HistoricalWeather(store.root, settings)
    provider.prepare(all_segments, refresh)
    station_path = store.root / "weather" / "stations.json"
    station_path.write_text(json.dumps(provider.stations, indent=2), encoding="utf-8")
    enriched, reused = 0, 0
    for i, ride in enumerate(rides.itertuples()):
        segments = store.segments([ride.id])
        if segments.empty:
            continue
        if (
            not refresh
            and ride._asdict().get("weather_version") == VERSION
            and "weather_available" in segments
            and "weather_wind_station_ids" in segments
            and segments.weather_available.mean() > 0.95
        ):
            reused += 1
            continue
        points = store.points(ride.id)
        segments["travel_bearing_deg"] = travel_bearing(segments)
        segments = provider.at(segments, segment_timestamps(points, segments), refresh)
        store.save_segments(ride.id, segments)
        store.update_metadata(
            ride.id,
            {"weather_version": VERSION, "weather_coverage": float(segments.weather_available.mean())},
        )
        enriched += 1
        if progress:
            progress(
                (i + 1) / len(rides),
                f"ECCC weather: {i + 1}/{len(rides)} rides, {provider.client.requests} requests, {provider.client.cache_hits} cached",
            )
    store.invalidate_model()
    report = weather_report(store)
    report.update(
        requests=provider.client.requests,
        cache_hits=provider.client.cache_hits,
        errors=sorted(set(provider.client.errors)),
        processed=enriched,
        reused=reused,
    )
    store.set_state("weather_coverage", report)
    (store.root / "reports" / "weather-coverage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def weather_report(store):
    segments = store.segments()
    if "weather_available" not in segments:
        return {"status": "not_enriched", "rides": 0, "segment_coverage": 0}
    valid = segments.weather_available.fillna(False)
    usable = segments[valid]
    wind_ids = usable.get("weather_wind_station_ids", usable.weather_station_ids)
    ids = sorted({x for value in wind_ids for x in value.split("|") if x})
    measurement_ids = sorted(
        {x for value in segments.weather_station_ids for x in value.split("|") if x}
    )
    return {
        "status": "complete",
        "rides": int(usable.ride_id.nunique()),
        "total_rides": int(segments.ride_id.nunique()),
        "stations": len(ids),
        "station_ids": ids,
        "measurement_stations": len(measurement_ids),
        "segments": len(segments),
        "usable_segments": int(valid.sum()),
        "segment_coverage": float(valid.mean()),
        "distance_coverage": float(usable.distance_m.sum() / segments.distance_m.sum()),
        "average_station_distance_km": float(
            np.average(usable.weather_station_distance_km, weights=usable.distance_m)
        )
        if len(usable)
        else None,
        "maximum_station_distance_km": float(usable.weather_station_distance_km.max())
        if len(usable)
        else None,
        "low_confidence_segments": int((usable.weather_confidence < 0.5).sum()),
        "missing_values": {k: int(segments[k].isna().sum()) for k in WEATHER_NUMERIC},
        "features": WEATHER_NUMERIC,
        "version": VERSION,
    }
