"""Adaptive segments with accumulated pacing/fatigue features and separate cruise labels."""

import numpy as np
import pandas as pd


def build_segments(points, ride, events=(), historical=True):
    df = points.copy()
    event_zone = np.zeros(len(df), dtype=bool)
    for event in events:
        event_zone |= (
            (df.distance_m >= event["start_m"])
            & (df.distance_m < event["end_m"])
            & (df.group == event["group"])
        ).to_numpy()
    df["event_zone"] = event_zone
    acceleration = df.speed_mps.diff().to_numpy() / np.maximum(0.1, df.dt_s.to_numpy())
    df["acceleration"] = acceleration
    rows = []
    total = float(df.ds_m.sum())
    distances = df.ds_m.to_numpy()
    gradients = df.gradient_pct.to_numpy()
    surfaces = df.surface.to_numpy()
    zones = df.event_zone.to_numpy()
    hearts = df.hr.to_numpy()
    times = df.elapsed_cumulative_s.to_numpy()
    for _, subset in df.groupby("group", sort=False):
        indices = list(subset.index)
        if len(indices) < 2:
            continue
        start = 1
        accumulated = 0.0
        for j in range(1, len(indices)):
            ix, prev = indices[j], indices[j - 1]
            accumulated += float(distances[ix])
            first = indices[start]
            changed = (
                surfaces[ix] != surfaces[prev]
                or abs(gradients[ix] - gradients[first]) > 3
                or zones[ix] != zones[prev]
                or (
                    np.isfinite(hearts[ix])
                    and np.isfinite(hearts[first])
                    and abs(hearts[ix] - hearts[first]) > 8
                )
            )
            elapsed = float(times[ix] - times[first])
            finish = (
                j == len(indices) - 1
                or accumulated >= 220
                or (accumulated >= 35 and (changed or elapsed >= 35))
            )
            if not finish:
                continue
            window = df.loc[indices[start : j + 1]]
            ds = float(window.ds_m.sum())
            if ds < 8:
                start, accumulated = j + 1, 0.0
                continue
            weights = window.ds_m.to_numpy()

            def mean(column, fallback=0.0):
                values = window[column].to_numpy(dtype=float)
                good = np.isfinite(values) & (weights > 0)
                return float(np.average(values[good], weights=weights[good])) if good.any() else fallback

            def category(column):
                values, inverse = np.unique(window[column].to_numpy().astype(str), return_inverse=True)
                return str(values[np.bincount(inverse, weights=weights).argmax()])

            moving_s = float(window.moving_s.sum())
            hr_weights = window.moving_s.to_numpy() if historical else weights
            hr_values = window.hr.to_numpy()
            valid_hr = np.isfinite(hr_values) & (hr_weights > 0)
            measured_hr = (
                float(np.average(hr_values[valid_hr], weights=hr_weights[valid_hr]))
                if valid_hr.any()
                else np.nan
            )
            moving_fraction = moving_s / max(1.0, float(window.dt_s.sum()))
            stable = (
                mean("speed_mps", 0) >= 1.5
                and (window.acceleration.abs().fillna(0) > 0.55).mean() < 0.25
            )
            # Sharp bends and interruption windows must not train the cruising speed target.
            cruise = bool(
                not window.event_zone.any()
                and mean("turn_angle_deg") < 25
                and stable
                and moving_fraction > 0.9
                and window.unobserved_s.sum() < 2
            )
            midpoint = window.iloc[len(window) // 2]
            date = ride.get("date") or "2000-01-01T00:00:00+00:00"
            date_value = pd.Timestamp(date)
            rows.append(
                {
                    "ride_id": str(ride.get("id", "route")),
                    "date": str(date),
                    "distance_m": ds,
                    "start_m": float(window.distance_m.iloc[0] - window.ds_m.iloc[0]),
                    "end_m": float(window.distance_m.iloc[-1]),
                    "lat": float(midpoint.lat),
                    "lon": float(midpoint.lon),
                    "start_lat": float(df.lat.iloc[indices[max(0, start - 1)]]),
                    "start_lon": float(df.lon.iloc[indices[max(0, start - 1)]]),
                    "end_lat": float(window.lat.iloc[-1]),
                    "end_lon": float(window.lon.iloc[-1]),
                    "gradient_pct": mean("gradient_pct"),
                    "elevation_m": mean("elevation_smooth", np.nan),
                    "elevation_missing": bool(window.elevation_missing.mean() > 0.5),
                    "surface": category("surface"),
                    "surface_source": category("surface_source"),
                    "surface_raw": category("surface_raw"),
                    "smoothness": category("smoothness"),
                    "tracktype": category("tracktype"),
                    "highway": category("highway"),
                    "bicycle": category("bicycle"),
                    "osm_tags": category("osm_tags"),
                    "map_confidence": mean("map_confidence"),
                    "surface_confidence": mean("surface_confidence"),
                    "planned_km": total / 1000,
                    "completed_km": float(midpoint.distance_m) / 1000,
                    "progress": float(midpoint.distance_m) / max(1, total),
                    "gain_m": float(midpoint.gain_cumulative_m),
                    "moving_hours": float(midpoint.moving_cumulative_s) / 3600,
                    "observation_day": float(date_value.toordinal()),
                    "day_of_year": date_value.dayofyear,
                    "bike": str(ride.get("bike", "unknown")),
                    "measured_hr": measured_hr,
                    "hr_measured": bool(valid_hr.any()),
                    "power": mean("power", np.nan),
                    "cadence": mean("cadence", np.nan),
                    "temperature": mean("temperature", np.nan),
                    "turn_angle_deg": mean("turn_angle_deg"),
                    "curve_radius_m": mean("curve_radius_m"),
                    "event_zone": bool(window.event_zone.any()),
                    "is_cruise": cruise,
                    "moving_s": moving_s,
                    "stationary_s": float(window.stationary_s.sum()),
                    "unobserved_s": float(window.unobserved_s.sum()),
                    "speed_kmh": ds / moving_s * 3.6 if moving_s > 0 else np.nan,
                }
            )
            start, accumulated = j + 1, 0.0
    return pd.DataFrame(rows)
