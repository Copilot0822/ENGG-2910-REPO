"""Trace cleaning and distance-based terrain geometry, without bridging track gaps."""

import numpy as np
import pandas as pd

EARTH_M = 6_371_008.8


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(np.asarray(lon2) - lon1)
    return (
        np.degrees(
            np.arctan2(
                np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
            )
        )
        + 360
    ) % 360


def angle_difference(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def clean_points(raw, settings, route=False):
    if len(raw) > settings.max_points:
        raise ValueError("Recording exceeds the configured point limit.")
    df = raw.copy().reset_index(drop=True)
    for col in ("lat", "lon", "elevation", "speed", "distance", "hr", "cadence", "power", "temperature"):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "time" not in df:
        df["time"] = pd.NaT
    df["time"] = pd.to_datetime(df.time, utc=True, errors="coerce")
    if "track_group" not in df:
        df["track_group"] = 0
    valid = df.lat.between(-90, 90) & df.lon.between(-180, 180)
    valid &= ~((df.lat == 0) & (df.lon == 0))
    if not route:
        valid &= df.time.notna()
    rejected = int((~valid).sum())
    df = df.loc[valid].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(
            "At least two valid GPS points"
            + (" with timestamps" if not route else "")
            + " are required."
        )
    for col, lower, upper in (
        ("hr", 35, 230),
        ("cadence", 0, 250),
        ("power", 0, 2500),
        ("temperature", -40, 60),
        ("elevation", -500, 9000),
        ("speed", 0, settings.max_speed_mps),
    ):
        df.loc[~df[col].between(lower, upper), col] = np.nan
    dt = df.time.diff().dt.total_seconds().to_numpy(copy=True)
    boundary = df.track_group.ne(df.track_group.shift()).to_numpy(copy=True)
    if not route:
        boundary |= ~np.isfinite(dt) | (dt <= 0)
    dist = haversine(df.lat.shift().fillna(df.lat), df.lon.shift().fillna(df.lon), df.lat, df.lon)
    gps_speed = np.divide(dist, dt, out=np.zeros(len(df)), where=np.isfinite(dt) & (dt > 0))
    jump = (gps_speed > settings.max_speed_mps) if not route else np.zeros(len(df), dtype=bool)
    # An implausible point invalidates both adjacent edges, rather than moving it to another road.
    boundary |= jump | np.roll(jump, 1)
    sensor_delta = df.distance.diff().to_numpy()
    sensor_ok = np.isfinite(sensor_delta) & (sensor_delta >= 0) & (sensor_delta < 5000)
    if not route:
        sensor_ok &= (
            np.divide(sensor_delta, dt, out=np.full(len(df), np.inf), where=dt > 0)
            < settings.max_speed_mps
        )
    edge_dist = np.where(sensor_ok, sensor_delta, dist)
    # GPS jitter at a reported stationary point should not create cycling distance.
    edge_dist[(df.speed.fillna(99).to_numpy() < 0.3) & (dist < 10)] = 0
    edge_dist[boundary] = 0
    df["ds_m"] = edge_dist
    df["distance_m"] = np.cumsum(edge_dist)
    df["group"] = np.cumsum(boundary)
    dt[boundary | ~np.isfinite(dt) | (dt < 0)] = 0
    df["dt_s"] = dt
    speed = df.speed.to_numpy()
    speed = np.where(np.isfinite(speed), speed, gps_speed)
    speed[boundary] = np.nan
    df["speed_mps"] = speed
    if route:
        df["moving_s"] = 0.0
        df["stationary_s"] = 0.0
        df["unobserved_s"] = 0.0
    else:
        gap = dt > settings.gap_seconds
        moving = (speed >= settings.moving_threshold_mps) & (edge_dist > 0)
        moving_s = np.where(moving & ~gap, dt, 0.0)
        # For sparse intervals estimate traversal only, keeping the rest explicitly unobserved.
        neighbor = pd.Series(speed).rolling(5, center=True, min_periods=1).median().to_numpy()
        estimated = np.divide(edge_dist, np.maximum(neighbor, settings.moving_threshold_mps))
        moving_s[gap & moving] = np.minimum(dt[gap & moving], estimated[gap & moving])
        explicit_stopped = df.get("timer_active", pd.Series(True, index=df.index)).eq(False).to_numpy()
        # Edges crossing a timer-stop event carry the documented stopped duration.
        timer_pause = df.get("timer_pause_s", pd.Series(0.0, index=df.index)).fillna(0).to_numpy()
        moving_s = np.minimum(moving_s, np.maximum(0, dt - timer_pause))
        stationary = np.where(~gap | explicit_stopped, np.maximum(0, dt - moving_s), timer_pause)
        stationary = np.minimum(stationary, dt - moving_s)
        df["moving_s"] = moving_s
        df["stationary_s"] = stationary
        df["unobserved_s"] = np.maximum(0, dt - moving_s - stationary)
    df["moving_cumulative_s"] = df.moving_s.cumsum()
    df["elapsed_cumulative_s"] = df.dt_s.cumsum()
    df["heading"] = bearing(df.lat.shift().fillna(df.lat), df.lon.shift().fillna(df.lon), df.lat, df.lon)
    df["heading"] = (
        df.heading.where(df.ds_m > 1).groupby(df.group).transform(lambda s: s.ffill().bfill()).fillna(0)
    )
    df["elevation_missing"] = df.elevation.isna()
    df["elevation_smooth"] = np.nan
    df["gradient_pct"] = 0.0
    df["turn_angle_deg"] = 0.0
    df["curve_radius_m"] = 10000.0
    df["elevation_step_m"] = 0.0
    for _, indices in df.groupby("group", sort=False).groups.items():
        ix = np.asarray(indices)
        x = df.loc[ix, "distance_m"].to_numpy()
        if len(ix) < 3 or x[-1] <= x[0]:
            continue
        # 5-point median rejects spikes; an evenly spaced 20m smooth avoids sample-rate bias.
        e = (
            df.loc[ix, "elevation"]
            .interpolate(limit_direction="both")
            .rolling(5, center=True, min_periods=1)
            .median()
        )
        if e.notna().any():
            grid = np.arange(x[0], x[-1] + 10, 10.0)
            z = np.interp(grid, x, e)
            z = pd.Series(z).rolling(5, center=True, min_periods=1).mean().to_numpy()
            smooth = np.interp(x, grid, z)
            df.loc[ix, "elevation_smooth"] = smooth
            left, right = np.maximum(x[0], x - 20), np.minimum(x[-1], x + 20)
            slope = (np.interp(right, grid, z) - np.interp(left, grid, z)) / np.maximum(1, right - left)
            df.loc[ix, "gradient_pct"] = np.clip(slope * 100, -35, 35)
            steps = np.r_[0, np.diff(smooth)]
            df.loc[ix, "elevation_step_m"] = steps
        left = np.searchsorted(x, x - 20).clip(0, len(x) - 1)
        right = np.searchsorted(x, x + 20).clip(0, len(x) - 1)
        lat, lon = df.loc[ix, "lat"].to_numpy(), df.loc[ix, "lon"].to_numpy()
        h1 = bearing(lat[left], lon[left], lat, lon)
        h2 = bearing(lat, lon, lat[right], lon[right])
        turn = np.abs(angle_difference(h2, h1))
        turn[(x - x[left] < 8) | (x[right] - x < 8)] = 0
        df.loc[ix, "turn_angle_deg"] = turn
        df.loc[ix, "curve_radius_m"] = np.clip(
            (x[right] - x[left]) / np.maximum(0.001, np.radians(turn)), 3, 10000
        )
    df["gain_cumulative_m"] = df.elevation_step_m.clip(lower=0).cumsum()
    diagnostics = {
        "rejected_points": rejected,
        "gps_jump_edges": int(jump.sum()),
        "unobserved_s": float(df.unobserved_s.sum()),
        "elevation_missing_fraction": float(df.elevation_missing.mean()),
        "distance_m": float(df.ds_m.sum()),
        "moving_s": float(df.moving_s.sum()),
        "elapsed_s": float(df.dt_s.sum()),
        "elevation_gain_m": float(df.elevation_step_m.clip(lower=0).sum())
        if df.elevation.notna().any()
        else None,
        "elevation_loss_m": float(-df.elevation_step_m.clip(upper=0).sum())
        if df.elevation.notna().any()
        else None,
    }
    return df, diagnostics
