"""Mutually exclusive interruption windows and descriptive historical behavior."""

from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def route_events(points):
    events = []
    for row in points[points.control_kind.ne("")].itertuples():
        events.append(
            {
                "index": row.Index,
                "distance_m": row.distance_m,
                "kind": row.control_kind,
                "identity": row.control_id,
                "lat": row.lat,
                "lon": row.lon,
                "turn_angle": float(row.turn_angle_deg),
                "radius_m": float(row.curve_radius_m),
                "gradient_pct": float(row.gradient_pct),
                "surface": row.surface,
                "highway": row.highway,
                "heading": float(row.heading),
                "group": row.group,
            }
        )
    entering = points.junction.eq("roundabout") & ~points.junction.shift().eq("roundabout")
    for row in points[entering].itertuples():
        events.append(
            {
                "index": row.Index,
                "distance_m": row.distance_m,
                "kind": "roundabout",
                "identity": f"roundabout:{row.way_id}",
                "lat": row.lat,
                "lon": row.lon,
                "turn_angle": float(row.turn_angle_deg),
                "radius_m": float(row.curve_radius_m),
                "gradient_pct": float(row.gradient_pct),
                "surface": row.surface,
                "highway": row.highway,
                "heading": float(row.heading),
                "group": row.group,
            }
        )
    for group, subset in points.groupby("group", sort=False):
        if len(subset) < 3:
            continue
        peaks, _ = find_peaks(subset.turn_angle_deg.to_numpy(), height=38, prominence=18)
        accepted = []
        # Retain strongest turn in a 70m geometry window, avoiding GPS micro-corners.
        for peak in sorted(peaks, key=lambda i: -subset.turn_angle_deg.iloc[i]):
            row = subset.iloc[peak]
            if all(abs(row.distance_m - distance) > 70 for distance in accepted):
                accepted.append(row.distance_m)
                identity = hashlib.sha256(
                    f"{row.lat:.4f},{row.lon:.4f},{row.heading:.0f}".encode()
                ).hexdigest()[:12]
                events.append(
                    {
                        "index": int(row.name),
                        "distance_m": float(row.distance_m),
                        "kind": "corner",
                        "identity": f"corner:{identity}",
                        "lat": float(row.lat),
                        "lon": float(row.lon),
                        "turn_angle": float(row.turn_angle_deg),
                        "radius_m": float(row.curve_radius_m),
                        "gradient_pct": float(row.gradient_pct),
                        "surface": row.surface,
                        "highway": row.highway,
                        "heading": float(row.heading),
                        "group": int(group),
                    }
                )
    events.sort(key=lambda e: e["distance_m"])
    # Co-located controls and cornering are one event: never charge a turn twice at a signal.
    priority = {"stop": 6, "signal": 5, "railway": 4, "yield": 3, "roundabout": 2, "corner": 1}
    merged = []
    for event in events:
        if (
            merged
            and event["group"] == merged[-1]["group"]
            and event["distance_m"] - merged[-1]["distance_m"] < 55
        ):
            previous = merged[-1]
            primary = event if priority[event["kind"]] > priority[previous["kind"]] else previous
            primary = dict(primary)
            primary["turn_angle"] = max(previous["turn_angle"], event["turn_angle"])
            primary["radius_m"] = min(previous["radius_m"], event["radius_m"])
            merged[-1] = primary
        else:
            merged.append(dict(event))
    # Voronoi boundaries make even nearby interruption windows mutually exclusive.
    for index, event in enumerate(merged):
        d = event["distance_m"]
        left = (
            (d + merged[index - 1]["distance_m"]) / 2
            if index and merged[index - 1]["group"] == event["group"]
            else d - 100
        )
        right = (
            (d + merged[index + 1]["distance_m"]) / 2
            if index + 1 < len(merged) and merged[index + 1]["group"] == event["group"]
            else d + 100
        )
        event["start_m"], event["end_m"] = max(d - 100, left), min(d + 100, right)
    return merged


def observe_events(points, events, ride_id):
    observations = []
    for event in events:
        subset = points[
            (points.distance_m >= event["start_m"])
            & (points.distance_m < event["end_m"])
            & (points.group == event["group"])
        ]
        if len(subset) < 4 or subset.ds_m.sum() < 20:
            continue
        speed = subset.speed_mps.dropna()
        if len(speed) < 3:
            continue
        approach = float(speed.iloc[: max(2, len(speed) // 4)].median())
        exit_speed = float(speed.iloc[-max(2, len(speed) // 4) :].median())
        cruise = max(approach, exit_speed)
        if cruise < 1.5:
            continue
        minimum = float(speed.min())
        stationary = float(subset.stationary_s.sum())
        reliable = bool(subset.unobserved_s.sum() < 8 and subset.dt_s[subset.dt_s > 0].median() <= 10)
        if stationary >= 2 and minimum < 0.8:
            behavior = "full_stop"
        elif reliable and minimum < min(2.8, cruise * 0.65):
            behavior = "rolling"
        elif reliable:
            behavior = "minimal"
        else:
            behavior = "unknown"
        moving_loss = max(0.0, float(subset.moving_s.sum() - subset.ds_m.sum() / cruise))
        minimum_ix = subset.speed_mps.idxmin()
        minimum_time = points.loc[minimum_ix, "time"]
        before, after = subset.loc[:minimum_ix], subset.loc[minimum_ix:]
        time_value = subset.time.dropna().iloc[0] if subset.time.notna().any() else pd.NaT
        observations.append(
            {
                **event,
                "ride_id": str(ride_id),
                "approach_mps": approach,
                "minimum_mps": minimum,
                "exit_mps": exit_speed,
                "braking_s": float(before.moving_s[before.speed_mps < cruise * 0.8].sum()),
                "acceleration_s": float(after.moving_s[after.speed_mps < cruise * 0.8].sum()),
                "stationary_s": stationary if reliable or stationary >= 2 else np.nan,
                "moving_loss_s": moving_loss,
                "reliable": reliable,
                "behavior": behavior,
                "turning": event["turn_angle"] >= 40,
                "hour": time_value.hour if pd.notna(time_value) else -1,
                "weekday": time_value.dayofweek if pd.notna(time_value) else -1,
                "minimum_time": str(minimum_time),
            }
        )
    return observations


def physics_loss(cruise_mps, minimum_mps, window_m=200.0, decel=0.9, accel=0.45):
    """Kinematic moving loss only, excluding stationary dwell; parameters are explicit fallbacks."""
    v = max(0.8, float(cruise_mps))
    u = max(0, min(float(minimum_mps), v))
    distance = (v * v - u * u) / 2 * (1 / decel + 1 / accel)
    if distance > window_m:
        # Reduce attainable speed change to fit the finite interruption window.
        u = np.sqrt(max(0, v * v - 2 * window_m / (1 / decel + 1 / accel)))
        distance = window_m
    travel = (v - u) * (1 / decel + 1 / accel)
    return max(0.0, float(travel - distance / v))


@dataclass
class BehaviorModel:
    observations: list = field(default_factory=list)

    def predict(self, event, cruise_mps, scenario="full_stop", hour=12, weekday=5):
        kind = event["kind"]
        usable = [o for o in self.observations if o["kind"] == kind and o["reliable"]]
        comparable = [
            o
            for o in usable
            if o["highway"] == event["highway"] and o["turning"] == (event["turn_angle"] >= 40)
        ]
        broad = comparable if len(comparable) >= 8 else usable
        local = [
            o
            for o in usable
            if o["identity"] == event["identity"]
            and abs(o["hour"] - hour) <= 3
            and (o["weekday"] >= 5) == (weekday >= 5)
        ]
        n = len(local)
        alpha = n / (n + 12.0)
        result = {
            "moving_loss_s": 0.0,
            "stationary_s": 0.0,
            "stop_probability": None,
            "source": "physics fallback",
            "encounters": len(broad),
            "local_encounters": n,
            "stationary_known": True,
            "behavior_probabilities": {},
        }
        if broad:
            global_loss = float(np.median([o["moving_loss_s"] for o in broad]))
            local_loss = float(np.median([o["moving_loss_s"] for o in local])) if local else global_loss
            observed_cruise = max(1.0, float(np.median([o["approach_mps"] for o in broad])))
            result["moving_loss_s"] = (alpha * local_loss + (1 - alpha) * global_loss) * np.clip(
                cruise_mps / observed_cruise, 0.5, 2
            )
            p_global = (sum(o["stationary_s"] >= 2 for o in broad) + 1) / (len(broad) + 2)
            p_local = (sum(o["stationary_s"] >= 2 for o in local) + 12 * p_global) / (n + 12)
            stopped = [o["stationary_s"] for o in broad if o["stationary_s"] >= 2]
            waits = [o["stationary_s"] for o in local if o["stationary_s"] >= 2]
            global_wait = float(np.mean(stopped)) if stopped else 0.0
            local_wait = float(np.mean(waits)) if waits else global_wait
            wait = local_wait * alpha + global_wait * (1 - alpha)
            result.update(
                {
                    "stop_probability": p_local,
                    "stationary_s": p_local * wait,
                    "source": "smoothed historical distribution",
                }
            )
            result["behavior_probabilities"] = {
                b: (sum(o["behavior"] == b for o in broad) + 1) / (len(broad) + 3)
                for b in ("full_stop", "rolling", "minimal")
            }
        else:
            lateral_accel = 0.9 if event["surface"] in ("loose_gravel", "rough_natural") else 1.5
            corner_speed = np.sqrt(max(3, event["radius_m"]) * lateral_accel)
            if kind in ("stop", "signal"):
                corner_speed = (
                    0
                    if kind == "stop" and scenario == "full_stop"
                    else min(corner_speed, cruise_mps * 0.65)
                )
                # A signal's red probability/wait cannot be inferred from geometry.
                result["stationary_known"] = False
                result["stationary_s"] = 0.0
            elif kind in ("yield", "roundabout", "railway"):
                corner_speed = min(corner_speed, cruise_mps * 0.65)
                if kind == "railway":
                    result["stationary_known"] = False
            result["moving_loss_s"] = physics_loss(
                cruise_mps, corner_speed, event["end_m"] - event["start_m"]
            )
        if kind == "stop" and scenario == "full_stop":
            full_observed = [o for o in broad if o["behavior"] == "full_stop"]
            # A stop means reaching zero; dwell is learned only if full-stop recordings exist.
            result["stationary_s"] = (
                float(np.median([o["stationary_s"] for o in full_observed])) if full_observed else 0.0
            )
            result["stationary_known"] = bool(full_observed)
            result["stop_probability"] = 1.0
            result["moving_loss_s"] = max(
                result["moving_loss_s"], physics_loss(cruise_mps, 0, event["end_m"] - event["start_m"])
            )
            result["source"] += "; complete stop imposed"
        if kind == "corner":
            # Geometry alone cannot identify why someone paused. Do not turn incidental stops
            # into implied traffic waits or silently assumed personal breaks.
            result["stationary_s"] = 0.0
            result["stop_probability"] = None
            result["stationary_known"] = True
            result["behavior_probabilities"] = {}
        return result
