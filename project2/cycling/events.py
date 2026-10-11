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
            and (
                event["kind"] == "corner"
                or merged[-1]["kind"] == "corner"
                or event["identity"] == merged[-1]["identity"]
            )
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
        # Moving-threshold time is not proof of reaching zero. Require recorded near-zero
        # samples close to the control, with consecutive short observation intervals.
        near = subset[abs(subset.distance_m - event["distance_m"]) <= 35]
        confirmed_zero = near.speed_mps.le(0.3) & near.dt_s.between(0, 10)
        stationary = float(near.loc[confirmed_zero, "stationary_s"].sum())
        reasons = []
        if subset.unobserved_s.sum() >= 8 or subset.dt_s.max() > 15:
            reasons.append("recording gap / sparse sampling / auto-pause")
        if (
            subset.distance_m.min() > event["distance_m"] - 15
            or subset.distance_m.max() < event["distance_m"] + 15
        ):
            reasons.append("incomplete approach or recovery")
        if stationary > 180:
            reasons.append("long dwell may be a personal break")
        if (
            event["kind"] != "corner"
            and float(near.get("map_confidence", pd.Series(1.0, index=near.index)).mean()) < 0.5
        ):
            reasons.append("uncertain road match")
        reliable = not reasons
        if reliable and stationary >= 2 and confirmed_zero.sum() >= 2:
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
        sign_ix = int(abs(subset.distance_m - event["distance_m"]).idxmin())
        time_value = points.loc[sign_ix, "time"]
        braking = before[before.speed_mps < cruise * 0.85]
        recovery = after[after.speed_mps >= cruise * 0.85]
        arrival_heading = float(subset.heading.iloc[0])
        departure_heading = float(subset.heading.iloc[-1])
        turn_signed = (departure_heading - arrival_heading + 180) % 360 - 180
        observations.append(
            {
                **event,
                "ride_id": str(ride_id),
                "approach_mps": approach,
                "minimum_mps": minimum,
                "exit_mps": exit_speed,
                "speed_at_sign_mps": float(points.loc[sign_ix, "speed_mps"]),
                "braking_distance_m": max(0.0, float(event["distance_m"] - braking.distance_m.iloc[0]))
                if len(braking)
                else 0.0,
                "recovery_distance_m": max(0.0, float(recovery.distance_m.iloc[0] - event["distance_m"]))
                if len(recovery)
                else float("nan"),
                "approach_heading": arrival_heading,
                "turn_direction": "straight"
                if abs(turn_signed) < 35
                else "right"
                if turn_signed > 0
                else "left",
                "braking_s": float(before.moving_s[before.speed_mps < cruise * 0.8].sum()),
                "acceleration_s": float(after.moving_s[after.speed_mps < cruise * 0.8].sum()),
                "stationary_s": stationary if reliable else np.nan,
                "moving_loss_s": moving_loss,
                "reliable": reliable,
                "reliability_reasons": reasons,
                "behavior_confidence": float(np.exp(-max(0, subset.dt_s.max() - 1) / 10))
                if reliable
                else 0.0,
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
        if u == 0:
            # Closely spaced mandatory stops cannot regain the unrestricted cruising
            # speed between them. Use a triangular profile that still reaches zero;
            # never relax a full-stop request into a rolling slowdown to fit the window.
            factor = 1 / decel + 1 / accel
            peak = np.sqrt(max(0, 2 * window_m / factor))
            return max(0.0, float(peak * factor - window_m / v))
        # Reduce attainable speed change to fit the finite interruption window.
        u = np.sqrt(max(0, v * v - 2 * window_m / (1 / decel + 1 / accel)))
        distance = window_m
    travel = (v - u) * (1 / decel + 1 / accel)
    return max(0.0, float(travel - distance / v))


@dataclass
class BehaviorModel:
    observations: list = field(default_factory=list)

    def pools(self, event, hour, weekday):
        if not hasattr(self, "_pool_cache"):
            self._pool_cache, self._kinds, self._identities = {}, {}, {}
            for observation in self.observations:
                if observation["reliable"] and observation["behavior"] in (
                    "full_stop",
                    "rolling",
                    "minimal",
                ):
                    self._kinds.setdefault(observation["kind"], []).append(observation)
                    self._identities.setdefault(observation["identity"], []).append(observation)
        key = self.context_key(event, hour, weekday)
        if key not in self._pool_cache:
            usable = self._kinds.get(event["kind"], [])
            comparable = [
                o
                for o in usable
                if o["highway"] == event["highway"] and o["turning"] == (event["turn_angle"] >= 40)
            ]
            terrain = [
                o
                for o in comparable
                if o["surface"] == event["surface"]
                and abs(o["gradient_pct"] - event["gradient_pct"]) <= 3
            ]
            clock = [o for o in terrain if self.clock_similar(o, hour, weekday)]
            broad = next((pool for pool in (clock, terrain, comparable) if len(pool) >= 12), usable)
            local_all = [
                o
                for o in self._identities.get(event["identity"], [])
                if o["kind"] == event["kind"]
                and abs((o.get("approach_heading", o["heading"]) - event["heading"] + 180) % 360 - 180)
                < 60
            ]
            local_clock = [o for o in local_all if self.clock_similar(o, hour, weekday)]
            local = local_clock if len(local_clock) >= 3 else local_all
            self._pool_cache[key] = broad, local
        return self._pool_cache[key]

    @staticmethod
    def clock_similar(observation, hour, weekday):
        return (
            observation["hour"] >= 0
            and min(abs(observation["hour"] - hour), 24 - abs(observation["hour"] - hour)) <= 3
            and (observation["weekday"] >= 5) == (weekday >= 5)
        )

    @staticmethod
    def context_key(event, hour, weekday):
        return (
            event["kind"],
            event["highway"],
            event["turn_angle"] >= 40,
            event["identity"],
            hour,
            weekday >= 5,
            event["surface"],
            round(event["gradient_pct"] / 3),
            round(event["heading"] / 45),
        )

    def historical_summary(self, event, hour, weekday, broad, local):
        from scipy.stats import beta

        if not hasattr(self, "_summary_cache"):
            self._summary_cache = {}
        key = self.context_key(event, hour, weekday)
        if key not in self._summary_cache:
            classes = ("full_stop", "rolling", "minimal")
            prior = np.array([sum(o["behavior"] == b for o in broad) + 0.5 for b in classes])
            prior /= prior.sum()
            counts = np.array([sum(o["behavior"] == b for o in local) for b in classes], dtype=float)
            posterior = counts + 12 * prior
            probability = posterior / posterior.sum()
            alpha = len(local) / (len(local) + 12.0)

            def blend(column, pool=None, default=0.0):
                pool = broad if pool is None else pool
                global_values = [
                    o.get(column, np.nan) for o in pool if np.isfinite(o.get(column, np.nan))
                ]
                local_values = [
                    o.get(column, np.nan)
                    for o in local
                    if o in pool and np.isfinite(o.get(column, np.nan))
                ]
                mean = float(np.mean(global_values)) if global_values else default
                return (
                    alpha * (float(np.mean(local_values)) if local_values else mean) + (1 - alpha) * mean
                )

            losses = [blend("moving_loss_s", [o for o in broad if o["behavior"] == b]) for b in classes]
            stopped = [
                o for o in broad if o["behavior"] == "full_stop" and np.isfinite(o["stationary_s"])
            ]
            local_stopped = [
                o for o in local if o["behavior"] == "full_stop" and np.isfinite(o["stationary_s"])
            ]
            global_wait = float(np.mean([o["stationary_s"] for o in stopped])) if stopped else 0.0
            wait_alpha = len(local_stopped) / (len(local_stopped) + 8)
            conditional_wait = (
                wait_alpha
                * (
                    float(np.mean([o["stationary_s"] for o in local_stopped]))
                    if local_stopped
                    else global_wait
                )
                + (1 - wait_alpha) * global_wait
            )
            self._summary_cache[key] = {
                "base_loss": float(np.dot(probability, losses)),
                "approach": max(1.0, blend("approach_mps")),
                "minimum": blend("minimum_mps"),
                "probability": float(probability[0]),
                "probability_interval": beta.ppf(
                    [0.05, 0.95], posterior[0], posterior[1:].sum()
                ).tolist(),
                "wait": float(probability[0] * conditional_wait),
                "conditional_wait": conditional_wait,
                "wait_samples": len(stopped),
                "full_wait": conditional_wait,
                "full_known": bool(stopped),
                "behaviors": dict(zip(classes, map(float, probability))),
            }
        return self._summary_cache[key]

    def predict(self, event, cruise_mps, scenario="full_stop", hour=12, weekday=5):
        kind = event["kind"]
        broad, local = self.pools(event, hour, weekday)
        n = len(local)
        result = {
            "moving_loss_s": 0.0,
            "stationary_s": 0.0,
            "stop_probability": None,
            "source": "generalized physics fallback",
            "encounters": len(broad),
            "local_encounters": n,
            "stationary_known": True,
            "behavior_probabilities": {},
            "expected_minimum_mps": None,
            "confidence": "unavailable",
            "stop_probability_interval": None,
            "conditional_wait_s": None,
        }
        if broad:
            summary = self.historical_summary(event, hour, weekday, broad, local)
            result.update(
                {
                    "moving_loss_s": summary["base_loss"]
                    * np.clip(cruise_mps / summary["approach"], 0.5, 2),
                    "stop_probability": summary["probability"],
                    "stationary_s": summary["wait"],
                    "behavior_probabilities": summary["behaviors"].copy(),
                    "expected_minimum_mps": float(
                        np.clip(summary["minimum"] * cruise_mps / summary["approach"], 0, cruise_mps)
                    ),
                    "conditional_wait_s": summary["conditional_wait"],
                    "stop_probability_interval": summary["probability_interval"],
                    "confidence": "moderate" if n >= 8 else "low",
                    "source": "location-specific smoothed history"
                    if n >= 3
                    else "generalized historical distribution",
                    "stationary_known": kind not in ("signal", "railway", "stop")
                    or summary["wait_samples"] > 0,
                }
            )
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
            summary = self.historical_summary(event, hour, weekday, broad, local) if broad else {}
            result["stationary_s"] = summary.get("full_wait", 0.0)
            result["stationary_known"] = summary.get("full_known", False)
            result["stop_probability"] = 1.0
            result["expected_minimum_mps"] = 0.0
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
