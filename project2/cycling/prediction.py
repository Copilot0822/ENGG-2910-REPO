"""Sequential fatigue features, moving-time HR constraint and time-correct aggregation."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .model import profile_features


def speed_from_time(distance_m, time_s):
    return float(distance_m / time_s * 3.6) if time_s > 0 else 0.0


def _allocate_events(
    frame, events, model, cruise, scenario, departure, moving, previous_waits, previous_breaks=None
):
    losses = np.zeros(len(frame))
    waits = np.zeros(len(frame))
    details = []
    previous_breaks = np.zeros(len(frame)) if previous_breaks is None else previous_breaks
    arrivals = (
        np.cumsum(moving + previous_waits + previous_breaks) - moving - previous_waits - previous_breaks
    )
    for event in events:
        ix = int(np.argmin(abs(frame.end_m.to_numpy() - event["distance_m"])))
        fraction = float(
            np.clip(
                (event["distance_m"] - frame.start_m.iloc[ix]) / max(1, frame.distance_m.iloc[ix]), 0, 1
            )
        )
        moment = departure + pd.Timedelta(seconds=float(arrivals[ix] + fraction * moving[ix]))
        estimate = model.behavior.predict(
            event, cruise[ix] / 3.6, scenario, moment.hour, moment.dayofweek
        )
        # Charge each consolidated event once. Braking belongs to moving time; wait belongs to elapsed.
        losses[ix] += estimate["moving_loss_s"]
        waits[ix] += estimate["stationary_s"]
        details.append({**event, **estimate, "segment": ix, "arrival_time_utc": moment.isoformat()})
    return losses, waits, details


def predict_route(
    model,
    segments,
    events=(),
    target_hr=None,
    bike=None,
    scenario="full_stop",
    personal_break_minutes=0.0,
    departure=None,
    use_fitness=False,
    weather_provider=None,
    weather_neutral=False,
    stopping_model=None,
    break_mode="custom",
):
    if len(segments) == 0:
        raise ValueError("The route contains no usable segments.")
    if scenario not in ("full_stop", "observed"):
        raise ValueError("Choose full_stop or observed historical behavior.")
    if personal_break_minutes < 0:
        raise ValueError("Personal break time cannot be negative.")
    if break_mode not in ("historical", "custom", "minimal"):
        raise ValueError("Choose historical, custom or minimal breaks.")
    if target_hr is not None and not 35 <= target_hr <= 230:
        raise ValueError("Target average HR must be between 35 and 230 BPM.")
    from .comparison import prediction_inputs

    frame, events = prediction_inputs(segments, events, getattr(model, "variant", "C"))
    if weather_neutral:
        from .weather import neutral_weather

        frame = neutral_weather(frame)
    # Only route geometry enters prediction; recorded speed, HR, elapsed time and power are ignored.
    frame["moving_hours"] = 0.0
    frame["date"] = str(departure or datetime.now(timezone.utc).isoformat())
    moment = pd.Timestamp(frame.date.iloc[0])
    # Recording encounter hours are stored in UTC; compare departure context on the same clock.
    event_moment = moment.tz_convert("UTC") if moment.tzinfo else moment.tz_localize("UTC")
    frame["observation_day"] = moment.toordinal()
    frame["day_of_year"] = moment.dayofyear
    frame["bike"] = bike or "unknown"
    distance = float(frame.distance_m.sum())
    warnings = []
    variant = getattr(model, "variant", "C")
    if variant in ("A", "B"):
        warnings.append(
            f"Model {variant} was validated without explicit mapped traffic penalties. Elapsed time is a lower bound; select model C for mapped waits and full-stop traffic-control scenarios."
        )
    if bike and bike not in model.support["bikes"]:
        warnings.append(
            "This bike was not observed during training; bike-specific effects are unavailable."
        )
    requested = target_hr
    if target_hr is not None and not model.hr_available:
        warnings.append(
            "Insufficient measured HR for a personalized effort relationship. This prediction uses the rider's learned terrain baseline; target HR is not modeled."
        )
        target_hr = None
    elif target_hr is not None:
        limited = float(np.clip(target_hr, *model.hr_range))
        if abs(limited - target_hr) > 0.01:
            warnings.append(
                f"Requested HR is outside measured support {model.hr_range[0]:.0f}–{model.hr_range[1]:.0f} BPM; modeled target limited to {limited:.0f} BPM."
            )
        target_hr = limited
    if distance / 1000 > model.support["max_distance_km"] * 1.05:
        warnings.append(
            "Route is longer than training rides. Pacing/fatigue features are capped at observed support; long-distance confidence is low."
        )
    if model.support["long_rides"] < 5 and distance > 100_000:
        warnings.append(
            "Fewer than five rides over 100 km: accumulated fatigue and long-distance pacing are weakly supported."
        )
    if frame.elevation_missing.mean() > 0.2:
        warnings.append(
            "Route elevation is missing or unreliable. Gradient is set to zero where unavailable; ascent and hill predictions are uncertain."
        )
    unknown_fraction = float(
        segments.loc[segments.surface_source.ne("known"), "distance_m"].sum() / distance
    )
    actual_coverage = segments.attrs.get("osm_coverage")
    if actual_coverage:
        unknown_fraction = 1 - actual_coverage["sources"].get("known", 0) / max(
            1e-9, actual_coverage["total_km"]
        )
    if unknown_fraction > 0.2:
        warnings.append(
            f"{unknown_fraction:.0%} of route distance lacks verified OSM surfaces. Missing controls may understate travel time."
        )
    hr_values = np.full(len(frame), np.nan)
    ds = frame.distance_m.to_numpy()
    cruise = model.predict_cruise(frame)
    moving = ds / (cruise / 3.6)
    waits = np.zeros(len(frame))
    from .stopping import allocate_breaks

    break_estimate = None

    def expected_breaks(duration):
        nonlocal break_estimate
        if break_mode == "historical" and stopping_model is not None:
            break_estimate = stopping_model.predict(frame, float(duration.sum()), event_moment, bike)
            return allocate_breaks(
                frame, break_estimate["expected_s"], break_estimate["progress_weights"]
            )
        return allocate_breaks(frame, personal_break_minutes * 60 if break_mode == "custom" else 0)

    break_delays = expected_breaks(moving)
    event_losses, waits, details = _allocate_events(
        frame, events, model, cruise, scenario, event_moment, moving, waits, break_delays
    )
    converged = False
    for iteration in range(20):
        frame["moving_hours"] = (np.cumsum(moving) - moving / 2) / 3600
        arrival_s = np.cumsum(moving + waits + break_delays) - moving - waits - break_delays + moving / 2
        frame["arrival_time_utc"] = event_moment + pd.to_timedelta(arrival_s, unit="s")
        if weather_provider is not None and not weather_neutral:
            frame = weather_provider.at(frame, frame.arrival_time_utc)
        if target_hr is not None:
            terrain_shape = np.clip(model.hr_profile.predict(profile_features(frame)), -25, 25)

            def profile(offset):
                return np.clip(target_hr + terrain_shape + offset, *model.hr_range)

            # Iterate speed/time and solve the moving-time-weighted average constraint together.
            def constraint(offset):
                return np.average(profile(offset), weights=moving) - target_hr

            hr_values = profile(brentq(constraint, -100, 100, xtol=0.01))
        cruise = model.predict_cruise(frame, hr_values if target_hr is not None else None, use_fitness)
        event_losses, waits, details = _allocate_events(
            frame, events, model, cruise, scenario, event_moment, moving, waits, break_delays
        )
        updated = ds / (cruise / 3.6) + event_losses
        updated_breaks = expected_breaks(updated)
        breaks_converged = abs(updated_breaks.sum() - break_delays.sum()) < max(
            1, break_delays.sum() * 0.002
        )
        break_delays = updated_breaks
        if np.max(abs(updated - moving) / np.maximum(1, moving)) < 0.002 and breaks_converged:
            moving = updated
            converged = True
            break
        moving = moving * 0.4 + updated * 0.6
    # Use the final exact time ledger for all metrics, never arithmetic mean segment speeds.
    moving = ds / (cruise / 3.6) + event_losses
    achieved_hr = float(np.average(hr_values, weights=moving)) if target_hr is not None else None
    if target_hr is not None and (not converged or abs(achieved_hr - target_hr) > 0.5):
        warnings.append(
            "HR/time optimization did not fully converge; inspect the achieved moving-time average HR."
        )
    frame["cruise_kmh"] = cruise
    frame["moving_time_s"] = moving
    frame["moving_loss_s"] = event_losses
    frame["stationary_delay_s"] = waits
    frame["rolling_kmh"] = ds / moving * 3.6
    frame["predicted_hr"] = hr_values
    frame["discretionary_delay_s"] = break_delays
    frame["elapsed_time_s"] = moving + waits + break_delays
    frame["accumulated_moving_s"] = np.cumsum(moving)
    frame["accumulated_elapsed_s"] = np.cumsum(moving + waits + break_delays)
    frame["segment_end_time_utc"] = event_moment + pd.to_timedelta(frame.accumulated_elapsed_s, unit="s")
    cruising_time = float((ds / (cruise / 3.6)).sum())
    moving_time = float(moving.sum())
    stationary_time = float(waits.sum())
    breaks = float(break_delays.sum())
    total_time = moving_time + stationary_time + breaks
    uncovered = [e for e in details if not e["stationary_known"]]
    weather_coverage = float(
        frame.loc[
            frame.get("weather_available", pd.Series(False, index=frame.index)).fillna(False),
            "distance_m",
        ].sum()
        / distance
    )
    if getattr(model, "neutral_model", None) is not None and weather_coverage < 0.95:
        warnings.append(
            f"Weather available for {weather_coverage:.0%} of route distance; missing sections use the fitted weather-neutral terrain model."
        )
    if weather_provider is not None:
        warnings.append(getattr(weather_provider, "note", "ECCC external weather observations"))
        warnings.extend(sorted(set(weather_provider.client.errors)))
    if breaks:
        warnings.append(
            "Expected personal stopping is distributed broadly through route progress and shifts downstream weather/traffic arrival times; precise stopping locations are not predicted."
        )
    if break_mode == "historical":
        warnings.extend(
            break_estimate["warnings"]
            if break_estimate
            else ["No fitted stopping history; usual personal delays are unknown."]
        )
    elif break_mode == "minimal":
        warnings.append(
            "Minimal personal breaks is an optimistic scenario; traffic delays and moving slowdowns are retained."
        )
    if uncovered:
        warnings.append(
            f"{len(uncovered)} traffic events lack calibrated stationary waits. Elapsed time excludes unknown waits; it may be understated. Full-stop fallback reaches zero speed without inventing dwell time."
        )
    surface = frame.groupby(["surface", "surface_source"]).distance_m.sum().reset_index()
    surface["percent"] = surface.distance_m / distance * 100
    interval = None
    residuals = model.validation.get("calibration_speed_residuals", [])
    if residuals:
        n = len(residuals)
        # Chronological calibration set was not used to select the half-life/model variant.
        rank = min(n - 1, int(np.ceil((n + 1) * 0.9)) - 1)
        margin = float(np.sort(residuals)[rank])
        rolling = speed_from_time(distance, moving_time)
        lower, upper = max(0.5, rolling - margin), rolling + margin
        interval = {
            "coverage_target": 0.9,
            "speed_lower_kmh": lower,
            "speed_upper_kmh": upper,
            "moving_lower_s": distance / (upper / 3.6),
            "moving_upper_s": distance / (lower / 3.6),
            "samples": n,
            "provisional": n < 20
            or unknown_fraction > 0.2
            or bool(uncovered)
            or weather_provider is not None
            or (variant == "CW" and weather_coverage < 0.8)
            or model.validation.get("metrics", {}).get("interval_coverage", 1.0) is None
            or (model.validation.get("metrics", {}).get("interval_coverage", 1.0) or 0) < 0.9,
            "method": "chronological held-out ride absolute-error calibration",
        }
    if interval is None:
        warnings.append(
            "No calibrated prediction interval yet. Run chronological validation with enough held-out rides."
        )
    losses_by_kind = {}
    waits_by_kind = {}
    for detail in details:
        kind = detail["kind"]
        losses_by_kind[kind] = losses_by_kind.get(kind, 0.0) + detail["moving_loss_s"]
        waits_by_kind[kind] = waits_by_kind.get(kind, 0.0) + detail["stationary_s"]
    return {
        "distance_m": distance,
        "cruise_kmh": speed_from_time(distance, cruising_time),
        "rolling_kmh": speed_from_time(distance, moving_time),
        "overall_kmh": speed_from_time(distance, total_time),
        "cruising_time_s": cruising_time,
        "moving_s": moving_time,
        "stationary_s": stationary_time,
        "personal_break_s": breaks,
        "moving_slowdown_s": float(event_losses.sum()),
        "break_mode": break_mode,
        "break_estimate": break_estimate,
        "elapsed_s": total_time,
        "departure_utc": event_moment.isoformat(),
        "finish_time_utc": (event_moment + pd.Timedelta(seconds=total_time)).isoformat(),
        "requested_hr": requested,
        "modeled_target_hr": target_hr,
        "achieved_hr": achieved_hr,
        "segments": frame,
        "events": details,
        "surfaces": surface,
        "interval": interval,
        "warnings": warnings,
        "losses_by_kind": losses_by_kind,
        "waits_by_kind": waits_by_kind,
        "scenario": scenario,
        "model_kind": model.kind,
        "model_variant": variant,
        "iterations": iteration + 1,
        "unknown_wait_events": len(uncovered),
        "weather_distance_coverage": weather_coverage,
        "weather_neutral": weather_neutral,
    }
