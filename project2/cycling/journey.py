"""Complete planning workflow layered on the preserved riding models."""

import json

import joblib
import pandas as pd

from .daylight import daylight_plan
from .elapsed import ElapsedCalibration, cutoff_probability
from .hr import recommend_hr
from .prediction import predict_route
from .stopping import StoppingModel, route_features


def load_planning(store):
    path = store.root / "models/journey.joblib"
    return joblib.load(path) if path.exists() else {"stopping": None, "calibrations": {}}


def plan_ride(
    model,
    route,
    rides,
    settings,
    planning=None,
    target_hr=None,
    bike=None,
    departure=None,
    scenario="full_stop",
    break_mode="historical",
    break_minutes=0,
    weather_provider=None,
    weather_neutral=False,
    daylight_reference="sunset",
    safety_minutes=30,
    manual_hr=False,
):
    from .pipeline import route_for_model

    planning = planning or {"stopping": None, "calibrations": {}}
    segments, events = route_for_model(route, model)
    departure = pd.Timestamp(departure or pd.Timestamp.now(tz="UTC"))
    if departure.tzinfo is None:
        raise ValueError("A planned departure must include a timezone.")
    recommendation = recommend_hr(
        rides,
        float(segments.distance_m.sum()) / 1000,
        departure,
        planning.get("hr_half_life", settings.hr_half_life_days),
        route_features(segments, bike),
        model.hr_range,
        settings.hr_min_comparable,
        settings.hr_distance_bandwidth,
    )
    if target_hr is None and not manual_hr:
        target_hr = recommendation["suggested_hr"]
    result = predict_route(
        model,
        segments,
        events,
        target_hr=target_hr,
        bike=bike,
        scenario=scenario,
        personal_break_minutes=break_minutes,
        departure=departure.isoformat(),
        use_fitness=model.validation.get("fitness_selected", False),
        weather_provider=weather_provider,
        weather_neutral=weather_neutral,
        stopping_model=planning.get("stopping"),
        break_mode=break_mode,
    )
    result["hr_recommendation"] = recommendation
    result["hr_selection"] = "manual override" if manual_hr else "automatic suggestion"
    points = route["points"]
    destination = (float(points.lat.iloc[-1]), float(points.lon.iloc[-1]))
    daylight = daylight_plan(
        departure, result["elapsed_s"], destination, daylight_reference, safety_minutes
    )
    key = "CW+stops-known" if manual_hr else "CW+stops-auto"
    if getattr(model, "variant", "CW") != "CW":
        key = getattr(model, "variant", "C") + ("-auto" if break_mode == "minimal" else "+stops-auto")
    calibration = planning.get("calibrations", {}).get(key, ElapsedCalibration())
    uncertainty = calibration.estimate(
        result["elapsed_s"],
        result["distance_m"] / 1000,
        departure,
        forecast=weather_provider is not None,
        break_mode=break_mode,
    )
    if getattr(model, "variant", "C") == "CW" and result["weather_distance_coverage"] < 0.8:
        uncertainty["provisional"] = True
        uncertainty["probability_supported"] = False
        uncertainty["warnings"].append(
            "Weather-neutral/fallback duration has not been calibrated against issued forecasts."
        )
    if result["unknown_wait_events"] or (
        break_mode == "historical" and (result.get("break_estimate") or {}).get("samples", 0) == 0
    ):
        uncertainty["provisional"] = True
        uncertainty["probability_supported"] = False
        uncertainty["warnings"].append("Unlearned waits or stopping history can understate duration.")
    available = (
        (pd.Timestamp(daylight["cutoff"]) - departure).total_seconds() if daylight["cutoff"] else None
    )
    daylight["before_cutoff_probability"] = (
        cutoff_probability(uncertainty, available) if available is not None else None
    )
    daylight["conservative_finish_local"] = (
        (departure + pd.Timedelta(seconds=uncertainty["conservative_s"]))
        .tz_convert(daylight["destination_timezone"])
        .isoformat()
        if uncertainty["conservative_s"] is not None
        else None
    )
    daylight["median_finish_local"] = (
        (departure + pd.Timedelta(seconds=uncertainty["median_s"]))
        .tz_convert(daylight["destination_timezone"])
        .isoformat()
        if uncertainty["median_s"] is not None
        else None
    )
    result["daylight"], result["elapsed_uncertainty"] = daylight, uncertainty
    result["warnings"] += (
        route.get("warnings", [])
        + recommendation["warnings"]
        + uncertainty["warnings"]
        + daylight["notes"]
    )
    if daylight["cutoff_margin_minutes"] is not None and daylight["cutoff_margin_minutes"] < 30:
        result["warnings"].append(
            "Expected arrival is close to or after the daylight cutoff; uncertain delays can make a before-dark plan optimistic."
        )
    return result


def fit_planning(store, settings, validation=None):
    rides = store.rides()
    validation = validation or store.get_state("journey_validation", {})
    method = validation.get("selected_stopping_method", "context")
    residual_rows = validation.get("residual_rows", [])
    model = StoppingModel.fit(rides, method, settings.stopping_half_life_days, residual_rows)
    metrics = {r["variant"]: r for r in validation.get("comparisons", [])}
    calibrations = {
        key: ElapsedCalibration(
            rows, metrics.get(key, {}).get("coverage_80"), metrics.get(key, {}).get("coverage_90")
        )
        for key, rows in validation.get("calibration_rows", {}).items()
    }
    planning = {
        "stopping": model,
        "calibrations": calibrations,
        "hr_half_life": validation.get("selected_hr_half_life", settings.hr_half_life_days),
        "train_ids": rides.id.astype(str).tolist(),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    joblib.dump(planning, store.root / "models/journey.joblib", compress=3)
    summary = {
        "trained_at": planning["trained_at"],
        "stopping_method": model.method,
        "stopping_rides": len(model.rides),
        "hr_half_life": planning["hr_half_life"],
    }
    store.set_state("journey_training", summary)
    store.set_state("journey_stale", False)
    (store.root / "reports/journey-training.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return planning


def train_planning(store, settings, progress=None, validate=True, archive=None):
    from .timing import audit_history
    from .journey_validation import run_validation

    audit_history(store, settings, archive, progress)
    report = (
        run_validation(store, settings, progress)
        if validate
        else store.get_state("journey_validation", {})
    )
    planning = fit_planning(store, settings, report)
    return {
        "training": store.get_state("journey_training"),
        "timing": store.get_state("timing_quality"),
        "comparisons": report.get("comparisons", []),
        "hr_metrics": report.get("hr_metrics"),
        "trained_at": planning["trained_at"],
    }
