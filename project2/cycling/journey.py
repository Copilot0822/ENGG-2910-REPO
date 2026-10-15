"""Complete planning workflow layered on the preserved riding models."""

import json

import joblib
import pandas as pd

from .daylight import daylight_plan
from .elapsed import ElapsedCalibration, cutoff_probability
from .hr import recommend_hr
from .prediction import predict_route
from .stopping import StoppingModel, route_features
from .elapsed_models import DurationCorrection, apply_duration_correction, historical_baseline
from .tail_calibration import TailCalibration
from .two_stage import HurdleStopping
from .release_policy import load_operational_planning, archive_artifact


def load_planning(store):
    phase5 = store.root / "models/journey-phase5.joblib"
    if phase5.exists():
        return load_operational_planning(store)
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
    journey_method="automatic",
):
    from .pipeline import route_for_model

    planning = planning or {"stopping": None, "calibrations": {}}
    rides = rides.copy()
    if "activity_start_utc" in rides:
        rides["date"] = rides.activity_start_utc.fillna(rides.date)
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
    chosen = (
        planning.get("operational_default", planning.get("recommended_method", "CW+stops-auto"))
        if journey_method == "automatic"
        else journey_method
    )
    reason = planning.get("selection_reason", "Existing personalized stopping model.")
    if journey_method != "automatic":
        reason = f"User selected {chosen}. Compare its own chronological diagnostics; it may differ from the supported automatic default."
    if getattr(model, "variant", "C") != "CW" or break_mode != "historical":
        chosen = "physical"
        reason = "Physical riding model with the selected break allowance; historical total-duration corrections are disabled for custom/minimal breaks and other riding models."
    stopping = planning.get("stopping")
    if chosen is None:
        raise ValueError(
            planning.get("operational_blocker", "Automatic has no verified operational default.")
        )
    if chosen in ("two-stage", "bias-corrected", "hybrid", "hybrid-stops") and not planning.get(
        "phase5"
    ):
        raise ValueError("This method requires a saved Phase 5 planning artifact.")
    if chosen == "two-stage":
        stopping = planning.get("hurdle")
        if not isinstance(stopping, HurdleStopping):
            raise ValueError(
                "Two-stage stopping is unavailable; legacy stopping cannot substitute under its label."
            )
    elif chosen == "hybrid-stops":
        stopping = planning.get("hybrid_hurdle")
        if not isinstance(stopping, HurdleStopping):
            raise ValueError("The saved hybrid stopping component is unavailable.")
    elif (
        chosen == "physical"
        and break_mode == "historical"
        and planning.get("operational_default") == "two-stage"
    ):
        stopping = planning.get("hurdle")
    correction = planning.get("corrections", {}).get(chosen)
    if chosen in ("bias-corrected", "hybrid", "hybrid-stops") and correction is None:
        raise ValueError(
            "The saved correction for the selected method is unavailable; no differently labelled fallback was applied."
        )
    physical_mode = "minimal" if chosen in ("CW-auto", "bias-corrected", "hybrid") else break_mode
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
        stopping_model=stopping,
        break_mode=physical_mode,
    )
    result["physical_elapsed_s"] = result["elapsed_s"]
    result["statistical_correction_s"] = 0.0
    if correction is not None:
        features = route_features(segments, bike)
        baseline = historical_baseline(rides, features["distance_km"], departure)
        total, detail = correction.predict(
            result["elapsed_s"], baseline, features["distance_km"], departure, features
        )
        result = apply_duration_correction(result, total, detail)
    result["journey_method"] = chosen
    result["journey_method_requested"] = journey_method
    result["journey_choice_reason"] = reason
    result["operational_default"] = planning.get(
        "operational_default", planning.get("recommended_method")
    )
    result["stopping_method"] = (
        "two-stage"
        if physical_mode == "historical" and isinstance(stopping, HurdleStopping)
        else "legacy residual stops"
        if physical_mode == "historical" and stopping is not None
        else physical_mode
    )
    result["break_mode"] = break_mode
    result["hr_recommendation"] = recommendation
    result["hr_selection"] = "manual override" if manual_hr else "automatic suggestion"
    points = route["points"]
    destination = (float(points.lat.iloc[-1]), float(points.lon.iloc[-1]))
    daylight = daylight_plan(
        departure, result["elapsed_s"], destination, daylight_reference, safety_minutes
    )
    key = chosen if chosen != "physical" else "CW+stops-auto"
    if getattr(model, "variant", "CW") != "CW":
        key = getattr(model, "variant", "C") + ("-auto" if break_mode == "minimal" else "+stops-auto")
    calibration = planning.get("calibrations", {}).get(key, TailCalibration())
    if chosen == "two-stage" and key not in planning.get("calibrations", {}):
        raise ValueError(
            "Two-stage elapsed calibration is unavailable; the old model's intervals cannot substitute."
        )
    uncertainty = calibration.estimate(
        result["elapsed_s"],
        result["distance_m"] / 1000,
        departure,
        forecast=weather_provider is not None,
        break_mode=break_mode,
    )
    uncertainty["calibration_key"] = key
    if planning.get("operational_blocker"):
        uncertainty["provisional"] = True
        uncertainty["probability_supported"] = False
        uncertainty["warnings"].append(planning["operational_blocker"])
    uncertainty["calibration_scenario"] = (
        "Observed traffic, automatic suggested HR, historical breaks and observed weather"
    )
    trained_rides = rides[rides.id.astype(str).isin(getattr(model, "train_ids", []))]
    if (
        len(trained_rides)
        and pd.to_datetime(trained_rides.date, utc=True, format="mixed").ge(departure).any()
    ):
        uncertainty["provisional"] = True
        uncertainty["probability_supported"] = False
        uncertainty["warnings"].append(
            "This past departure uses the current fitted riding model, which includes later observations; it is a planning simulation, not a historical backtest."
        )
    if manual_hr or scenario != "observed":
        uncertainty["provisional"] = True
        uncertainty["probability_supported"] = False
        if manual_hr:
            uncertainty["warnings"].append(
                "Manual target HR is a planning scenario, not retrospective actual measured HR. Automatic-HR calibration is only a reference estimate."
            )
        if scenario != "observed":
            uncertainty["warnings"].append(
                "Full-stop planning is a hypothetical scenario. Historical aggregate errors and intervals were validated in observed-behavior mode."
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
    conservative = uncertainty["conservative_s"]
    daylight["conservative_before_cutoff"] = (
        bool(conservative <= available) if conservative is not None and available is not None else None
    )
    daylight["conservative_cutoff_margin_minutes"] = (
        (available - conservative) / 60 if conservative is not None and available is not None else None
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


def fit_phase5(store, settings, report=None):
    """Save a separate artifact; retain every previous riding and journey model."""
    report = report or store.get_state("phase5_validation", {})
    if report.get("status") != "complete":
        raise ValueError("Complete Phase 5 chronological validation before fitting planning.")
    path = store.root / "models/journey.joblib"
    original = joblib.load(path) if path.exists() else fit_planning(store, settings)
    planning = dict(original)
    families = report["frozen_selection"]["families"]
    rides = store.rides()
    rides["date"] = rides.activity_start_utc.fillna(rides.date)
    planning["hurdle"] = HurdleStopping.fit(
        rides, families["two-stage"]["half_life"], families["two-stage"]["conditional"]
    )
    from .phase5_validation import STOP_GRID

    planning["hybrid_hurdle"] = HurdleStopping.fit(rides, **STOP_GRID[families["hybrid-stops"]["stop"]])
    planning["corrections"] = {
        key: DurationCorrection(params["kind"], params["parameter"], rows=report["correction_rows"])
        for key, params in families.items()
        if key != "two-stage"
    }
    metrics = {r["variant"]: r for r in report["all_later"]["comparisons"]}
    planning["calibrations"] = {
        key: TailCalibration(
            rows, metrics.get(key, {}).get("coverage_80"), metrics.get(key, {}).get("coverage_90")
        )
        for key, rows in report["calibration_rows"].items()
    }
    planning["recommended_method"] = report.get("recommended_method")
    planning["selection_reason"] = (
        f"Earlier chronological selection chose {report['selected_method']}. "
        + (
            "Reference-gate result is recorded separately; the operational default requires artifact and calibration verification."
        )
    )
    planning["phase5"] = True
    planning["trained_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    archive_artifact(store.root / "models/journey-phase5.joblib")
    joblib.dump(planning, store.root / "models/journey-phase5.joblib", compress=3)
    store.set_state("phase5_stale", False)
    store.set_state(
        "phase5_training",
        {
            "trained_at": planning["trained_at"],
            "recommended_method": planning["recommended_method"],
            "selection_reason": planning["selection_reason"],
        },
    )
    (store.root / "reports/phase5-training.json").write_text(
        json.dumps(store.get_state("phase5_training"), indent=2), encoding="utf-8"
    )
    return load_operational_planning(store)


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
