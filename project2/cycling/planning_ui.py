"""Elapsed-first controls and displays within the existing four-page app."""

from datetime import datetime, time
import hashlib
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .daylight import coordinate_timezone, latest_departure, local_departure
from .hr import mark_manual_hr, recommend_hr, update_hr_state, use_suggested_hr
from .journey import load_planning, plan_ride
from .pipeline import prepare_route, route_for_model
from .stopping import route_features


def duration(seconds):
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m"


def clock(value, zone=None):
    if not value:
        return "Unavailable"
    moment = pd.Timestamp(value)
    if zone:
        moment = moment.tz_convert(zone)
    return moment.strftime("%a %b %d, %H:%M %Z")


def route_prediction_page(store, settings, model):
    st.title("Plan your finishing time")
    if model is None:
        st.info("Import rides and train the riding model on Historical Data first.")
        return
    choices = [
        v for v in ("CW", "C", "B", "A") if (store.root / "models" / f"model-{v}.joblib").exists()
    ]
    chosen = st.session_state.get(
        "prediction_variant", choices[0] if choices else getattr(model, "variant", "C")
    )
    if chosen in choices and chosen != getattr(model, "variant", "C"):
        model = joblib.load(store.root / "models" / f"model-{chosen}.joblib")
    planning = load_planning(store)
    if store.get_state("model_stale", False):
        st.warning("The riding model predates changes to the history; retrain on Historical Data.")
    if not planning.get("stopping"):
        st.warning(
            "Usual personal stops have not been fitted. Train elapsed-time planning on Historical Data."
        )
    elif store.get_state("journey_stale", False):
        st.warning(
            "Elapsed/HR planning predates changes to the history. Revalidate planning on Historical Data."
        )
    upload = st.file_uploader("Planned Komoot / GPX route", type=["gpx"], key="planning_gpx")
    correction = st.session_state.get("planning_elevation")
    if upload is not None:
        data, elevation_data = upload.getvalue(), correction.getvalue() if correction else None
        token = hashlib.sha256(data + (elevation_data or b"")).hexdigest()
        if st.session_state.get("loaded_route_token") != token:
            try:
                with st.spinner("Preparing terrain from the GPX and existing OSM cache…"):
                    st.session_state.route = prepare_route(
                        data, upload.name, store, settings, elevation_data
                    )
                st.session_state.loaded_route_token = token
                points = st.session_state.route["points"]
                st.session_state.departure_timezone = coordinate_timezone(
                    points.lat.iloc[0], points.lon.iloc[0]
                )
            except (ValueError, OSError) as exc:
                st.error(f"The GPX could not be prepared: {exc}")
                return
    route = st.session_state.get("route")
    token = st.session_state.get("loaded_route_token", "cached-route")
    zone_name = st.session_state.get("departure_timezone", "America/Toronto")
    zone_valid = True
    try:
        departure_zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError:
        departure_zone = ZoneInfo("America/Toronto")
        zone_valid = False
        st.error("Unknown departure timezone. Correct the IANA name in the timezone settings below.")
    dates, times = st.columns(2)
    day = dates.date_input(
        "Planned departure date", datetime.now(departure_zone).date(), key="planning_date"
    )
    departure_clock = times.time_input("Planned departure time", time(9, 0), key="planning_time")
    if route:
        features = route_features(route["segments"], st.session_state.get("planning_bike"))
        reference = datetime.combine(day, departure_clock).replace(tzinfo=departure_zone).isoformat()
        suggestion = recommend_hr(
            store.rides(),
            features["distance_km"],
            reference,
            planning.get("hr_half_life", settings.hr_half_life_days),
            features,
            model.hr_range,
            settings.hr_min_comparable,
            settings.hr_distance_bandwidth,
        )
        update_hr_state(st.session_state, token, suggestion)
        st.caption(
            f"Loaded {route['filename']} · {features['distance_km']:.1f} km. Upload another GPX to change the route."
        )
    else:
        suggestion = {
            "suggested_hr": None,
            "evidence": "unavailable",
            "comparable_rides": 0,
            "warnings": [],
        }
        st.session_state.setdefault("target_hr", None)
    target = st.number_input(
        "Target average heart rate (BPM)",
        min_value=35,
        max_value=230,
        value=None,
        step=1,
        key="target_hr",
        on_change=mark_manual_hr,
        args=(st.session_state,),
    )
    if suggestion.get("suggested_hr") is not None:
        st.caption(
            f"Suggested HR: {suggestion['suggested_hr']} BPM · {suggestion['comparable_rides']} comparable measured-HR rides · {suggestion['evidence']} evidence · {suggestion['method']}."
        )
        bounds = suggestion["distance_range_km"]
        st.caption(
            f"Evidence distances {bounds[0]:.0f}–{bounds[1]:.0f} km; weighted median {suggestion['median_distance_km']:.0f} km. Recency half-life {suggestion['half_life_days']:g} days. Imputed HR is excluded."
        )
        st.button("Use suggested HR", on_click=use_suggested_hr, args=(st.session_state,))
    else:
        st.info(
            "Upload a route for a measured-HR suggestion. If reliable HR history is unavailable, enter your target manually."
        )
    if st.session_state.get("hr_manual", False):
        st.caption(
            "Manual HR override is active. It is preserved across reruns and route changes until you use the suggestion."
        )
    for warning in suggestion.get("warnings", []):
        st.warning(warning)
    st.caption(
        "HR is averaged over moving time, including braking and acceleration; personal and traffic stops are excluded."
    )
    if model.hr_available:
        st.caption(
            f"Fitted effort support {model.hr_range[0]:.0f}–{model.hr_range[1]:.0f} BPM. Requests outside this support remain visible but the modeled effort is limited."
        )
    else:
        st.warning(
            "The fitted riding model cannot condition on HR yet; terrain-based pacing will be used."
        )
    bike = st.selectbox("Bicycle", ["Any / unknown"] + model.support["bikes"], key="planning_bike")
    traffic_model = getattr(model, "variant", "C") in ("C", "CW")
    scenario = st.selectbox(
        "Stop-sign behavior scenario",
        ["Full stops", "Observed historical behavior"]
        if traffic_model
        else ["Rolling baseline; mapped traffic excluded"],
    )
    break_label = st.selectbox(
        "Break-planning mode", ["Historical behavior", "Custom break allowance", "Minimal breaks"]
    )
    break_mode = {
        "Historical behavior": "historical",
        "Custom break allowance": "custom",
        "Minimal breaks": "minimal",
    }[break_label]
    break_minutes = (
        st.number_input("Custom personal break allowance (minutes)", 0.0, 1440.0, 0.0, step=5.0)
        if break_mode == "custom"
        else 0.0
    )
    if break_mode == "minimal":
        st.info("Optimistic personal-stop scenario. Traffic waits and braking/acceleration still apply.")
    elif break_mode == "custom":
        st.caption(
            "Custom allowance replaces usual personal stopping; traffic waiting remains separate."
        )
    safety = st.number_input(
        "Daylight safety margin (minutes)", 0, 240, settings.daylight_safety_minutes, step=5
    )
    reference_label = st.selectbox("Daylight reference", ["Sunset", "End of civil twilight"])
    daylight_reference = "sunset" if reference_label == "Sunset" else "civil_twilight"
    if daylight_reference == "civil_twilight":
        st.caption("Civil twilight provides reduced light and is not equivalent to full daylight.")
    with st.expander("Weather, model, timezone and elevation settings"):
        if choices:
            st.selectbox(
                "Prediction model",
                choices,
                key="prediction_variant",
                format_func=lambda v: {
                    "CW": "C + Weather (default)",
                    "C": "C · Terrain and traffic",
                    "B": "B · Surface and road",
                    "A": "A · Geometry baseline",
                }[v],
            )
        st.text_input("Departure timezone", zone_name, key="departure_timezone")
        occurrence = st.selectbox(
            "Repeated daylight-saving hour",
            ["Require unambiguous time", "First occurrence", "Second occurrence"],
        )
        st.file_uploader("Optional elevation profile CSV", type=["csv"], key="planning_elevation")
        st.caption("Elevation correction requires distance_m and elevation_m covering the whole route.")
        forecasts = st.checkbox(
            "Use available ECCC hourly forecasts",
            value=getattr(model, "variant", "C") == "CW",
            disabled=getattr(model, "variant", "C") != "CW",
            key="planning_forecast",
        )
        neutral = st.checkbox("Weather-neutral prediction", value=False)
        st.caption(
            "Short-horizon regional City Page forecasts supply wind and temperature. Missing sections use the fitted terrain baseline. Predicted personal stops shift downstream arrivals."
        )
    if st.button("Predict total ride time", type="primary", disabled=route is None or not zone_valid):
        try:
            departure = local_departure(
                day,
                departure_clock,
                st.session_state.departure_timezone,
                {"First occurrence": "first", "Second occurrence": "second"}.get(occurrence),
            )
            forecast = None
            if forecasts and not neutral and getattr(model, "variant", "C") == "CW":
                from .forecast import CityForecast

                forecast = CityForecast(store.root, settings)
                forecast.prepare(route_for_model(route, model)[0])
            with st.spinner("Solving riding, expected stops and segment arrival weather…"):
                result = plan_ride(
                    model,
                    route,
                    store.rides(),
                    settings,
                    planning,
                    target,
                    None if bike == "Any / unknown" else bike,
                    departure,
                    "full_stop" if scenario == "Full stops" else "observed",
                    break_mode,
                    break_minutes,
                    forecast,
                    neutral,
                    daylight_reference,
                    safety,
                    st.session_state.get("hr_manual", False),
                )
                result["weather_forecast_enabled"] = forecast is not None
                result["bike"] = None if bike == "Any / unknown" else bike
                if forecast is not None:
                    comparator = plan_ride(
                        model,
                        route,
                        store.rides(),
                        settings,
                        planning,
                        target,
                        result["bike"],
                        departure,
                        result["scenario"],
                        break_mode,
                        break_minutes,
                        weather_neutral=True,
                        manual_hr=st.session_state.get("hr_manual", False),
                    )
                    result["neutral_comparison"] = {
                        k: comparator[k] for k in ("rolling_kmh", "moving_s", "elapsed_s")
                    }
                st.session_state.result = result
            st.success("Plan ready. Open Results for finishing time, daylight and uncertainty.")
            a, b, c = st.columns(3)
            a.metric("Estimated total ride time", duration(result["elapsed_s"]))
            b.metric(
                "Expected finish", pd.Timestamp(result["daylight"]["finish_local"]).strftime("%H:%M %Z")
            )
            c.metric(
                "Remaining daylight",
                f"{result['daylight']['remaining_daylight_minutes']:.0f} min"
                if result["daylight"]["remaining_daylight_minutes"] is not None
                else "No sunset",
            )
        except Exception as exc:
            st.error(f"Could not predict the complete ride: {exc}")
    if route and route.get("coverage"):
        coverage = route["coverage"]
        st.caption(
            f"OSM route coverage: {coverage['matched_km'] / max(1e-9, coverage['total_km']):.0%} matched · {coverage['surfaces'].get('unknown', 0) / max(1e-9, coverage['total_km']):.0%} unknown surface."
        )


def finish_results(result, route, store, settings):
    daylight = result.get("daylight")
    if not daylight:
        st.info("Recalculate this earlier prediction to include learned personal stops and daylight.")
        return
    zone = daylight["destination_timezone"]
    a, b, c, d = st.columns(4)
    a.metric("Estimated total ride time", duration(result["elapsed_s"]))
    b.metric("Expected finish time", pd.Timestamp(daylight["finish_local"]).strftime("%H:%M %Z"))
    c.metric(
        "Sunset at destination",
        pd.Timestamp(daylight["sunset"]).strftime("%H:%M %Z") if daylight["sunset"] else "No sunset",
    )
    remaining = daylight["remaining_daylight_minutes"]
    d.metric("Remaining daylight", f"{remaining:+.0f} min" if remaining is not None else "Unavailable")
    st.caption(
        f"Departure {clock(result['departure_utc'], zone)} · expected arrival {clock(daylight['finish_local'])} · destination timezone {zone}. Cutoff: {clock(daylight['cutoff'])}, including {daylight['safety_minutes']} min margin."
    )
    st.caption(
        f"Requested moving HR {result['requested_hr'] if result['requested_hr'] is not None else 'unavailable'} BPM · {result.get('hr_selection', 'manual')} · break mode {result.get('break_mode', 'custom')}. Civil twilight ends {clock(daylight['civil_twilight'])}."
    )
    uncertainty = result["elapsed_uncertainty"]
    if daylight["conservative_finish_local"]:
        st.metric(
            "Conservative finish estimate (90th percentile)",
            clock(daylight["conservative_finish_local"]),
        )
        st.caption(
            f"Median estimated finish {clock(daylight['median_finish_local'])}. {'Provisional' if uncertainty['provisional'] else 'Historically calibrated'} using {uncertainty['samples']} earlier held-out elapsed errors. This is separate from the ledger's expected duration."
        )
    else:
        st.warning(
            "A conservative finish estimate is unavailable: too little elapsed calibration evidence."
        )
    for coverage, limits in uncertainty["intervals_s"].items():
        start = pd.Timestamp(result["departure_utc"])
        lower, upper = (start + pd.Timedelta(seconds=seconds) for seconds in limits)
        st.write(
            f"{coverage}% empirical finish interval: {clock(lower.isoformat(), zone)} to {clock(upper.isoformat(), zone)}{' · provisional' if uncertainty['provisional'] else ''}."
        )
    probability = daylight["before_cutoff_probability"]
    if probability is None:
        st.warning(
            "Likelihood of finishing before the cutoff is not statistically supported; no percentage is shown. Use the conservative estimate and leave additional room for unexpected delays."
        )
    else:
        st.metric("Likelihood of finishing before the cutoff", f"{probability:.0%}")
    if daylight["cutoff_margin_minutes"] is not None and daylight["cutoff_margin_minutes"] < 0:
        st.error("Expected arrival is after the selected daylight cutoff.")
    elif (
        daylight["conservative_finish_local"]
        and daylight["cutoff"]
        and pd.Timestamp(daylight["conservative_finish_local"]) > pd.Timestamp(daylight["cutoff"])
    ):
        st.warning("The conservative arrival estimate crosses the daylight cutoff.")
    breakdown = pd.DataFrame(
        {
            "Component": ["Cruising", "Moving slowdowns", "Traffic waiting", "Discretionary stopping"],
            "Minutes": [
                result["cruising_time_s"] / 60,
                (result["moving_s"] - result["cruising_time_s"]) / 60,
                result["stationary_s"] / 60,
                result["personal_break_s"] / 60,
            ],
        }
    )
    st.plotly_chart(
        px.bar(breakdown, x="Minutes", y="Component", orientation="h", title="Complete time ledger"),
        width="stretch",
    )
    estimate = result.get("break_estimate")
    if estimate:
        st.caption(
            f"Personal stopping: {estimate['method']} distribution from {estimate['samples']} comparable reliable rides, {estimate['effective_samples']:.1f} effective observations; {estimate['evidence']} evidence. Some unattributed stops may be unmapped traffic."
        )
    frame = result["segments"].copy()
    frame["Distance (km)"] = frame.end_m / 1000
    frame["Expected clock time"] = (
        pd.to_datetime(frame.segment_end_time_utc, utc=True).dt.tz_convert(zone).dt.tz_localize(None)
    )
    figure = px.line(
        frame,
        x="Distance (km)",
        y="Expected clock time",
        title=f"Expected route progress · clock in {zone}",
    )
    for name, value in [
        ("Sunset", daylight["sunset"]),
        ("Civil twilight", daylight["civil_twilight"]),
        ("Safety cutoff", daylight["cutoff"]),
    ]:
        if value:
            y = pd.Timestamp(value).tz_localize(None).isoformat()
            figure.add_shape(type="line", xref="paper", x0=0, x1=1, y0=y, y1=y, line={"dash": "dash"})
            figure.add_annotation(xref="paper", x=1, y=y, text=name, showarrow=False)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Personal stopping is an expected distribution across broad progress bands, not predicted café or bathroom locations."
    )
    checkpoints = np.arange(25, result["distance_m"] / 1000, 25)
    if len(checkpoints):
        elapsed = np.interp(
            checkpoints, np.r_[0, frame["Distance (km)"]], np.r_[0, frame.accumulated_elapsed_s]
        )
        arrivals = pd.Timestamp(result["departure_utc"]) + pd.to_timedelta(elapsed, unit="s")
        st.dataframe(
            pd.DataFrame(
                {
                    "Distance (km)": checkpoints,
                    "Expected arrival": [clock(x.isoformat(), zone) for x in arrivals],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    if daylight["sunset"]:
        seconds = (
            pd.Timestamp(daylight["sunset"]) - pd.Timestamp(result["departure_utc"])
        ).total_seconds()
        if 0 <= seconds <= result["elapsed_s"]:
            position = np.interp(
                seconds, np.r_[0, frame.accumulated_elapsed_s], np.r_[0, frame["Distance (km)"]]
            )
            st.caption(
                f"Expected progress at sunset: approximately {position:.1f} km. Timing uncertainty also makes this position uncertain."
            )
    criterion_label = st.selectbox(
        "Latest departure criterion",
        ["Conservative 90th percentile", "Median estimated finish", "Expected finish (optimistic)"],
    )
    criterion = {
        "Conservative 90th percentile": "conservative",
        "Median estimated finish": "median",
        "Expected finish (optimistic)": "expected",
    }[criterion_label]
    if daylight["cutoff"] and st.button("Calculate latest departure", key="latest-departure"):
        model = joblib.load(store.root / "models" / f"model-{result['model_variant']}.joblib")
        planning = load_planning(store)
        provider = None
        if result.get("weather_forecast_enabled"):
            from .forecast import CityForecast

            provider = CityForecast(store.root, settings)
            provider.prepare(route_for_model(route, model)[0])

        def candidate(departure):
            return plan_ride(
                model,
                route,
                store.rides(),
                settings,
                planning,
                result["requested_hr"],
                result.get("bike"),
                departure,
                result["scenario"],
                result["break_mode"],
                result["personal_break_s"] / 60 if result["break_mode"] == "custom" else 0,
                provider,
                result["weather_neutral"],
                daylight["reference"],
                daylight["safety_minutes"],
                result.get("hr_selection") == "manual override",
            )

        with st.spinner("Recalculating candidate departures with changing weather and traffic…"):
            result["latest_departure"] = latest_departure(
                candidate, daylight["cutoff"], criterion=criterion
            )
    if result.get("latest_departure"):
        latest = result["latest_departure"]
        st.write(
            f"Latest {'provisional ' if latest['provisional'] else ''}departure: {clock(latest['departure_utc'], zone)}"
        )
        st.caption(f"{latest['evaluations']} recalculated departures. {latest['note']}")


def elapsed_performance(store):
    st.title("Elapsed-time performance")
    report = store.get_state("journey_validation")
    quality = store.get_state("timing_quality", {})
    st.caption(
        f"Reliable complete duration: {quality.get('reliable_elapsed', 0)} of {quality.get('rides', 0)} rides. Reliable measured-HR recommendation evidence: {quality.get('reliable_measured_hr', 0)} rides."
    )
    if not report:
        st.info(
            "Train and validate elapsed planning on Historical Data to evaluate complete finishing time."
        )
        return
    comparison = pd.DataFrame(report["comparisons"])
    primary = comparison[comparison.variant.eq("CW+stops-auto")].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Total elapsed-time MAE", f"{primary.elapsed_mae_minutes:.1f} min")
    b.metric("Median absolute elapsed error", f"{primary.median_absolute_elapsed_minutes:.1f} min")
    c.metric("Elapsed bias", f"{primary.elapsed_bias_minutes:+.1f} min")
    st.dataframe(comparison, hide_index=True, width="stretch")
    baseline = comparison[comparison.variant.eq("simple-elapsed")]
    if len(baseline) and primary.elapsed_mae_minutes > baseline.elapsed_mae_minutes.iloc[0]:
        st.warning(
            f"The learned elapsed model did not beat the simple duration baseline on these final rides ({primary.elapsed_mae_minutes:.1f} versus {baseline.elapsed_mae_minutes.iloc[0]:.1f} min MAE). The C+Weather default is retained; greater complexity has not established superior overall accuracy."
        )
    st.caption(
        f"{report['final_rides']} same final route/departure folds; metrics exclude unreliable durations identically. Model selection used {report['selection_rides']} earlier rides ending {report['selection_last'][:10]}. Each effort suggestion, fit and calibration sees only earlier activities."
    )
    st.caption(report["weather_evaluation"])
    st.warning(report["probability_note"])
    hr = report["hr_metrics"]
    a, b, c = st.columns(3)
    a.metric(
        "Suggested HR MAE", f"{hr['mae_bpm']:.1f} BPM" if hr["mae_bpm"] is not None else "Unavailable"
    )
    b.metric(
        "Global-median HR baseline MAE",
        f"{hr['global_median_mae_bpm']:.1f} BPM"
        if hr["global_median_mae_bpm"] is not None
        else "Unavailable",
    )
    c.metric("Measured-HR post-selection tests", hr["rides"])
    st.caption(
        f"HR recency half-life selected separately: {report['selected_hr_half_life']} days. Stopping approach: {report['selected_stopping_method']}."
    )
    st.subheader("Automatic versus known measured HR · paired eligible rides")
    st.dataframe(pd.DataFrame(report["measured_hr_comparisons"]), hide_index=True, width="stretch")
    diagnostic = st.selectbox(
        "Elapsed diagnostic model",
        comparison.variant.tolist(),
        index=comparison.variant.tolist().index("CW+stops-auto"),
    )
    st.dataframe(
        pd.DataFrame(
            [{"Distance group": key, **value} for key, value in report["groups"][diagnostic].items()]
        ),
        hide_index=True,
        width="stretch",
    )
    rows = pd.DataFrame(report["rows"])
    selected = rows[rows.variant.eq(diagnostic)].copy()
    selected["Actual elapsed (min)"] = selected.actual_elapsed_s / 60
    selected["Predicted elapsed (min)"] = selected.predicted_elapsed_s / 60
    st.plotly_chart(
        px.scatter(
            selected[selected.elapsed_reliable],
            x="Actual elapsed (min)",
            y="Predicted elapsed (min)",
            hover_data=["date", "distance_km"],
            title="Complete duration · chronological held-out rides",
        ),
        width="stretch",
    )
    selected["Signed error (min)"] = selected.elapsed_error_s / 60
    selected["Absolute error (min)"] = selected.absolute_elapsed_error_s / 60
    st.dataframe(
        selected.drop(columns=["warnings", "hr_training_ids"], errors="ignore").sort_values(
            "Absolute error (min)", ascending=False
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Click table headers to sort. Moving, traffic-attribution and unattributed-stop errors are accounting diagnostics; they do not establish causal effects of fatigue, weather or rider intent."
    )
    with st.expander("Selection candidates and HR distance-group accuracy"):
        st.dataframe(pd.DataFrame(report["stopping_candidates"]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(report["hr_candidates"]), hide_index=True, width="stretch")
        st.dataframe(
            pd.DataFrame(
                [{"Distance group": key, **value} for key, value in report["hr_groups"].items()]
            ),
            hide_index=True,
            width="stretch",
        )
