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
from .phase5_ui import phase5_performance


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
        st.selectbox(
            "Total-duration method",
            ["automatic", "CW+stops-auto", "CW-auto"]
            + (
                ["two-stage", "bias-corrected", "hybrid", "hybrid-stops"]
                if planning.get("phase5")
                else []
            ),
            key="journey_method",
            format_func=lambda value: {
                "automatic": "Automatic · use supported default",
                "CW+stops-auto": "Existing residual stopping model",
                "CW-auto": "C + Weather · no discretionary allowance",
                "two-stage": "Two-stage stopping candidate",
                "bias-corrected": "Bias-corrected elapsed candidate",
                "hybrid": "Hybrid elapsed candidate",
                "hybrid-stops": "Hybrid with two-stage stopping candidate",
            }[value],
        )
        st.caption(
            planning.get(
                "selection_reason",
                "Existing personalized stopping default. New methods become available after chronological validation.",
            )
        )
        st.caption(
            "Observed behavior matches historical elapsed validation. Full stops, manual HR, custom/minimal breaks and future forecasts are planning scenarios with provisional confidence."
        )
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
                    journey_method=st.session_state.get("journey_method", "automatic"),
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
                        journey_method=st.session_state.get("journey_method", "automatic"),
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
    c.metric("Conservative finish time", clock(daylight.get("conservative_finish_local")))
    margin = daylight.get("conservative_cutoff_margin_minutes")
    d.metric(
        "Conservative daylight margin", f"{margin:+.0f} min" if margin is not None else "Unavailable"
    )
    a, b, c = st.columns(3)
    a.metric("Sunset at destination", clock(daylight["sunset"]))
    b.metric("Daylight safety cutoff", clock(daylight["cutoff"]))
    before = daylight.get("conservative_before_cutoff")
    c.metric(
        "Conservative finish before cutoff",
        "Yes · provisional"
        if before and result["elapsed_uncertainty"]["provisional"]
        else "Yes"
        if before
        else "No"
        if before is False
        else "Unknown",
    )
    st.caption(
        f"Total-duration method: {result.get('journey_method', 'existing stopping')} · {result.get('journey_choice_reason', '')}"
    )
    phase5 = store.get_state("phase5_validation", {})
    if phase5:
        metrics = next(
            (m for m in phase5["fresh"]["comparisons"] if m["variant"] == result.get("journey_method")),
            None,
        )
        if metrics:
            st.caption(
                f"Historical aggregate total-time MAPE {metrics['elapsed_mape']:.2f}% on {metrics['rides']} fresh later rides in observed traffic mode with automatic HR. This is not an error bound for this route or full-stop scenario."
            )
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
    correction = result.get("statistical_correction_s", 0)
    if correction:
        breakdown.loc[len(breakdown)] = ["Statistical total-duration adjustment", correction / 60]
        st.write(
            f"Physical ledger {duration(result['physical_elapsed_s'])}; statistical adjustment {correction / 60:+.1f} min. This adjustment contains interruption and riding-model bias, rather than a measured stop."
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
        title=f"Physical route progress · clock in {zone}",
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
    if correction:
        st.caption(
            "The statistical total-duration adjustment is unallocated. This physical progress chart and checkpoints exclude it; use the adjusted headline and conservative finish for daylight decisions."
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
                journey_method=result.get("journey_method_requested", "automatic"),
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
    phase5 = store.get_state("phase5_validation", {})
    if phase5.get("status") == "complete":
        phase5_performance(phase5, store)
        return
    from .accuracy import MODEL_LABELS, PRIMARY_VARIANTS
    from .journey_validation import DISTANCE_LABELS, distance_group, percentage_report

    st.title("Percentage-based elapsed-time accuracy")
    report = percentage_report(store.get_state("journey_validation", {}))
    quality = store.get_state("timing_quality", {})
    st.caption(
        f"Reliable complete duration: {quality.get('reliable_elapsed', 0)} of {quality.get('rides', 0)} rides. Reliable measured-HR recommendation evidence: {quality.get('reliable_measured_hr', 0)} rides."
    )
    if not report or not report.get("rows"):
        st.info(
            "Train and validate elapsed planning on Historical Data to evaluate complete finishing time."
        )
        return
    comparison = pd.DataFrame(report["comparisons"])
    variants = comparison.variant.tolist()
    diagnostic = st.selectbox(
        "Elapsed diagnostic model",
        variants,
        index=variants.index("CW+stops-auto") if "CW+stops-auto" in variants else 0,
        format_func=lambda value: MODEL_LABELS.get(value, value),
    )
    primary = comparison[comparison.variant.eq(diagnostic)].iloc[0]

    def pct(value, signed=False):
        return (
            "Unavailable"
            if value is None or pd.isna(value)
            else f"{value:+.1f}%"
            if signed
            else f"{value:.1f}%"
        )

    def percentage_table(values, label):
        frame = pd.DataFrame(values)
        return frame.reindex(
            columns=[
                label,
                "rides",
                "elapsed_mape",
                "elapsed_mdape",
                "elapsed_mspe",
                "elapsed_p90_ape",
            ]
        ).rename(
            columns={
                "rides": "Evaluated rides",
                "elapsed_mape": "MAPE (%)",
                "elapsed_mdape": "Median APE (%)",
                "elapsed_mspe": "Signed bias (%)",
                "elapsed_p90_ape": "P90 APE (%)",
            }
        )

    st.caption(
        f"Accuracy model: {MODEL_LABELS.get(diagnostic, diagnostic)}. The initial view is the current C+Weather + stopping planning model."
    )
    a, b, c = st.columns(3)
    a.metric("Total Elapsed-Time MAPE (%)", pct(primary.elapsed_mape))
    b.metric("Median absolute percentage error", pct(primary.elapsed_mdape))
    c.metric("Signed percentage bias", pct(primary.elapsed_mspe, signed=True))
    a, b, c = st.columns(3)
    a.metric("P90 absolute percentage error", pct(primary.elapsed_p90_ape))
    b.metric("Rides underestimated", pct(primary.underprediction_percentage))
    c.metric("Total evaluated rides", int(primary.rides))
    st.caption(
        "MAPE averages one absolute percentage error per ride using that ride's actual total duration. "
        "Negative signed bias means optimistic early finishes; positive means durations are overstated. "
        "Median APE describes the typical middle error, while MAPE includes every eligible error."
    )
    st.info(
        "P90 APE is a historical error percentile, not a calibrated 90% prediction interval or a before-dark guarantee. "
        "MAPE is not a ± confidence range. Daylight planning retains the independently calibrated asymmetric elapsed intervals."
    )
    cohort = report["percentage_comparison_cohort"]
    st.subheader("Same-ride model comparison")
    main = comparison[comparison.variant.isin(PRIMARY_VARIANTS)].copy()
    main["Model"] = main.variant.map(MODEL_LABELS)
    summary = percentage_table(main, "Model")
    st.dataframe(summary, hide_index=True, width="stretch")
    if main.elapsed_mape.notna().any():
        st.plotly_chart(
            px.bar(
                summary,
                x="Model",
                y="MAPE (%)",
                hover_data=["Evaluated rides", "Median APE (%)", "Signed bias (%)", "P90 APE (%)"],
                title="Total elapsed-time MAPE · identical held-out rides",
            ),
            width="stretch",
        )
    ranking = report["percentage_ranking"]
    if ranking["lowest_mape_variant"]:
        st.write(
            f"Lowest final-set MAPE: **{MODEL_LABELS[ranking['lowest_mape_variant']]}**. "
            f"Lowest minute MAE: **{MODEL_LABELS[ranking['lowest_minute_mae_variant']]}**. "
            f"{'The rankings differ.' if ranking['winners_differ'] else 'Both criteria select the same observed winner.'}"
        )
    default = comparison[comparison.variant.eq("CW+stops-auto")]
    baseline = comparison[comparison.variant.eq("simple-elapsed")]
    if (
        len(default)
        and len(baseline)
        and pd.notna(default.elapsed_mape.iloc[0])
        and pd.notna(baseline.elapsed_mape.iloc[0])
        and default.elapsed_mape.iloc[0] > baseline.elapsed_mape.iloc[0]
    ):
        st.warning(
            f"The deployed stopping model did not beat the simple baseline in percentage accuracy "
            f"({default.elapsed_mape.iloc[0]:.2f}% versus {baseline.elapsed_mape.iloc[0]:.2f}% MAPE). "
            "C+Weather and the configured stopping allowance are retained; the final-set ranking does not retune or replace them."
        )
    st.caption(
        f"{cohort['evaluated_rides']} of {cohort['candidate_rides']} saved final rides are shared by all four primary models. "
        f"{len(cohort['excluded_rides'])} rides excluded from this paired comparison. "
        "Unreliable, nonpositive/nonfinite actual durations, invalid predictions and missing model predictions are explicitly audited."
    )
    if cohort["excluded_rides"]:
        st.dataframe(pd.DataFrame(cohort["excluded_rides"]), hide_index=True, width="stretch")
    st.caption(
        f"Model selection used {report['selection_rides']} earlier folds ending {report['selection_last'][:10]} "
        f"with {'per-ride MAPE' if report.get('stopping_selection_metric') == 'elapsed_mape' else 'minute-based MAE'}. "
        "Final percentages are reporting only. Each effort suggestion, fit and calibration sees only earlier activities."
    )
    st.caption(report["weather_evaluation"])
    st.warning(report["probability_note"])

    st.subheader("Underprediction · late-arrival risk")
    a, b, c = st.columns(3)
    a.metric("Underpredicted rides", f"{int(primary.underpredicted_rides)} / {int(primary.rides)}")
    b.metric("Mean underprediction", pct(primary.underprediction_mape))
    c.metric("Median underprediction", pct(primary.underprediction_median_percentage))
    a, b, c = st.columns(3)
    a.metric("Largest underprediction", pct(primary.underprediction_max_percentage))
    b.metric("P90 underprediction", pct(primary.underprediction_p90_percentage))
    c.metric("P90 late-arrival error · all rides", pct(primary.late_arrival_p90_percentage))
    st.caption(
        "Underprediction magnitudes are positive percentages of actual duration, conditional on underestimated rides. "
        "All-ride late-arrival P90 assigns zero to overestimated/exact rides. Opposite-direction errors never cancel these magnitudes."
    )

    st.subheader("Percentage error by distance")
    groups = percentage_table(
        [{"Distance group": key, **value} for key, value in report["groups"][diagnostic].items()],
        "Distance group",
    )
    st.dataframe(groups, hide_index=True, width="stretch")
    if groups["MAPE (%)"].notna().any():
        st.plotly_chart(
            px.bar(
                groups,
                x="Distance group",
                y="MAPE (%)",
                hover_data=["Evaluated rides", "Median APE (%)", "Signed bias (%)", "P90 APE (%)"],
                title=f"MAPE by distance · {MODEL_LABELS.get(diagnostic, diagnostic)}",
            ),
            width="stretch",
        )
    st.caption(
        "Counts accompany every subgroup. Groups with only a few rides do not establish statistical confidence."
    )
    st.subheader("Percentage error by actual ride duration")
    st.dataframe(
        percentage_table(
            [
                {"Actual duration": key, **value}
                for key, value in report["duration_groups"][diagnostic].items()
            ],
            "Actual duration",
        ),
        hide_index=True,
        width="stretch",
    )

    rows = pd.DataFrame(report["rows"])
    selected = rows[rows.variant.eq(diagnostic)].copy()
    selected["Actual elapsed (hours)"] = selected.actual_elapsed_s / 3600
    if selected.percentage_eligible.any():
        st.plotly_chart(
            px.scatter(
                selected[selected.percentage_eligible],
                x="Actual elapsed (hours)",
                y="absolute_percentage_error",
                hover_data=["date", "distance_km", "signed_percentage_error"],
                labels={
                    "absolute_percentage_error": "Absolute percentage error (%)",
                    "signed_percentage_error": "Signed error (%)",
                },
                title="Per-ride percentage error versus actual duration",
            ),
            width="stretch",
        )
    selected["Distance group"] = selected.distance_km.map(distance_group)
    distance_filter = st.selectbox(
        "Validation distance group", ["All distance groups"] + DISTANCE_LABELS
    )
    if distance_filter != "All distance groups":
        selected = selected[selected["Distance group"].eq(distance_filter)]
    selected["Actual elapsed (min)"] = selected.actual_elapsed_s / 60
    selected["Predicted elapsed (min)"] = selected.predicted_elapsed_s / 60
    selected["Signed error (min)"] = (selected.predicted_elapsed_s - selected.actual_elapsed_s) / 60
    selected["Model"] = selected.variant.map(MODEL_LABELS)
    selected = selected.sort_values("absolute_percentage_error", ascending=False, na_position="last")
    st.subheader("Individual held-out rides · largest percentage error first")
    view = selected.reindex(
        columns=[
            "date",
            "distance_km",
            "Distance group",
            "Actual elapsed (min)",
            "Predicted elapsed (min)",
            "signed_percentage_error",
            "absolute_percentage_error",
            "actual_overall_kmh",
            "predicted_overall_kmh",
            "Model",
            "prediction_direction",
            "actual_finish_utc",
            "predicted_finish_utc",
            "Signed error (min)",
            "percentage_eligible",
            "paired_comparison_eligible",
            "percentage_exclusion_reason",
        ]
    ).rename(
        columns={
            "date": "Departure (UTC)",
            "distance_km": "Ride distance (km)",
            "signed_percentage_error": "Signed error (%)",
            "absolute_percentage_error": "Absolute error (%)",
            "actual_overall_kmh": "Actual overall speed (km/h)",
            "predicted_overall_kmh": "Predicted overall speed (km/h)",
            "prediction_direction": "Estimate direction",
            "actual_finish_utc": "Actual finish (UTC)",
            "predicted_finish_utc": "Predicted finish (UTC)",
            "percentage_eligible": "Percentage calculable",
            "paired_comparison_eligible": "In primary paired comparison",
            "percentage_exclusion_reason": "Exclusion reason",
        }
    )
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption(
        "Use the model selector and distance filter above; click table headers to change the sort. Excluded rows remain visible with reasons. Finish timestamps are departure plus the saved actual/predicted duration."
    )
    st.download_button(
        "Download filtered percentage validation",
        view.to_csv(index=False),
        "elapsed-percentage-validation.csv",
        "text/csv",
    )
    with st.expander("Secondary minute errors, daylight and component diagnostics"):
        st.dataframe(
            comparison.reindex(
                columns=[
                    "variant",
                    "rides",
                    "elapsed_mae_minutes",
                    "median_absolute_elapsed_minutes",
                    "elapsed_bias_minutes",
                    "overall_speed_mae_kmh",
                    "coverage_80",
                    "coverage_90",
                    "daylight_accuracy",
                    "optimistic_daylight_errors",
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.dataframe(
            selected.drop(columns=["warnings", "hr_training_ids"], errors="ignore"),
            hide_index=True,
            width="stretch",
        )
        if selected.percentage_eligible.any():
            st.plotly_chart(
                px.scatter(
                    selected[selected.percentage_eligible],
                    x="Actual elapsed (min)",
                    y="Predicted elapsed (min)",
                    hover_data=["date", "distance_km"],
                    title="Complete duration · chronological held-out rides",
                ),
                width="stretch",
            )
        st.caption(
            "Moving, traffic-attribution and unattributed-stop errors are accounting diagnostics; they do not establish causal effects of fatigue, weather or rider intent."
        )

    st.subheader("Automatic HR recommendation diagnostics")
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
    measured = pd.DataFrame(report["measured_hr_comparisons"])
    measured["Model"] = measured.variant.map(MODEL_LABELS)
    st.dataframe(percentage_table(measured, "Model"), hide_index=True, width="stretch")
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
