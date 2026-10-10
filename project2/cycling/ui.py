"""Local Streamlit workflow and route visualization."""

from dataclasses import asdict
from datetime import datetime, time
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from .config import Settings
from .importer import impute_ride_hr
from .model import load_model
from .pipeline import import_archive, prepare_route, process_history, train
from .prediction import predict_route
from .storage import Store

COLORS = {
    "paved": [36, 126, 105],
    "hardpack_gravel": [207, 155, 66],
    "loose_gravel": [222, 113, 50],
    "rough_natural": [156, 83, 61],
    "unknown": [130, 138, 143],
}


def duration(seconds):
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m"


def settings_panel(settings):
    with st.sidebar.expander("Data, OSM and training settings"):
        provider = st.selectbox(
            "OSM source",
            ["offline", "overpass", "local"],
            index=["offline", "overpass", "local"].index(settings.osm_provider),
        )
        st.caption(
            "Offline uses cached OSM or unknown labels. Overpass sends coarse road-area queries; the ZIP and recordings stay local."
        )
        endpoint = st.text_input("Overpass endpoint", settings.overpass_url)
        local = st.text_input("Local .osm / .osm.gz / .json / .pbf path", settings.local_osm_path)
        tiles = st.number_input("Maximum new OSM tiles per batch", 1, 100, settings.osm_max_tiles)
        valhalla = st.text_input(
            "Optional Valhalla URL",
            settings.valhalla_url,
            help="Use a local service such as http://127.0.0.1:8002. Enabling a remote URL sends route coordinates to that provider.",
        )
        half_life = st.number_input("Recency half-life (days)", 15, 1000, int(settings.half_life_days))
        optimize = st.checkbox("Select half-life by earlier-only backtests", settings.optimize_half_life)
        validation = st.number_input("Chronological test rides", 3, 60, settings.validation_rides)
        ebikes = st.checkbox("Include outdoor e-bike rides", settings.include_ebikes)
        if st.button("Save settings", width="stretch"):
            settings.osm_provider, settings.overpass_url, settings.local_osm_path = (
                provider,
                endpoint,
                local,
            )
            settings.osm_max_tiles, settings.valhalla_url = int(tiles), valhalla
            settings.half_life_days, settings.optimize_half_life = half_life, optimize
            settings.validation_rides, settings.include_ebikes = int(validation), ebikes
            settings.save()
            st.success("Saved. Refresh OSM processing or retrain to apply changes to historical data.")
            st.rerun()
        st.caption(f"Private cache: {settings.data_dir}")


def _run_progress(function):
    bar = st.progress(0, text="Starting…")

    def progress(value, message):
        bar.progress(min(1.0, max(0.0, float(value))), text=message)

    try:
        result = function(progress)
        bar.progress(1, text="Complete")
        return result
    finally:
        bar.empty()


def historical_page(store, settings):
    st.title("Historical data")
    st.write("Import your Strava export, inspect the recordings, then train a model of your riding.")
    rides = store.rides()
    if len(rides):
        hr, median = impute_ride_hr(rides)
        columns = st.columns(4)
        columns[0].metric("Outdoor rides", len(rides))
        columns[1].metric("Cycling distance", f"{rides.distance_m.sum() / 1000:,.0f} km")
        columns[2].metric("Moving hours", f"{rides.moving_s.sum() / 3600:,.1f}")
        columns[3].metric("Median measured HR", f"{median:.0f} BPM" if median else "Unavailable")
        st.caption(
            f"{rides.date.min()[:10]} to {rides.date.max()[:10]} · {(~hr.hr_imputed).sum()} rides with measured HR · {hr.hr_imputed.sum()} rides with imputed HR"
        )
        st.info(
            f"Missing-HR rides receive the median recorded ride-average HR ({median:.0f} BPM). They keep full terrain-learning weight. Only real timestamped HR trains the effort relationship."
            if median
            else "No measured HR yet. Missing-HR rides still train the terrain baseline; a personalized HR relationship is unavailable."
        )
    with st.container(border=True):
        st.subheader("1 · Import the complete archive")
        upload = st.file_uploader("Strava account ZIP", type=["zip"])
        with st.expander("Import a ZIP directly from disk"):
            local_path = st.text_input(
                "Absolute ZIP path", placeholder="C:\\Users\\you\\Downloads\\export.zip"
            )
        if st.button("Import activities", type="primary", disabled=not (upload or local_path)):
            try:
                source = upload.getvalue() if upload else local_path.strip().strip('"')
                report = _run_progress(lambda p: import_archive(source, store, settings, p))
                st.success(
                    f"Imported {report['imported']} rides; {report['duplicates']} duplicates, {report['excluded']} excluded, {report['failed']} failed. {report['partial']} imported with warnings."
                )
                st.rerun()
            except Exception as exc:
                st.error(
                    f"Import failed: {exc}. Check that this is the full Strava export and review size limits."
                )
        report = store.get_state("last_import")
        if report:
            st.caption(
                f"Last import: {report['imported']} imported · {report['partial']} partial · {report['duplicates']} duplicates · {report['excluded']} excluded · {report['failed']} failed"
            )
            with st.expander("Import report"):
                st.dataframe(pd.DataFrame(report["items"]), hide_index=True, width="stretch")
                st.download_button(
                    "Download import report",
                    json.dumps(report, indent=2),
                    "import-report.json",
                    "application/json",
                )
    with st.container(border=True):
        st.subheader("2 · Process terrain and train")
        refresh = st.checkbox("Refresh OSM annotations for previously processed rides", False)
        run_validation = st.checkbox("Run chronological validation", True)
        a, b = st.columns(2)
        if a.button("Process recordings and OSM", disabled=len(rides) == 0, width="stretch"):
            try:
                result = _run_progress(lambda p: process_history(store, settings, p, refresh))
                st.success(
                    f"Processed {result['processed']} rides; {result['observations']} historical traffic/corner encounters."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Processing failed: {exc}. Use offline mode or verify the local OSM file.")
        if b.button("Train / retrain model", disabled=len(rides) == 0, type="primary", width="stretch"):
            try:
                _run_progress(lambda p: process_history(store, settings, p, refresh))
                model = _run_progress(lambda p: train(store, settings, p, run_validation))
                st.success(f"Trained {model.kind}. Selected half-life: {model.half_life:g} days.")
            except Exception as exc:
                st.error(
                    f"Training failed: {exc}. Review the import report and add reliable timestamped outdoor rides."
                )
        training = store.get_state("training")
        if training:
            st.caption(
                f"Last training: {training['trained_at']} · {training['kind']} · half-life {training['half_life']:g} days"
            )
        for warning in store.get_state("osm_warnings", []):
            st.warning(warning)
    if len(rides):
        table = rides.copy()
        table["distance_km"] = table.distance_m / 1000
        table["moving_hours"] = table.moving_s / 3600
        columns = [
            c
            for c in (
                "date",
                "name",
                "bike",
                "distance_km",
                "moving_hours",
                "measured_avg_hr",
                "hr_source",
                "known_surface_fraction",
                "segments",
                "matched_segments",
                "timestamp_reliable",
                "unobserved_s",
            )
            if c in table
        ]
        st.subheader("Imported rides")
        st.dataframe(table[columns], hide_index=True, width="stretch")


def prediction_page(store, settings, model):
    st.title("Route prediction")
    if model is None:
        st.info("Import your archive and train a model on the Historical Data page first.")
        return
    if store.get_state("model_stale", False):
        st.warning(
            "Historical data changed after training. Retrain for the latest rides and OSM annotations."
        )
    upload = st.file_uploader("Planned Komoot / GPX route", type=["gpx"])
    with st.expander("Optional elevation correction"):
        st.caption(
            "Upload a trusted elevation profile with distance_m and elevation_m columns covering the whole route. Otherwise the GPX elevations are cleaned and used as supplied."
        )
        correction = st.file_uploader("Elevation profile CSV", type=["csv"])
    left, right = st.columns(2)
    default_hr = int(round(model.median_hr)) if model.median_hr else 140
    target = left.number_input("Target average heart rate (BPM)", 35, 230, default_hr)
    left.caption(
        "Average over moving time, including braking and acceleration. Stationary waits and personal breaks are excluded."
    )
    if model.hr_available:
        left.caption(
            f"Measured HR support: {model.hr_range[0]:.0f}–{model.hr_range[1]:.0f} BPM. Terrain and learned drift shape the pacing profile."
        )
    else:
        left.warning(
            "Target HR cannot currently be modeled. The rider's learned terrain baseline will be used."
        )
    bikes = ["Any / unknown"] + model.support["bikes"]
    bike = right.selectbox("Bike", bikes)
    scenario = right.selectbox("Stop-sign scenario", ["Full stops", "Observed historical behavior"])
    right.caption(
        "Observed behavior describes past recordings. Full stops is the default for traffic-law-compliant planning."
    )
    breaks = right.number_input("Personal breaks (minutes)", 0.0, 1440.0, 0.0, step=5.0)
    departure_date = left.date_input("Planned date", datetime.now(ZoneInfo("America/Toronto")).date())
    departure_time = left.time_input("Departure time", time(9, 0))
    timezone_name = left.text_input("Departure timezone", "America/Toronto")
    if st.button("Predict route", type="primary", disabled=upload is None):
        try:
            departure = (
                datetime.combine(departure_date, departure_time)
                .replace(tzinfo=ZoneInfo(timezone_name))
                .isoformat()
            )
            data = upload.getvalue()
            elevation_data = correction.getvalue() if correction else None
            key = hashlib.sha256(
                data + (elevation_data or b"") + json.dumps(asdict(settings), sort_keys=True).encode()
            ).hexdigest()
            with st.spinner("Preparing route and looking up cached OSM roads…"):
                if st.session_state.get("route_key") != key:
                    st.session_state.route = prepare_route(
                        data, upload.name, store, settings, elevation_data
                    )
                    st.session_state.route_key = key
                route = st.session_state.route
                result = predict_route(
                    model,
                    route["segments"],
                    route["events"],
                    target,
                    None if bike == "Any / unknown" else bike,
                    "full_stop" if scenario == "Full stops" else "observed",
                    breaks,
                    departure,
                    model.validation.get("fitness_selected", False),
                )
                result["warnings"] += route["warnings"]
                st.session_state.result = result
            st.success("Prediction ready. Open Results in the sidebar.")
            a, b, c = st.columns(3)
            a.metric("Rolling speed", f"{result['rolling_kmh']:.1f} km/h")
            b.metric("Moving time", duration(result["moving_s"]))
            c.metric("Total elapsed", duration(result["elapsed_s"]))
        except Exception as exc:
            st.error(
                f"Could not predict this route: {exc}. Check GPX geometry, elevation profile and timezone."
            )


def route_map(result, route):
    overlay = st.selectbox(
        "Route overlay", ["Road surface", "Rolling speed", "Gradient", "Map-match confidence"]
    )
    basemap = st.checkbox(
        "Load an online basemap",
        False,
        help="Downloads map tiles from CARTO/OSM. Disable for an offline map canvas.",
    )
    frame = result["segments"].copy()
    rows = []
    points = route["points"]
    for segment in frame.itertuples():
        shape = points[(points.distance_m >= segment.start_m) & (points.distance_m <= segment.end_m)]
        path = (
            [[segment.start_lon, segment.start_lat]]
            + [
                [float(r.lon), float(r.lat)]
                for r in shape.iloc[:: max(1, len(shape) // 50)].itertuples()
            ]
            + [[segment.end_lon, segment.end_lat]]
        )
        if overlay == "Road surface":
            color = COLORS[segment.surface]
        elif overlay == "Rolling speed":
            value = np.clip(segment.rolling_kmh / 45, 0, 1)
            color = [int(230 * (1 - value)), int(80 + 100 * value), int(80 + 100 * value)]
        elif overlay == "Gradient":
            value = np.clip(abs(segment.gradient_pct) / 12, 0, 1)
            color = (
                [int(70 + 160 * value), int(165 * (1 - value)), 95]
                if segment.gradient_pct >= 0
                else [50, 120, int(120 + 130 * value)]
            )
        else:
            color = [int(210 * (1 - segment.map_confidence)), int(160 * segment.map_confidence), 100]
        rows.append(
            {
                "path": path,
                "color": color,
                "surface": segment.surface,
                "source": segment.surface_source,
                "speed": round(segment.rolling_kmh, 1),
                "gradient": round(segment.gradient_pct, 1),
                "confidence": round(segment.map_confidence, 2),
                "km": round(segment.end_m / 1000, 2),
                "hr": round(segment.predicted_hr)
                if np.isfinite(segment.predicted_hr)
                else "unavailable",
            }
        )
    layers = [
        pdk.Layer(
            "PathLayer",
            data=rows,
            get_path="path",
            get_color="color",
            get_width=5,
            width_min_pixels=4,
            pickable=True,
            auto_highlight=True,
        )
    ]
    markers = st.multiselect(
        "Traffic and geometry markers",
        ["stop", "signal", "corner", "yield", "roundabout", "railway"],
        default=["stop", "signal", "corner"],
    )
    events = [
        {
            **e,
            "position": [e["lon"], e["lat"]],
            "color": [210, 67, 67] if e["kind"] == "stop" else [65, 102, 190],
            "surface": e["kind"],
            "source": e["source"],
            "speed": "—",
            "gradient": round(e["gradient_pct"], 1),
            "confidence": "—",
            "km": round(e["distance_m"] / 1000, 2),
            "hr": "—",
        }
        for e in result["events"]
        if e["kind"] in markers
    ]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            events,
            get_position="position",
            get_fill_color="color",
            get_radius=12,
            radius_min_pixels=5,
            pickable=True,
        )
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            map_provider="carto",
            map_style="light" if basemap else None,
            initial_view_state=pdk.ViewState(
                latitude=float(points.lat.mean()), longitude=float(points.lon.mean()), zoom=10
            ),
            tooltip={
                "text": "{km} km · {surface} ({source})\n{speed} km/h · gradient {gradient}%\nHR {hr} · map confidence {confidence}"
            },
        ),
        height=480,
    )
    st.caption(
        "Map data © OpenStreetMap contributors · optional basemap © CARTO. Gray = unknown surface. Inferred surfaces are labeled in tooltips."
    )


def results_page():
    st.title("Results")
    result = st.session_state.get("result")
    if result is None:
        st.info("Upload and predict a GPX on Route Prediction first.")
        return
    route = st.session_state.route
    st.caption(
        f"{route['filename']} · {result['distance_m'] / 1000:.1f} km · {result['model_kind']} · {result['scenario']} scenario"
    )
    a, b, c = st.columns(3)
    a.metric("Cruising speed", f"{result['cruise_kmh']:.1f} km/h")
    b.metric("Rolling speed", f"{result['rolling_kmh']:.1f} km/h")
    c.metric("Overall average", f"{result['overall_kmh']:.1f} km/h")
    a, b, c = st.columns(3)
    a.metric("Moving time", duration(result["moving_s"]))
    b.metric("Total elapsed time", duration(result["elapsed_s"]))
    gain = route["diagnostics"]["elevation_gain_m"]
    c.metric("Elevation gain", f"{gain:,.0f} m" if gain is not None else "Unknown")
    if result["achieved_hr"] is not None:
        st.caption(
            f"Requested HR {result['requested_hr']:.0f} · modeled target {result['modeled_target_hr']:.1f} · achieved moving-time average {result['achieved_hr']:.1f} BPM"
        )
    for warning in dict.fromkeys(result["warnings"]):
        st.warning(warning)
    if result["interval"]:
        interval = result["interval"]
        st.info(
            f"{'Provisional' if interval['provisional'] else 'Calibrated'} 90% rolling-speed interval: {interval['speed_lower_kmh']:.1f}–{interval['speed_upper_kmh']:.1f} km/h, moving time {duration(interval['moving_lower_s'])}–{duration(interval['moving_upper_s'])}. Based on {interval['samples']} chronological calibration rides. Traffic waits need separate support."
        )
    st.subheader("Where time goes")
    ledger = [
        {
            "Category": "Uninterrupted cruising",
            "Minutes": result["cruising_time_s"] / 60,
            "Time type": "Moving",
        }
    ]
    ledger += [
        {
            "Category": f"{kind.title()} braking / acceleration",
            "Minutes": value / 60,
            "Time type": "Moving",
        }
        for kind, value in result["losses_by_kind"].items()
    ]
    ledger += [
        {"Category": f"{kind.title()} stationary wait", "Minutes": value / 60, "Time type": "Stationary"}
        for kind, value in result["waits_by_kind"].items()
    ]
    ledger.append(
        {
            "Category": "Personal breaks",
            "Minutes": result["personal_break_s"] / 60,
            "Time type": "Personal",
        }
    )
    st.dataframe(pd.DataFrame(ledger), hide_index=True, width="stretch")
    route_map(result, route)
    frame = result["segments"].copy()
    frame["Distance (km)"] = frame.end_m / 1000
    frame["Elapsed (hours)"] = frame.accumulated_elapsed_s / 3600
    st.plotly_chart(
        px.line(
            frame,
            x="Distance (km)",
            y=["cruise_kmh", "rolling_kmh"],
            labels={"value": "Speed (km/h)"},
            title="Speed over the route",
        ),
        width="stretch",
    )
    elevation = route["points"]
    st.plotly_chart(
        px.area(
            elevation,
            x=elevation.distance_m / 1000,
            y="elevation_smooth",
            labels={"x": "Distance (km)", "elevation_smooth": "Elevation (m)"},
            title="Cleaned GPX elevation",
        ),
        width="stretch",
    )
    st.plotly_chart(
        px.line(
            frame,
            x="Distance (km)",
            y="Elapsed (hours)",
            title="Accumulated journey time, before personal breaks",
        ),
        width="stretch",
    )
    if result["achieved_hr"] is not None:
        st.plotly_chart(
            px.line(
                frame,
                x="Distance (km)",
                y="predicted_hr",
                title="Estimated pacing HR profile",
                labels={"predicted_hr": "HR (BPM)"},
            ),
            width="stretch",
        )
    st.subheader("Surface composition by distance")
    st.dataframe(result["surfaces"], hide_index=True, width="stretch")
    with st.expander("Inspect a route segment"):
        number = st.number_input("Segment number", 1, len(frame), 1)
        segment = frame.iloc[int(number) - 1]
        keys = [
            "start_m",
            "end_m",
            "distance_m",
            "gradient_pct",
            "surface",
            "surface_source",
            "surface_raw",
            "smoothness",
            "tracktype",
            "highway",
            "map_confidence",
            "predicted_hr",
            "cruise_kmh",
            "rolling_kmh",
            "moving_loss_s",
            "stationary_delay_s",
        ]
        values = segment[keys].map(
            lambda v: (
                "Unavailable"
                if pd.isna(v)
                else f"{v:.2f}"
                if isinstance(v, (float, np.floating))
                else str(v)
            )
        )
        st.dataframe(values.rename("Value").to_frame(), width="stretch")
    if result["events"]:
        with st.expander("Traffic and corner estimates"):
            st.dataframe(
                pd.DataFrame(result["events"])[
                    [
                        "kind",
                        "distance_m",
                        "identity",
                        "moving_loss_s",
                        "stationary_s",
                        "stop_probability",
                        "encounters",
                        "local_encounters",
                        "source",
                        "stationary_known",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
    st.download_button(
        "Download segment predictions", frame.to_csv(index=False), "route-prediction.csv", "text/csv"
    )


def performance_page(store, model):
    st.title("Model performance")
    if model is None:
        st.info("Train a model on Historical Data to inspect performance.")
        return
    support = model.support
    rides = store.rides()
    a, b, c = st.columns(3)
    a.metric("Cruising training segments", f"{support['cruise_segments']:,}")
    b.metric("Selected half-life", f"{model.half_life:g} days")
    c.metric("Rides over 100 km", support["long_rides"])
    known = float(
        (
            rides.get("known_surface_fraction", pd.Series(0, index=rides.index)).fillna(0)
            * rides.distance_m
        ).sum()
        / rides.distance_m.sum()
    )
    matched = int(rides.get("matched_segments", pd.Series(0, index=rides.index)).fillna(0).sum())
    observations = store.get_state("observations", [])
    st.caption(
        f"{known:.0%} of historical distance has verified OSM surface · {matched:,} map-matched segments · {sum(o['kind'] == 'stop' for o in observations)} observed stop signs · {sum(o['kind'] == 'signal' for o in observations)} observed signals"
    )
    st.caption(
        f"{support['measured_hr_rides']} measured-HR rides · {support['imputed_hr_rides']} imputed-HR rides · last trained {model.trained_at}"
    )
    report = model.validation
    if report.get("status") != "complete":
        st.info(
            report.get("reason", "Validation was not run. Enable chronological validation and retrain.")
        )
        return
    metrics = report["metrics"]
    a, b, c = st.columns(3)
    a.metric("Held-out rolling speed MAE", f"{metrics['speed_mae_kmh']:.2f} km/h")
    b.metric("Moving time MAE", f"{metrics['moving_mae_minutes']:.1f} min")
    c.metric("Median absolute time error", f"{metrics['median_absolute_percentage_error']:.1f}%")
    elapsed = metrics["elapsed_mae_minutes"]
    st.caption(
        f"Elapsed time MAE: {elapsed:.1f} min"
        if elapsed is not None
        else "Elapsed-time MAE unavailable on rides with recording gaps / unlearned waits."
    )
    st.caption(report["elapsed_note"])
    st.write(
        f"Historical median speed baseline MAE: {metrics['median_baseline_mae_kmh']:.2f} km/h · gradient baseline: {metrics['gradient_baseline_mae_kmh']:.2f} km/h"
    )
    if metrics["speed_mae_kmh"] >= min(
        metrics["median_baseline_mae_kmh"], metrics["gradient_baseline_mae_kmh"]
    ):
        st.warning(
            "The segment model did not outperform the best simple baseline on this calibration set. More varied rides, OSM coverage, or model refinement may be needed."
        )
    coverage = metrics["interval_coverage"]
    st.write(
        f"Earlier-residual interval coverage: {coverage:.0%} across {metrics['coverage_rides']} rides"
        if coverage is not None
        else "Too few chronological residuals to measure interval coverage."
    )
    if report.get("interval_provisional"):
        st.info(
            "Intervals are provisional: fewer than 20 independent calibration rides. Cycling conditions can change, so coverage is not guaranteed."
        )
    st.caption(
        "Half-life and fitness variant use the earlier selection folds. The later rides calibrate uncertainty. Each model and behavioral distribution uses rides strictly before its test ride."
    )
    rows = pd.DataFrame(report["rows"])
    st.plotly_chart(
        px.scatter(
            rows,
            x="actual_rolling_kmh",
            y="predicted_rolling_kmh",
            color="split",
            hover_data=["date", "distance_km"],
            title="Chronological predictions versus recorded rolling speed",
        ),
        width="stretch",
    )
    st.dataframe(rows.drop(columns=["warnings"], errors="ignore"), hide_index=True, width="stretch")
    if report.get("candidates"):
        st.subheader("Recency and recent-fitness comparison")
        st.dataframe(pd.DataFrame(report["candidates"]), hide_index=True, width="stretch")
        st.caption(
            f"Separate recent-fitness correction {'selected' if report.get('fitness_selected') else 'not selected'} by earlier validation. Current multiplier: {model.fitness_adjustment:.3f}."
        )
    st.subheader("Errors by terrain and ride length")
    for dimension, grouped in report["groups"].items():
        st.write(dimension.replace("_", " ").title())
        st.dataframe(pd.DataFrame(grouped), hide_index=True, width="stretch")
    trend = rides[["date", "distance_m", "moving_s", "measured_avg_hr"]].copy()
    trend["Rolling speed (km/h)"] = trend.distance_m / trend.moving_s * 3.6
    st.plotly_chart(
        px.scatter(
            trend,
            x="date",
            y="Rolling speed (km/h)",
            color="measured_avg_hr",
            title="Historical ride speed and measured HR",
        ),
        width="stretch",
    )
    st.caption(
        "Ride averages mix terrain and weather; the separate fitness correction is evaluated using terrain-controlled prediction residuals."
    )
    st.download_button(
        "Download validation report",
        json.dumps(report, indent=2, default=str),
        "validation.json",
        "application/json",
    )


def main():
    st.set_page_config(page_title="Personal cycling predictor", page_icon="🚲", layout="wide")
    settings = Settings.load()
    store = Store(settings.data_dir)
    st.sidebar.title("Personal cycling predictor")
    st.sidebar.caption("Your rides. Your pacing. Local data.")
    page = st.sidebar.radio(
        "Workflow", ["Historical Data", "Route Prediction", "Results", "Model Performance"]
    )
    settings_panel(settings)
    try:
        model = load_model(store)
    except Exception as exc:
        st.sidebar.error(f"Saved model cannot be loaded: {exc}. Retrain after a dependency upgrade.")
        model = None
    if model:
        st.sidebar.success(f"{model.support['rides']} rides trained")
    if page == "Historical Data":
        historical_page(store, settings)
    elif page == "Route Prediction":
        prediction_page(store, settings, model)
    elif page == "Results":
        results_page()
    else:
        performance_page(store, model)
