"""Local Streamlit workflow and route visualization."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from .config import Settings
from .importer import impute_ride_hr
from .model import load_model
from .pipeline import import_archive, process_history, train
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


def behavior_label(event, key, label):
    probabilities = event.get("behavior_probabilities") or {}
    return f"{label}: {probabilities[key]:.0%}" if key in probabilities else f"{label}: unavailable"


def settings_panel(settings):
    with st.sidebar.expander("Data, OSM and training settings"):
        provider = st.selectbox(
            "OSM source",
            ["offline", "overpass", "local"],
            index=["offline", "overpass", "local"].index(settings.osm_provider),
        )
        st.caption(
            "Local uses persistent regional indexes. Offline uses cached lookups. Overpass is for small routes; bulk history requires local extracts."
        )
        endpoint = st.text_input("Overpass endpoint", settings.overpass_url)
        local = st.text_input("Local .osm / .osm.gz / .json / .pbf path", settings.local_osm_path)
        locals_text = st.text_area(
            "Additional extract paths (one per line)", "\n".join(settings.local_osm_paths or [])
        )
        st.caption(
            "Leave paths empty to use data/osm/extracts/*.osm.pbf. New regions need a public extract downloaded locally."
        )
        tiles = st.number_input("Maximum new OSM tiles per batch", 1, 100, settings.osm_max_tiles)
        valhalla = st.text_input(
            "Optional Valhalla URL",
            settings.valhalla_url,
            help="Only local endpoints such as http://127.0.0.1:8002 are permitted. The built-in HMM needs no service.",
        )
        half_life = st.number_input("Recency half-life (days)", 15, 1000, int(settings.half_life_days))
        optimize = st.checkbox("Select half-life by earlier-only backtests", settings.optimize_half_life)
        validation = st.number_input("Chronological test rides", 3, 60, settings.validation_rides)
        ebikes = st.checkbox("Include outdoor e-bike rides", settings.include_ebikes)
        weather_enabled = st.checkbox("Enrich ECCC weather before UI training", settings.weather_enabled)
        weather_offline = st.checkbox("Use only cached ECCC data", settings.weather_offline)
        if st.button("Save settings", width="stretch"):
            settings.osm_provider, settings.overpass_url, settings.local_osm_path = (
                provider,
                endpoint,
                local,
            )
            settings.osm_max_tiles, settings.valhalla_url = int(tiles), valhalla
            settings.local_osm_paths = [
                p.strip().strip('"') for p in locals_text.splitlines() if p.strip()
            ]
            settings.half_life_days, settings.optimize_half_life = half_life, optimize
            settings.validation_rides, settings.include_ebikes = int(validation), ebikes
            settings.weather_enabled, settings.weather_offline = weather_enabled, weather_offline
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
    settings.weather_offline = st.session_state.get("weather-cache-only", settings.weather_offline)
    st.title("Historical data")
    st.write("Import your Strava export, inspect the recordings, then train a model of your riding.")
    rides = store.rides()
    quality = store.get_state("timing_quality", {})
    if quality:
        st.caption(
            f"Complete duration audit: {quality.get('reliable_elapsed', 0)}/{quality.get('rides', 0)} reliable · {quality.get('reliable_measured_hr', 0)} trustworthy measured-HR rides for suggestions."
        )
        with st.expander("Complete duration and HR quality"):
            st.json(quality)
    if st.button("Train elapsed-time planning and validate", disabled=len(rides) < 20):
        from .journey import train_planning

        try:
            result = _run_progress(lambda p: train_planning(store, settings, p))
            st.success(
                f"Elapsed planning trained on {result['training']['stopping_rides']} reliable stopping histories. Open Model Performance for actual held-out errors."
            )
        except Exception as exc:
            st.error(f"Elapsed/HR training failed: {exc}")
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
                if settings.weather_enabled:
                    from .weather import enrich_history

                    _run_progress(lambda p: enrich_history(store, settings, p))
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
        weather_dashboard(store, settings, rides)
        osm_dashboard(store, rides)
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


def weather_dashboard(store, settings, rides):
    from .weather import enrich_history
    from .traffic import refresh_behavior

    with st.container(border=True):
        st.subheader("ECCC weather and observed traffic behavior")
        report = store.get_state("weather_coverage", {})
        a, b, c = st.columns(3)
        a.metric("Rides with ECCC wind", f"{report.get('rides', 0)} / {len(rides)}")
        b.metric("Segments with weather", f"{report.get('segment_coverage', 0):.1%}")
        c.metric("Stations used", report.get("stations", 0))
        st.caption(
            "Station winds are usually measured at 10 m. They describe regional conditions; shelter and cyclist-height wind differ. Requests contain station IDs and time ranges, never complete traces."
        )
        offline = st.checkbox(
            "Weather enrichment cache only (offline)", settings.weather_offline, key="weather-cache-only"
        )
        settings.weather_offline = offline
        if st.button("Enrich / retry missing ECCC weather"):
            try:
                _run_progress(lambda p: enrich_history(store, settings, p))
                st.rerun()
            except Exception as exc:
                st.error(
                    f"Weather enrichment failed: {exc}. Cached weather and neutral predictions remain usable."
                )
        if st.button("Reanalyse stop-sign and signal speed traces"):
            _run_progress(lambda p: refresh_behavior(store, p))
            st.rerun()
        with st.expander("Weather coverage, missing values and station metadata"):
            st.json(report)
            path = store.root / "weather" / "stations.json"
            if path.exists():
                stations = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
                st.dataframe(
                    stations[stations.id.isin(report.get("station_ids", []))],
                    hide_index=True,
                    width="stretch",
                )
        behavior = store.get_state("traffic_behavior", {})
        if behavior:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "kind": k,
                            **v,
                            "full_stop_percent": 100 * (v["fractions"]["full_stop"] or 0),
                            "rolling_percent": 100 * (v["fractions"]["rolling"] or 0),
                            "minimal_percent": 100 * (v["fractions"]["minimal"] or 0),
                        }
                        for k, v in behavior["kinds"].items()
                        if k in ("stop", "signal")
                    ]
                ).drop(columns=["counts", "fractions"]),
                hide_index=True,
                width="stretch",
            )
            st.caption(behavior["note"])


def osm_dashboard(store, rides):
    st.subheader("OSM enrichment and data quality")
    report = store.get_state("osm_coverage")
    with st.expander("Public extracts and indexed regions"):
        st.caption(
            "Province downloads contain public OSM data. No recorded GPS traces are uploaded. Current maps may differ from roads on historical ride dates."
        )
        sources = store.get_state("osm_sources", [])
        if sources:
            st.dataframe(
                pd.DataFrame(sources)[["source", "ways", "controls", "indexed_at", "sha256"]],
                hide_index=True,
                width="stretch",
            )
        from .network import REGIONS, NetworkIndex, download_region, source_paths

        regions = st.multiselect("Extracts to download", list(REGIONS), default=["ontario"])
        if st.button("Download missing regional extracts", disabled=not regions):
            try:
                _run_progress(lambda p: [download_region(store.root, r, p) for r in regions])
                st.success("Extracts downloaded locally. Build the index, then process history.")
            except Exception as exc:
                st.error(f"Download failed: {exc}")
        if st.button("Build / update local road indexes"):
            try:
                index = _run_progress(
                    lambda p: NetworkIndex(store.root, source_paths(Settings.load(), store.root), p)
                )
                store.set_state("osm_sources", index.report())
                st.success(
                    "Persistent spatial indexes are ready. Set OSM source to local and process history."
                )
            except Exception as exc:
                st.error(f"Index failed: {exc}")
    if report:
        a, b, c, d = st.columns(4)
        a.metric(
            "Road-matched distance",
            f"{report['matched_km']:,.1f} km",
            f"{report['matched_percent']:.1f}% of history",
        )
        b.metric("High confidence / matched", f"{report['high_confidence_percent_of_matched']:.1f}%")
        c.metric("Analyzed encounters", f"{report['analyzed_encounters']:,}")
        d.metric("Reliable traffic encounters", f"{report['reliable_traffic_encounters']:,}")
        left, right = st.columns(2)
        left.dataframe(pd.DataFrame(report["surfaces"]), hide_index=True, width="stretch")
        right.dataframe(pd.DataFrame(report["surface_provenance"]), hide_index=True, width="stretch")
        st.caption(
            f"Distance weighting uses represented GPS distance, not sample counts. Confirmed matches require confidence ≥0.50; high confidence ≥0.75. Ambiguous matches retain raw tags for inspection but receive unknown model labels. {report['uncertain_controls']:,} uncertain approach detections are excluded."
        )
        if "high_confidence_segment_percent" in report:
            st.caption(
                f"{report['high_confidence_segment_percent']:.1f}% of {report['matched_segments']:,} matched adaptive segments have high-confidence road identification. Observed stop behavior (reliable encounters): {report.get('stop_behaviors', {})}."
            )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "kind": k,
                        "encounters": count,
                        "unique_directional_controls": report["unique_directional_controls"].get(k, 0),
                    }
                    for k, count in report["control_encounters"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("Unmatched and low-confidence regions"):
            st.dataframe(pd.DataFrame(report["weak_regions"]), hide_index=True, width="stretch")
            cells = [
                {
                    "position": [row["longitude_cell"] + 0.05, row["latitude_cell"] + 0.05],
                    "km": round(row["unmatched_km"], 1),
                    "radius": max(1000, np.sqrt(row["unmatched_km"]) * 250),
                }
                for row in report["weak_regions"]
            ]
            if cells:
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[
                            pdk.Layer(
                                "ScatterplotLayer",
                                cells,
                                get_position="position",
                                get_radius="radius",
                                get_fill_color=[225, 115, 35, 150],
                                pickable=True,
                            )
                        ],
                        map_style=None,
                        initial_view_state=pdk.ViewState(
                            latitude=cells[0]["position"][1], longitude=cells[0]["position"][0], zoom=7
                        ),
                        tooltip={"text": "{km} km unmatched / low confidence in this 0.1° cell"},
                    ),
                    height=300,
                )
                st.caption(
                    "Regional overview of weak road identification. Use the inspector below to see original GPS sections and unknown surfaces."
                )
    else:
        st.info("Process historical recordings to measure actual OSM coverage.")
    with st.expander("Inspect recorded GPS and selected OSM roads"):
        if st.checkbox("Open local map-matching inspector"):
            if st.button("Choose a random activity"):
                st.session_state.inspection_ride = str(rides.sample(1).id.iloc[0])
            identifiers = rides.id.astype(str).tolist()
            selected = st.selectbox(
                "Recorded activity",
                identifiers,
                index=identifiers.index(st.session_state.get("inspection_ride", identifiers[0])),
                format_func=lambda value: (
                    f"{rides.loc[rides.id.astype(str).eq(value), 'date'].iloc[0][:10]} · {value}"
                ),
            )
            matching_inspector(store.points(selected))


def matching_inspector(points):
    if "matched_geometry" not in points:
        st.info("This trace has not been processed with the new matcher.")
        return
    total = float(points.distance_m.max() / 1000)
    start = st.number_input("Inspection start (km)", 0.0, max(0.01, total), 0.0, step=0.5)
    span = st.number_input("Inspection length (km)", 0.1, 20.0, 2.0, step=0.5)
    sample = points[points.distance_m.between(start * 1000, (start + span) * 1000)]
    if len(sample) < 2:
        st.info("Choose a section with at least two recorded points.")
        return
    paths = []
    for _, section in sample.groupby("group"):
        paths.append(
            {
                "path": section[["lon", "lat"]].to_numpy().tolist(),
                "color": [205, 65, 145],
                "detail": "Original GPS",
            }
        )
    roads = [
        {
            "path": json.loads(row.matched_geometry),
            "color": [40, 115, 220],
            "detail": f"Way {row.way_id} · {row.surface_raw or 'untagged'} · confidence {row.map_confidence:.2f}",
        }
        for row in sample[sample.matched_geometry.ne("[]")]
        .drop_duplicates("matched_geometry")
        .itertuples()
    ]
    weak = [
        {"position": [r.lon, r.lat], "color": [235, 135, 30], "detail": r.match_status}
        for r in sample[sample.map_confidence.lt(0.5) | sample.surface.eq("unknown")].itertuples()
    ]
    layers = [
        pdk.Layer(
            "PathLayer", paths, get_path="path", get_color="color", width_min_pixels=3, pickable=True
        ),
        pdk.Layer(
            "PathLayer", roads, get_path="path", get_color="color", width_min_pixels=2, pickable=True
        ),
        pdk.Layer(
            "ScatterplotLayer",
            weak,
            get_position="position",
            get_fill_color="color",
            radius_min_pixels=3,
            get_radius=2,
            pickable=True,
        ),
    ]
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(sample.lat.mean()), longitude=float(sample.lon.mean()), zoom=14
            ),
            tooltip={"text": "{detail}"},
        ),
        height=460,
    )
    st.caption(
        "Pink: original GPS · blue: selected OSM edge geometry · orange: uncertain identification or unknown surface. No basemap requests. Spatial proximity is not independently surveyed ground truth."
    )
    columns = [
        c
        for c in (
            "distance_m",
            "way_id",
            "edge_id",
            "forward",
            "map_confidence",
            "map_distance_m",
            "surface",
            "surface_raw",
            "highway",
            "osm_tags",
            "control_kind",
            "control_id",
        )
        if c in sample
    ]
    st.dataframe(sample[columns].iloc[:: max(1, len(sample) // 500)], hide_index=True, width="stretch")


def prediction_page(store, settings, model):
    from .planning_ui import route_prediction_page

    route_prediction_page(store, settings, model)


def route_map(result, route):
    overlay = st.selectbox(
        "Route overlay",
        [
            "Road surface",
            "Rolling speed",
            "Headwind / tailwind",
            "Gradient",
            "Map-match confidence",
            "Low-confidence matches",
        ],
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
            surface = shape.groupby("surface").ds_m.sum().idxmax() if len(shape) else segment.surface
            color = COLORS.get(surface, COLORS["unknown"])
        elif overlay == "Rolling speed":
            value = np.clip(segment.rolling_kmh / 45, 0, 1)
            color = [int(230 * (1 - value)), int(80 + 100 * value), int(80 + 100 * value)]
        elif overlay == "Headwind / tailwind":
            wind = getattr(segment, "headwind_kmh", np.nan)
            value = np.clip(abs(wind) / 30, 0, 1) if np.isfinite(wind) else 0
            color = (
                [int(130 + 120 * value), 70, 60]
                if wind > 0
                else [45, 100, int(130 + 120 * value)]
                if np.isfinite(wind)
                else COLORS["unknown"]
            )
        elif overlay == "Gradient":
            value = np.clip(abs(segment.gradient_pct) / 12, 0, 1)
            color = (
                [int(70 + 160 * value), int(165 * (1 - value)), 95]
                if segment.gradient_pct >= 0
                else [50, 120, int(120 + 130 * value)]
            )
        else:
            confidence = (
                float(np.average(shape.map_confidence, weights=np.maximum(0.01, shape.ds_m)))
                if len(shape)
                else segment.map_confidence
            )
            color = (
                ([225, 90, 45] if confidence < 0.5 else [60, 160, 110])
                if overlay == "Low-confidence matches"
                else [int(210 * (1 - confidence)), int(160 * confidence), 100]
            )
        rows.append(
            {
                "path": path,
                "color": color,
                "surface": shape.groupby("surface").ds_m.sum().idxmax()
                if len(shape)
                else segment.surface,
                "source": shape.groupby("surface_source").ds_m.sum().idxmax()
                if len(shape)
                else segment.surface_source,
                "speed": round(segment.rolling_kmh, 1),
                "gradient": round(segment.gradient_pct, 1),
                "confidence": round(float(shape.map_confidence.mean()), 2)
                if len(shape)
                else round(segment.map_confidence, 2),
                "km": round(segment.end_m / 1000, 2),
                "wind": f"{getattr(segment, 'headwind_kmh', np.nan):.1f} km/h headwind",
                "traffic": "",
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
            "source": e.get("source", "OSM / geometry; not charged by this model"),
            "speed": "—",
            "gradient": round(e["gradient_pct"], 1),
            "confidence": "—",
            "km": round(e["distance_m"] / 1000, 2),
            "hr": "—",
            "wind": "",
            "traffic": " | ".join(
                [
                    behavior_label(e, "full_stop", "Historical full stop"),
                    behavior_label(e, "rolling", "Historical rolling slowdown"),
                    behavior_label(e, "minimal", "Historical minimal slowdown"),
                    f"Minimum speed: {e.get('expected_minimum_mps') * 3.6:.1f} km/h"
                    if e.get("expected_minimum_mps") is not None
                    else "Minimum speed unavailable",
                    f"Moving loss {e.get('moving_loss_s', 0):.1f}s; wait {e.get('stationary_s', 0):.1f}s",
                    f"{e.get('local_encounters', 0)} local / {e.get('encounters', 0)} relevant encounters; {e.get('confidence', 'unavailable')} confidence",
                    e.get("source", "Uncharged by selected baseline"),
                ]
            ),
        }
        for e in list(result["events"])
        + [
            e
            for e in route["events"]
            if not any(
                e["identity"] == charged["identity"] and abs(e["distance_m"] - charged["distance_m"]) < 3
                for charged in result["events"]
            )
        ]
        if e["kind"] in markers
    ]
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            events,
            id="traffic-controls",
            get_position="position",
            get_fill_color="color",
            get_radius=12,
            radius_min_pixels=5,
            pickable=True,
        )
    )
    selected = st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            map_provider="carto",
            map_style="light" if basemap else None,
            initial_view_state=pdk.ViewState(
                latitude=float(points.lat.mean()), longitude=float(points.lon.mean()), zoom=10
            ),
            tooltip={
                "text": "{km} km · {surface} ({source})\n{speed} km/h · gradient {gradient}%\nHR {hr} · map confidence {confidence}\n{wind}\n{traffic}"
            },
        ),
        height=480,
        key="route-traffic-map",
        on_select="rerun",
        selection_mode="single-object",
    )
    for marker in selected.selection.objects.get("traffic-controls", []):
        st.info(marker.get("traffic", ""))
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
    from .planning_ui import finish_results

    settings = Settings.load()
    finish_results(result, route, Store(settings.data_dir), settings)
    st.subheader("Riding and route diagnostics")
    st.caption(
        f"{route['filename']} · {result['distance_m'] / 1000:.1f} km · model {result.get('model_variant', 'C')} · {result['model_kind']} · {result['scenario']} scenario"
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
    if "headwind_kmh" in frame:
        st.caption(
            f"ECCC weather available for {result.get('weather_distance_coverage', 0):.1%} of route distance. Positive headwind opposes travel; negative values are tailwind."
        )
        st.plotly_chart(
            px.line(
                frame,
                x="Distance (km)",
                y=["headwind_kmh", "crosswind_kmh"],
                title="Headwind / tailwind and signed crosswind",
                labels={"value": "Wind component (km/h)"},
            ),
            width="stretch",
        )
        if result.get("neutral_comparison"):
            neutral = result["neutral_comparison"]
            st.info(
                f"Weather-neutral comparison: {neutral['rolling_kmh']:.1f} km/h rolling, {duration(neutral['moving_s'])} moving, {duration(neutral['elapsed_s'])} elapsed."
            )
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
    coverage = route.get("coverage")
    composition = (
        pd.DataFrame(
            [
                {"surface": k, "distance_km": v, "percent": v / max(1e-9, coverage["total_km"]) * 100}
                for k, v in coverage["surfaces"].items()
            ]
        )
        if coverage
        else result["surfaces"]
    )
    st.dataframe(composition, hide_index=True, width="stretch")
    with st.expander("Inspect original GPX and selected OSM geometry"):
        if st.checkbox("Open planned-route matching inspector"):
            matching_inspector(route["points"])
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
        keys += [
            k
            for k in [
                "headwind_kmh",
                "crosswind_kmh",
                "wind_speed_kmh",
                "wind_from_deg",
                "air_temperature_c",
                "humidity_pct",
                "precipitation_mm",
                "pressure_kpa",
                "weather_source",
                "weather_station_ids",
                "weather_station_distance_km",
                "weather_age_minutes",
                "weather_confidence",
                "weather_uncertainty_kmh",
                "weather_forecast_issue_utc",
                "arrival_time_utc",
            ]
            if k in segment.index
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
                        "expected_minimum_mps",
                        "behavior_probabilities",
                        "stop_probability_interval",
                        "confidence",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
    st.download_button(
        "Download segment predictions", frame.to_csv(index=False), "route-prediction.csv", "text/csv"
    )


def speed_performance_page(store, model):
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
    if report.get("comparisons"):
        st.subheader("Paired model accuracy")
        comparison = pd.DataFrame(
            [{k: v for k, v in c.items() if k != "groups"} for c in report["comparisons"]]
        )
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption(
            f"Default {store.get_state('training', {}).get('variant', report['selected_variant'])} follows your Model C preference. Half-life and fitness use earlier selection folds ({report['selection_rides']} rides, ending {report['selection_last'][:10]}). All variants use the same {report['final_rides']} final test rides starting {report['final_first'][:10]}. Long rides were included before looking at errors."
        )
        baseline_variant = "B" if "B" in set(comparison.variant) else "A"
        baseline = comparison.loc[comparison.variant.eq(baseline_variant), "speed_mae_kmh"].iloc[0]
        for candidate in report["comparisons"]:
            if candidate["variant"] != baseline_variant and candidate["speed_mae_kmh"] > baseline:
                st.warning(
                    f"Model {candidate['variant']} regressed by {candidate['speed_mae_kmh'] - baseline:.2f} km/h on final folds. Missing surface tags, ambiguous parallel roads and sparse/control-biased observations may limit these features. Final fold errors are not used to tune the model."
                )
        if report.get("weather_evaluation"):
            st.caption(report["weather_evaluation"])
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
            "Intervals are provisional when there are fewer than 20 calibration rides or measured coverage is below the 90% target. Changing conditions and correlated rides limit confidence."
        )
    st.caption(
        "Half-life and fitness variant use the earlier selection folds. The later rides calibrate uncertainty. Each model and behavioral distribution uses rides strictly before its test ride."
    )
    rows = pd.DataFrame(report.get("paired_rows", report["rows"]))
    st.plotly_chart(
        px.scatter(
            rows,
            x="actual_rolling_kmh",
            y="predicted_rolling_kmh",
            color="variant" if "variant" in rows else "split",
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
    st.subheader("Errors by terrain, wind, HR and ride length")
    groups = report["groups"]
    if report.get("comparisons"):
        grouped_variant = st.selectbox(
            "Diagnostic model",
            [c["variant"] for c in report["comparisons"]],
            index=next(
                (
                    i
                    for i, c in enumerate(report["comparisons"])
                    if c["variant"] == report["selected_variant"]
                ),
                0,
            ),
        )
        groups = next(c["groups"] for c in report["comparisons"] if c["variant"] == grouped_variant)
    for dimension, grouped in groups.items():
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


def performance_page(store, model):
    from .planning_ui import elapsed_performance

    elapsed_performance(store)
    with st.expander("Secondary riding-model and speed diagnostics"):
        speed_performance_page(store, model)


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
