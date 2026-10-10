"""Independent import, annotation, training and route preparation workflows."""

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .events import observe_events, route_events
from .features import build_segments
from .geometry import clean_points
from .importer import import_archive
from .model import fit_model
from .osm import OSMProvider
from .recordings import parse_recording
from .validation import backtest


def process_history(store, settings, progress=None, refresh_osm=False):
    rides = store.rides()
    if len(rides) == 0:
        raise ValueError("Import a Strava archive first.")
    provider = OSMProvider(store, settings)
    previous = store.get_state("observations", [])
    observations_by_ride = {}
    for observation in previous:
        observations_by_ride.setdefault(str(observation["ride_id"]), []).append(observation)
    processed = []
    existing_segments = store.segments()
    existing_ids = set(existing_segments.ride_id.astype(str)) if len(existing_segments) else set()
    for index, ride in rides.iterrows():
        ride_id = str(ride.id)
        if progress:
            progress(index / len(rides), f"Terrain and events: ride {index + 1}/{len(rides)}")
        if not refresh_osm and ride_id in existing_ids:
            continue
        points = provider.annotate(store.points(ride_id))
        events = route_events(points)
        observations_by_ride[ride_id] = observe_events(points, events, ride_id)
        segments = build_segments(points, ride.to_dict(), events)
        if not len(segments):
            continue
        # Keep OSM annotations next to the trace so historical maps can be reused in validation.
        store._write_frame(points, store._frame_path("points", ride_id))
        store.save_segments(ride_id, segments)
        known = float(
            points.loc[points.surface_source.eq("known"), "ds_m"].sum() / max(1, points.ds_m.sum())
        )
        matched = int(segments.map_confidence.ge(0.25).sum())
        store.update_metadata(
            ride_id,
            {
                "known_surface_fraction": known,
                "matched_segments": matched,
                "segments": len(segments),
                "event_count": len(events),
                "osm_controls_uncertain": int(points.control_uncertain.sum()),
            },
        )
        processed.append(ride_id)
    observations = [o for values in observations_by_ride.values() for o in values]
    # SQLite state is JSON; convert NaN sensor-derived waits to null for a portable state file.
    observations = (
        json.loads(pd.DataFrame(observations).to_json(orient="records")) if observations else []
    )
    store.set_state("observations", observations)
    store.set_state("osm_warnings", sorted(provider.warnings))
    store.set_state("processed_at", datetime.now(timezone.utc).isoformat())
    if processed:
        store.invalidate_model()
    if progress:
        progress(1, "Historical processing complete")
    return {
        "processed": len(processed),
        "observations": len(observations),
        "warnings": sorted(provider.warnings),
        "osm_tiles_requested": provider.requested,
    }


def train(store, settings, progress=None, validate=True):
    rides, segments = store.rides(), store.segments()
    observations = store.get_state("observations", [])
    if not len(segments):
        raise ValueError("Process historical recordings before training.")
    ids = set(segments.ride_id.astype(str))
    rides = rides[rides.id.astype(str).isin(ids)]
    validation = (
        backtest(store, rides, segments, observations, settings, progress)
        if validate
        else {
            "status": "not_run",
            "selected_half_life": settings.half_life_days,
            "rows": [],
            "calibration_speed_residuals": [],
        }
    )
    if progress:
        progress(0.95, "Fitting all historical rides")
    model = fit_model(
        rides,
        segments,
        observations,
        validation["selected_half_life"],
        reference_date=datetime.now(timezone.utc).isoformat(),
    )
    model.validation = validation
    model.fitness_adjustment = validation.get("fitness_multiplier", 1.0)
    model.save(store.model_path())
    store.set_state("validation", validation)
    store.set_state(
        "training",
        {
            "trained_at": model.trained_at,
            "half_life": model.half_life,
            "kind": model.kind,
            "hr_available": model.hr_available,
            "support": model.support,
        },
    )
    store.set_state("model_stale", False)
    if progress:
        progress(1, "Training complete")
    return model


def prepare_route(data, filename, store, settings, elevation_csv=None):
    raw, _ = parse_recording(data, filename, settings)
    # A timed GPX used as a planned route must not feed measured effort or speed into prediction.
    raw = raw[[c for c in ("lat", "lon", "elevation", "track_group") if c in raw]].copy()
    points, diagnostics = clean_points(raw, settings, route=True)
    if elevation_csv is not None:
        correction = pd.read_csv(BytesIO(elevation_csv))
        if not {"distance_m", "elevation_m"}.issubset(correction.columns):
            raise ValueError("Elevation correction CSV needs distance_m and elevation_m columns.")
        correction = (
            correction[["distance_m", "elevation_m"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .sort_values("distance_m")
        )
        if len(correction) < 2 or not correction.elevation_m.between(-500, 9000).all():
            raise ValueError("Elevation correction must contain at least two plausible elevations.")
        if correction.distance_m.min() > 0 or correction.distance_m.max() < points.distance_m.max():
            raise ValueError("Elevation correction must cover the whole GPX distance.")
        raw = points[["lat", "lon"]].copy()
        raw["track_group"] = points.group
        raw["elevation"] = np.interp(points.distance_m, correction.distance_m, correction.elevation_m)
        points, diagnostics = clean_points(raw, settings, route=True)
    provider = OSMProvider(store, settings)
    points = provider.annotate(points)
    events = route_events(points)
    ride = {"id": "route", "bike": "unknown", "date": datetime.now(timezone.utc).isoformat()}
    segments = build_segments(points, ride, events, historical=False)
    warnings = sorted(provider.warnings)
    groups = points.group.nunique()
    if groups > 1:
        warnings.append(
            f"GPX has {groups} disconnected track sections. Gaps are excluded, not connected or rerouted."
        )
    if points.control_uncertain.any():
        warnings.append(
            f"{int(points.control_uncertain.sum())} OSM controls have uncertain approach directions and were not charged as confirmed stops."
        )
    return {
        "points": points,
        "segments": segments,
        "events": events,
        "diagnostics": diagnostics,
        "warnings": warnings,
        "filename": Path(filename).name,
    }


__all__ = ["import_archive", "process_history", "train", "prepare_route"]
