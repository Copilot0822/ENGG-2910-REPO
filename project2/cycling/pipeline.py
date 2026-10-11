"""Independent import, annotation, training and route preparation workflows."""

from datetime import datetime, timezone
from io import BytesIO
import json
import hashlib
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
    if settings.osm_provider == "overpass":
        raise ValueError(
            "Bulk history processing requires local OSM extracts or offline caches. Use Overpass only for small routes."
        )
    provider = OSMProvider(store, settings)
    previous = store.get_state("observations", [])
    observations_by_ride = {}
    for observation in previous:
        observations_by_ride.setdefault(str(observation["ride_id"]), []).append(observation)
    for ride_id in rides.id.astype(str):
        saved = store.get_state(f"encounters:{ride_id}")
        if saved is not None:
            observations_by_ride[ride_id] = saved
    processed = []
    existing_segments = store.segments()
    existing_ids = set(existing_segments.ride_id.astype(str)) if len(existing_segments) else set()
    work = []
    for _, ride in rides.iterrows():
        signature = provider.signature(store.points(str(ride.id))) + ":segments-3"
        if (
            refresh_osm
            or str(ride.id) not in existing_ids
            or ride.get("enrichment_signature") != signature
        ):
            work.append((ride.to_dict(), signature))
    if progress:
        progress(0, f"{len(work)} rides need enrichment; {len(rides) - len(work)} cached rides reused")
    workers = (
        min(3, max(1, settings.osm_workers))
        if settings.osm_provider == "local" and len(work) >= 8
        else 1
    )
    if workers == 1:
        results = (_enrich_one(store, provider, ride, signature) for ride, signature in work)
        executor = None
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        executor = ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_worker, initargs=(str(store.root), settings)
        )
        futures = [executor.submit(_enrich_worker, item) for item in work]
        results = (future.result() for future in as_completed(futures))
    try:
        for completed, result in enumerate(results, 1):
            ride_id, encountered, warnings = result
            observations_by_ride[ride_id] = encountered
            provider.warnings.update(warnings)
            processed.append(ride_id)
            if progress:
                progress(
                    completed / max(1, len(work)),
                    f"Enriched {completed}/{len(work)} rides; {len(rides) - len(work)} cached, {workers} local workers",
                )
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    observations = [o for values in observations_by_ride.values() for o in values]
    # SQLite state is JSON; convert NaN sensor-derived waits to null for a portable state file.
    observations = (
        json.loads(pd.DataFrame(observations).to_json(orient="records")) if observations else []
    )
    store.set_state("observations", observations)
    store.set_state("osm_warnings", sorted(provider.warnings))
    store.set_state("processed_at", datetime.now(timezone.utc).isoformat())
    from .coverage import history_coverage

    store.set_state("osm_coverage", history_coverage(store))
    if provider.network:
        store.set_state("osm_sources", provider.network.report())
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


def _enrich_one(store, provider, ride, signature):
    ride_id = str(ride["id"])
    points = provider.annotate(store.points(ride_id))
    events = route_events(points)
    encountered = observe_events(points, events, ride_id)
    segments = build_segments(points, ride, events)
    if not len(segments):
        raise ValueError(f"Ride {ride_id} has no usable segments.")
    store._write_frame(points, store._frame_path("points", ride_id))
    store.save_segments(ride_id, segments)
    store.set_state(f"encounters:{ride_id}", encountered)
    store.update_metadata(
        ride_id,
        {
            "known_surface_fraction": float(
                points.loc[points.surface_source.eq("known"), "ds_m"].sum() / max(1, points.ds_m.sum())
            ),
            "matched_segments": int(segments.map_confidence.ge(0.5).sum()),
            "segments": len(segments),
            "event_count": len(events),
            "osm_controls_uncertain": int(points.control_uncertain.sum()),
            "enrichment_signature": signature,
            "traffic_stationary_s": sum(
                o["stationary_s"] for o in encountered if o["reliable"] and o["kind"] != "corner"
            ),
        },
    )
    store.invalidate_model()
    return ride_id, encountered, sorted(provider.warnings)


def _initialize_worker(root, settings):
    global _worker_store, _worker_provider
    from .storage import Store

    _worker_store = Store(root)
    _worker_provider = OSMProvider(_worker_store, settings)


def _enrich_worker(item):
    return _enrich_one(_worker_store, _worker_provider, *item)


def train(store, settings, progress=None, validate=True):
    rides, segments = store.rides(), store.segments()
    observations = store.get_state("observations", [])
    if not len(segments):
        raise ValueError("Process historical recordings before training.")
    if validate and settings.osm_provider == "local" and len(rides) >= 20:
        from .comparison import compare_models
        from .model import load_model

        compare_models(store, settings, progress)
        return load_model(store)
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
        weather=settings.weather_enabled and "weather_available" in segments,
    )
    model.variant = "CW" if getattr(model, "neutral_model", None) is not None else "C"
    model.validation = validation
    model.fitness_adjustment = validation.get("fitness_multiplier", 1.0)
    model.save(store.model_path())
    model.save(store.root / "models" / f"model-{model.variant}.joblib")
    store.set_state("validation", validation)
    store.set_state(
        "training",
        {
            "trained_at": model.trained_at,
            "half_life": model.half_life,
            "kind": model.kind,
            "hr_available": model.hr_available,
            "support": model.support,
            "variant": model.variant,
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
    # Private route products are keyed by content, cleaning settings and relevant OSM versions.
    key = hashlib.sha256(
        data
        + (elevation_csv or b"")
        + json.dumps(
            {
                "signature": provider.signature(points),
                "max_speed": settings.max_speed_mps,
                "route_algorithm": "route-3",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = store.root / "routes" / key
    if (cache / "metadata.json").exists():
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        return {
            **metadata,
            "points": pd.read_parquet(cache / "points.parquet"),
            "segments": pd.read_parquet(cache / "segments.parquet"),
            "filename": Path(filename).name,
        }
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
    from .coverage import trace_coverage

    result = {
        "points": points,
        "segments": segments,
        "events": events,
        "diagnostics": diagnostics,
        "warnings": warnings,
        "filename": Path(filename).name,
        "coverage": trace_coverage(points),
    }
    cache.mkdir(parents=True, exist_ok=True)
    store._write_frame(points, cache / "points.parquet")
    store._write_frame(segments, cache / "segments.parquet")
    (cache / "metadata.json").write_text(
        json.dumps({k: v for k, v in result.items() if k not in ("points", "segments")}, default=str),
        encoding="utf-8",
    )
    return result


def route_for_model(route, model):
    """Rebuild geometry-only A/B labels before prediction, matching their training targets.

    Full OSM annotations remain in route['points'] for coverage and inspection. Controls
    are neither charged twice nor added to a model trained on their observed slowdowns.
    """
    variant = getattr(model, "variant", "C")
    if variant in ("C", "CW"):
        return route["segments"], route["events"]
    from .comparison import variant_points

    points = variant_points(route["points"], variant)
    events = route_events(points)
    segments = build_segments(
        points,
        {"id": "route", "date": datetime.now(timezone.utc).isoformat(), "bike": "unknown"},
        events,
        historical=False,
    )
    segments.attrs["osm_coverage"] = route.get("coverage")
    return segments, events


__all__ = ["import_archive", "process_history", "train", "prepare_route"]
