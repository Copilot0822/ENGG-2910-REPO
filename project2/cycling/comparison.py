"""Paired chronological A/B/C comparisons with selection preceding final evaluation."""

from datetime import datetime, timezone
import json
import hashlib

import numpy as np
import pandas as pd

from .events import observe_events, route_events
from .features import build_segments
from .matching import DEFAULTS
from .model import fit_model
from .storage import Store
from .validation import _fitness_multiplier, _score, held_out_route

WEATHER_VARIANTS = {"B": "Surface and road", "C": "Surface and intersections", "CW": "C + ECCC weather"}

VARIANTS = {"A": "Baseline (geometry)", "B": "Surface and road", "C": "Surface and intersections"}


def variant_points(points, variant):
    result = points.copy()
    if variant == "A":
        for key, value in DEFAULTS.items():
            result[key] = value
    elif variant == "B":
        result["control_kind"], result["control_id"], result["junction"] = "", "", ""
    return result


def prediction_inputs(segments, events, variant):
    frame = segments.copy()
    if variant == "A":
        for key, value in DEFAULTS.items():
            if key in frame:
                frame[key] = value
    if variant in ("A", "B"):
        events = [e for e in events if e["kind"] == "corner"]
    return frame, events


def variant_data(store, rides, progress=None, variants=None):
    variants = variants or VARIANTS
    frames, observations = {k: [] for k in variants}, {k: [] for k in variants}
    # Existing Phase 1 segments/observations preserve the actual original baseline.
    old = (
        Store(store.root / "baseline_phase1")
        if (store.root / "baseline_phase1" / "manifest.json").exists()
        else None
    )
    old_ids = set(old.rides().id.astype(str)) if old and len(old.rides()) else set()
    old_segments = old.segments() if old else pd.DataFrame()
    old_observations = old.get_state("observations", []) if old else []
    full_segments = store.segments()
    full_observations = store.get_state("observations", [])
    for index, ride in rides.iterrows():
        points = store.points(ride.id)
        if progress:
            progress(
                index / max(1, len(rides)), f"Preparing A/B/C labels: {index + 1}/{len(rides)} rides"
            )
        for variant in variants:
            if variant in ("C", "CW"):
                frames[variant].append(full_segments[full_segments.ride_id.astype(str).eq(str(ride.id))])
                observations[variant].extend(
                    o for o in full_observations if str(o["ride_id"]) == str(ride.id)
                )
                continue
            if variant == "A" and str(ride.id) in old_ids:
                frames[variant].append(old_segments[old_segments.ride_id.astype(str).eq(str(ride.id))])
                observations[variant].extend(
                    o for o in old_observations if str(o["ride_id"]) == str(ride.id)
                )
                continue
            signature = hashlib.sha256(
                f"{ride.id}:{ride.get('enrichment_signature')}:{variant}:labels-3".encode()
            ).hexdigest()
            directory = store.root / "variants" / variant / signature
            metadata_path = directory / "encounters.json"
            if not metadata_path.exists():
                annotated = variant_points(points, variant)
                events = route_events(annotated)
                directory.mkdir(parents=True, exist_ok=True)
                store._write_frame(
                    build_segments(annotated, ride.to_dict(), events), directory / "segments.parquet"
                )
                metadata_path.write_text(
                    json.dumps(observe_events(annotated, events, str(ride.id)), default=str),
                    encoding="utf-8",
                )
            frames[variant].append(pd.read_parquet(directory / "segments.parquet"))
            if not (directory / "behavior-2.version").exists():
                annotated = variant_points(points, variant)
                metadata_path.write_text(
                    json.dumps(
                        observe_events(annotated, route_events(annotated), str(ride.id)), default=str
                    ),
                    encoding="utf-8",
                )
                (directory / "behavior-2.version").write_text("behavior-2", encoding="utf-8")
            observations[variant].extend(json.loads(metadata_path.read_text(encoding="utf-8")))
    return {v: (pd.concat(frames[v], ignore_index=True), observations[v]) for v in variants}


def comparison_folds(rides, final_count=32, selection_count=12):
    ordered = rides.sort_values("date").reset_index(drop=True)
    if len(ordered) < 20:
        raise ValueError(
            "Paired comparison needs at least 20 rides; ordinary validation remains available."
        )
    minimum = min(50, max(4, len(ordered) // 4))
    boundary = max(minimum + 5, int(len(ordered) * 0.63))

    def sample(begin, end, count):
        available = ordered.iloc[begin:end]
        # Stratify by distance before looking at errors, retaining all long rides when practical.
        chosen = set(available[available.distance_m.ge(100_000)].index)
        chosen.update(np.linspace(begin, end - 1, min(count, end - begin), dtype=int))
        if len(chosen) > count:
            long = set(available[available.distance_m.ge(100_000)].index)
            rest = sorted(chosen - long)
            remaining = max(0, count - len(long))
            chosen = long | {rest[i] for i in np.linspace(0, len(rest) - 1, remaining, dtype=int)}
        return sorted(chosen)

    selection = sample(minimum, boundary, selection_count)
    final = sample(boundary, len(ordered), final_count)
    folds = []
    for split, indices in (("selection", selection), ("calibration", final)):
        for index in indices:
            test = ordered.iloc[index]
            train = ordered[pd.to_datetime(ordered.date, utc=True).lt(pd.Timestamp(test.date))]
            folds.append((split, train, test))
    return folds


def metrics(rows):
    frame = pd.DataFrame(rows)
    errors = frame.predicted_rolling_kmh - frame.actual_rolling_kmh
    reliable = frame.elapsed_error_s.dropna()
    result = {
        "speed_mae_kmh": float(errors.abs().mean()),
        "speed_rmse_kmh": float(np.sqrt(np.mean(errors**2))),
        "moving_mae_minutes": float(frame.moving_error_s.abs().mean() / 60),
        "moving_bias_minutes": float(frame.moving_error_s.mean() / 60),
        "elapsed_mae_minutes": float(reliable.abs().mean() / 60) if len(reliable) else None,
        "elapsed_rides": len(reliable),
        "calibration_rides": len(rows),
        "median_absolute_percentage_error": float(frame.moving_absolute_percentage_error.median()),
        "median_baseline_mae_kmh": float(frame.median_baseline_error_kmh.mean()),
        "gradient_baseline_mae_kmh": float(frame.gradient_baseline_error_kmh.mean()),
        "interval_coverage": float(frame.interval_covered.dropna().mean())
        if "interval_covered" in frame
        else None,
        "coverage_rides": int(frame.interval_covered.notna().sum())
        if "interval_covered" in frame
        else 0,
        "long_rides": int(frame.distance_km.ge(100).sum()),
    }
    frame["distance_band"] = pd.cut(
        frame.distance_km,
        [0, 30, 75, 150, np.inf],
        labels=["<30 km", "30–75 km", "75–150 km", ">150 km"],
    )
    frame["elevation_band"] = pd.cut(
        pd.to_numeric(frame.elevation_gain_m),
        [-1, 300, 1000, np.inf],
        labels=["<300 m", "300–1000 m", ">1000 m"],
    )
    frame["composition"] = np.where(
        frame.gravel_fraction.ge(0.3),
        "≥30% gravel/natural",
        np.where(frame.paved_fraction.ge(0.7), "≥70% paved", "mixed/unknown"),
    )
    frame["recency"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m")
    dimensions = ["distance_band", "elevation_band", "composition", "recency"]
    if "mean_wind_kmh" in frame:
        frame["wind"] = pd.cut(
            pd.to_numeric(frame.mean_wind_kmh, errors="coerce"),
            [-1, 8, 20, np.inf],
            labels=["calm <8 km/h", "moderate 8–20 km/h", "windy ≥20 km/h"],
        ).astype(str)
        frame.loc[frame.mean_wind_kmh.isna(), "wind"] = "missing wind"
        frame["weather_coverage"] = np.where(
            frame.weather_distance_coverage >= 0.8, "≥80% weather", "partial / missing weather"
        )
        dimensions.extend(["wind", "weather_coverage"])
    if "measured_average_hr" in frame:
        frame["average_hr"] = pd.cut(
            pd.to_numeric(frame.measured_average_hr, errors="coerce"),
            [0, 130, 150, 170, np.inf],
            labels=["<130", "130–150", "150–170", "≥170"],
        ).astype(str)
        frame.loc[frame.measured_average_hr.isna(), "average_hr"] = "unmeasured HR"
        frame["traffic_density"] = np.where(
            frame.traffic_density_per_km >= 0.5,
            "urban / ≥0.5 controls per km",
            "rural / <0.5 controls per km",
        )
        dimensions.extend(["average_hr", "traffic_density"])
    groups = {}
    for dimension in dimensions:
        groups[dimension] = [
            {
                "group": str(k),
                "rides": len(g),
                "speed_mae_kmh": float(g.speed_absolute_error.mean()),
                "moving_mae_minutes": float(g.moving_error_s.abs().mean() / 60),
            }
            for k, g in frame.groupby(dimension, observed=True)
        ]
    groups["long_distance"] = [
        {
            "group": f">{threshold} km",
            "rides": int(frame.distance_km.gt(threshold).sum()),
            "speed_mae_kmh": float(
                frame.loc[frame.distance_km.gt(threshold), "speed_absolute_error"].mean()
            )
            if frame.distance_km.gt(threshold).any()
            else None,
            "moving_mae_minutes": float(
                frame.loc[frame.distance_km.gt(threshold), "moving_error_s"].abs().mean() / 60
            )
            if frame.distance_km.gt(threshold).any()
            else None,
        }
        for threshold in (100, 150, 200)
    ]
    encounters = [e for row in rows for e in row.get("traffic_validation", [])]
    for kind in ("stop", "signal"):
        observed = [e for e in encounters if e["kind"] == kind]
        if observed:
            result[f"{kind}_validation_encounters"] = len(observed)
            result[f"{kind}_brier"] = float(np.mean([e["brier"] for e in observed]))
            result[f"{kind}_wait_mae_s"] = float(np.mean([e["wait_absolute_error_s"] for e in observed]))
            stopped = [
                e["conditional_wait_absolute_error_s"]
                for e in observed
                if e["conditional_wait_absolute_error_s"] is not None
            ]
            result[f"{kind}_conditional_wait_mae_s"] = float(np.mean(stopped)) if stopped else None
            result[f"{kind}_classification_accuracy"] = float(
                np.mean([e["actual_behavior"] == e["predicted_behavior"] for e in observed])
            )
    return result, groups


def _selection_candidate(variant, half, folds, datasets, routes, composition, updates):
    segments, observations = datasets[variant]
    rows = []
    for _, train, test in folds:
        model = fit_model(
            train, segments, observations, half, reference_date=test.date, weather=variant == "CW"
        )
        model.variant = variant
        route, events = routes[variant][str(test.id)]
        row = _score(
            model,
            test,
            route,
            events,
            actual_encounters=[o for o in observations if str(o["ride_id"]) == str(test.id)],
        )
        row.update(composition[str(test.id)])
        row["raw_predicted_kmh"] = row["predicted_rolling_kmh"]
        model.fitness_adjustment = _fitness_multiplier(rows)
        trend = _score(model, test, route, events, fitness=True)
        trend.update(composition[str(test.id)])
        row["fitness_result"] = trend
        rows.append(row)
        updates.put(f"Selection {variant}, half-life {half:g}")
    raw_mae = np.mean([r["speed_absolute_error"] for r in rows])
    trend_mae = np.mean([r["fitness_result"]["speed_absolute_error"] for r in rows])
    fitness = bool(len(rows) >= 3 and trend_mae < raw_mae * 0.97)
    return {
        "variant": variant,
        "half_life": half,
        "fitness_selected": fitness,
        "selection_mae_kmh": float(trend_mae if fitness else raw_mae),
        "raw_mae_kmh": float(raw_mae),
        "fitness_mae_kmh": float(trend_mae),
        "rows": rows,
    }


def _final_variant(variant, selected, folds, datasets, routes, composition, updates):
    segments, observations = datasets[variant]
    previous, residuals, rows = selected["rows"].copy(), [], []
    for _, train, test in folds:
        model = fit_model(
            train,
            segments,
            observations,
            selected["half_life"],
            reference_date=test.date,
            weather=variant == "CW",
        )
        model.variant = variant
        route, events = routes[variant][str(test.id)]
        raw = _score(model, test, route, events)
        model.fitness_adjustment = _fitness_multiplier(previous) if selected["fitness_selected"] else 1.0
        row = _score(
            model,
            test,
            route,
            events,
            fitness=selected["fitness_selected"],
            actual_encounters=[o for o in observations if str(o["ride_id"]) == str(test.id)],
        )
        row.update(composition[str(test.id)])
        row.update(raw_predicted_kmh=raw["predicted_rolling_kmh"], split="calibration", variant=variant)
        if len(residuals) >= 3:
            rank = min(len(residuals) - 1, int(np.ceil((len(residuals) + 1) * 0.9)) - 1)
            row["interval_covered"] = bool(row["speed_absolute_error"] <= np.sort(residuals)[rank])
        residuals.append(row["speed_absolute_error"])
        rows.append(row)
        previous.append(row)
        updates.put(f"Final paired backtest {variant}")
    return variant, rows


def _collect(futures, updates, completed, total, progress):
    from concurrent.futures import wait, FIRST_COMPLETED
    from queue import Empty

    pending, results = set(futures), []
    while pending:
        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
        while True:
            try:
                message = updates.get_nowait()
            except Empty:
                break
            completed += 1
            if progress:
                progress(completed / total, f"{message}: {completed}/{total} forecasts")
        results.extend(future.result() for future in done)
    return results, completed


def _initialize_evaluation(datasets, routes, composition, updates):
    global _evaluation_context
    _evaluation_context = datasets, routes, composition, updates


def _selection_worker(variant, half, folds):
    return _selection_candidate(variant, half, folds, *_evaluation_context)


def _final_worker(variant, chosen, folds):
    return _final_variant(variant, chosen, folds, *_evaluation_context)


def compare_models(store, settings, progress=None, deploy=True):
    rides = store.rides()
    variants = (
        WEATHER_VARIANTS
        if settings.weather_enabled and "weather_available" in store.segments()
        else VARIANTS
    )
    datasets = variant_data(store, rides, progress, variants)
    folds = comparison_folds(
        rides,
        final_count=max(12, settings.validation_rides * 2 // 3),
        selection_count=max(6, settings.validation_rides // 4),
    )
    routes = {v: {} for v in variants}
    composition = {}
    for route_index, (_, _, test) in enumerate(folds):
        if progress:
            progress(
                route_index / max(1, len(folds)),
                f"Preparing held-out geometry for A/B/C: {route_index + 1}/{len(folds)} rides",
            )
        points = store.points(test.id)
        total_distance = max(1.0, float(points.ds_m.sum()))
        composition[str(test.id)] = {
            "paved_fraction": float(
                points.loc[points.surface.eq("paved"), "ds_m"].sum() / total_distance
            ),
            "gravel_fraction": float(
                points.loc[
                    points.surface.isin(["hardpack_gravel", "loose_gravel", "rough_natural"]), "ds_m"
                ].sum()
                / total_distance
            ),
            "known_surface_fraction": float(
                points.loc[points.surface_source.eq("known"), "ds_m"].sum() / total_distance
            ),
            "map_matched_fraction": float(
                points.loc[points.map_confidence.ge(0.5), "ds_m"].sum() / total_distance
            ),
            "measured_average_hr": float(test.measured_avg_hr)
            if pd.notna(test.measured_avg_hr)
            else None,
            "traffic_density_per_km": sum(
                o["kind"] in ("stop", "signal", "yield", "railway", "roundabout")
                for o in datasets["C"][1]
                if str(o["ride_id"]) == str(test.id)
            )
            / max(0.1, float(test.distance_m / 1000)),
        }
        weather_rows = datasets["C"][0].loc[datasets["C"][0].ride_id.astype(str).eq(str(test.id))]
        if "weather_available" in weather_rows:
            good = weather_rows.weather_available.fillna(False)
            wx = weather_rows[good]
            composition[str(test.id)].update(
                weather_segment_coverage=float(good.mean()),
                weather_distance_coverage=float(wx.distance_m.sum() / weather_rows.distance_m.sum()),
                mean_wind_kmh=float(np.average(wx.wind_speed_kmh, weights=wx.distance_m))
                if len(wx)
                else None,
                mean_abs_headwind_kmh=float(np.average(wx.headwind_kmh.abs(), weights=wx.distance_m))
                if len(wx)
                else None,
            )
        for v in variants:
            weather_segments = (
                datasets["CW"][0].loc[datasets["CW"][0].ride_id.astype(str).eq(str(test.id))]
                if v == "CW"
                else None
            )
            routes[v][str(test.id)] = held_out_route(
                variant_points(points, v), test, settings, weather_segments
            )
    selection = [f for f in folds if f[0] == "selection"]
    final = [f for f in folds if f[0] == "calibration"]
    half_lives = [60, 90, 120, 180, 365] if settings.optimize_half_life else [settings.half_life_days]
    candidates, chosen, final_rows = [], {}, {}
    total = len(variants) * (len(half_lives) * len(selection) + len(final))
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from queue import Queue
    import multiprocessing

    parallel = len(rides) >= 50 and settings.comparison_workers > 1
    manager = multiprocessing.get_context("spawn").Manager() if parallel else None
    updates = manager.Queue() if manager else Queue()
    if parallel:
        executor = ProcessPoolExecutor(
            max_workers=min(3, settings.comparison_workers),
            initializer=_initialize_evaluation,
            initargs=(datasets, routes, composition, updates),
        )
    else:
        executor = ThreadPoolExecutor(max_workers=1)
    try:
        with executor:
            futures = [
                executor.submit(_selection_worker, variant, half, selection)
                if parallel
                else executor.submit(
                    _selection_candidate,
                    variant,
                    half,
                    selection,
                    datasets,
                    routes,
                    composition,
                    updates,
                )
                for variant in variants
                for half in half_lives
            ]
            candidates, completed = _collect(futures, updates, 0, total, progress)
            candidates.sort(key=lambda c: (c["variant"], c["half_life"]))
            chosen = {
                variant: min(
                    [c for c in candidates if c["variant"] == variant],
                    key=lambda c: c["selection_mae_kmh"],
                )
                for variant in variants
            }
            # All choices are frozen before evaluating the final chronological sample.
            # The user's traffic-aware default is persistent; report accuracy rankings separately.
            selected_variant = "CW" if "CW" in variants else "C"
            futures = [
                executor.submit(_final_worker, variant, chosen[variant], final)
                if parallel
                else executor.submit(
                    _final_variant,
                    variant,
                    chosen[variant],
                    final,
                    datasets,
                    routes,
                    composition,
                    updates,
                )
                for variant in variants
            ]
            results, completed = _collect(futures, updates, completed, total, progress)
            final_rows = dict(sorted(results))
    finally:
        if manager:
            manager.shutdown()
    comparisons = []
    for variant, rows in final_rows.items():
        measured, groups = metrics(rows)
        selected = chosen[variant]
        comparisons.append(
            {
                "variant": variant,
                "name": variants[variant],
                "half_life": selected["half_life"],
                "fitness_selected": selected["fitness_selected"],
                "selection_mae_kmh": selected["selection_mae_kmh"],
                **measured,
                "training_segments": len(datasets[variant][0]),
                "cruise_training_segments": int(datasets[variant][0].is_cruise.sum()),
                "groups": groups,
            }
        )
    best = next(c for c in comparisons if c["variant"] == selected_variant)
    selected_rows = final_rows[selected_variant]
    report = {
        "status": "complete",
        "selected_variant": selected_variant,
        "default_policy": "User-requested Model C; weather-aware CW when available. Accuracy ranking does not change the default.",
        "weather_evaluation": "Retrospective observed weather at actual recorded passage times; this does not measure future forecast skill.",
        "selected_half_life": best["half_life"],
        "fitness_selected": best["fitness_selected"],
        "fitness_multiplier": _fitness_multiplier(selected_rows) if best["fitness_selected"] else 1.0,
        "comparisons": comparisons,
        "metrics": {k: v for k, v in best.items() if k not in ("groups", "name")},
        "groups": best["groups"],
        "rows": selected_rows,
        "paired_rows": [r for rows in final_rows.values() for r in rows],
        "selection_rows": [
            {**r, "variant": v, "split": "selection"} for v, c in chosen.items() for r in c["rows"]
        ],
        "candidates": [{k: val for k, val in c.items() if k != "rows"} for c in candidates],
        "calibration_speed_residuals": [r["speed_absolute_error"] for r in selected_rows],
        "interval_provisional": len(selected_rows) < 20 or (best["interval_coverage"] or 0) < 0.9,
        "selection_rides": len(selection),
        "final_rides": len(final),
        "selection_last": str(selection[-1][2].date),
        "final_first": str(final[0][2].date),
        "elapsed_note": "Elapsed errors require ≥70% road coverage, known modeled waits, reliable timestamps and ≥90% of recorded stationary time inside observed traffic windows. Current OSM is not a historical map. Variant selection precedes final evaluation.",
    }
    report = json.loads(json.dumps(report, default=str))
    (store.root / "reports" / "model-comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    store.set_state("comparison", report)
    if deploy:
        for variant in variants:
            candidate = next(c for c in comparisons if c["variant"] == variant)
            segments, observations = datasets[variant]
            fitted = fit_model(
                rides,
                segments,
                observations,
                candidate["half_life"],
                reference_date=datetime.now(timezone.utc).isoformat(),
                weather=variant == "CW",
            )
            fitted.variant = variant
            fitted.validation = {
                **report,
                "selected_variant": variant,
                "metrics": candidate,
                "fitness_selected": candidate["fitness_selected"],
                "calibration_speed_residuals": [r["speed_absolute_error"] for r in final_rows[variant]],
                "interval_provisional": len(final_rows[variant]) < 20
                or (candidate["interval_coverage"] or 0) < 0.9,
            }
            fitted.fitness_adjustment = (
                _fitness_multiplier(final_rows[variant]) if candidate["fitness_selected"] else 1.0
            )
            fitted.save(store.root / "models" / f"model-{variant}.joblib")
            if variant == selected_variant:
                model = fitted
                model.validation = report
                model.save(store.model_path())
        store.set_state("validation", report)
        store.set_state(
            "training",
            {
                "trained_at": model.trained_at,
                "half_life": model.half_life,
                "kind": model.kind,
                "variant": selected_variant,
                "hr_available": model.hr_available,
                "support": model.support,
            },
        )
        store.set_state("model_stale", False)
    if progress:
        progress(1, f"Comparison complete; selected model {selected_variant}")
    return report
