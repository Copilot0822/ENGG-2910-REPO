"""Reproducible local workflows, useful for imports too large for the browser uploader."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import Settings
from .model import load_model
from .pipeline import import_archive, prepare_route, process_history, route_for_model, train
from .storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Override local private cache directory")
    sub = parser.add_subparsers(dest="command", required=True)
    importer = sub.add_parser("import", help="Incrementally import a complete Strava export")
    importer.add_argument("zip")
    process = sub.add_parser("process", help="Build terrain segments and traffic encounters")
    process.add_argument("--refresh-osm", action="store_true")
    trainer = sub.add_parser("train", help="Fit personalized models and chronological validation")
    trainer.add_argument("--no-validation", action="store_true")
    trainer.add_argument("--optimize-half-life", action="store_true")
    trainer.add_argument("--validation-rides", type=int)
    trainer.add_argument(
        "--enrich-weather", action="store_true", help="Retrieve/reuse ECCC weather before fitting"
    )
    predictor = sub.add_parser("predict", help="Predict a GPX without the web interface")
    predictor.add_argument("gpx")
    predictor.add_argument("--hr", type=float)
    predictor.add_argument("--bike")
    predictor.add_argument(
        "--model-variant",
        choices=["A", "B", "C", "CW"],
        help="Retained variant; default is traffic-aware C + Weather when trained",
    )
    predictor.add_argument("--scenario", choices=["full_stop", "observed"], default="full_stop")
    predictor.add_argument("--break-minutes", type=float)
    predictor.add_argument("--break-mode", choices=["historical", "custom", "minimal"])
    predictor.add_argument(
        "--daylight-reference", choices=["sunset", "civil_twilight"], default="sunset"
    )
    predictor.add_argument("--safety-minutes", type=float, default=30)
    predictor.add_argument("--latest-departure", action="store_true")
    predictor.add_argument(
        "--latest-criterion", choices=["conservative", "median", "expected"], default="conservative"
    )
    predictor.add_argument("--output", help="Write private segment predictions to CSV")
    predictor.add_argument("--departure", help="Planned ISO timestamp with timezone")
    predictor.add_argument(
        "--forecast", action="store_true", help="Use cached/bounded ECCC City Page hourly forecasts"
    )
    predictor.add_argument("--weather-neutral", action="store_true")
    sub.add_parser("status", help="Show local dataset and trained model status")
    sub.add_parser("config", help="Show effective configuration")
    weather = sub.add_parser("weather", help="Cache ECCC historical observations and enrich segments")
    weather.add_argument("action", choices=["enrich", "report"])
    weather.add_argument(
        "--retry", action="store_true", help="Retry missing coverage using cached successful requests"
    )
    weather.add_argument("--refresh", action="store_true", help="Explicitly refresh ECCC records")
    weather.add_argument("--offline", action="store_true")
    osm = sub.add_parser("osm", help="Manage public extracts and local indexes")
    osm.add_argument("action", choices=["download", "index", "report", "regions"])
    osm.add_argument(
        "--regions", nargs="+", default=["ontario", "quebec", "new-brunswick", "prince-edward-island"]
    )
    compare = sub.add_parser(
        "compare", help="Paired chronological comparisons; retain the traffic-aware C default"
    )
    compare.add_argument("--validation-rides", type=int)
    planning = sub.add_parser(
        "planning", help="Audit and train measured-HR, total duration and daylight planning"
    )
    planning.add_argument("action", choices=["audit", "train", "validate", "report"])
    planning.add_argument("--archive", help="Optional original export for missing FIT timing metadata")
    planning.add_argument("--validation-rides", type=int)
    args = parser.parse_args()
    settings = Settings.load()
    if args.data_dir:
        settings.data_dir = args.data_dir
    if args.command == "config":
        print(json.dumps(asdict(settings), indent=2))
        return
    store = Store(settings.data_dir)
    last = [-1]

    def progress(value, message):
        bucket = int(value * 20)
        if bucket != last[0]:
            print(f"{value:5.0%} {message}", flush=True)
            last[0] = bucket

    try:
        if args.command == "import":
            report = import_archive(args.zip, store, settings, progress)
            summary = {k: v for k, v in report.items() if k != "items"}
            store.set_state("last_import", report)
        elif args.command == "process":
            summary = process_history(store, settings, progress, args.refresh_osm)
        elif args.command == "weather":
            from .weather import enrich_history, weather_report

            settings.weather_offline = args.offline or settings.weather_offline
            summary = (
                enrich_history(store, settings, progress, args.refresh)
                if args.action == "enrich"
                else weather_report(store)
            )
        elif args.command == "osm":
            from .network import NetworkIndex, download_region, source_paths
            from .coverage import history_coverage

            if args.action == "download":
                summary = {
                    "files": [str(download_region(store.root, r, progress)) for r in args.regions]
                }
            elif args.action == "index":
                summary = {
                    "sources": NetworkIndex(
                        store.root, source_paths(settings, store.root), progress
                    ).report()
                }
                store.set_state("osm_sources", summary["sources"])
            elif args.action == "report":
                summary = history_coverage(store)
                store.set_state("osm_coverage", summary)
            else:
                import numpy as np
                import pandas as pd

                cells = []
                for ride in store.rides().itertuples():
                    p = store.points(ride.id)
                    cells.append(
                        pd.DataFrame(
                            {
                                "lat_degree": np.floor(p.lat),
                                "lon_degree": np.floor(p.lon),
                                "km": p.ds_m / 1000,
                            }
                        )
                    )
                summary = (
                    pd.concat(cells)
                    .groupby(["lat_degree", "lon_degree"])
                    .km.sum()
                    .sort_values(ascending=False)
                    .reset_index()
                    .to_dict("records")
                    if cells
                    else []
                )
        elif args.command == "compare":
            from .comparison import compare_models

            if args.validation_rides:
                settings.validation_rides = args.validation_rides
            summary = compare_models(store, settings, progress)
            summary = {
                k: v
                for k, v in summary.items()
                if k not in ("rows", "paired_rows", "selection_rows", "candidates", "groups")
            }
        elif args.command == "planning":
            from .timing import audit_history
            from .journey import train_planning

            if args.validation_rides:
                settings.validation_rides = args.validation_rides
            if args.action == "audit":
                summary = audit_history(store, settings, args.archive, progress)
            elif args.action in ("train", "validate"):
                summary = train_planning(
                    store, settings, progress, args.action == "validate", args.archive
                )
            else:
                report = store.get_state("journey_validation", {})
                summary = {
                    "training": store.get_state("journey_training"),
                    "timing": store.get_state("timing_quality"),
                    "comparisons": report.get("comparisons"),
                    "hr_metrics": report.get("hr_metrics"),
                }
        elif args.command == "train":
            settings.optimize_half_life = args.optimize_half_life or settings.optimize_half_life
            if args.validation_rides:
                settings.validation_rides = args.validation_rides
            if args.enrich_weather:
                from .weather import enrich_history

                enrich_history(store, settings, progress)
            model = train(store, settings, progress, validate=not args.no_validation)
            summary = {
                "kind": model.kind,
                "support": model.support,
                "hr_available": model.hr_available,
                "half_life": model.half_life,
                "validation": model.validation.get("metrics"),
            }
        elif args.command == "predict":
            model = load_model(store)
            if model is None:
                raise ValueError("Train a model first.")
            requested_variant = args.model_variant
            candidate_path = store.root / "models" / f"model-{requested_variant}.joblib"
            if requested_variant and candidate_path.exists():
                import joblib

                model = joblib.load(candidate_path)
            if args.scenario == "full_stop" and getattr(model, "variant", "C") in ("A", "B"):
                raise ValueError(
                    "Full-stop mapped traffic predictions require model C. Use --model-variant C or --scenario observed for the rolling baseline."
                )
            route = prepare_route(Path(args.gpx).read_bytes(), args.gpx, store, settings)
            route_segments, route_controls = route_for_model(route, model)
            forecast = None
            if args.forecast and not args.weather_neutral:
                from .forecast import CityForecast

                forecast = CityForecast(store.root, settings)
                forecast.prepare(route_segments)
            from .journey import plan_ride, load_planning

            planning = load_planning(store)

            def candidate(departure):
                return plan_ride(
                    model,
                    route,
                    store.rides(),
                    settings,
                    planning,
                    args.hr,
                    args.bike,
                    departure,
                    args.scenario,
                    args.break_mode or ("custom" if args.break_minutes is not None else "historical"),
                    args.break_minutes or 0,
                    forecast,
                    args.weather_neutral,
                    args.daylight_reference,
                    args.safety_minutes,
                    manual_hr=args.hr is not None,
                )

            result = candidate(args.departure)
            if args.latest_departure and result["daylight"]["cutoff"]:
                from .daylight import latest_departure

                result["latest_departure"] = latest_departure(
                    candidate, result["daylight"]["cutoff"], criterion=args.latest_criterion
                )
            summary = {k: v for k, v in result.items() if k not in ("segments", "surfaces", "events")}
            summary["warnings"] += route["warnings"]
            if args.output:
                result["segments"].to_csv(args.output, index=False)
        else:
            rides = store.rides()
            summary = {
                "data_dir": str(store.root),
                "rides": len(rides),
                "distance_km": float(rides.distance_m.sum() / 1000) if len(rides) else 0,
                "training": store.get_state("training"),
                "stale": store.get_state("model_stale", True),
            }
        print(json.dumps(summary, indent=2, default=str, allow_nan=False))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
