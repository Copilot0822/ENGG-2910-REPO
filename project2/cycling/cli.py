"""Reproducible local workflows, useful for imports too large for the browser uploader."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import Settings
from .model import load_model
from .pipeline import import_archive, prepare_route, process_history, train
from .prediction import predict_route
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
    predictor = sub.add_parser("predict", help="Predict a GPX without the web interface")
    predictor.add_argument("gpx")
    predictor.add_argument("--hr", type=float)
    predictor.add_argument("--bike")
    predictor.add_argument("--scenario", choices=["full_stop", "observed"], default="full_stop")
    predictor.add_argument("--break-minutes", type=float, default=0)
    predictor.add_argument("--output", help="Write private segment predictions to CSV")
    sub.add_parser("status", help="Show local dataset and trained model status")
    sub.add_parser("config", help="Show effective configuration")
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
        elif args.command == "train":
            settings.optimize_half_life = args.optimize_half_life or settings.optimize_half_life
            if args.validation_rides:
                settings.validation_rides = args.validation_rides
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
            route = prepare_route(Path(args.gpx).read_bytes(), args.gpx, store, settings)
            result = predict_route(
                model,
                route["segments"],
                route["events"],
                args.hr,
                args.bike,
                args.scenario,
                args.break_minutes,
                use_fitness=model.validation.get("fitness_selected", False),
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
