"""Command-line interface for the NMEA 2000 dataset simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulator.canboat_loader import load_canboat
from simulator.dataset_generator import DatasetGenerator
from simulator.output import write_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = PROJECT_ROOT / "vendor" / "canboat" / "canboat.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "simulated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simulator",
        description=(
            "Generate deterministic physical NMEA 2000 CAN frames and synchronized "
            "known physical values from the pinned Canboat database."
        ),
    )
    parser.add_argument("--mode", choices=("random", "coverage"), default="random")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--min-observations-per-field", type=int, default=20)
    parser.add_argument("--messages-per-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    database = load_canboat(arguments.database)
    generator = DatasetGenerator(database, arguments.seed)
    if arguments.mode == "random":
        per_sample = arguments.messages_per_sample or 6
        result = generator.random_mode(arguments.samples, per_sample)
    else:
        per_sample = arguments.messages_per_sample or 12
        result = generator.coverage_mode(
            arguments.min_observations_per_field, per_sample
        )
    result.options.update(
        {
            "database": str(arguments.database),
            "output": str(arguments.output),
            "samples": arguments.samples,
            "min_observations_per_field": arguments.min_observations_per_field,
        }
    )
    return write_outputs(arguments.output, database, result)


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    metadata = run(arguments)
    summary = {
        "Canboat version": metadata["canboat_version"],
        "Schema version": metadata["canboat_schema_version"],
        "PGNs found": metadata["number_of_pgns_found"],
        "Definitions/variants found": metadata["number_of_variants_found"],
        "Supported": metadata["number_supported"],
        "Partial": metadata["number_partially_supported"],
        "Skipped": metadata["number_skipped"],
        "Eligible linear target signals": metadata[
            "number_of_eligible_linear_target_signals"
        ],
        "Simulation samples": metadata["number_of_time_steps"],
        "Logical N2K messages": metadata["number_of_logical_messages"],
        "Physical CAN frames": metadata["number_of_physical_can_frames"],
        "Target observations": metadata["total_target_observations"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
