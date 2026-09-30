"""Command-line interface for NMEA 2000 and SAE J1939 simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulator.canboat_loader import load_canboat
from simulator.dataset_generator import DatasetGenerator
from simulator.output import write_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_N2K_DATABASE = PROJECT_ROOT / "vendor" / "canboat" / "canboat.json"
DEFAULT_J1939_DATABASE = (
    PROJECT_ROOT / "vendor" / "j1939" / "J1939_2024-06_v2.4.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simulator",
        description=(
            "Generate deterministic physical NMEA 2000 or SAE J1939 CAN frames "
            "and synchronized known physical values."
        ),
    )
    parser.add_argument("--protocol", choices=("nmea2000", "j1939"), default="nmea2000")
    parser.add_argument("--mode", choices=("random", "coverage"), default="random")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--min-observations-per-field", type=int, default=20)
    parser.add_argument("--messages-per-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument(
        "--category",
        choices=("all_linear", "aligned_8_16_32"),
        default="aligned_8_16_32",
        help="J1939 target category",
    )
    parser.add_argument(
        "--include-transport",
        action="store_true",
        help="include J1939 messages longer than eight bytes using TP.CM/TP.DT",
    )
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.protocol == "j1939":
        from simulator.j1939.generator import J1939DatasetGenerator
        from simulator.j1939.loader import load_j1939_database
        from simulator.j1939.output import write_j1939_outputs

        database_path = arguments.database or DEFAULT_J1939_DATABASE
        output = arguments.output or PROJECT_ROOT / "data" / "j1939" / "simulated"
        database = load_j1939_database(database_path)
        generator = J1939DatasetGenerator(database, arguments.seed)
        if arguments.mode == "random":
            result = generator.random_mode(
                arguments.samples,
                category=arguments.category,
                include_transport=arguments.include_transport,
                messages_per_sample=arguments.messages_per_sample or 12,
            )
        else:
            result = generator.coverage_mode(
                arguments.min_observations_per_field,
                category=arguments.category,
                include_transport=arguments.include_transport,
                messages_per_sample=arguments.messages_per_sample or 48,
            )
        result.options.update(
            {
                "database": str(database_path),
                "output": str(output),
                "samples": arguments.samples,
            }
        )
        return write_j1939_outputs(output, database, result)

    database_path = arguments.database or DEFAULT_N2K_DATABASE
    output = arguments.output or PROJECT_ROOT / "data" / "simulated"
    database = load_canboat(database_path)
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
            "database": str(database_path),
            "output": str(output),
            "samples": arguments.samples,
            "min_observations_per_field": arguments.min_observations_per_field,
        }
    )
    return write_outputs(output, database, result)


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    metadata = run(arguments)
    if arguments.protocol == "j1939":
        summary = {
            "Protocol": "J1939",
            "Database version": metadata["database_version"],
            "Messages/PGNs loaded": metadata["number_of_messages_loaded"],
            "Signals loaded": metadata["number_of_signals_loaded"],
            "Aligned 8/16/32 targets": metadata["number_of_aligned_8_16_32_targets"],
            "Eligible targets in dataset": metadata["number_of_eligible_targets_in_dataset"],
            "Single-frame messages": metadata["number_of_single_frame_messages"],
            "Transport messages": metadata["number_of_transport_messages"],
            "Simulation samples": metadata["number_of_samples"],
            "Logical messages": metadata["number_of_logical_messages"],
            "Physical CAN frames": metadata["number_of_physical_can_frames"],
            "Target observations": metadata["total_target_observations"],
        }
        print(json.dumps(summary, indent=2))
        return
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
