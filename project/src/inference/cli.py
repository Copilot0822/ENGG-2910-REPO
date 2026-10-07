"""Command-line interface for discovery, evaluation, and experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.evaluate import evaluate_results, write_evaluation
from inference.infer import infer_signals
from inference.output import write_inference_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", choices=("nmea2000", "j1939", "auto"), default="nmea2000")
    parser.add_argument("--mode", choices=("aligned", "arbitrary", "extended"), default="aligned")
    parser.add_argument("--bit-lengths", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument(
        "--byte-orders",
        nargs="+",
        choices=("little_endian", "big_endian"),
        default=["little_endian", "big_endian"],
    )
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--min-pairing-fraction", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-signals", type=int)
    parser.add_argument("--signal", action="append", dest="signals")
    parser.add_argument("--candidate-lengths", nargs="+", type=int)
    parser.add_argument("--min-bit-length", type=int)
    parser.add_argument("--max-bit-length", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference",
        description="Infer linear CAN signal encodings and evaluate them separately.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer", help="discover signals from inference inputs")
    infer_parser.add_argument("--can", type=Path, required=True)
    infer_parser.add_argument("--known", type=Path, required=True)
    infer_parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "inference"
    )
    _add_search_arguments(infer_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="compare completed inference with ground truth"
    )
    evaluate_parser.add_argument("--inferred", type=Path, required=True)
    evaluate_parser.add_argument("--ground-truth", type=Path, required=True)
    evaluate_parser.add_argument("--protocol", choices=("nmea2000", "j1939"), default="nmea2000")
    evaluate_parser.add_argument("--output", type=Path)
    evaluate_parser.add_argument("--scale-relative-tolerance", type=float, default=1e-5)
    evaluate_parser.add_argument("--scale-absolute-tolerance", type=float, default=1e-8)
    evaluate_parser.add_argument("--offset-relative-tolerance", type=float, default=1e-5)
    evaluate_parser.add_argument("--offset-absolute-tolerance", type=float, default=1e-6)

    experiment_parser = subparsers.add_parser(
        "experiment", help="measure recovery as observation count changes"
    )
    experiment_parser.add_argument(
        "--observation-counts", nargs="+", type=int, default=[10, 25, 50, 100, 200]
    )
    experiment_parser.add_argument("--seed", type=int, default=42)
    experiment_parser.add_argument(
        "--modes", nargs="+", choices=("aligned", "arbitrary", "extended"), default=["aligned", "arbitrary"]
    )
    experiment_parser.add_argument("--max-signals", type=int, default=25)
    experiment_parser.add_argument("--min-pairs", type=int, default=8)
    experiment_parser.add_argument("--min-pairing-fraction", type=float, default=0.5)
    experiment_parser.add_argument("--top-k", type=int, default=10)
    experiment_parser.add_argument("--messages-per-sample", type=int, default=24)
    experiment_parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="database JSON; J1939 defaults to J1939_DATABASE_PATH, then vendor/j1939",
    )
    experiment_parser.add_argument(
        "--output", type=Path, default=None
    )
    experiment_parser.add_argument(
        "--protocol", choices=("nmea2000", "j1939"), default="nmea2000"
    )
    experiment_parser.add_argument(
        "--skip-cross-protocol",
        action="store_true",
        help="do not rewrite the NMEA 2000 versus J1939 comparison CSV",
    )

    comparison_parser = subparsers.add_parser(
        "compare-search",
        help="compare aligned and generic extended J1939 candidate spaces",
    )
    comparison_parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "j1939" / "extended_dataset",
    )
    comparison_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "j1939" / "extended_comparison",
    )
    comparison_parser.add_argument("--min-bit-length", type=int, default=1)
    comparison_parser.add_argument("--max-bit-length", type=int, default=32)
    comparison_parser.add_argument("--signals-per-length", type=int, default=1)
    comparison_parser.add_argument("--min-pairs", type=int, default=8)
    comparison_parser.add_argument("--min-pairing-fraction", type=float, default=0.5)
    comparison_parser.add_argument("--top-k", type=int, default=10)

    full_parser = subparsers.add_parser(
        "full-library",
        help="run the complete 13,774-signal J1939 experiment",
    )
    full_parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="licensed JSON; defaults to J1939_DATABASE_PATH, then vendor/j1939",
    )
    full_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "j1939" / "full_library",
    )
    full_parser.add_argument("--observations-per-signal", type=int, default=200)
    full_parser.add_argument("--seed", type=int, default=42)
    full_parser.add_argument("--batch-size", type=int, default=250)
    full_parser.add_argument("--candidate-batch-size", type=int, default=512)
    full_parser.add_argument("--top-k", type=int, default=10)
    full_parser.add_argument("--resume", action="store_true")
    full_parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="refresh evaluation and plots from completed inference outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "infer":
        bit_lengths = arguments.candidate_lengths or arguments.bit_lengths
        if arguments.min_bit_length is not None or arguments.max_bit_length is not None:
            minimum = arguments.min_bit_length or 1
            maximum = arguments.max_bit_length or 32
            if minimum > maximum:
                raise SystemExit("--min-bit-length cannot exceed --max-bit-length")
            bit_lengths = list(range(minimum, maximum + 1))
        run = infer_signals(
            arguments.can,
            arguments.known,
            mode=arguments.mode,
            bit_lengths=tuple(bit_lengths),
            byte_orders=tuple(arguments.byte_orders),
            min_pairs=arguments.min_pairs,
            min_pairing_fraction=arguments.min_pairing_fraction,
            top_k=arguments.top_k,
            max_signals=arguments.max_signals,
            selected_signals=set(arguments.signals) if arguments.signals else None,
            protocol=arguments.protocol,
        )
        write_inference_outputs(arguments.output, run)
        print(json.dumps(run.metadata, indent=2))
        return
    if arguments.command == "evaluate":
        evaluation = evaluate_results(
            arguments.inferred,
            arguments.ground_truth,
            scale_relative_tolerance=arguments.scale_relative_tolerance,
            scale_absolute_tolerance=arguments.scale_absolute_tolerance,
            offset_relative_tolerance=arguments.offset_relative_tolerance,
            offset_absolute_tolerance=arguments.offset_absolute_tolerance,
        )
        output = arguments.output or arguments.inferred.parent / "evaluation_results.json"
        write_evaluation(output, evaluation)
        print(json.dumps(evaluation["summary"], indent=2))
        return
    if arguments.command == "compare-search":
        from simulator.j1939.experiments import run_j1939_search_space_comparison

        rows = run_j1939_search_space_comparison(
            dataset_directory=arguments.dataset,
            output_directory=arguments.output,
            min_bit_length=arguments.min_bit_length,
            max_bit_length=arguments.max_bit_length,
            signals_per_length=arguments.signals_per_length,
            min_pairs=arguments.min_pairs,
            min_pairing_fraction=arguments.min_pairing_fraction,
            top_k=arguments.top_k,
        )
        print(json.dumps(rows, indent=2))
        return
    if arguments.command == "full-library":
        from simulator.j1939.full_library import (
            refresh_full_library_evaluation,
            run_full_library_experiment,
        )

        if arguments.evaluation_only:
            summary = refresh_full_library_evaluation(arguments.output)
        else:
            summary = run_full_library_experiment(
                database_path=arguments.database,
                output_directory=arguments.output,
                observations_per_signal=arguments.observations_per_signal,
                seed=arguments.seed,
                batch_size=arguments.batch_size,
                candidate_batch_size=arguments.candidate_batch_size,
                top_k=arguments.top_k,
                resume=arguments.resume,
            )
        print(json.dumps(summary, indent=2))
        return
    if arguments.protocol == "j1939":
        from simulator.j1939.experiments import run_j1939_experiment

        rows = run_j1939_experiment(
            database_path=arguments.database,
            output_directory=arguments.output
            or PROJECT_ROOT / "data" / "j1939" / "experiments",
            observation_counts=arguments.observation_counts,
            seed=arguments.seed,
            modes=tuple(arguments.modes),
            max_signals=arguments.max_signals,
            min_pairs=arguments.min_pairs,
            min_pairing_fraction=arguments.min_pairing_fraction,
            top_k=arguments.top_k,
            messages_per_sample=arguments.messages_per_sample,
            n2k_results_path=(
                None
                if arguments.skip_cross_protocol
                else PROJECT_ROOT
                / "data"
                / "experiments"
                / "recovery_vs_observations.csv"
            ),
            cross_protocol_output=(
                None
                if arguments.skip_cross_protocol
                else PROJECT_ROOT / "data" / "experiments" / "n2k_vs_j1939.csv"
            ),
        )
        print(json.dumps(rows, indent=2))
        return

    # Imported lazily so the discovery command never imports either simulator.
    from inference.experiments import run_experiment

    rows = run_experiment(
        database_path=arguments.database
        or PROJECT_ROOT / "vendor" / "canboat" / "canboat.json",
        output_directory=arguments.output or PROJECT_ROOT / "data" / "experiments",
        observation_counts=arguments.observation_counts,
        seed=arguments.seed,
        modes=tuple(arguments.modes),
        max_signals=arguments.max_signals,
        min_pairs=arguments.min_pairs,
        min_pairing_fraction=arguments.min_pairing_fraction,
        top_k=arguments.top_k,
        messages_per_sample=arguments.messages_per_sample,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
