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
    parser.add_argument("--mode", choices=("aligned", "arbitrary"), default="aligned")
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
        "--modes", nargs="+", choices=("aligned", "arbitrary"), default=["aligned", "arbitrary"]
    )
    experiment_parser.add_argument("--max-signals", type=int, default=25)
    experiment_parser.add_argument("--min-pairs", type=int, default=8)
    experiment_parser.add_argument("--min-pairing-fraction", type=float, default=0.5)
    experiment_parser.add_argument("--top-k", type=int, default=10)
    experiment_parser.add_argument("--messages-per-sample", type=int, default=24)
    experiment_parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "vendor" / "canboat" / "canboat.json",
    )
    experiment_parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "experiments"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "infer":
        run = infer_signals(
            arguments.can,
            arguments.known,
            mode=arguments.mode,
            bit_lengths=tuple(arguments.bit_lengths),
            byte_orders=tuple(arguments.byte_orders),
            min_pairs=arguments.min_pairs,
            min_pairing_fraction=arguments.min_pairing_fraction,
            top_k=arguments.top_k,
            max_signals=arguments.max_signals,
            selected_signals=set(arguments.signals) if arguments.signals else None,
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
    # Imported lazily so the discovery command never imports the simulator.
    from inference.experiments import run_experiment

    rows = run_experiment(
        database_path=arguments.database,
        output_directory=arguments.output,
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
