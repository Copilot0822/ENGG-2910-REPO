"""Reproducible orchestration for simulator/inference/evaluation experiments.

This module is deliberately outside discovery: it may generate simulator data
and evaluate against ground truth, while :mod:`inference.infer` cannot.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference.evaluate import evaluate_results, write_evaluation
from inference.infer import infer_signals
from inference.output import write_inference_outputs
from simulator.canboat_loader import load_canboat
from simulator.dataset_generator import DatasetGenerator
from simulator.output import write_outputs


EXPERIMENT_COLUMNS = [
    "observations_per_signal",
    "mode",
    "total_signals",
    "recoverable_signals",
    "top1_location_accuracy",
    "top3_location_accuracy",
    "top5_location_accuracy",
    "top10_location_accuracy",
    "full_encoding_accuracy",
    "average_scale_error",
    "average_offset_error",
    "runtime_seconds",
    "candidates_tested",
]

_SIGNAL_ID = re.compile(r"\[(signal_\d+)\]\s*$")


def _subset_known_values(source: Path, destination: Path, limit: int) -> None:
    counts: dict[str, int] = {}
    with source.open(newline="", encoding="utf-8") as input_handle, destination.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"{source} has no CSV header")
        writer = csv.DictWriter(
            output_handle, fieldnames=reader.fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in reader:
            signal = row["signal_name"]
            count = counts.get(signal, 0)
            if count < limit:
                writer.writerow(row)
                counts[signal] = count + 1


def _select_standard_width_signals(
    known_path: Path, ground_truth_path: Path, limit: int
) -> set[str]:
    """Select a declared experiment population, not features for discovery.

    The population is limited to Stage 1/2 widths (8/16/32). Any unaligned
    members are placed first so an aligned-versus-arbitrary comparison actually
    exercises the distinction. The inference engine receives only their opaque
    labels and still discovers all candidate definitions from the two CSVs.
    """

    truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))["signals"]
    with known_path.open(newline="", encoding="utf-8") as handle:
        names = sorted({row["signal_name"] for row in csv.DictReader(handle)})
    eligible: list[tuple[bool, str]] = []
    for name in names:
        match = _SIGNAL_ID.search(name)
        if not match or match.group(1) not in truth:
            continue
        definition = truth[match.group(1)]
        if int(definition["bit_length"]) not in {8, 16, 32}:
            continue
        eligible.append((int(definition["start_bit"]) % 8 == 0, name))
    # False (unaligned) sorts before True (aligned).
    return {name for _aligned, name in sorted(eligible)[:limit]}


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(
    rows: list[dict[str, object]], metric: str, label: str, destination: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    modes = sorted({str(row["mode"]) for row in rows})
    for mode in modes:
        selected = sorted(
            (row for row in rows if row["mode"] == mode),
            key=lambda row: int(row["observations_per_signal"]),
        )
        axis.plot(
            [int(row["observations_per_signal"]) for row in selected],
            [float(row[metric]) for row in selected],
            marker="o",
            linewidth=2,
            label=mode,
        )
    axis.set_xlabel("Known observations per signal")
    axis.set_ylabel(label)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def run_experiment(
    *,
    database_path: str | Path,
    output_directory: str | Path,
    observation_counts: Iterable[int],
    seed: int,
    modes: tuple[str, ...] = ("aligned", "arbitrary"),
    max_signals: int = 25,
    min_pairs: int = 8,
    min_pairing_fraction: float = 0.5,
    top_k: int = 10,
    messages_per_sample: int = 24,
) -> list[dict[str, object]]:
    counts = sorted(set(int(value) for value in observation_counts))
    if not counts or counts[0] < min_pairs:
        raise ValueError("observation counts must be nonempty and at least min_pairs")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "dataset"
    database = load_canboat(database_path)
    simulation = DatasetGenerator(database, seed).coverage_mode(
        max(counts), messages_per_sample
    )
    write_outputs(dataset, database, simulation)
    can_path = dataset / "can_frames.csv"
    full_known = dataset / "known_values.csv"
    truth_path = dataset / "ground_truth.json"
    selected_signals = _select_standard_width_signals(
        full_known, truth_path, max_signals
    )

    rows: list[dict[str, object]] = []
    for count in counts:
        known_subset = output / f"known_values_{count}.csv"
        _subset_known_values(full_known, known_subset, count)
        for mode in modes:
            run = infer_signals(
                can_path,
                known_subset,
                mode=mode,
                min_pairs=min_pairs,
                min_pairing_fraction=min_pairing_fraction,
                top_k=top_k,
                selected_signals=selected_signals,
            )
            run_directory = output / "runs" / f"{mode}_{count}"
            write_inference_outputs(run_directory, run)
            evaluation = evaluate_results(
                run_directory / "inferred_signals.json", truth_path
            )
            write_evaluation(run_directory / "evaluation_results.json", evaluation)
            summary = evaluation["summary"]
            rows.append(
                {
                    "observations_per_signal": count,
                    "mode": mode,
                    "total_signals": summary["total_signals"],
                    "recoverable_signals": summary["recoverable_signals_in_mode"],
                    "top1_location_accuracy": summary["top1_location_accuracy"],
                    "top3_location_accuracy": summary["top3_location_accuracy"],
                    "top5_location_accuracy": summary["top5_location_accuracy"],
                    "top10_location_accuracy": summary["top10_location_accuracy"],
                    "full_encoding_accuracy": summary["full_encoding_accuracy"],
                    "average_scale_error": summary["average_scale_absolute_error"],
                    "average_offset_error": summary["average_offset_absolute_error"],
                    "runtime_seconds": run.metadata["runtime_seconds"],
                    "candidates_tested": run.metadata["number_of_candidates_tested"],
                }
            )

    _write_rows(output / "recovery_vs_observations.csv", rows)
    maximum_rows = [row for row in rows if row["observations_per_signal"] == max(counts)]
    _write_rows(output / "aligned_vs_arbitrary.csv", maximum_rows)
    _plot_metric(
        rows,
        "top1_location_accuracy",
        "Top-1 location accuracy",
        output / "top1_location_accuracy.png",
    )
    _plot_metric(
        rows,
        "top5_location_accuracy",
        "Top-5 location accuracy",
        output / "top5_location_accuracy.png",
    )
    _plot_metric(
        rows,
        "full_encoding_accuracy",
        "Full encoding accuracy",
        output / "full_encoding_accuracy.png",
    )
    metadata = {
        "observation_counts": counts,
        "seed": seed,
        "modes": list(modes),
        "max_signals": max_signals,
        "signal_selection": "8/16/32-bit population, unaligned member first",
        "min_pairs": min_pairs,
        "min_pairing_fraction": min_pairing_fraction,
        "top_k": top_k,
        "messages_per_sample": messages_per_sample,
        "canboat_version": database.version,
        "canboat_sha256": database.sha256,
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows
