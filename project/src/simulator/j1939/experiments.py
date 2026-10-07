"""J1939 observation-count and cross-protocol experiment orchestration."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

from inference.evaluate import evaluate_results, write_evaluation
from inference.experiments import (
    EXPERIMENT_COLUMNS,
    _plot_metric,
    _subset_known_values,
    _write_rows,
)
from inference.infer import infer_signals
from inference.output import write_inference_outputs
from inference.output_privacy import protect_private_outputs
from simulator.j1939.generator import J1939DatasetGenerator
from simulator.j1939.loader import load_j1939_database
from simulator.j1939.models import J1939Database, J1939Message
from simulator.j1939.output import write_j1939_outputs


_SIGNAL_ID_SUFFIX = re.compile(r"\[(j1939_signal_\d+)\]\s*$")


def _stratified_extended_targets(
    dataset_directory: Path,
    *,
    min_bit_length: int,
    max_bit_length: int,
    signals_per_length: int,
) -> tuple[set[str], list[dict[str, object]]]:
    """Select opaque labels for a width-stratified post-generation benchmark.

    Ground truth is used here only by experiment orchestration to define the
    population.  The discovery call still receives just the two clean CSVs,
    the opaque selected labels, and a caller-supplied generic length range.
    """

    if min_bit_length <= 0 or max_bit_length < min_bit_length:
        raise ValueError("invalid extended bit-length range")
    if signals_per_length <= 0:
        raise ValueError("signals_per_length must be positive")
    truth = json.loads(
        (dataset_directory / "ground_truth.json").read_text(encoding="utf-8")
    )["signals"]
    names_by_id: dict[str, str] = {}
    with (dataset_directory / "known_values.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            match = _SIGNAL_ID_SUFFIX.search(row["signal_name"])
            if match:
                names_by_id.setdefault(match.group(1), row["signal_name"])

    by_length: dict[int, list[tuple[str, str, dict[str, object]]]] = {}
    for signal_id, definition in truth.items():
        bit_length = int(definition["bit_length"])
        signal_name = names_by_id.get(signal_id)
        if signal_name is None or not min_bit_length <= bit_length <= max_bit_length:
            continue
        by_length.setdefault(bit_length, []).append(
            (signal_id, signal_name, definition)
        )

    selected_names: set[str] = set()
    selected_metadata: list[dict[str, object]] = []
    for bit_length in sorted(by_length):
        for signal_id, signal_name, definition in sorted(by_length[bit_length])[
            :signals_per_length
        ]:
            selected_names.add(signal_name)
            selected_metadata.append(
                {
                    "signal_id": signal_id,
                    "opaque_signal_label": signal_name,
                    "bit_length": bit_length,
                    "start_bit": int(definition["start_bit"]),
                }
            )
    if not selected_names:
        raise ValueError("no benchmark targets matched the requested length range")
    return selected_names, selected_metadata


def run_j1939_search_space_comparison(
    *,
    dataset_directory: str | Path,
    output_directory: str | Path,
    min_bit_length: int = 1,
    max_bit_length: int = 32,
    signals_per_length: int = 1,
    min_pairs: int = 8,
    min_pairing_fraction: float = 0.5,
    top_k: int = 10,
) -> list[dict[str, object]]:
    """Measure aligned 8/16/32 search against generic extended search."""

    dataset = Path(dataset_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    protect_private_outputs(output)
    selected, selected_metadata = _stratified_extended_targets(
        dataset,
        min_bit_length=min_bit_length,
        max_bit_length=max_bit_length,
        signals_per_length=signals_per_length,
    )
    can_path = dataset / "can_frames.csv"
    known_path = dataset / "known_values.csv"
    truth_path = dataset / "ground_truth.json"
    rows: list[dict[str, object]] = []
    searches = (
        ("aligned", (8, 16, 32)),
        ("extended", tuple(range(min_bit_length, max_bit_length + 1))),
    )
    for mode, bit_lengths in searches:
        run = infer_signals(
            can_path,
            known_path,
            protocol="j1939",
            mode=mode,  # type: ignore[arg-type]
            bit_lengths=bit_lengths,
            min_pairs=min_pairs,
            min_pairing_fraction=min_pairing_fraction,
            top_k=top_k,
            selected_signals=selected,
        )
        run_directory = output / mode
        write_inference_outputs(run_directory, run)
        evaluation = evaluate_results(
            run_directory / "inferred_signals.json", truth_path
        )
        write_evaluation(run_directory / "evaluation_results.json", evaluation)
        summary = evaluation["summary"]
        total = int(summary["total_signals"])
        recoverable = int(summary["recoverable_signals_in_mode"])
        rows.append(
            {
                "mode": mode,
                "candidate_lengths": " ".join(str(item) for item in bit_lengths),
                "target_signals": total,
                "recoverable_signals": recoverable,
                "target_coverage": recoverable / total if total else 0.0,
                "top1_location_accuracy": summary["top1_location_accuracy"],
                "top5_location_accuracy": summary["top5_location_accuracy"],
                "top10_location_accuracy": summary["top10_location_accuracy"],
                "top1_location_accuracy_within_scope": summary[
                    "top1_location_accuracy_within_scope"
                ],
                "full_encoding_accuracy": summary["full_encoding_accuracy"],
                "full_encoding_accuracy_within_scope": summary[
                    "full_encoding_accuracy_within_scope"
                ],
                "candidates_tested": run.metadata["number_of_candidates_tested"],
                "runtime_seconds": run.metadata["runtime_seconds"],
                "mean_scale_error": summary["average_scale_absolute_error"],
                "mean_offset_error": summary["average_offset_absolute_error"],
            }
        )

    columns = list(rows[0])
    with (output / "aligned_vs_extended.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "J1939",
                "dataset_directory": str(dataset),
                "generic_extended_range": [min_bit_length, max_bit_length],
                "signals_per_available_length": signals_per_length,
                "selected_targets": selected_metadata,
                "answer_key_separation": (
                    "Ground truth selected and evaluated the benchmark population; "
                    "discovery received only can_frames.csv, known_values.csv, opaque "
                    "labels, and the configured generic candidate range."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def _select_aligned_population(
    database: J1939Database, limit: int
) -> tuple[set[str], list[J1939Message]]:
    eligible = sorted(
        (
            f"signal_{signal.signal_id.rsplit('_', 1)[-1]} [{signal.signal_id}]",
            message.definition_id,
        )
        for message in database.single_frame_messages
        for signal in message.target_signals("aligned_8_16_32")
    )
    selected = eligible[:limit]
    selected_names = {name for name, _ in selected}
    definition_ids = {definition_id for _, definition_id in selected}
    definitions = [
        message
        for message in database.messages
        if message.definition_id in definition_ids
    ]
    return selected_names, definitions


def _cross_protocol_rows(
    j1939_rows: list[dict[str, object]], n2k_path: Path
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if n2k_path.exists():
        with n2k_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["mode"] != "aligned":
                    continue
                result.append(
                    {
                        "protocol": "NMEA2000",
                        "number_of_target_signals": row["total_signals"],
                        "observations_per_signal": row["observations_per_signal"],
                        "top1_location_accuracy": row["top1_location_accuracy"],
                        "top5_location_accuracy": row["top5_location_accuracy"],
                        "full_encoding_accuracy": row["full_encoding_accuracy"],
                        "candidate_count": row["candidates_tested"],
                        "runtime_seconds": row["runtime_seconds"],
                        "mean_scale_error": row["average_scale_error"],
                        "mean_offset_error": row["average_offset_error"],
                    }
                )
    for row in j1939_rows:
        if row["mode"] != "aligned":
            continue
        result.append(
            {
                "protocol": "J1939",
                "number_of_target_signals": row["total_signals"],
                "observations_per_signal": row["observations_per_signal"],
                "top1_location_accuracy": row["top1_location_accuracy"],
                "top5_location_accuracy": row["top5_location_accuracy"],
                "full_encoding_accuracy": row["full_encoding_accuracy"],
                "candidate_count": row["candidates_tested"],
                "runtime_seconds": row["runtime_seconds"],
                "mean_scale_error": row["average_scale_error"],
                "mean_offset_error": row["average_offset_error"],
            }
        )
    return result


def _write_cross_protocol(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "protocol",
        "number_of_target_signals",
        "observations_per_signal",
        "top1_location_accuracy",
        "top5_location_accuracy",
        "full_encoding_accuracy",
        "candidate_count",
        "runtime_seconds",
        "mean_scale_error",
        "mean_offset_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_j1939_experiment(
    *,
    database_path: str | Path | None,
    output_directory: str | Path,
    observation_counts: Iterable[int],
    seed: int,
    modes: tuple[str, ...] = ("aligned",),
    max_signals: int = 25,
    min_pairs: int = 8,
    min_pairing_fraction: float = 0.5,
    top_k: int = 10,
    messages_per_sample: int = 96,
    cross_protocol_output: str | Path | None = None,
    n2k_results_path: str | Path | None = None,
) -> list[dict[str, object]]:
    counts = sorted(set(int(value) for value in observation_counts))
    if not counts or counts[0] < min_pairs:
        raise ValueError("observation counts must be nonempty and at least min_pairs")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    protect_private_outputs(output)
    database = load_j1939_database(database_path)
    selected_signals, selected_messages = _select_aligned_population(
        database, max_signals
    )
    simulation = J1939DatasetGenerator(database, seed).targeted_coverage_mode(
        max(counts),
        selected_messages,
        category="aligned_8_16_32",
        include_transport=False,
    )
    dataset = output / "dataset"
    write_j1939_outputs(dataset, database, simulation)
    can_path = dataset / "can_frames.csv"
    full_known = dataset / "known_values.csv"
    truth_path = dataset / "ground_truth.json"
    rows: list[dict[str, object]] = []
    for count in counts:
        known_subset = output / f"known_values_{count}.csv"
        _subset_known_values(full_known, known_subset, count)
        for mode in modes:
            run = infer_signals(
                can_path,
                known_subset,
                protocol="j1939",
                mode=mode,  # type: ignore[arg-type]
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
        "protocol": "J1939",
        "observation_counts": counts,
        "seed": seed,
        "modes": list(modes),
        "max_signals": max_signals,
        "min_pairs": min_pairs,
        "min_pairing_fraction": min_pairing_fraction,
        "top_k": top_k,
        "messages_per_sample": 1,
        "requested_messages_per_sample": messages_per_sample,
        "selected_message_count": len(selected_messages),
        "generation_strategy": "targeted messages on disjoint sample ranges",
        "database_version": database.database_version,
        "single_frame_baseline": True,
        "target_category": "aligned_8_16_32",
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if cross_protocol_output is not None and n2k_results_path is not None:
        cross_rows = _cross_protocol_rows(rows, Path(n2k_results_path))
        _write_cross_protocol(Path(cross_protocol_output), cross_rows)
    return rows
