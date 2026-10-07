"""Orchestrate the complete J1939 simulation, discovery, and evaluation run."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from inference.full_library import infer_full_library
from inference.output_privacy import protect_private_outputs
from inference.full_library_evaluate import (
    _effective_raw_bounds,
    evaluate_full_library,
    summarize_rows,
    write_evaluation_json,
    write_full_library_plots,
)
from simulator.j1939.generator import J1939DatasetGenerator, valid_raw_limits
from simulator.j1939.loader import load_j1939_database
from simulator.j1939.output import write_j1939_outputs


FULL_CANDIDATE_LENGTHS = tuple(range(1, 33)) + (40, 48, 56, 64)
ALIGNED_CANDIDATE_LENGTHS = (8, 16, 32)


def _unsupported_ids_from_truth(truth: dict[str, dict[str, Any]]) -> set[str]:
    supported_lengths = set(FULL_CANDIDATE_LENGTHS)
    exact_limit = 1 << 53
    unsupported: set[str] = set()
    for signal_id, definition in truth.items():
        lower, upper = _effective_raw_bounds(definition)
        if int(definition["bit_length"]) not in supported_lengths or (
            int(definition["bit_length"]) > 53
            and (lower < -exact_limit or upper > exact_limit)
        ):
            unsupported.add(signal_id)
    return unsupported


def _accounting(database: Any) -> dict[str, Any]:
    signals = [signal for message in database.messages for signal in message.signals]
    aligned = sum(signal.aligned_8_16_32 for signal in signals)
    at_most_32 = sum(1 <= signal.bit_length <= 32 for signal in signals)
    larger_signals = [signal for signal in signals if signal.bit_length > 32]
    float64_exact_limit = 1 << 53
    unsupported_signals = []
    for signal in signals:
        lower, upper = valid_raw_limits(signal)
        if signal.bit_length > 53 and (
            lower < -float64_exact_limit or upper > float64_exact_limit
        ):
            unsupported_signals.append(signal)
    larger_inferable = [
        signal for signal in larger_signals if signal not in unsupported_signals
    ]
    supported_lengths = set(FULL_CANDIDATE_LENGTHS)
    unsupported_widths = [
        signal for signal in signals if signal.bit_length not in supported_lengths
    ]
    unsupported = len({signal.signal_id for signal in unsupported_signals + unsupported_widths})
    bam_messages = [
        message
        for message in database.transport_messages
        if message.pdu_type != "PDU1"
    ]
    rts_messages = [
        message
        for message in database.transport_messages
        if message.pdu_type == "PDU1"
    ]
    return {
        "total_j1939_signals": len(signals),
        "total_messages_pgns": len(database.messages),
        "category_a_aligned_8_16_32": aligned,
        "category_b_all_signals_1_through_32": at_most_32,
        "category_b_additional_non_aligned_or_other_width": at_most_32 - aligned,
        "category_c_larger_than_32_total": len(larger_signals),
        "category_c_larger_than_32_inferable": len(larger_inferable),
        "category_d_all_inferable": len(signals) - unsupported,
        "unsupported_for_inference": unsupported,
        "unsupported_signal_ids": sorted(
            {signal.signal_id for signal in unsupported_signals + unsupported_widths}
        ),
        "unsupported_reason": (
            "Raw integer range exceeds exact IEEE-754 float64 representation "
            "or width is absent from the generic candidate policy."
        ),
        "single_frame_messages": len(database.single_frame_messages),
        "single_frame_signals": sum(
            len(message.signals) for message in database.single_frame_messages
        ),
        "bam_messages": len(bam_messages),
        "bam_signals": sum(len(message.signals) for message in bam_messages),
        "rts_cts_messages": len(rts_messages),
        "rts_cts_signals": sum(len(message.signals) for message in rts_messages),
        "generic_candidate_lengths": list(FULL_CANDIDATE_LENGTHS),
        "partition_reconciliation": {
            "aligned": aligned,
            "additional_le_32": at_most_32 - aligned,
            "larger_than_32_inferable": len(larger_inferable),
            "unsupported": unsupported,
            "sum": (
                aligned
                + (at_most_32 - aligned)
                + len(larger_inferable)
                + unsupported
            ),
        },
    }


def _write_breakdown(
    path: Path, group_name: str, groups: dict[str, dict[str, Any]]
) -> None:
    columns = [
        group_name,
        "signals",
        "top1_location_accuracy",
        "top3_location_accuracy",
        "top5_location_accuracy",
        "top10_location_accuracy",
        "raw_full_encoding_accuracy",
        "identifiable_full_encoding_accuracy",
        "ambiguous_signals",
        "incorrect_signals",
        "candidate_regressions",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name, summary in groups.items():
            writer.writerow(
                {group_name: name, **{key: summary[key] for key in columns[1:]}}
            )


def _public_accounting(accounting: dict[str, Any]) -> dict[str, Any]:
    """Retain aggregate counts without original database ID lists."""
    return {key: value for key, value in accounting.items() if key != "unsupported_signal_ids"}


def _write_search_analysis(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "mode",
        "target_signals",
        "candidate_lengths",
        "candidate_definitions_extracted",
        "candidate_regressions",
        "candidates_per_target_average",
        "candidate_extraction_seconds",
        "regression_scoring_seconds",
        "input_loading_seconds",
        "runtime_seconds",
        "peak_candidate_matrix_megabytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_full_library_experiment(
    *,
    database_path: str | Path | None,
    output_directory: str | Path,
    observations_per_signal: int = 200,
    seed: int = 42,
    batch_size: int = 250,
    candidate_batch_size: int = 512,
    top_k: int = 10,
    resume: bool = False,
) -> dict[str, Any]:
    if observations_per_signal != 200:
        raise ValueError("the declared full-library experiment requires 200 observations")
    total_started = time.perf_counter()
    output = Path(output_directory)
    simulated = output / "simulated"
    inference = output / "inference"
    evaluation_directory = output / "evaluation"
    metadata_directory = output / "metadata"
    for directory in (output, simulated, inference, evaluation_directory, metadata_directory):
        directory.mkdir(parents=True, exist_ok=True)

    database = load_j1939_database(database_path)
    protect_private_outputs(output, private_derived_output=database.private_derived_output)
    # Install guards before the resume branch, which may skip the output writer.
    protect_private_outputs(simulated, private_derived_output=database.private_derived_output)
    accounting = _accounting(database)
    print(f"Total J1939 signals: {accounting['total_j1939_signals']}", flush=True)
    print(
        "Eligible for aligned search: "
        f"{accounting['category_a_aligned_8_16_32']}",
        flush=True,
    )
    print(
        "Eligible for extended search (1..32): "
        f"{accounting['category_b_all_signals_1_through_32']}",
        flush=True,
    )
    print(
        "Eligible for >32-bit search: "
        f"{accounting['category_c_larger_than_32_inferable']}",
        flush=True,
    )
    print(
        f"Unsupported for inference: {accounting['unsupported_for_inference']}",
        flush=True,
    )
    (metadata_directory / "full_library_accounting.json").write_text(
        json.dumps(_public_accounting(accounting), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    simulation_started = time.perf_counter()
    required_dataset = (
        simulated / "can_frames.csv",
        simulated / "known_values.csv",
        simulated / "ground_truth.json",
        simulated / "simulation_metadata.json",
    )
    existing_metadata = (
        json.loads((simulated / "simulation_metadata.json").read_text(encoding="utf-8"))
        if resume and all(path.exists() for path in required_dataset)
        else {}
    )
    if (
        existing_metadata.get("output_schema_version") == 3
        and existing_metadata.get("private_derived_output") == database.private_derived_output
    ):
        simulation_metadata = json.loads(
            (simulated / "simulation_metadata.json").read_text(encoding="utf-8")
        )
        simulation_seconds = 0.0
        print("Reusing completed full-library simulation dataset", flush=True)
    else:
        result = J1939DatasetGenerator(database, seed).targeted_coverage_mode(
            observations_per_signal,
            list(database.messages),
            category="all_linear",
            include_transport=True,
        )
        simulation_metadata = write_j1939_outputs(simulated, database, result)
        simulation_seconds = time.perf_counter() - simulation_started
        print(
            f"Generated {simulation_metadata['number_of_logical_messages']} logical "
            f"messages and {simulation_metadata['number_of_physical_can_frames']} "
            f"physical frames in {simulation_seconds:.1f}s",
            flush=True,
        )

    truth = json.loads((simulated / "ground_truth.json").read_text(encoding="utf-8"))[
        "signals"
    ]
    aligned_ids = {
        signal_id
        for signal_id, definition in truth.items()
        if bool(definition["aligned_8_16_32"])
    }
    unsupported_ids = set(accounting["unsupported_signal_ids"])
    aligned_labels = {
        f"signal_{signal_id.rsplit('_', 1)[-1]} [{signal_id}]"
        for signal_id, definition in truth.items()
        if signal_id in aligned_ids
    }

    aligned_metadata = infer_full_library(
        simulated / "can_frames.csv",
        simulated / "known_values.csv",
        inference / "aligned",
        mode="aligned",
        bit_lengths=ALIGNED_CANDIDATE_LENGTHS,
        selected_signals=aligned_labels,
        batch_size=batch_size,
        candidate_batch_size=candidate_batch_size,
        top_k=top_k,
        resume=resume,
    )
    full_metadata = infer_full_library(
        simulated / "can_frames.csv",
        simulated / "known_values.csv",
        inference,
        mode="extended",
        bit_lengths=FULL_CANDIDATE_LENGTHS,
        batch_size=batch_size,
        candidate_batch_size=candidate_batch_size,
        top_k=top_k,
        resume=resume,
    )

    aligned_evaluation = evaluate_full_library(
        inference / "aligned" / "inferred_signals.json",
        simulated / "ground_truth.json",
        evaluation_directory / "aligned_results.csv",
        allowed_signal_ids=aligned_ids,
    )
    write_evaluation_json(
        evaluation_directory / "aligned_evaluation_results.json", aligned_evaluation
    )
    full_evaluation = evaluate_full_library(
        inference / "inferred_signals.json",
        simulated / "ground_truth.json",
        output / "full_library_results.csv",
    )
    write_evaluation_json(
        evaluation_directory / "evaluation_results.json", full_evaluation
    )

    full_rows = full_evaluation["rows"]
    category_b = [row for row in full_rows if int(row["true_bit_length"]) <= 32]
    category_c = [
        row
        for row in full_rows
        if int(row["true_bit_length"]) > 32
        and row["signal_id"] not in unsupported_ids
    ]
    category_d = [
        row for row in full_rows if row["signal_id"] not in unsupported_ids
    ]
    categories = {
        "category_a_aligned_baseline": aligned_evaluation["summary"],
        "category_b_extended_le_32": summarize_rows(category_b),
        "category_c_larger_linear_fields": summarize_rows(category_c),
        "category_d_all_inferable": summarize_rows(category_d),
        "unsupported_for_float64_inference": summarize_rows(
            [row for row in full_rows if row["signal_id"] in unsupported_ids]
        ),
    }
    search_rows = [
        {
            "mode": "aligned",
            "target_signals": aligned_metadata["number_of_target_signals"],
            "candidate_lengths": "8 16 32",
            "candidate_definitions_extracted": aligned_metadata[
                "candidate_definitions_extracted"
            ],
            "candidate_regressions": aligned_metadata["candidate_regressions"],
            "candidates_per_target_average": aligned_metadata[
                "candidate_definitions_per_target_average"
            ],
            "candidate_extraction_seconds": aligned_metadata[
                "candidate_extraction_seconds"
            ],
            "regression_scoring_seconds": aligned_metadata[
                "regression_scoring_seconds"
            ],
            "input_loading_seconds": aligned_metadata["input_loading_seconds"],
            "runtime_seconds": aligned_metadata["runtime_seconds"],
            "peak_candidate_matrix_megabytes": aligned_metadata[
                "peak_candidate_matrix_megabytes"
            ],
        },
        {
            "mode": "full_library_extended",
            "target_signals": full_metadata["number_of_target_signals"],
            "candidate_lengths": " ".join(map(str, FULL_CANDIDATE_LENGTHS)),
            "candidate_definitions_extracted": full_metadata[
                "candidate_definitions_extracted"
            ],
            "candidate_regressions": full_metadata["candidate_regressions"],
            "candidates_per_target_average": full_metadata[
                "candidate_definitions_per_target_average"
            ],
            "candidate_extraction_seconds": full_metadata[
                "candidate_extraction_seconds"
            ],
            "regression_scoring_seconds": full_metadata[
                "regression_scoring_seconds"
            ],
            "input_loading_seconds": full_metadata["input_loading_seconds"],
            "runtime_seconds": full_metadata["runtime_seconds"],
            "peak_candidate_matrix_megabytes": full_metadata[
                "peak_candidate_matrix_megabytes"
            ],
        },
    ]
    _write_search_analysis(metadata_directory / "search_space_analysis.csv", search_rows)
    _write_breakdown(
        evaluation_directory / "accuracy_by_bit_length.csv",
        "bit_length",
        full_evaluation["by_bit_length"],
    )
    _write_breakdown(
        evaluation_directory / "accuracy_by_alignment.csv",
        "alignment_category",
        full_evaluation["by_alignment"],
    )
    _write_breakdown(
        evaluation_directory / "accuracy_by_transport.csv",
        "transport_category",
        full_evaluation["by_transport"],
    )
    write_full_library_plots(full_evaluation, evaluation_directory, search_rows)

    summary = {
        "accounting": _public_accounting(accounting),
        "dataset": simulation_metadata,
        "categories": categories,
        "by_alignment": full_evaluation["by_alignment"],
        "by_transport": full_evaluation["by_transport"],
        "search_space": search_rows,
        "ambiguity_rule": full_evaluation["ambiguity_rule"],
        "timing": {
            "simulation_seconds": simulation_seconds,
            "aligned_discovery_seconds": aligned_metadata["runtime_seconds"],
            "full_discovery_seconds": full_metadata["runtime_seconds"],
            "aligned_evaluation_seconds": aligned_evaluation[
                "evaluation_runtime_seconds"
            ],
            "full_evaluation_seconds": full_evaluation["evaluation_runtime_seconds"],
            "total_runtime_seconds": time.perf_counter() - total_started,
        },
        "configuration": {
            "seed": seed,
            "observations_per_signal": observations_per_signal,
            "batch_size": batch_size,
            "candidate_batch_size": candidate_batch_size,
            "top_k": top_k,
            "resume": resume,
            "candidate_lengths": list(FULL_CANDIDATE_LENGTHS),
        },
    }
    (output / "full_library_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (metadata_directory / "runtime_metadata.json").write_text(
        json.dumps(summary["timing"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def refresh_full_library_evaluation(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Re-evaluate completed inference without rerunning simulation/discovery."""

    started = time.perf_counter()
    output = Path(output_directory)
    simulated = output / "simulated"
    inference = output / "inference"
    evaluation_directory = output / "evaluation"
    metadata_directory = output / "metadata"
    summary_path = output / "full_library_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    private_derived_output = summary.get("dataset", {}).get("private_derived_output") is not False
    protect_private_outputs(output, private_derived_output=private_derived_output)
    protect_private_outputs(simulated, private_derived_output=private_derived_output)
    truth = json.loads((simulated / "ground_truth.json").read_text(encoding="utf-8"))[
        "signals"
    ]
    unsupported_ids = _unsupported_ids_from_truth(truth)
    larger_total = sum(int(item["bit_length"]) > 32 for item in truth.values())
    summary["accounting"].update(
        {
            "category_c_larger_than_32_total": larger_total,
            "category_c_larger_than_32_inferable": larger_total
            - len(unsupported_ids),
            "category_d_all_inferable": len(truth) - len(unsupported_ids),
            "unsupported_for_inference": len(unsupported_ids),
            "unsupported_reason": (
                "Raw integer range exceeds exact IEEE-754 float64 "
                "representation or width is absent from the generic candidate policy."
            ),
        }
    )
    summary["accounting"].pop("category_c_larger_than_32", None)
    summary["accounting"].pop("unsupported_signal_ids", None)
    summary["accounting"]["partition_reconciliation"].update(
        {
            "larger_than_32_inferable": larger_total - len(unsupported_ids),
            "unsupported": len(unsupported_ids),
            "sum": len(truth),
        }
    )
    summary["accounting"]["partition_reconciliation"].pop(
        "larger_than_32", None
    )
    (metadata_directory / "full_library_accounting.json").write_text(
        json.dumps(summary["accounting"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aligned_ids = {
        signal_id
        for signal_id, definition in truth.items()
        if bool(definition["aligned_8_16_32"])
    }
    aligned_evaluation = evaluate_full_library(
        inference / "aligned" / "inferred_signals.json",
        simulated / "ground_truth.json",
        evaluation_directory / "aligned_results.csv",
        allowed_signal_ids=aligned_ids,
    )
    write_evaluation_json(
        evaluation_directory / "aligned_evaluation_results.json", aligned_evaluation
    )
    full_evaluation = evaluate_full_library(
        inference / "inferred_signals.json",
        simulated / "ground_truth.json",
        output / "full_library_results.csv",
    )
    write_evaluation_json(
        evaluation_directory / "evaluation_results.json", full_evaluation
    )
    rows = full_evaluation["rows"]
    summary["categories"] = {
        "category_a_aligned_baseline": aligned_evaluation["summary"],
        "category_b_extended_le_32": summarize_rows(
            [row for row in rows if int(row["true_bit_length"]) <= 32]
        ),
        "category_c_larger_linear_fields": summarize_rows(
            [
                row
                for row in rows
                if int(row["true_bit_length"]) > 32
                and row["signal_id"] not in unsupported_ids
            ]
        ),
        "category_d_all_inferable": summarize_rows(
            [row for row in rows if row["signal_id"] not in unsupported_ids]
        ),
        "unsupported_for_float64_inference": summarize_rows(
            [row for row in rows if row["signal_id"] in unsupported_ids]
        ),
    }
    summary["by_alignment"] = full_evaluation["by_alignment"]
    summary["by_transport"] = full_evaluation["by_transport"]
    summary["ambiguity_rule"] = full_evaluation["ambiguity_rule"]
    summary["timing"]["aligned_evaluation_seconds"] = aligned_evaluation[
        "evaluation_runtime_seconds"
    ]
    summary["timing"]["full_evaluation_seconds"] = full_evaluation[
        "evaluation_runtime_seconds"
    ]
    summary["timing"]["evaluation_refresh_seconds"] = (
        time.perf_counter() - started
    )
    _write_breakdown(
        evaluation_directory / "accuracy_by_bit_length.csv",
        "bit_length",
        full_evaluation["by_bit_length"],
    )
    _write_breakdown(
        evaluation_directory / "accuracy_by_alignment.csv",
        "alignment_category",
        full_evaluation["by_alignment"],
    )
    _write_breakdown(
        evaluation_directory / "accuracy_by_transport.csv",
        "transport_category",
        full_evaluation["by_transport"],
    )
    write_full_library_plots(
        full_evaluation, evaluation_directory, summary["search_space"]
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (metadata_directory / "runtime_metadata.json").write_text(
        json.dumps(summary["timing"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
