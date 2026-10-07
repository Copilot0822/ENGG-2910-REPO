"""Protect local definition exports and serialize only public evaluation fields.

This module has no simulator or database dependency. Discovery may use the
output guard without receiving any answer-key information.
"""

import math
from pathlib import Path
from typing import Any


PRIVATE_OUTPUT_PATTERNS = (
    "ground_truth*.json", "signal_catalog*.csv", "inferred_signals*.json",
    "candidate_rankings*.csv", "simulation_metadata.json", "experiment_metadata.json",
    "comparison_metadata.json", "inference_metadata.json", "run_config.json", "checkpoints/", "private/",
    "*decoded*.json", "*decoded*.csv", "*definition*.json", "*definition*.csv",
    "*signal_mapping*.json", "*signal_mapping*.csv", "*debug_dump*",
)
LICENSED_DATASET_PATTERNS = ("can_frames*.csv", "known_values*.csv")


def protect_private_outputs(
    directory: str | Path, *, private_derived_output: bool = False,
) -> None:
    """Protect answer keys, plus reconstructable datasets when classified private.

    Raw datasets are ignored locally only for licensed-derived J1939 runs;
    unrelated NMEA and deliberately invented synthetic inputs stay usable.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    ignore = destination / ".gitignore"
    existing = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    patterns = PRIVATE_OUTPUT_PATTERNS + (
        LICENSED_DATASET_PATTERNS if private_derived_output else ()
    )
    missing = [pattern for pattern in patterns
               if pattern not in existing.splitlines()]
    if missing:
        separator = "\n" if existing and not existing.endswith("\n") else ""
        ignore.write_text(
            existing + separator + "# Private local CAN definition exports\n"
            + "\n".join(missing) + "\n", encoding="utf-8",
        )


def protect_private_file(path: str | Path) -> None:
    """Protect a private artifact even when a caller chooses an arbitrary name."""
    destination = Path(path)
    protect_private_outputs(destination.parent)
    ignore = destination.parent / ".gitignore"
    content = ignore.read_text(encoding="utf-8")
    name = destination.name
    for char in ("\\", "[", "*", "?", "#", "!"):
        name = name.replace(char, "\\" + char)
    if name not in content.splitlines():
        ignore.write_text(content + name + "\n", encoding="utf-8")


PUBLIC_SIGNAL_FIELDS = (
    "location_rank", "top1_location_correct", "top3_location_correct",
    "top5_location_correct", "top10_location_correct", "full_encoding_correct",
    "identifiable_full_encoding_correct", "identifiability_status", "result_status",
    "correct_pgn", "correct_start_bit", "correct_width", "correct_byte_order",
    "correct_signedness", "scale_absolute_error", "scale_relative_error",
    "offset_absolute_error", "offset_relative_error", "scale_error", "offset_error",
    "candidate_count",
)

PUBLIC_SUMMARY_FIELDS = (
    "total_signals", "signals", "recoverable_signals_in_mode",
    "top1_location_accuracy", "top3_location_accuracy", "top5_location_accuracy",
    "top10_location_accuracy", "full_encoding_accuracy", "raw_full_encoding_accuracy",
    "identifiable_full_encoding_accuracy", "top1_location_accuracy_within_scope",
    "top3_location_accuracy_within_scope", "top5_location_accuracy_within_scope",
    "top10_location_accuracy_within_scope", "full_encoding_accuracy_within_scope",
    "average_scale_absolute_error", "average_scale_relative_error",
    "average_offset_absolute_error", "average_offset_relative_error",
    "average_scale_absolute_error_when_location_correct",
    "average_offset_absolute_error_when_location_correct",
    "scale_relative_tolerance", "scale_absolute_tolerance",
    "offset_relative_tolerance", "offset_absolute_tolerance",
    "correct_pgn_accuracy", "correct_start_bit_accuracy", "correct_width_accuracy",
    "correct_byte_order_accuracy", "correct_signedness_accuracy",
    "mean_scale_absolute_error", "mean_offset_absolute_error",
    "ambiguous_signals", "identifiable_signals", "incorrect_signals", "candidate_regressions",
)


def _numeric_metric(value: Any) -> bool:
    return value is None or (
        type(value) in (int, float) and math.isfinite(value)
    )


def public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Allow only numerical aggregate metrics, never nested definitions or IDs."""
    return {key: summary[key] for key in PUBLIC_SUMMARY_FIELDS
            if key in summary and _numeric_metric(summary[key])}


def _public_signal_metric(key: str, value: Any) -> bool:
    if key in {"identifiability_status", "result_status"}:
        return isinstance(value, str) and value in {"AMBIGUOUS", "IDENTIFIABLE", "INCORRECT"}
    if key.startswith("correct_") or key.endswith("_correct"):
        return type(value) is bool
    return _numeric_metric(value)


def public_signal_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Discard names, units, true definitions, candidates and original ID mappings."""
    return [
        {"signal_id": f"signal_{index:06d}",
         **{key: row[key] for key in PUBLIC_SIGNAL_FIELDS
            if key in row and _public_signal_metric(key, row[key])}}
        for index, row in enumerate(rows, start=1)
    ]
