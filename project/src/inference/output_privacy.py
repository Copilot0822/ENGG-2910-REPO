"""Protect local definition exports and serialize only public evaluation fields.

This module has no simulator or database dependency. Discovery may use the
output guard without receiving any answer-key information.
"""

from pathlib import Path
from typing import Any


PRIVATE_OUTPUT_PATTERNS = (
    "ground_truth*.json", "signal_catalog*.csv", "inferred_signals*.json",
    "candidate_rankings*.csv", "simulation_metadata.json", "experiment_metadata.json",
    "comparison_metadata.json", "inference_metadata.json", "run_config.json", "checkpoints/", "private/",
    "*decoded*.json", "*decoded*.csv", "*definition*.json", "*definition*.csv",
    "*signal_mapping*.json", "*signal_mapping*.csv", "*debug_dump*",
)


def protect_private_outputs(directory: str | Path) -> None:
    """Add output-local ignores so custom output paths also protect answer keys."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    ignore = destination / ".gitignore"
    existing = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    missing = [pattern for pattern in PRIVATE_OUTPUT_PATTERNS
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


def public_signal_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Discard names, units, true definitions, candidates and original ID mappings."""
    return [
        {"signal_id": f"signal_{index:06d}",
         **{key: row[key] for key in PUBLIC_SIGNAL_FIELDS if key in row}}
        for index, row in enumerate(rows, start=1)
    ]
