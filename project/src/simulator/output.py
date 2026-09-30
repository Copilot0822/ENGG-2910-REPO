"""Write inference inputs, evaluation data, and reproducibility metadata."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from simulator import __version__
from simulator.models import CanboatDatabase, FieldDefinition, SimulationResult, SupportStatus


def _number(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def _timestamp(value: Decimal) -> str:
    return f"{value:.2f}"


def _write_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _field_truth(definition: Any, field: FieldDefinition) -> dict[str, Any]:
    repeating = next(
        (
            {
                "start_field_order": group.start_order,
                "field_count": group.size,
                "count_field_order": group.count_field_order,
            }
            for group in definition.repeating_groups
            if group.start_order <= field.order < group.start_order + group.size
        ),
        None,
    )
    return {
        "signal_id": field.signal_id,
        "name": f"{definition.description} / {field.name}",
        "pgn": definition.pgn,
        "pgn_variant": definition.canboat_id,
        "variant_id": definition.variant_id,
        "pgn_description": definition.description,
        "packet_type": definition.packet_type,
        "field_id": field.field_id,
        "field_name": field.name,
        "start_bit": field.bit_offset,
        "bit_length": field.bit_length,
        "signed": field.signed,
        "scale": _number(field.resolution),
        "offset": _number(field.offset),
        "unit": field.unit,
        "field_type": field.field_type,
        "match_conditions": definition.match_conditions,
        "repeating_field": repeating,
        "is_linear_target": field.is_linear_target,
    }


def write_outputs(
    output_directory: str | Path,
    database: CanboatDatabase,
    result: SimulationResult,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    _write_csv(
        output / "can_frames.csv",
        ["sample_id", "timestamp", "message_id", "frame_index", "can_id", "data_hex"],
        (
            [
                frame.sample_id,
                _timestamp(frame.timestamp),
                frame.message_id,
                frame.frame_index,
                f"0x{frame.can_id:08X}",
                " ".join(f"{byte:02X}" for byte in frame.data),
            ]
            for frame in result.frames
        ),
    )
    _write_csv(
        output / "known_values.csv",
        ["sample_id", "timestamp", "observation_id", "signal_name", "value", "unit"],
        (
            [
                item.sample_id,
                _timestamp(item.timestamp),
                item.observation_id,
                item.signal_name,
                str(item.value),
                item.unit,
            ]
            for item in result.observations
        ),
    )

    signals = {
        field.signal_id: _field_truth(definition, field)
        for definition in database.definitions
        for field in definition.eligible_fields
    }
    (output / "ground_truth.json").write_text(
        json.dumps({"signals": signals}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_csv(
        output / "signal_catalog.csv",
        [
            "signal_id",
            "pgn",
            "pgn_description",
            "variant",
            "field_name",
            "field_type",
            "start_bit",
            "bit_length",
            "signed",
            "scale",
            "offset",
            "unit",
            "packet_type",
            "is_linear_target",
            "support_status",
        ],
        (
            [
                field.signal_id,
                definition.pgn,
                definition.description,
                definition.canboat_id,
                field.name,
                field.field_type,
                "" if field.bit_offset is None else field.bit_offset,
                "" if field.bit_length is None else field.bit_length,
                str(field.signed).lower(),
                str(field.resolution),
                str(field.offset),
                field.unit,
                definition.packet_type,
                str(field.is_linear_target).lower(),
                definition.status.value,
            ]
            for definition in database.definitions
            for field in definition.fields
        ),
    )

    unsupported = [
        {
            "pgn": definition.pgn,
            "definition": definition.canboat_id,
            "variant_id": definition.variant_id,
            "status": definition.status.value,
            "reason": definition.support_reason,
        }
        for definition in database.definitions
        if definition.status != SupportStatus.SUPPORTED
    ]
    (output / "unsupported_pgns.json").write_text(
        json.dumps({"definitions": unsupported}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status_counts = {
        status.value: sum(item.status == status for item in database.definitions)
        for status in SupportStatus
    }
    metadata = {
        "simulator_version": __version__,
        "seed": result.options["seed"],
        "canboat_version": database.version,
        "canboat_schema_version": database.schema_version,
        "canboat_sha256": database.sha256,
        "command_line_options": result.options,
        "generation_mode": result.options["mode"],
        "number_of_time_steps": result.time_step_count,
        "number_of_logical_messages": result.logical_message_count,
        "number_of_physical_can_frames": len(result.frames),
        "number_of_pgns_found": len({item.pgn for item in database.definitions}),
        "number_of_variants_found": len(database.definitions),
        "number_supported": status_counts[SupportStatus.SUPPORTED.value],
        "number_partially_supported": status_counts[SupportStatus.PARTIAL.value],
        "number_skipped": status_counts[SupportStatus.SKIPPED.value],
        "number_of_eligible_linear_target_signals": database.eligible_signal_count,
        "total_target_observations": len(result.observations),
    }
    (output / "simulation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata

