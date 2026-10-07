"""Write J1939 inference inputs, answer keys, catalog, and metadata."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from inference.output_privacy import protect_private_outputs
from simulator import __version__
from simulator.j1939.models import J1939Database, J1939SimulationResult, TargetCategory


def _number(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def _write_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_j1939_outputs(
    output_directory: str | Path,
    database: J1939Database,
    result: J1939SimulationResult,
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    protect_private_outputs(output)
    category = str(result.options["category"])
    include_transport = bool(result.options["include_transport"])
    eligible_messages = [
        message
        for message in database.messages
        if (include_transport or not message.requires_transport)
        and message.target_signals(category)  # type: ignore[arg-type]
    ]
    eligible_signals = [
        signal
        for message in eligible_messages
        for signal in message.target_signals(category)  # type: ignore[arg-type]
    ]
    eligible_ids = {signal.signal_id for signal in eligible_signals}

    _write_csv(
        output / "can_frames.csv",
        ["sample_id", "timestamp", "message_id", "frame_index", "can_id", "data_hex"],
        (
            [
                frame.sample_id,
                f"{frame.timestamp:.2f}",
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
                f"{item.timestamp:.2f}",
                item.observation_id,
                f"signal_{item.signal_id.rsplit('_', 1)[-1]} [{item.signal_id}]",
                str(item.value),
                "",
            ]
            for item in result.observations
        ),
    )
    truths: dict[str, dict[str, object]] = {}
    for message in database.messages:
        for signal in message.signals:
            if signal.signal_id not in eligible_ids:
                continue
            truths[signal.signal_id] = {
                "signal_id": signal.signal_id,
                "spn": signal.spn,
                "name": signal.name,
                "message_name": message.name,
                "pgn": message.pgn,
                "start_bit": signal.start_bit,
                "bit_length": signal.bit_length,
                "byte_order": signal.byte_order,
                "signed": signal.signed,
                "scale": _number(signal.factor),
                "factor": _number(signal.factor),
                "offset": _number(signal.offset),
                "minimum": _number(signal.minimum),
                "maximum": _number(signal.maximum),
                "raw_min": signal.raw_min,
                "raw_max": signal.raw_max,
                "unit": signal.unit,
                "logical_message_length": message.length_bytes,
                "transport_category": (
                    "single_frame"
                    if not message.requires_transport
                    else "RTS_CTS"
                    if message.pdu_type == "PDU1"
                    else "BAM"
                ),
                "source_database": database.path.name,
                "is_linear_target": signal.is_linear,
                "aligned_8_16_32": signal.aligned_8_16_32,
            }
    (output / "ground_truth.json").write_text(
        json.dumps({"signals": truths}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / "signal_catalog.csv",
        [
            "signal_id",
            "spn",
            "pgn",
            "message_name",
            "signal_name",
            "start_bit",
            "bit_length",
            "byte_order",
            "signed",
            "factor",
            "offset",
            "minimum",
            "maximum",
            "unit",
            "aligned_8_16_32",
            "message_length",
        ],
        (
            [
                signal.signal_id,
                signal.spn,
                message.pgn,
                message.name,
                signal.name,
                signal.start_bit,
                signal.bit_length,
                signal.byte_order,
                str(signal.signed).lower(),
                str(signal.factor),
                str(signal.offset),
                str(signal.minimum),
                str(signal.maximum),
                signal.unit,
                str(signal.aligned_8_16_32).lower(),
                message.length_bytes,
            ]
            for message in database.messages
            for signal in message.signals
        ),
    )
    metadata: dict[str, object] = {
        "simulator_version": __version__,
        "protocol": "J1939",
        "database_version": database.database_version,
        "schema_version": database.schema_version,
        "output_schema_version": 2,
        "number_of_messages_loaded": len(database.messages),
        "number_of_pgns_loaded": len({message.pgn for message in database.messages}),
        "number_of_signals_loaded": database.signal_count,
        "number_of_all_linear_targets": database.signal_count,
        "number_of_aligned_8_16_32_targets": database.aligned_signal_count,
        "number_of_eligible_targets_in_dataset": len(eligible_signals),
        "number_of_single_frame_messages": len(database.single_frame_messages),
        "number_of_transport_messages": len(database.transport_messages),
        "seed": result.options["seed"],
        "generation_mode": result.options["mode"],
        "target_category": category,
        "transport_included": include_transport,
        "number_of_samples": result.time_step_count,
        "number_of_logical_messages": result.logical_message_count,
        "number_of_physical_can_frames": len(result.frames),
        "total_target_observations": len(result.observations),
        "command_line_options": {
            key: value for key, value in result.options.items()
            if key not in {"database", "output"}
        },
    }
    (output / "simulation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
