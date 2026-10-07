"""Small invented definitions for public tests; no licensed database rows."""

import json
from dataclasses import asdict
from pathlib import Path

from simulator.j1939.identifier import build_j1939_identifier, parse_j1939_identifier


def _signal(index, start, width, *, signed=False, factor=0.5, offset=-7,
            byte_order="little_endian", raw_limits=None):
    lower, upper = raw_limits or (
        (-(1 << (width - 1)), (1 << (width - 1)) - 1)
        if signed else (0, (1 << width) - 1)
    )
    endpoints = (lower * factor + offset, upper * factor + offset)
    return {
        "name": f"Synthetic field {index}",
        "spn": 900000 + index,
        "start_bit": start,
        "bit_length": width,
        "byte_order": byte_order,
        "signed": signed,
        "factor": factor,
        "offset": offset,
        "minimum": min(endpoints),
        "maximum": max(endpoints),
        "raw_min": lower,
        "raw_max": upper,
        "unit": "synthetic_units",
        "comment": "Invented for this project's tests.",
        "is_linear_numeric": True,
        "aligned_8_16_32_candidate": start % 8 == 0 and width in (8, 16, 32),
    }


def write_synthetic_database(path: Path) -> Path:
    definitions = (
        (0xFF10, 8, [
            _signal(1, 0, 8),
            _signal(2, 8, 16, factor=0.25, offset=11),
            _signal(3, 24, 32, signed=True, factor=-2, raw_limits=(-4096, 4095)),
        ]),
        (0xFF11, 8, [
            _signal(4, 3, 5, factor=1.5, offset=4),
            _signal(5, 16, 12, signed=True, offset=3),
            _signal(6, 32, 16, byte_order="big_endian", factor=0.125, offset=-2),
        ]),
        (0xFF12, 12, [
            _signal(7, 0, 16, signed=True),
            _signal(8, 24, 40, raw_limits=(0, 65535)),
        ]),
        (0xEF00, 12, [
            _signal(9, 0, 16),
            _signal(10, 32, 64, raw_limits=(0, 32767)),
        ]),
    )
    messages = []
    for index, (pgn, length, signals) in enumerate(definitions, start=1):
        can_id = build_j1939_identifier(6, pgn, 0x80, destination=0x17)
        identifier = asdict(parse_j1939_identifier(can_id))
        identifier["source_address"] = identifier.pop("source")
        identifier["destination_address"] = identifier.pop("destination")
        messages.append({
            "name": f"Synthetic message {index}",
            "comment": "Invented test message, not a standard signal catalog.",
            "dbc_id": can_id | (1 << 31),
            "can_id": can_id,
            "length_bytes": length,
            "j1939": identifier,
            "signals": signals,
        })
    document = {
        "metadata": {
            "database_version": "synthetic-test-v1",
            "schema_version": "1.0",
            "message_count": 4,
            "unique_pgn_count": 4,
            "signal_count": 10,
            "aligned_8_16_32_signal_count": 6,
        },
        "messages": messages,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
