"""Load the normalized CSS Electronics J1939 JSON conversion."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from simulator.j1939.identifier import parse_j1939_identifier
from simulator.j1939.models import J1939Database, J1939Message, J1939Signal


DEFAULT_J1939_DATABASE = (
    Path(__file__).resolve().parents[3]
    / "vendor" / "j1939" / "J1939_2024-06_v2.4.json"
)


def resolve_j1939_database_path(path: str | Path | None = None) -> Path:
    """Select an explicit path, the local environment setting, or the default."""
    configured = path if path is not None else os.environ.get("J1939_DATABASE_PATH")
    return Path(configured or DEFAULT_J1939_DATABASE).expanduser().resolve()


def _decimal(value: object) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def load_j1939_database(path: str | Path | None = None) -> J1939Database:
    source = resolve_j1939_database_path(path)
    if not source.is_file():
        raise FileNotFoundError(
            "Licensed J1939 database not found. This repository does not "
            "redistribute the CSS Electronics J1939 database. Provide your "
            "licensed local copy with --database or J1939_DATABASE_PATH. "
            f"Selected path: {source}"
        )
    raw_bytes = source.read_bytes()
    document = json.loads(raw_bytes, parse_float=Decimal)
    metadata = document["metadata"]

    messages: list[J1939Message] = []
    signal_counter = 0
    aligned_count = 0
    for message_index, raw_message in enumerate(document["messages"], start=1):
        parsed = parse_j1939_identifier(int(raw_message["can_id"]))
        declared = raw_message["j1939"]
        if parsed.pgn != int(declared["pgn"]):
            raise ValueError(f"PGN mismatch in message {raw_message['name']}")
        signals: list[J1939Signal] = []
        for raw_signal in raw_message["signals"]:
            signal_counter += 1
            signal = J1939Signal(
                signal_id=f"j1939_signal_{signal_counter:06d}",
                name=str(raw_signal["name"]),
                spn=int(raw_signal["spn"]),
                start_bit=int(raw_signal["start_bit"]),
                bit_length=int(raw_signal["bit_length"]),
                byte_order=str(raw_signal["byte_order"]),
                signed=bool(raw_signal["signed"]),
                factor=_decimal(raw_signal["factor"]),
                offset=_decimal(raw_signal["offset"]),
                minimum=_decimal(raw_signal["minimum"]),
                maximum=_decimal(raw_signal["maximum"]),
                raw_min=int(raw_signal["raw_min"]),
                raw_max=int(raw_signal["raw_max"]),
                unit=str(raw_signal["unit"]),
                comment=str(raw_signal.get("comment") or ""),
                is_linear=bool(raw_signal.get("is_linear_numeric", True)),
                aligned_8_16_32=bool(
                    raw_signal.get("aligned_8_16_32_candidate", False)
                ),
            )
            if signal.start_bit + signal.bit_length > int(raw_message["length_bytes"]) * 8:
                raise ValueError(f"signal {signal.name} extends beyond its message")
            aligned_count += int(signal.aligned_8_16_32)
            signals.append(signal)
        messages.append(
            J1939Message(
                definition_id=f"j1939_message_{message_index:04d}",
                dbc_id=int(raw_message["dbc_id"]),
                template_can_id=int(raw_message["can_id"]),
                name=str(raw_message["name"]),
                comment=str(raw_message.get("comment") or ""),
                length_bytes=int(raw_message["length_bytes"]),
                priority=int(declared["priority"]),
                reserved_bit=int(declared["reserved_bit"]),
                data_page=int(declared["data_page"]),
                pdu_format=int(declared["pdu_format"]),
                pdu_specific=int(declared["pdu_specific"]),
                pdu_type=str(declared["pdu_type"]),
                pgn=int(declared["pgn"]),
                template_source=int(declared["source_address"]),
                template_destination=(
                    None
                    if declared["destination_address"] is None
                    else int(declared["destination_address"])
                ),
                signals=tuple(signals),
            )
        )

    if len(messages) != int(metadata["message_count"]):
        raise ValueError("metadata message_count does not match messages array")
    if signal_counter != int(metadata["signal_count"]):
        raise ValueError("metadata signal_count does not match loaded signals")
    if aligned_count != int(metadata["aligned_8_16_32_signal_count"]):
        raise ValueError("metadata aligned signal count does not match loaded signals")
    if len({message.pgn for message in messages}) != int(metadata["unique_pgn_count"]):
        raise ValueError("metadata unique PGN count does not match loaded messages")
    return J1939Database(
        path=source,
        database_version=str(metadata["database_version"]),
        schema_version=str(metadata["schema_version"]),
        messages=tuple(messages),
        signal_count=signal_counter,
        aligned_signal_count=aligned_count,
    )
