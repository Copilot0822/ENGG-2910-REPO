"""Load the normalized CSS Electronics J1939 JSON conversion."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from simulator.j1939.identifier import parse_j1939_identifier
from simulator.j1939.models import J1939Database, J1939Message, J1939Signal


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _decimal(value: object) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _verify_sidecar(path: Path, actual_hash: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise ValueError(f"missing pinned JSON checksum sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
    if not _SHA256.fullmatch(expected) or expected != actual_hash:
        raise ValueError(
            f"J1939 JSON SHA-256 mismatch: expected {expected}, calculated {actual_hash}"
        )


def load_j1939_database(path: str | Path) -> J1939Database:
    source = Path(path).resolve()
    raw_bytes = source.read_bytes()
    json_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    _verify_sidecar(source, json_sha256)
    document = json.loads(raw_bytes, parse_float=Decimal)
    metadata = document["metadata"]
    source_hash = str(metadata["source_sha256"]).lower()
    if not _SHA256.fullmatch(source_hash):
        raise ValueError("metadata source_sha256 is not a valid SHA-256 digest")

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
        json_sha256=json_sha256,
        source_dbc_sha256=source_hash,
        database_version=str(metadata["database_version"]),
        schema_version=str(metadata["schema_version"]),
        messages=tuple(messages),
        signal_count=signal_counter,
        aligned_signal_count=aligned_count,
    )

