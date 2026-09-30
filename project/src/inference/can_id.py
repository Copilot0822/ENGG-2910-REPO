"""Independent 29-bit NMEA 2000/J1939 identifier parser."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedIdentifier:
    priority: int
    pgn: int
    source: int
    destination: int | None
    is_pdu1: bool


def parse_can_id(can_id: int) -> ParsedIdentifier:
    if not 0 <= can_id <= 0x1FFFFFFF:
        raise ValueError(f"0x{can_id:X} is not a 29-bit CAN identifier")
    priority = (can_id >> 26) & 0x07
    page = (can_id >> 24) & 0x03
    pdu_format = (can_id >> 16) & 0xFF
    pdu_specific = (can_id >> 8) & 0xFF
    source = can_id & 0xFF
    if pdu_format < 240:
        return ParsedIdentifier(
            priority=priority,
            pgn=(page << 16) | (pdu_format << 8),
            source=source,
            destination=pdu_specific,
            is_pdu1=True,
        )
    return ParsedIdentifier(
        priority=priority,
        pgn=(page << 16) | (pdu_format << 8) | pdu_specific,
        source=source,
        destination=None,
        is_pdu1=False,
    )

