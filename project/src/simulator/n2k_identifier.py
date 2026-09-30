"""Construction and parsing of 29-bit NMEA 2000/J1939 CAN identifiers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedIdentifier:
    priority: int
    pgn: int
    source: int
    destination: int | None
    is_pdu1: bool


def build_can_id(priority: int, pgn: int, source: int, destination: int = 0xFF) -> int:
    if not 0 <= priority <= 7:
        raise ValueError("priority must fit in 3 bits")
    if not 0 <= pgn <= 0x3FFFF:
        raise ValueError("PGN must fit in 18 bits")
    if not 0 <= source <= 0xFF or not 0 <= destination <= 0xFF:
        raise ValueError("addresses must fit in one byte")
    reserved_and_page = (pgn >> 16) & 0x03
    pdu_format = (pgn >> 8) & 0xFF
    group_extension = pgn & 0xFF
    if pdu_format < 240:
        if group_extension != 0:
            raise ValueError("PDU1 PGNs must have a zero group-extension byte")
        pdu_specific = destination
    else:
        pdu_specific = group_extension
    return (
        (priority << 26)
        | (reserved_and_page << 24)
        | (pdu_format << 16)
        | (pdu_specific << 8)
        | source
    )


def parse_can_id(can_id: int) -> ParsedIdentifier:
    if not 0 <= can_id <= 0x1FFFFFFF:
        raise ValueError("CAN identifier is not a 29-bit extended identifier")
    priority = (can_id >> 26) & 0x07
    reserved_and_page = (can_id >> 24) & 0x03
    pdu_format = (can_id >> 16) & 0xFF
    pdu_specific = (can_id >> 8) & 0xFF
    source = can_id & 0xFF
    if pdu_format < 240:
        pgn = (reserved_and_page << 16) | (pdu_format << 8)
        destination: int | None = pdu_specific
        is_pdu1 = True
    else:
        pgn = (reserved_and_page << 16) | (pdu_format << 8) | pdu_specific
        destination = None
        is_pdu1 = False
    return ParsedIdentifier(priority, pgn, source, destination, is_pdu1)

