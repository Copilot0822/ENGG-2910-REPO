"""Build and parse SAE J1939 29-bit identifiers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class J1939Identifier:
    priority: int
    reserved_bit: int
    data_page: int
    pdu_format: int
    pdu_specific: int
    source: int
    destination: int | None
    pgn: int
    pdu_type: str


def parse_j1939_identifier(can_id: int) -> J1939Identifier:
    if not 0 <= can_id <= 0x1FFFFFFF:
        raise ValueError("J1939 identifiers must fit in 29 bits")
    priority = (can_id >> 26) & 0x07
    reserved = (can_id >> 25) & 0x01
    data_page = (can_id >> 24) & 0x01
    pf = (can_id >> 16) & 0xFF
    ps = (can_id >> 8) & 0xFF
    source = can_id & 0xFF
    page = (reserved << 17) | (data_page << 16)
    if pf < 240:
        pgn = page | (pf << 8)
        destination: int | None = ps
        pdu_type = "PDU1"
    else:
        pgn = page | (pf << 8) | ps
        destination = None
        pdu_type = "PDU2"
    return J1939Identifier(
        priority, reserved, data_page, pf, ps, source, destination, pgn, pdu_type
    )


def build_j1939_identifier(
    priority: int,
    pgn: int,
    source: int,
    destination: int = 0xFF,
) -> int:
    if not 0 <= priority <= 7 or not 0 <= pgn <= 0x3FFFF:
        raise ValueError("priority or PGN is out of range")
    if not 0 <= source <= 0xFF or not 0 <= destination <= 0xFF:
        raise ValueError("J1939 addresses must fit in one byte")
    reserved = (pgn >> 17) & 0x01
    data_page = (pgn >> 16) & 0x01
    pf = (pgn >> 8) & 0xFF
    group_extension = pgn & 0xFF
    if pf < 240:
        if group_extension:
            raise ValueError("PDU1 PGN group extension must be zero")
        ps = destination
    else:
        ps = group_extension
    return (
        (priority << 26)
        | (reserved << 25)
        | (data_page << 24)
        | (pf << 16)
        | (ps << 8)
        | source
    )

