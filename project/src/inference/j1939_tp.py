"""Metadata-free J1939 TP.CM/TP.DT reassembly for discovery inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from inference.can_id import parse_can_id


TP_CM_PGN = 0xEC00
TP_DT_PGN = 0xEB00
CM_RTS = 16
CM_BAM = 32


class FrameLike(Protocol):
    can_id: int
    data: bytes


@dataclass(frozen=True, slots=True)
class ReassembledJ1939TP:
    payload: bytes
    pgn: int
    source: int
    destination: int | None


def is_transport_group(frames: list[FrameLike]) -> bool:
    has_start = any(
        parse_can_id(frame.can_id).pgn == TP_CM_PGN
        and len(frame.data) == 8
        and frame.data[0] in {CM_RTS, CM_BAM}
        for frame in frames
    )
    has_data = any(parse_can_id(frame.can_id).pgn == TP_DT_PGN for frame in frames)
    return has_start and has_data


def reassemble_j1939_tp(frames: list[FrameLike]) -> ReassembledJ1939TP:
    starts = [
        frame
        for frame in frames
        if parse_can_id(frame.can_id).pgn == TP_CM_PGN
        and len(frame.data) == 8
        and frame.data[0] in {CM_RTS, CM_BAM}
    ]
    if len(starts) != 1:
        raise ValueError("J1939 TP group must contain exactly one BAM/RTS start")
    start = starts[0]
    parsed_start = parse_can_id(start.can_id)
    size = int.from_bytes(start.data[1:3], "little")
    packet_count = start.data[3]
    target_pgn = int.from_bytes(start.data[5:8], "little")
    dt = [
        frame.data
        for frame in frames
        if parse_can_id(frame.can_id).pgn == TP_DT_PGN
        and parse_can_id(frame.can_id).source == parsed_start.source
    ]
    dt.sort(key=lambda payload: payload[0])
    if len(dt) != packet_count:
        raise ValueError("J1939 TP packet count does not match TP.CM")
    if [payload[0] for payload in dt] != list(range(1, packet_count + 1)):
        raise ValueError("J1939 TP.DT sequence is incomplete")
    payload = b"".join(frame[1:8] for frame in dt)[:size]
    if len(payload) != size:
        raise ValueError("J1939 TP payload is shorter than declared")
    broadcast = start.data[0] == CM_BAM
    return ReassembledJ1939TP(
        payload=payload,
        pgn=target_pgn,
        source=parsed_start.source,
        destination=None if broadcast else parsed_start.destination,
    )

