"""J1939-21 TP.CM/TP.DT BAM and deterministic RTS/CTS transport."""

from __future__ import annotations

from dataclasses import dataclass

from simulator.j1939.identifier import build_j1939_identifier, parse_j1939_identifier


TP_CM_PGN = 0xEC00
TP_DT_PGN = 0xEB00
CM_RTS = 16
CM_CTS = 17
CM_END_ACK = 19
CM_BAM = 32


@dataclass(frozen=True, slots=True)
class TransportFrame:
    can_id: int
    data: bytes


def _cm_payload(control: int, size: int, packets: int, value4: int, pgn: int) -> bytes:
    return bytes(
        [control, size & 0xFF, size >> 8, packets, value4]
    ) + pgn.to_bytes(3, "little")


def fragment_transport(
    payload: bytes,
    target_pgn: int,
    source: int,
    destination: int = 0xFF,
) -> list[TransportFrame]:
    if not 9 <= len(payload) <= 1785:
        raise ValueError("J1939 TP payload must contain 9..1785 bytes")
    packets = (len(payload) + 6) // 7
    broadcast = destination == 0xFF
    frames: list[TransportFrame] = []
    if broadcast:
        frames.append(
            TransportFrame(
                build_j1939_identifier(7, TP_CM_PGN, source, 0xFF),
                _cm_payload(CM_BAM, len(payload), packets, 0xFF, target_pgn),
            )
        )
    else:
        frames.append(
            TransportFrame(
                build_j1939_identifier(7, TP_CM_PGN, source, destination),
                _cm_payload(CM_RTS, len(payload), packets, packets, target_pgn),
            )
        )
        cts = bytes([CM_CTS, packets, 1, 0xFF, 0xFF]) + target_pgn.to_bytes(3, "little")
        frames.append(
            TransportFrame(
                build_j1939_identifier(7, TP_CM_PGN, destination, source), cts
            )
        )
    dt_can_id = build_j1939_identifier(7, TP_DT_PGN, source, destination)
    for sequence in range(1, packets + 1):
        chunk = payload[(sequence - 1) * 7 : sequence * 7]
        frames.append(
            TransportFrame(dt_can_id, bytes([sequence]) + chunk.ljust(7, b"\xFF"))
        )
    if not broadcast:
        frames.append(
            TransportFrame(
                build_j1939_identifier(7, TP_CM_PGN, destination, source),
                _cm_payload(CM_END_ACK, len(payload), packets, 0xFF, target_pgn),
            )
        )
    return frames


@dataclass(frozen=True, slots=True)
class ReassembledTransport:
    payload: bytes
    target_pgn: int
    source: int
    destination: int | None
    broadcast: bool


def reassemble_transport(frames: list[TransportFrame]) -> ReassembledTransport:
    starts = []
    for frame in frames:
        parsed = parse_j1939_identifier(frame.can_id)
        if parsed.pgn == TP_CM_PGN and frame.data[0] in {CM_BAM, CM_RTS}:
            starts.append((frame, parsed))
    if len(starts) != 1:
        raise ValueError("transport group must contain exactly one BAM or RTS start")
    start, parsed_start = starts[0]
    control = start.data[0]
    size = int.from_bytes(start.data[1:3], "little")
    packet_count = start.data[3]
    target_pgn = int.from_bytes(start.data[5:8], "little")
    dt: list[bytes] = []
    for frame in frames:
        parsed = parse_j1939_identifier(frame.can_id)
        if parsed.pgn == TP_DT_PGN and parsed.source == parsed_start.source:
            dt.append(frame.data)
    dt.sort(key=lambda data: data[0])
    if len(dt) != packet_count or [data[0] for data in dt] != list(
        range(1, packet_count + 1)
    ):
        raise ValueError("TP.DT packet sequence is incomplete")
    payload = b"".join(data[1:8] for data in dt)[:size]
    broadcast = control == CM_BAM
    return ReassembledTransport(
        payload=payload,
        target_pgn=target_pgn,
        source=parsed_start.source,
        destination=None if broadcast else parsed_start.destination,
        broadcast=broadcast,
    )
