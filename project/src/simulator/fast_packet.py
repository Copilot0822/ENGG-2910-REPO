"""NMEA 2000 Fast Packet fragmentation and reassembly."""

from __future__ import annotations


MAX_FAST_PACKET_PAYLOAD = 223


def fragment_fast_packet(payload: bytes, sequence_id: int) -> list[bytes]:
    if not 1 <= len(payload) <= MAX_FAST_PACKET_PAYLOAD:
        raise ValueError("Fast Packet payload length must be between 1 and 223 bytes")
    if not 0 <= sequence_id <= 7:
        raise ValueError("Fast Packet sequence ID must fit in 3 bits")
    frames: list[bytes] = []
    first = bytearray([0xFF] * 8)
    first[0] = sequence_id << 5
    first[1] = len(payload)
    first_chunk = payload[:6]
    first[2 : 2 + len(first_chunk)] = first_chunk
    frames.append(bytes(first))
    position = len(first_chunk)
    counter = 1
    while position < len(payload):
        if counter > 31:
            raise ValueError("Fast Packet frame counter overflow")
        frame = bytearray([0xFF] * 8)
        frame[0] = (sequence_id << 5) | counter
        chunk = payload[position : position + 7]
        frame[1 : 1 + len(chunk)] = chunk
        frames.append(bytes(frame))
        position += len(chunk)
        counter += 1
    return frames


def reassemble_fast_packet(frames: list[bytes]) -> bytes:
    if not frames:
        raise ValueError("at least one frame is required")
    if any(len(frame) != 8 for frame in frames):
        raise ValueError("Fast Packet physical frames must contain exactly 8 bytes")
    sequence_id = frames[0][0] >> 5
    if frames[0][0] & 0x1F:
        raise ValueError("first Fast Packet counter must be zero")
    length = frames[0][1]
    if not 1 <= length <= MAX_FAST_PACKET_PAYLOAD:
        raise ValueError("invalid declared Fast Packet length")
    result = bytearray(frames[0][2:8])
    for expected_counter, frame in enumerate(frames[1:], start=1):
        if frame[0] >> 5 != sequence_id or frame[0] & 0x1F != expected_counter:
            raise ValueError("Fast Packet sequence/counter discontinuity")
        result.extend(frame[1:8])
    required_frames = 1 + max(0, (length - 6 + 6) // 7)
    if len(frames) != required_frames:
        raise ValueError("incorrect number of Fast Packet frames")
    return bytes(result[:length])
