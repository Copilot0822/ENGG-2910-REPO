"""Fast Packet recognition and reassembly using only physical frame bytes."""

from __future__ import annotations


def looks_like_single_frame_fast_packet(frame: bytes) -> bool:
    """Conservatively recognize a one-frame Fast Packet payload."""

    if len(frame) != 8 or frame[0] & 0x1F:
        return False
    declared_length = frame[1]
    if not 1 <= declared_length <= 6:
        return False
    return all(byte == 0xFF for byte in frame[2 + declared_length :])


def reassemble_physical_frames(frames: list[bytes]) -> bytes:
    if not frames:
        raise ValueError("at least one Fast Packet frame is required")
    if any(len(frame) != 8 for frame in frames):
        raise ValueError("Fast Packet frames must have an eight-byte payload")
    sequence = frames[0][0] >> 5
    if frames[0][0] & 0x1F:
        raise ValueError("first Fast Packet frame counter is not zero")
    logical_length = frames[0][1]
    if not 1 <= logical_length <= 223:
        raise ValueError("invalid Fast Packet logical length")
    payload = bytearray(frames[0][2:8])
    for expected, frame in enumerate(frames[1:], start=1):
        if frame[0] >> 5 != sequence:
            raise ValueError("Fast Packet sequence ID changed")
        if frame[0] & 0x1F != expected:
            raise ValueError("Fast Packet frame counter is discontinuous")
        payload.extend(frame[1:8])
    required = 1 + max(0, (logical_length - 6 + 6) // 7)
    if len(frames) != required:
        raise ValueError(
            f"Fast Packet declares {logical_length} bytes but has {len(frames)} frames"
        )
    return bytes(payload[:logical_length])

