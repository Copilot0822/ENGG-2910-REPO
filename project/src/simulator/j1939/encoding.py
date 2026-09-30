"""DBC signal bit packing and decoding."""

from __future__ import annotations

from simulator.j1939.models import J1939Signal


def _motorola_positions(start_bit: int, bit_length: int) -> list[int]:
    positions: list[int] = []
    position = start_bit
    for _ in range(bit_length):
        positions.append(position)
        position = position + 15 if position % 8 == 0 else position - 1
    return positions


def encode_signal(buffer: bytearray, signal: J1939Signal, raw_value: int) -> None:
    width = signal.bit_length
    encoded = raw_value & ((1 << width) - 1)
    if signal.byte_order == "little_endian":
        positions = [signal.start_bit + bit for bit in range(width)]
        value_bits = range(width)
    elif signal.byte_order == "big_endian":
        positions = _motorola_positions(signal.start_bit, width)
        value_bits = range(width - 1, -1, -1)
    else:
        raise ValueError(f"unsupported DBC byte order {signal.byte_order}")
    for position, value_bit in zip(positions, value_bits, strict=True):
        if not 0 <= position < len(buffer) * 8:
            raise ValueError(f"signal {signal.name} extends beyond payload")
        mask = 1 << (position % 8)
        if encoded & (1 << value_bit):
            buffer[position // 8] |= mask
        else:
            buffer[position // 8] &= ~mask


def decode_signal(payload: bytes, signal: J1939Signal) -> int:
    width = signal.bit_length
    raw = 0
    if signal.byte_order == "little_endian":
        positions = [signal.start_bit + bit for bit in range(width)]
        value_bits = range(width)
    elif signal.byte_order == "big_endian":
        positions = _motorola_positions(signal.start_bit, width)
        value_bits = range(width - 1, -1, -1)
    else:
        raise ValueError(f"unsupported DBC byte order {signal.byte_order}")
    for position, value_bit in zip(positions, value_bits, strict=True):
        if payload[position // 8] & (1 << (position % 8)):
            raw |= 1 << value_bit
    if signal.signed and raw & (1 << (width - 1)):
        raw -= 1 << width
    return raw

