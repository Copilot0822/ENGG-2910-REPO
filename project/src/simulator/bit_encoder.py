"""NMEA 2000 little-endian bit-field primitives."""

from __future__ import annotations


def set_bits(buffer: bytearray, bit_offset: int, bit_length: int, value: int) -> None:
    """Set an arbitrary little-endian bit field in *buffer*."""

    if bit_offset < 0 or bit_length <= 0:
        raise ValueError("bit offset must be non-negative and length must be positive")
    if bit_offset + bit_length > len(buffer) * 8:
        raise ValueError("field does not fit in destination buffer")
    mask = (1 << bit_length) - 1
    encoded = value & mask
    for bit in range(bit_length):
        target = bit_offset + bit
        byte_index, bit_index = divmod(target, 8)
        bit_mask = 1 << bit_index
        if encoded & (1 << bit):
            buffer[byte_index] |= bit_mask
        else:
            buffer[byte_index] &= ~bit_mask


def get_bits(
    data: bytes | bytearray,
    bit_offset: int,
    bit_length: int,
    *,
    signed: bool = False,
) -> int:
    """Read an arbitrary little-endian bit field, optionally as two's complement."""

    if bit_offset < 0 or bit_length <= 0:
        raise ValueError("bit offset must be non-negative and length must be positive")
    if bit_offset + bit_length > len(data) * 8:
        raise ValueError("field extends past input data")
    value = 0
    for bit in range(bit_length):
        target = bit_offset + bit
        byte_index, bit_index = divmod(target, 8)
        if data[byte_index] & (1 << bit_index):
            value |= 1 << bit
    if signed and value & (1 << (bit_length - 1)):
        value -= 1 << bit_length
    return value

