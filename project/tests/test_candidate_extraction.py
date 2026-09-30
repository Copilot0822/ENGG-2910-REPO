from inference.candidate import extract_raw
from inference.candidate import generate_specs


def test_aligned_8_16_and_32_bit_extraction():
    payload = bytes.fromhex("7F 34 12 EF CD AB 89")
    assert extract_raw(payload, 0, 8, signed=False, byte_order="little_endian") == 0x7F
    assert extract_raw(payload, 8, 16, signed=False, byte_order="little_endian") == 0x1234
    assert extract_raw(payload, 24, 32, signed=False, byte_order="little_endian") == 0x89ABCDEF
    assert extract_raw(payload, 8, 16, signed=False, byte_order="big_endian") == 0x3412


def test_signed_and_arbitrary_bit_extraction():
    payload = bytearray(4)
    raw = (1 << 12) - 321
    for bit in range(12):
        if raw & (1 << bit):
            absolute = 5 + bit
            payload[absolute // 8] |= 1 << (absolute % 8)
    assert extract_raw(bytes(payload), 5, 12, signed=True, byte_order="little_endian") == -321


def test_extended_non_byte_lengths_do_not_create_invalid_big_endian_candidates():
    specs = generate_specs(
        0xFEEE,
        1,
        8,
        "extended",
        (1, 5, 12, 16),
        ("little_endian", "big_endian"),
    )
    assert specs
    assert all(
        spec.byte_order == "little_endian" or spec.bit_length % 8 == 0
        for spec in specs
    )
