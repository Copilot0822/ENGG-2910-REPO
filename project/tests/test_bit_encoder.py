from simulator.bit_encoder import get_bits, set_bits


def test_non_byte_aligned_field_round_trip_without_touching_neighbors():
    data = bytearray([0xA5, 0x5A, 0xC3])
    before_low = get_bits(data, 0, 3)
    before_high = get_bits(data, 16, 8)
    set_bits(data, 3, 13, 0x137A)
    assert get_bits(data, 3, 13) == 0x137A
    assert get_bits(data, 0, 3) == before_low
    assert get_bits(data, 16, 8) == before_high


def test_multibyte_unsigned_field():
    data = bytearray(5)
    set_bits(data, 7, 24, 0xABCDEF)
    assert get_bits(data, 7, 24) == 0xABCDEF


def test_signed_twos_complement_field():
    data = bytearray(3)
    set_bits(data, 5, 12, -321)
    assert get_bits(data, 5, 12, signed=True) == -321
    assert get_bits(data, 5, 12, signed=False) == (1 << 12) - 321

