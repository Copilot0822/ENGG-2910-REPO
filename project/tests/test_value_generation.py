import random

from simulator.value_generator import ValueGenerator, choose_valid_raw, physical_from_raw


def _field(database, pgn, field_id):
    definition = next(item for item in database.definitions if item.pgn == pgn)
    return next(item for item in definition.fields if item.field_id == field_id)


def test_linear_generation_obeys_scale_offset_and_avoids_sentinels(database):
    field = _field(database, 127488, "speed")
    rng = random.Random(42)
    values = [choose_valid_raw(field, rng) for _ in range(1000)]
    assert not set(values) & field.sentinels
    assert all(physical_from_raw(field, raw) == field.resolution * raw + field.offset for raw in values)


def test_signed_and_unsigned_values_encode_correctly(database):
    signed = _field(database, 127488, "tiltTrim")
    unsigned = _field(database, 127488, "speed")
    generator = ValueGenerator(database, random.Random(8))
    signed_buffer = bytearray(1)
    unsigned_buffer = bytearray(2)
    signed_value = generator.encode(signed_buffer, signed, 0, forced_raw=-17)
    unsigned_value = generator.encode(unsigned_buffer, unsigned, 0, forced_raw=1234)
    assert signed_value is not None and signed_value.raw_value == -17
    assert unsigned_value is not None and unsigned_value.raw_value == 1234
    assert signed_buffer == bytes([0xEF])
    assert unsigned_buffer == bytes([0xD2, 0x04])

