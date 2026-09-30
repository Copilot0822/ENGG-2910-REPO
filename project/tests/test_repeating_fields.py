import random

from simulator.bit_encoder import get_bits
from simulator.message_generator import MessageGenerator, field_occurrences
from simulator.models import SupportStatus


def test_supported_repeating_group_offsets_and_count(database):
    definition = next(
        item
        for item in database.definitions
        if item.canboat_id == "gnssSatsInView" and item.status != SupportStatus.SKIPPED
    )
    occurrences = field_occurrences(definition, 3)
    prn_offsets = [
        item.bit_offset for item in occurrences if item.field.field_id == "prn"
    ]
    assert len(prn_offsets) == 3
    assert prn_offsets[1] - prn_offsets[0] == prn_offsets[2] - prn_offsets[1]

    message = MessageGenerator(database, random.Random(9)).generate(definition)
    count_field = next(item for item in definition.fields if item.field_id == "satsInView")
    count = get_bits(message.payload, count_field.bit_offset, count_field.bit_length)
    assert 1 <= count <= 3
    expected_bytes = 3 + count * 12
    assert len(message.payload) == expected_bytes

