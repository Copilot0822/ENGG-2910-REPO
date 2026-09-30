import random

from simulator.bit_encoder import get_bits
from simulator.message_generator import MessageGenerator
from simulator.models import SupportStatus


def test_variant_match_fields_are_encoded(database):
    candidates = [
        definition
        for definition in database.definitions
        if definition.match_conditions and definition.status != SupportStatus.SKIPPED
    ]
    assert len(candidates) > 1
    generator = MessageGenerator(database, random.Random(123))
    for definition in candidates[:25]:
        message = generator.generate(definition)
        for field in definition.fields:
            if field.match is None:
                continue
            actual = get_bits(message.payload, field.bit_offset, field.bit_length)
            assert actual == field.match


def test_indirect_lookup_fields_use_defined_pairs(database):
    definition = next(
        item for item in database.generatable_definitions if item.canboat_id == "isoAddressClaim"
    )
    message = MessageGenerator(database, random.Random(77)).generate(definition)
    indirect = next(field for field in definition.fields if field.field_type == "INDIRECT_LOOKUP")
    selector_field = next(
        field for field in definition.fields if field.order == indirect.indirect_field_order
    )
    selector = get_bits(message.payload, selector_field.bit_offset, selector_field.bit_length)
    value = get_bits(message.payload, indirect.bit_offset, indirect.bit_length)
    legal = {
        (item[0], item[1])
        for item in database.indirect_lookups[indirect.indirect_lookup_name]
    }
    assert (selector, value) in legal
