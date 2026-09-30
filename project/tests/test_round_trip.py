import random

from simulator.fast_packet import fragment_fast_packet, reassemble_fast_packet
from simulator.message_generator import MessageGenerator, decode_linear_fields
from simulator.models import SupportStatus


def test_ground_truth_decode_matches_generated_values(database):
    generator = MessageGenerator(database, random.Random(31415))
    candidates = [
        item
        for item in database.definitions
        if item.status != SupportStatus.SKIPPED and item.eligible_fields
    ]
    for definition in candidates[:100]:
        message = generator.generate(definition)
        decoded = decode_linear_fields(message)
        for value in message.values:
            if value.repetition_index is None:
                assert decoded[value.field.signal_id] == value.physical_value
        if definition.packet_type == "Fast":
            assert reassemble_fast_packet(fragment_fast_packet(message.payload, 7)) == message.payload


def test_every_non_skipped_variant_can_be_synthesized(database):
    generator = MessageGenerator(database, random.Random(2026))
    for definition in database.generatable_definitions:
        message = generator.generate(definition)
        assert 1 <= len(message.payload) <= (8 if definition.packet_type == "Single" else 223)
