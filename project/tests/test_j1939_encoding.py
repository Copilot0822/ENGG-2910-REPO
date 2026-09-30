import random

import pytest

from simulator.j1939.encoding import decode_signal, encode_signal
from simulator.j1939.generator import J1939MessageGenerator, valid_raw_limits


@pytest.mark.parametrize("width", [8, 16, 32, 5])
def test_multiple_j1939_widths_round_trip(j1939_database, width):
    pair = next(
        (message, signal)
        for message in j1939_database.messages
        for signal in message.signals
        if signal.bit_length == width
    )
    message, signal = pair
    lower, upper = valid_raw_limits(signal)
    raw = (lower + upper) // 2
    payload = bytearray(message.length_bytes)
    encode_signal(payload, signal, raw)
    assert decode_signal(bytes(payload), signal) == raw
    physical = signal.factor * raw + signal.offset
    assert signal.minimum <= physical <= signal.maximum


def test_generator_respects_ranges_factors_offsets_and_round_trip(j1939_database):
    generator = J1939MessageGenerator(random.Random(91))
    chosen = [
        next(
            message
            for message in j1939_database.messages
            if any(signal.bit_length == width for signal in message.signals)
        )
        for width in (8, 16, 32, 12)
    ]
    assert any(signal.offset != 0 for message in j1939_database.messages for signal in message.signals)
    for message in chosen:
        generated = generator.generate(message, "all_linear")
        for value in generated.values:
            assert decode_signal(generated.payload, value.signal) == value.raw_value
            assert value.physical_value == value.raw_value * value.signal.factor + value.signal.offset
            assert value.signal.minimum <= value.physical_value <= value.signal.maximum

