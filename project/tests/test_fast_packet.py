import pytest

from simulator.fast_packet import fragment_fast_packet, reassemble_fast_packet


@pytest.mark.parametrize("length", [1, 6, 7, 13, 20, 134, 223])
def test_fragmentation_and_exact_reassembly(length):
    payload = bytes((index * 17) % 256 for index in range(length))
    frames = fragment_fast_packet(payload, sequence_id=5)
    assert all(len(frame) == 8 for frame in frames)
    assert frames[0][0] == 0xA0
    assert frames[0][1] == length
    assert reassemble_fast_packet(frames) == payload


def test_bad_continuation_counter_is_rejected():
    frames = fragment_fast_packet(bytes(range(20)), 2)
    altered = list(frames)
    altered[1] = bytes([altered[1][0] + 1]) + altered[1][1:]
    with pytest.raises(ValueError):
        reassemble_fast_packet(altered)

