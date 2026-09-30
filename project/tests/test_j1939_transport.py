import pytest

from inference.j1939_tp import reassemble_j1939_tp
from inference.loader import PhysicalFrame, reconstruct_logical_messages
from simulator.j1939.transport import fragment_transport, reassemble_transport


@pytest.mark.parametrize("destination", [0xFF, 0x17])
def test_j1939_tp_bam_and_rts_cts_fragment_reassemble(destination):
    payload = bytes((index * 13) % 256 for index in range(200))
    frames = fragment_transport(payload, 0xFEEE, 0x01, destination)
    assert all(len(frame.data) == 8 for frame in frames)
    rebuilt = reassemble_transport(frames)
    assert rebuilt.payload == payload
    assert rebuilt.target_pgn == 0xFEEE
    physical = [
        PhysicalFrame(1, 0.1, 7, index, frame.can_id, frame.data)
        for index, frame in enumerate(frames)
    ]
    logical = reconstruct_logical_messages(physical, "j1939")
    assert len(logical) == 1
    assert logical[0].payload == payload
    assert logical[0].pgn == 0xFEEE
    assert logical[0].transport == "j1939_tp"

