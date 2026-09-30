import csv

from inference.loader import load_logical_messages
from simulator.fast_packet import fragment_fast_packet
from simulator.n2k_identifier import build_can_id


def test_identifier_and_simulator_fast_packet_are_reconstructed(tmp_path):
    payload = bytes(range(26))
    can_id = build_can_id(2, 127489, 0x01)
    frames = fragment_fast_packet(payload, 5)
    path = tmp_path / "can_frames.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "timestamp", "message_id", "frame_index", "can_id", "data_hex"]
        )
        for index, frame in enumerate(frames):
            writer.writerow([7, "0.70", 12, index, f"0x{can_id:08X}", frame.hex(" ")])
    messages = load_logical_messages(path)
    assert len(messages) == 1
    message = messages[0]
    assert message.payload == payload
    assert message.is_fast_packet
    assert (message.pgn, message.source, message.priority) == (127489, 0x01, 2)

