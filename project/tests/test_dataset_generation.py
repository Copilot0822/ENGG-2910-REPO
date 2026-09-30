import csv
from collections import Counter

from simulator.dataset_generator import DatasetGenerator
from simulator.output import write_outputs


def test_physical_frames_and_inference_headers(database, tmp_path):
    result = DatasetGenerator(database, 42).random_mode(8, 5)
    assert result.frames
    assert all(len(frame.data) <= 8 for frame in result.frames)
    write_outputs(tmp_path, database, result)
    with (tmp_path / "can_frames.csv").open(newline="", encoding="utf-8") as handle:
        frame_header = next(csv.reader(handle))
    with (tmp_path / "known_values.csv").open(newline="", encoding="utf-8") as handle:
        known_header = next(csv.reader(handle))
    forbidden = {"pgn", "start_bit", "bit_length", "scale", "offset", "signed"}
    assert not forbidden.intersection(frame_header)
    assert not forbidden.intersection(known_header)
    assert all(
        str(definition.pgn) not in observation.signal_name
        for observation in result.observations
        for definition in database.definitions
    )
    assert frame_header == [
        "sample_id",
        "timestamp",
        "message_id",
        "frame_index",
        "can_id",
        "data_hex",
    ]


def test_coverage_reaches_every_eligible_target(database):
    result = DatasetGenerator(database, 17).coverage_mode(1, 24)
    counts = Counter(item.signal_id for item in result.observations)
    expected = {
        field.signal_id
        for definition in database.generatable_definitions
        for field in definition.eligible_fields
    }
    assert expected
    assert expected == set(counts)
    assert min(counts.values()) >= 1
