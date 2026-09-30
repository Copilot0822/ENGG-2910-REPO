import csv
import random

from inference.infer import infer_signals
from simulator.bit_encoder import set_bits
from simulator.n2k_identifier import build_can_id


def _write_synthetic_dataset(directory, *, start_bit=8, bit_length=16, count=40):
    can_path = directory / "can_frames.csv"
    known_path = directory / "known_values.csv"
    rng = random.Random(42)
    can_id = build_can_id(2, 127488, 1)
    with can_path.open("w", newline="", encoding="utf-8") as can_handle, known_path.open(
        "w", newline="", encoding="utf-8"
    ) as known_handle:
        can_writer = csv.writer(can_handle)
        known_writer = csv.writer(known_handle)
        can_writer.writerow(
            ["sample_id", "timestamp", "message_id", "frame_index", "can_id", "data_hex"]
        )
        known_writer.writerow(
            ["sample_id", "timestamp", "observation_id", "signal_name", "value", "unit"]
        )
        for sample in range(count):
            raw = rng.randint(0, (1 << bit_length) - 4)
            payload = bytearray(rng.getrandbits(8) for _ in range(8))
            set_bits(payload, start_bit, bit_length, raw)
            can_writer.writerow(
                [sample, f"{sample / 10:.2f}", sample, 0, f"0x{can_id:08X}", bytes(payload).hex(" ")]
            )
            known_writer.writerow(
                [sample, f"{sample / 10:.2f}", sample, "Test Value [signal_999999]", raw * 0.25 - 3, "V"]
            )
    return can_path, known_path


def test_aligned_signal_is_recovered_using_only_two_input_files(tmp_path):
    can_path, known_path = _write_synthetic_dataset(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "can_frames.csv",
        "known_values.csv",
    ]
    run = infer_signals(can_path, known_path, mode="aligned", min_pairs=10, top_k=5)
    best = run.signals[0].candidates[0]
    assert (best.spec.pgn, best.spec.start_bit, best.spec.bit_length) == (127488, 8, 16)
    assert not best.spec.signed
    assert best.scale == pytest.approx(0.25)
    assert best.offset == pytest.approx(-3.0)


def test_arbitrary_bit_signal_is_recovered(tmp_path):
    can_path, known_path = _write_synthetic_dataset(
        tmp_path, start_bit=3, bit_length=16
    )
    run = infer_signals(can_path, known_path, mode="arbitrary", min_pairs=10, top_k=5)
    best = run.signals[0].candidates[0]
    assert (best.spec.start_bit, best.spec.bit_length, best.spec.signed) == (3, 16, False)


def test_insufficient_observations_are_rejected(tmp_path):
    can_path, known_path = _write_synthetic_dataset(tmp_path, count=5)
    run = infer_signals(can_path, known_path, min_pairs=10)
    assert run.signals[0].candidates == []


import pytest

