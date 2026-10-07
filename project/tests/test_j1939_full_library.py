import json

import numpy as np

from inference.candidate import extract_raw
from inference.full_library import (
    CandidateVectorCache,
    FullCandidateSpec,
    FullStream,
    FullTarget,
    infer_full_library,
    load_batch_checkpoint,
    make_stream_batches,
    write_batch_checkpoint,
)
from inference.full_library_evaluate import (
    evaluate_full_library,
    signedness_is_inherently_ambiguous,
    width_is_inherently_ambiguous,
)
from simulator.j1939.full_library import FULL_CANDIDATE_LENGTHS, _accounting
from simulator.j1939.generator import J1939DatasetGenerator
from simulator.j1939.output import write_j1939_outputs


def test_candidate_cache_reuses_vectors_and_extracts_larger_fields():
    payloads = np.asarray(
        [
            list(bytes.fromhex("0123456789ABCDEF1032")),
            list(bytes.fromhex("FFEEDDCCBBA987654321")),
        ],
        dtype=np.uint8,
    )
    cache = CandidateVectorCache(payloads, pgn=61444)
    for width, start in ((40, 3), (64, 7)):
        spec = FullCandidateSpec(start, width, False, "little_endian", 0)
        first = cache.vector(spec)
        second = cache.vector(spec)
        expected = np.asarray(
            [
                extract_raw(
                    bytes(row),
                    start,
                    width,
                    signed=False,
                    byte_order="little_endian",
                )
                for row in payloads
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)
    assert cache.cache_hits == 2
    assert cache.cache_misses == 2


def test_full_library_accounting_reconciles(j1939_database):
    accounting = _accounting(j1939_database)
    assert accounting["total_j1939_signals"] == 10
    assert accounting["category_a_aligned_8_16_32"] == 6
    assert accounting["category_b_all_signals_1_through_32"] == 8
    assert accounting["category_c_larger_than_32_total"] == 2
    assert accounting["category_c_larger_than_32_inferable"] == 2
    assert accounting["unsupported_for_inference"] == 0
    assert accounting["partition_reconciliation"]["sum"] == 10
    assert set(FULL_CANDIDATE_LENGTHS) >= {40, 56, 64}


def _fake_stream(pgn, targets):
    return FullStream(
        pgn,
        1,
        8,
        np.asarray([0, 1, 2], dtype=np.int64),
        np.zeros((3, 8), dtype=np.uint8),
        [
            FullTarget(
                f"target_{pgn}_{index}",
                f"j1939_signal_{pgn + index:06d}",
                "",
                np.asarray([0.0, 1.0, 2.0]),
            )
            for index in range(targets)
        ],
    )


def test_stream_batches_are_deterministic_and_keep_streams_together():
    streams = [_fake_stream(1, 2), _fake_stream(2, 2), _fake_stream(3, 1)]
    batches = make_stream_batches(streams, 3)
    assert [[stream.pgn for stream in batch] for batch in batches] == [[1], [2, 3]]
    assert sum(len(stream.targets) for batch in batches for stream in batch) == 5


def test_checkpoint_round_trip_and_configuration_guard(tmp_path):
    path = tmp_path / "batch.json"
    write_batch_checkpoint(path, 3, "abc", {"signals": {"x": {}}, "metrics": {}})
    loaded = load_batch_checkpoint(path, "abc")
    assert loaded["batch_index"] == 3
    assert loaded["signals"] == {"x": {}}
    try:
        load_batch_checkpoint(path, "different")
    except ValueError as exc:
        assert "configuration mismatch" in str(exc)
    else:
        raise AssertionError("configuration mismatch was not rejected")


def test_ambiguity_classification_uses_complete_valid_range():
    ambiguous = {
        "bit_length": 8,
        "raw_min": 0,
        "raw_max": 255,
        "minimum": 0,
        "maximum": 125,
        "scale": 1,
        "offset": 0,
    }
    identifiable = {
        "bit_length": 8,
        "raw_min": 0,
        "raw_max": 255,
        "minimum": 0,
        "maximum": 250,
        "scale": 1,
        "offset": 0,
    }
    assert signedness_is_inherently_ambiguous(ambiguous)
    assert not signedness_is_inherently_ambiguous(identifiable)
    assert width_is_inherently_ambiguous(ambiguous)
    assert not width_is_inherently_ambiguous(identifiable)


def test_full_result_rows_reconcile_with_inferred_targets(tmp_path):
    truth = {
        "signals": {
            "j1939_signal_000001": {
                "pgn": 100,
                "start_bit": 0,
                "bit_length": 8,
                "byte_order": "little_endian",
                "signed": False,
                "scale": 1,
                "offset": 0,
                "raw_min": 0,
                "raw_max": 100,
                "minimum": 0,
                "maximum": 100,
                "transport_category": "single_frame",
            }
        }
    }
    candidate = {
        "pgn": 100,
        "start_bit": 0,
        "bit_length": 8,
        "byte_order": "little_endian",
        "signed": True,
        "scale": 1,
        "offset": 0,
    }
    inferred = {
        "metadata": {},
        "signals": {
            "opaque [j1939_signal_000001]": {
                "signal_id": "j1939_signal_000001",
                "candidate_count": 64,
                "candidates": [candidate],
            }
        },
    }
    truth_path = tmp_path / "truth.json"
    inferred_path = tmp_path / "inferred.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    inferred_path.write_text(json.dumps(inferred), encoding="utf-8")
    evaluation = evaluate_full_library(
        inferred_path, truth_path, tmp_path / "results.csv"
    )
    assert evaluation["summary"]["signals"] == 1
    assert evaluation["summary"]["ambiguous_signals"] == 1
    assert evaluation["summary"]["raw_full_encoding_accuracy"] == 0
    assert evaluation["summary"]["identifiable_full_encoding_accuracy"] == 1
    assert len(evaluation["rows"]) == 1


def test_cached_full_pipeline_recovers_from_clean_inputs(j1939_database, tmp_path):
    definition = next(
        message
        for message in j1939_database.single_frame_messages
        if any(
            signal.bit_length == 8 and signal.start_bit % 8 == 0
            for signal in message.signals
        )
    )
    target = next(
        signal
        for signal in definition.signals
        if signal.bit_length == 8 and signal.start_bit % 8 == 0
    )
    result = J1939DatasetGenerator(j1939_database, 42).targeted_coverage_mode(
        30,
        [definition],
        category="all_linear",
        include_transport=False,
    )
    dataset = tmp_path / "dataset"
    write_j1939_outputs(dataset, j1939_database, result)
    label = next(
        observation.signal_name
        for observation in result.observations
        if observation.signal_id == target.signal_id
    )
    output = tmp_path / "inference"
    metadata = infer_full_library(
        dataset / "can_frames.csv",
        dataset / "known_values.csv",
        output,
        mode="extended",
        bit_lengths=tuple(range(1, 17)),
        selected_signals={label},
        batch_size=1,
        candidate_batch_size=64,
        top_k=10,
    )
    evaluation = evaluate_full_library(
        output / "inferred_signals.json",
        dataset / "ground_truth.json",
        tmp_path / "results.csv",
    )
    assert metadata["number_of_target_signals"] == 1
    assert evaluation["summary"]["top1_location_accuracy"] == 1
