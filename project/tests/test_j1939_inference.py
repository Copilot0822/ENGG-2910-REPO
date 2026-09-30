from collections import Counter

from inference.evaluate import evaluate_results
from inference.infer import infer_signals
from inference.output import write_inference_outputs
from simulator.j1939.generator import J1939DatasetGenerator
from simulator.j1939.models import J1939SimulationResult
from simulator.j1939.output import write_j1939_outputs


def test_j1939_signal_recovery_uses_only_clean_csvs(j1939_database, tmp_path):
    definition = next(
        message
        for message in j1939_database.single_frame_messages
        if any(
            signal.aligned_8_16_32 and signal.bit_length == 16
            for signal in message.signals
        )
    )
    generator = J1939DatasetGenerator(j1939_database, 123)
    result = J1939SimulationResult(
        options={
            "protocol": "j1939",
            "mode": "targeted_test",
            "category": "aligned_8_16_32",
            "include_transport": False,
            "seed": 123,
        }
    )
    for sample in range(30):
        generator._emit_sample(result, sample, [definition], "aligned_8_16_32")
    result.time_step_count = 30
    full = tmp_path / "full"
    write_j1939_outputs(full, j1939_database, result)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "can_frames.csv").write_bytes((full / "can_frames.csv").read_bytes())
    (clean / "known_values.csv").write_bytes((full / "known_values.csv").read_bytes())
    assert {path.name for path in clean.iterdir()} == {"can_frames.csv", "known_values.csv"}

    selected = {
        observation.signal_name
        for observation in result.observations
        if observation.signal_id
        == next(
            signal.signal_id
            for signal in definition.signals
            if signal.aligned_8_16_32 and signal.bit_length == 16
        )
    }
    run = infer_signals(
        clean / "can_frames.csv",
        clean / "known_values.csv",
        protocol="j1939",
        mode="aligned",
        min_pairs=10,
        selected_signals=selected,
    )
    assert run.signals[0].candidates
    output = tmp_path / "inference"
    write_inference_outputs(output, run)
    evaluation = evaluate_results(output / "inferred_signals.json", full / "ground_truth.json")
    assert evaluation["summary"]["top1_location_accuracy"] == 1
    assert evaluation["summary"]["full_encoding_accuracy"] == 1


def test_j1939_tp_signal_is_reassembled_and_recovered(j1939_database, tmp_path):
    definition = next(
        message
        for message in j1939_database.transport_messages
        if any(
            signal.aligned_8_16_32 and signal.bit_length == 16
            for signal in message.signals
        )
    )
    target = next(
        signal
        for signal in definition.signals
        if signal.aligned_8_16_32 and signal.bit_length == 16
    )
    generator = J1939DatasetGenerator(j1939_database, 321)
    result = J1939SimulationResult(
        options={
            "protocol": "j1939",
            "mode": "targeted_tp_test",
            "category": "aligned_8_16_32",
            "include_transport": True,
            "seed": 321,
        }
    )
    for sample in range(30):
        generator._emit_sample(result, sample, [definition], "aligned_8_16_32")
    result.time_step_count = 30
    dataset = tmp_path / "tp"
    write_j1939_outputs(dataset, j1939_database, result)
    selected = {
        observation.signal_name
        for observation in result.observations
        if observation.signal_id == target.signal_id
    }

    run = infer_signals(
        dataset / "can_frames.csv",
        dataset / "known_values.csv",
        protocol="j1939",
        mode="aligned",
        min_pairs=10,
        selected_signals=selected,
    )
    assert run.metadata["number_of_logical_messages"] == 30
    assert len(result.frames) > 30
    output = tmp_path / "tp_inference"
    write_inference_outputs(output, run)
    evaluation = evaluate_results(
        output / "inferred_signals.json", dataset / "ground_truth.json"
    )
    assert evaluation["summary"]["top1_location_accuracy"] == 1
