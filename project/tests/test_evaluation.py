import json

from inference.evaluate import evaluate_results


def _candidate(start_bit):
    return {
        "pgn": 127488,
        "source": 1,
        "payload_length": 8,
        "start_bit": start_bit,
        "bit_length": 16,
        "signed": False,
        "byte_order": "little_endian",
        "scale": 0.25,
        "offset": 0.0,
        "r2": 1.0,
        "rmse": 0.0,
        "mae": 0.0,
        "observation_count": 50,
        "raw_variance": 1.0,
        "target_variance": 1.0,
        "normalized_rmse": 0.0,
        "pairing_fraction": 1.0,
        "ranking_score": 1.05,
    }


def test_evaluation_distinguishes_success_and_failure(tmp_path):
    truth = {
        "signals": {
            "signal_1": {
                "pgn": 127488,
                "start_bit": 8,
                "bit_length": 16,
                "signed": False,
                "scale": 0.25,
                "offset": 0,
            }
        }
    }
    inferred = {
        "metadata": {"inference_mode": "aligned", "supported_bit_lengths": [8, 16, 32]},
        "signals": {
            "Speed [signal_1]": {
                "signal_id": "signal_1",
                "candidates": [_candidate(0), _candidate(8)],
            }
        },
    }
    truth_path = tmp_path / "truth.json"
    inferred_path = tmp_path / "inferred.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    inferred_path.write_text(json.dumps(inferred), encoding="utf-8")
    result = evaluate_results(inferred_path, truth_path)
    assert result["summary"]["top1_location_accuracy"] == 0
    assert result["summary"]["top3_location_accuracy"] == 1
    assert result["summary"]["full_encoding_accuracy"] == 0

    inferred["signals"]["Speed [signal_1]"]["candidates"] = [_candidate(8)]
    inferred_path.write_text(json.dumps(inferred), encoding="utf-8")
    result = evaluate_results(inferred_path, truth_path)
    assert result["summary"]["top1_location_accuracy"] == 1
    assert result["summary"]["full_encoding_accuracy"] == 1

