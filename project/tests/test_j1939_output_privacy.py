import csv
import json
import subprocess

from inference.evaluate import write_evaluation
from inference.full_library import _config_digest, write_batch_checkpoint
from inference.full_library_evaluate import evaluate_full_library, write_evaluation_json
from inference.output_privacy import protect_private_outputs, public_signal_results
from simulator.j1939.generator import J1939DatasetGenerator
from simulator.j1939.output import write_j1939_outputs


def test_clean_simulation_csvs_do_not_export_names_units_or_checksums(j1939_database, tmp_path):
    result = J1939DatasetGenerator(j1939_database, 42).coverage_mode(
        3, category="all_linear", include_transport=True,
    )
    metadata = write_j1939_outputs(tmp_path, j1939_database, result)
    with (tmp_path / "known_values.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["signal_name"].startswith("signal_") and row["unit"] == "" for row in rows)
    text = (tmp_path / "known_values.csv").read_text(encoding="utf-8")
    for message in j1939_database.messages:
        for signal in message.signals:
            assert signal.name not in text
            assert signal.unit not in text
    assert metadata["output_schema_version"] == 2
    assert not any("sha256" in key or "digest" in key for key in metadata)
    assert (tmp_path / "ground_truth.json").exists()
    assert (tmp_path / "signal_catalog.csv").exists()
    assert (tmp_path / ".gitignore").exists()


def test_custom_output_directory_is_protected_even_outside_project(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    output = tmp_path / "custom"
    protect_private_outputs(output)
    protect_private_outputs(output)
    private_names = (
        "ground_truth.json", "signal_catalog.csv", "inferred_signals.json",
        "candidate_rankings.csv", "run_config.json", "checkpoints/batch_00000.json",
    )
    for name in private_names:
        result = subprocess.run(["git", "check-ignore", "custom/" + name], cwd=tmp_path)
        assert result.returncode == 0
    write_batch_checkpoint(output / "custom_cache.json", 0, _config_digest({}), {"signals": {}})
    for name in ("custom_cache.json", "custom_cache.json.tmp"):
        result = subprocess.run(["git", "check-ignore", "custom/" + name], cwd=tmp_path)
        assert result.returncode == 0
    for name in ("can_frames.csv", "known_values.csv", "evaluation_results.json", "full_library_results.csv"):
        result = subprocess.run(["git", "check-ignore", "custom/" + name], cwd=tmp_path)
        assert result.returncode == 1
    content = (output / ".gitignore").read_text(encoding="utf-8")
    assert content.count("ground_truth*.json") == 1


def test_public_evaluation_uses_opaque_ids_and_discards_definition_fields(tmp_path):
    private = {
        "signal_id": "untrusted_named_identifier", "signal_name": "Invented secret label",
        "pgn": 100, "spn": 900001, "start_bit": 0, "bit_length": 16,
        "byte_order": "little_endian", "signed": False, "factor": 0.5, "offset": 7,
        "unit": "private_unit", "truth": {"name": "private"},
        "candidates": [{"pgn": 100}], "top1_location_correct": True,
        "full_encoding_correct": True, "scale_absolute_error": 0,
    }
    assert public_signal_results([private]) == [{
        "signal_id": "signal_000001", "top1_location_correct": True,
        "full_encoding_correct": True, "scale_absolute_error": 0,
    }]
    path = tmp_path / "evaluation_results.json"
    write_evaluation(path, {"summary": {"total_signals": 1}, "signals": [private]})
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["signals"] == public_signal_results([private])
    assert "Invented secret label" not in path.read_text(encoding="utf-8")


def test_full_evaluation_csv_keeps_only_opaque_metrics(tmp_path):
    truth = {"signals": {"j1939_signal_000001": {
        "pgn": 100, "start_bit": 0, "bit_length": 8,
        "byte_order": "little_endian", "signed": False, "scale": 1, "offset": 0,
        "raw_min": 0, "raw_max": 100, "minimum": 0, "maximum": 100,
        "transport_category": "single_frame",
    }}}
    inferred = {"signals": {"Invented private label": {
        "signal_id": "j1939_signal_000001", "candidate_count": 64,
        "candidates": [{"pgn": 100, "start_bit": 0, "bit_length": 8,
                        "byte_order": "little_endian", "signed": False,
                        "scale": 1, "offset": 0}],
    }}}
    truth_path, inferred_path = tmp_path / "truth.json", tmp_path / "inferred.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    inferred_path.write_text(json.dumps(inferred), encoding="utf-8")
    path = tmp_path / "results.csv"
    evaluation = evaluate_full_library(inferred_path, truth_path, path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["signal_id"] == "signal_000001"
    assert "signal_name" not in rows[0]
    assert "true_bit_length" not in rows[0]
    assert "transport_category" not in rows[0]
    assert "pgn" not in rows[0]
    assert "Invented private label" not in path.read_text(encoding="utf-8")
    summary = tmp_path / "summary.json"
    write_evaluation_json(summary, evaluation)
    assert "rows" not in json.loads(summary.read_text(encoding="utf-8"))


def test_checkpoint_configuration_compares_full_values_without_hashes():
    signature = _config_digest({"top_k": 10, "mode": "aligned"})
    assert json.loads(signature) == {"top_k": 10, "mode": "aligned"}
    assert signature != _config_digest({"top_k": 5, "mode": "aligned"})
