import csv
import json
import subprocess
from dataclasses import replace

import pytest

from inference.evaluate import write_evaluation
from inference.full_library import _config_digest, write_batch_checkpoint
from inference.full_library_evaluate import evaluate_full_library, write_evaluation_json
from inference.output_privacy import protect_private_outputs, public_signal_results
from simulator.j1939.generator import J1939DatasetGenerator
from simulator.j1939.loader import load_j1939_database
from simulator.j1939.full_library import _accounting, _public_accounting
from simulator.j1939 import full_library as orchestration
from simulator.j1939 import experiments
from simulator.j1939.output import write_j1939_outputs


@pytest.mark.parametrize("private_derived_output", [True, False])
def test_simulation_classification_protects_licensed_inputs(
    j1939_database, tmp_path, private_derived_output,
):
    # Both test modes use invented rows; True exercises the licensed-source policy.
    database = replace(j1939_database, private_derived_output=private_derived_output)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    result = J1939DatasetGenerator(database, 42).coverage_mode(
        3, category="all_linear", include_transport=True,
    )
    metadata = write_j1939_outputs(tmp_path, database, result)
    for name in ("can_frames.csv", "known_values.csv"):
        ignored = subprocess.run(["git", "check-ignore", name], cwd=tmp_path).returncode == 0
        assert ignored == private_derived_output
    with (tmp_path / "known_values.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["signal_name"].startswith("signal_") and row["unit"] == "" for row in rows)
    text = (tmp_path / "known_values.csv").read_text(encoding="utf-8")
    for message in j1939_database.messages:
        for signal in message.signals:
            assert signal.name not in text
            assert signal.unit not in text
    assert metadata["output_schema_version"] == 3
    assert metadata["private_derived_output"] == private_derived_output
    assert not any("sha256" in key or "digest" in key for key in metadata)
    assert (tmp_path / "ground_truth.json").exists()
    assert (tmp_path / "signal_catalog.csv").exists()
    assert (tmp_path / ".gitignore").exists()


def test_custom_output_directory_is_protected_even_outside_project(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    output = tmp_path / "custom"
    protect_private_outputs(output, private_derived_output=True)
    protect_private_outputs(output, private_derived_output=True)
    private_names = (
        "ground_truth.json", "signal_catalog.csv", "inferred_signals.json",
        "candidate_rankings.csv", "run_config.json", "checkpoints/batch_00000.json",
        "checkpoints/batch_00000.json.tmp", "can_frames.csv", "known_values.csv",
        "known_values_200.csv", "decoded_export.csv", "converted_definitions.json",
    )
    for name in private_names:
        result = subprocess.run(["git", "check-ignore", "custom/" + name], cwd=tmp_path)
        assert result.returncode == 0
    write_batch_checkpoint(output / "custom_cache.json", 0, _config_digest({}), {"signals": {}})
    for name in ("custom_cache.json", "custom_cache.json.tmp"):
        result = subprocess.run(["git", "check-ignore", "custom/" + name], cwd=tmp_path)
        assert result.returncode == 0
    for name in ("evaluation_results.json", "full_library_results.csv", "accuracy_by_bit_length.csv"):
        result = subprocess.run(["git", "check-ignore", "custom/" + name], cwd=tmp_path)
        assert result.returncode == 1
    content = (output / ".gitignore").read_text(encoding="utf-8")
    assert content.count("ground_truth*.json") == 1
    assert content.count("can_frames*.csv") == 1


def test_unrelated_inputs_and_synthetic_sources_do_not_receive_raw_data_ignores(tmp_path, synthetic_j1939_path):
    default = load_j1939_database(synthetic_j1939_path)
    invented = load_j1939_database(synthetic_j1939_path, private_derived_output=False)
    assert default.private_derived_output is True
    assert invented.private_derived_output is False
    assert invented.signal_count == 10
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    protect_private_outputs(tmp_path / "nmea")
    for name in ("can_frames.csv", "known_values.csv"):
        assert subprocess.run(["git", "check-ignore", "nmea/" + name], cwd=tmp_path).returncode == 1


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
    write_evaluation(path, {"summary": {"total_signals": 1, "definitions": [private],
                                       "signal_name": "Invented secret label"}, "signals": [private]})
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["signals"] == public_signal_results([private])
    assert document["summary"] == {"total_signals": 1}
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
    forbidden = {
        "signal_name", "name", "spn", "pgn", "true_pgn", "start_bit", "true_start_bit",
        "bit_length", "true_bit_length", "byte_order", "signed", "factor", "scale", "offset",
        "unit", "minimum", "maximum", "raw_min", "raw_max", "candidates", "truth",
        "transport_category", "true_alignment_category", "original_signal_id", "signal_mappings",
    }
    assert not forbidden.intersection(rows[0])
    assert "Invented private label" not in path.read_text(encoding="utf-8")
    summary = tmp_path / "summary.json"
    write_evaluation_json(summary, evaluation)
    assert "rows" not in json.loads(summary.read_text(encoding="utf-8"))


def test_public_serializers_reject_definition_payloads_inside_metric_fields(tmp_path):
    definition = {"pgn": 100, "spn": 900001, "start_bit": 0, "bit_length": 8,
                  "factor": 0.5, "offset": -7, "name": "Invented private label"}
    assert public_signal_results([{
        "signal_id": "original_database_id", "candidate_count": definition,
        "correct_pgn": 100, "result_status": "Invented private label",
        "top1_location_correct": True,
    }]) == [{"signal_id": "signal_000001", "top1_location_correct": True}]
    path = tmp_path / "summary.json"
    write_evaluation_json(path, {
        "summary": {"signals": 1, "candidate_regressions": definition, "truth": definition},
        "by_bit_length": {"8": {"signals": 1, "truth": definition}, "Invented private label": definition},
        "by_alignment": {"aligned_8_16_32": {"signals": 1, "signal_mappings": definition}},
        "by_transport": {"single_frame": {"signals": 1, "original_signal_id": "original_database_id"}},
        "ambiguity_rule": "Invented private label", "evaluation_runtime_seconds": 0.1,
        "rows": [definition],
    })
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["summary"] == {"signals": 1}
    assert document["by_bit_length"] == {"8": {"signals": 1}}
    assert document["by_alignment"] == {"aligned_8_16_32": {"signals": 1}}
    assert document["by_transport"] == {"single_frame": {"signals": 1}}
    assert "Invented private label" not in path.read_text(encoding="utf-8")
    assert "original_database_id" not in path.read_text(encoding="utf-8")


def test_public_full_library_accounting_omits_original_database_ids(j1939_database):
    accounting = _accounting(j1939_database)
    accounting["unsupported_signal_ids"] = ["original_database_id"]
    public = _public_accounting(accounting)
    assert "unsupported_signal_ids" not in public
    assert public["total_j1939_signals"] == 10


def test_resumed_licensed_run_reinstalls_raw_dataset_guards(j1939_database, tmp_path, monkeypatch):
    database = replace(j1939_database, private_derived_output=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    output = tmp_path / "custom"
    simulated = output / "simulated"
    result = J1939DatasetGenerator(database, 42).coverage_mode(
        3, category="all_linear", include_transport=True,
    )
    write_j1939_outputs(simulated, database, result)
    # Mimic a copied dataset whose output-local ignore file was left behind.
    (simulated / ".gitignore").unlink()
    monkeypatch.setattr(orchestration, "load_j1939_database", lambda path: database)

    def stop_before_discovery(*args, **kwargs):
        raise RuntimeError("stop before discovery")

    def unexpected_regeneration(*args, **kwargs):
        raise AssertionError("completed simulation should be reused")

    monkeypatch.setattr(orchestration, "infer_full_library", stop_before_discovery)
    monkeypatch.setattr(orchestration, "write_j1939_outputs", unexpected_regeneration)
    with pytest.raises(RuntimeError, match="stop before discovery"):
        orchestration.run_full_library_experiment(
            database_path=None, output_directory=output, resume=True,
        )
    for name in ("can_frames.csv", "known_values.csv", "ground_truth.json", "signal_catalog.csv"):
        assert subprocess.run(["git", "check-ignore", "custom/simulated/" + name], cwd=tmp_path).returncode == 0
    accounting = json.loads((output / "metadata/full_library_accounting.json").read_text())
    assert "unsupported_signal_ids" not in accounting


def test_licensed_experiment_protects_observation_subsets(j1939_database, tmp_path, monkeypatch):
    database = replace(j1939_database, private_derived_output=True)
    monkeypatch.setattr(experiments, "load_j1939_database", lambda path: database)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    output = tmp_path / "custom"
    experiments.run_j1939_experiment(
        database_path=None, output_directory=output, observation_counts=(3,),
        seed=42, max_signals=1, min_pairs=3,
    )
    assert (output / "known_values_3.csv").is_file()
    for name in ("known_values_3.csv", "dataset/can_frames.csv", "dataset/known_values.csv"):
        assert subprocess.run(["git", "check-ignore", "custom/" + name], cwd=tmp_path).returncode == 0
    assert subprocess.run(["git", "check-ignore", "custom/recovery_vs_observations.csv"], cwd=tmp_path).returncode == 1


def test_checkpoint_configuration_compares_full_values_without_hashes():
    signature = _config_digest({"top_k": 10, "mode": "aligned"})
    assert json.loads(signature) == {"top_k": 10, "mode": "aligned"}
    assert signature != _config_digest({"top_k": 5, "mode": "aligned"})
