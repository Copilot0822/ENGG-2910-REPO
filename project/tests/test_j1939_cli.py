import subprocess
import sys

import pytest

from inference import cli as inference_cli
from simulator import cli as simulator_cli
from simulator.j1939.loader import load_j1939_database


def test_simulator_cli_uses_environment_and_explicit_override(monkeypatch, tmp_path, synthetic_j1939_path):
    monkeypatch.setenv("J1939_DATABASE_PATH", str(synthetic_j1939_path))
    arguments = simulator_cli.build_parser().parse_args([
        "--protocol", "j1939", "--samples", "1", "--output", str(tmp_path / "env"),
    ])
    assert simulator_cli.run(arguments)["number_of_signals_loaded"] == 10
    monkeypatch.setenv("J1939_DATABASE_PATH", str(tmp_path / "missing.json"))
    arguments.database = synthetic_j1939_path
    arguments.output = tmp_path / "explicit"
    assert simulator_cli.run(arguments)["number_of_signals_loaded"] == 10


@pytest.mark.parametrize("command", ["experiment", "full-library"])
@pytest.mark.parametrize("explicit", [False, True])
def test_experiment_cli_preserves_database_selection(
    monkeypatch, tmp_path, synthetic_j1939_path, command, explicit,
):
    from simulator.j1939 import experiments, full_library

    selected = []

    def run(**kwargs):
        selected.append(load_j1939_database(kwargs["database_path"]).path)
        return []

    if command == "experiment":
        monkeypatch.setattr(experiments, "run_j1939_experiment", run)
    else:
        monkeypatch.setattr(full_library, "run_full_library_experiment", run)
    monkeypatch.setenv(
        "J1939_DATABASE_PATH",
        str(tmp_path / "missing.json") if explicit else str(synthetic_j1939_path),
    )
    argv = [command, "--output", str(tmp_path / "output")]
    if command == "experiment":
        argv.extend(["--protocol", "j1939"])
    if explicit:
        argv.extend(["--database", str(synthetic_j1939_path)])
    inference_cli.main(argv)
    assert selected == [synthetic_j1939_path.resolve()]


@pytest.mark.parametrize("module,args", [
    ("simulator", ["--protocol", "j1939"]),
    ("inference", ["experiment", "--protocol", "j1939"]),
    ("inference", ["full-library"]),
])
def test_all_database_dependent_clis_fail_clearly_when_absent(monkeypatch, tmp_path, module, args):
    monkeypatch.setenv("J1939_DATABASE_PATH", str(tmp_path / "missing.json"))
    result = subprocess.run(
        [sys.executable, "-m", module, *args, "--output", str(tmp_path / "output")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Licensed J1939 database not found" in result.stderr
    assert "--database or J1939_DATABASE_PATH" in result.stderr
