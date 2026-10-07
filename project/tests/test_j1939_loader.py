import pytest

from simulator.j1939 import loader
from simulator.j1939.loader import load_j1939_database
from synthetic_j1939 import write_synthetic_database


def test_j1939_database_counts_and_spn_metadata(j1939_database):
    assert len(j1939_database.messages) == 4
    assert len({message.pgn for message in j1939_database.messages}) == 4
    assert j1939_database.signal_count == 10
    assert j1939_database.aligned_signal_count == 6
    assert len(j1939_database.single_frame_messages) == 2
    assert len(j1939_database.transport_messages) == 2
    assert all(signal.spn >= 0 for message in j1939_database.messages for signal in message.signals)


def test_hashless_local_json_loads(tmp_path):
    path = write_synthetic_database(tmp_path / "local.json")
    assert load_j1939_database(path).signal_count == 10


def test_database_path_precedence(monkeypatch, tmp_path, synthetic_j1939_path):
    default = write_synthetic_database(tmp_path / "default.json")
    monkeypatch.setattr(loader, "DEFAULT_J1939_DATABASE", default)
    monkeypatch.delenv("J1939_DATABASE_PATH", raising=False)
    assert load_j1939_database().path == default.resolve()
    monkeypatch.setenv("J1939_DATABASE_PATH", str(synthetic_j1939_path))
    assert load_j1939_database().path == synthetic_j1939_path.resolve()
    assert load_j1939_database(default).path == default.resolve()


def test_missing_configured_database_does_not_fall_back(monkeypatch, tmp_path, synthetic_j1939_path):
    monkeypatch.setattr(loader, "DEFAULT_J1939_DATABASE", synthetic_j1939_path)
    monkeypatch.setenv("J1939_DATABASE_PATH", str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match="--database or J1939_DATABASE_PATH"):
        load_j1939_database()
    monkeypatch.setenv("J1939_DATABASE_PATH", str(synthetic_j1939_path))
    with pytest.raises(FileNotFoundError, match="Licensed J1939 database not found"):
        load_j1939_database(tmp_path / "absent.json")


def test_missing_default_database_has_actionable_message(monkeypatch, tmp_path):
    monkeypatch.delenv("J1939_DATABASE_PATH", raising=False)
    monkeypatch.setattr(loader, "DEFAULT_J1939_DATABASE", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="does not redistribute the CSS Electronics"):
        load_j1939_database()


@pytest.mark.integration
def test_licensed_local_database_loads(licensed_j1939_database):
    database = licensed_j1939_database
    assert database.messages
    assert database.signal_count == sum(len(message.signals) for message in database.messages)
    assert database.aligned_signal_count == sum(
        signal.aligned_8_16_32 for message in database.messages for signal in message.signals
    )
